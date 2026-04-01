"""
eval_multi.py — многорядовая оценка методов CPD.

Запускает все (или выбранные) методы на N тестовых рядах с последовательными
сидами и усредняет метрики. Для LSTM/GRU/CatBoost веса загружаются из
weights_path (без переобучения), если файл существует.

Использование:
    python eval_multi.py
    python eval_multi.py --n-series 50 --test-seed-start 2000
    python eval_multi.py --methods LSTM,GRU,CatBoost --n-series 30
    python eval_multi.py --config configs/base.yaml --output-dir results/eval_multi
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from data_source_policy import (
    PolicyError,
    build_meta,
    load_config,
    save_json,
    validate_and_normalize_config,
    validate_meta,
)
from algoritms.data_utils import generate_dataset
from algoritms.evaluation import compute_metrics
from run_all import REGISTRY


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Многорядовая оценка CPD-методов")
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Путь к YAML-конфигу")
    parser.add_argument("--n-series", type=int, default=20,
                        help="Количество тестовых рядов (по умолчанию 20)")
    parser.add_argument("--test-seed-start", type=int, default=1000,
                        help="Первый сид тестовых рядов (по умолчанию 1000)")
    parser.add_argument("--methods", default="all",
                        help="Методы через запятую или 'all'")
    parser.add_argument("--output-dir", default=None,
                        help="Папка для результатов (по умолчанию results/eval_multi)")
    return parser.parse_args()


def _resolve_methods(methods_arg: str) -> list[str]:
    if methods_arg == "all":
        return list(REGISTRY.keys())
    names = [m.strip() for m in methods_arg.split(",") if m.strip()]
    unknown = [m for m in names if m not in REGISTRY]
    if unknown:
        print(f"Предупреждение: неизвестные методы проигнорированы: {unknown}")
    return [m for m in names if m in REGISTRY]


def main() -> int:
    args = _parse_args()

    try:
        raw_config = load_config(args.config)
        normalized = validate_and_normalize_config(raw_config)
    except PolicyError as exc:
        print(f"Policy check failed: {exc}")
        return 2

    gen = normalized["generation"]
    dataset_source = normalized["dataset_source"]
    method_params = raw_config.get("method_params") or {}
    eval_cfg = raw_config.get("evaluation") or {}
    margin = int(eval_cfg.get("margin", 50))
    nan_mae = float(eval_cfg.get("nan_mae", 9999.0))

    base_output = Path(raw_config.get("output_dir", "results"))
    output_dir = Path(args.output_dir) if args.output_dir else base_output / "eval_multi"
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = _resolve_methods(args.methods)
    n_series = args.n_series
    test_seed_start = args.test_seed_start

    print(f"Методы    : {methods}")
    print(f"Рядов     : {n_series}  (сиды {test_seed_start}..{test_seed_start + n_series - 1})")
    print(f"Параметры : length={gen['length']}, dt={gen['dt']}, D={gen['D']}, noise={gen['noise_type']}")
    print(f"Margin    : ±{margin}")
    print()

    per_series_rows: list[dict] = []
    method_accum: dict[str, list[dict]] = {m: [] for m in methods}

    for series_idx in range(n_series):
        test_seed = test_seed_start + series_idx
        x_values, cp_labels = generate_dataset(
            length=gen["length"],
            dt=gen["dt"],
            D=gen["D"],
            noise_type=gen["noise_type"],
            seed=test_seed,
        )
        true_cps = np.where(cp_labels == 1)[0].tolist()

        series_gen_params = {**gen, "seed": test_seed}
        meta_base = build_meta(dataset_source, series_gen_params)
        base_kwargs = {
            "dataset_source": meta_base["dataset_source"],
            "generation_params": meta_base["generation_params"],
            "source_path": meta_base["source_path"],
        }

        print(
            f"[{series_idx + 1}/{n_series}] seed={test_seed}  "
            f"len={len(x_values)}  true_cps={len(true_cps)} at {true_cps}"
        )

        for method_name in methods:
            fn = REGISTRY[method_name]
            call_kwargs = {**base_kwargs, **method_params.get(method_name, {})}

            t_start = time.time()
            try:
                change_points, scores, method_meta = fn(x_values, test_seed, **call_kwargs)
            except Exception as exc:
                print(f"  {method_name:20s}  ERROR: {exc}")
                continue
            elapsed = time.time() - t_start

            try:
                validate_meta(method_meta)
            except PolicyError as exc:
                print(f"  {method_name:20s}  meta validation failed: {exc}")
                continue

            metrics = compute_metrics(
                true_cps, change_points, scores, len(x_values), margin, nan_mae
            )

            per_series_rows.append({
                "method": method_name,
                "series_seed": test_seed,
                "n_pred": len(change_points),
                "n_true": len(true_cps),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mae": metrics["mae"],
                "roc_auc": metrics.get("roc_auc"),
                "pr_auc": metrics.get("pr_auc"),
                "time_s": round(elapsed, 2),
            })
            method_accum[method_name].append(metrics)

            auc_str = ""
            if metrics.get("roc_auc") is not None:
                auc_str = f"  ROC={metrics['roc_auc']:.3f}  PR={metrics['pr_auc']:.3f}"
            print(
                f"  {method_name:20s}  CP={len(change_points):3d}"
                f"  P={metrics['precision']:.3f}"
                f"  R={metrics['recall']:.3f}"
                f"  F1={metrics['f1']:.3f}"
                f"  MAE={metrics['mae']:8.1f}"
                f"{auc_str}"
                f"  ({elapsed:.1f}s)"
            )

        print()

    per_series_path = output_dir / "per_series.csv"
    if per_series_rows:
        fieldnames = [
            "method", "series_seed", "n_pred", "n_true",
            "precision", "recall", "f1", "mae",
            "roc_auc", "pr_auc", "time_s",
        ]
        with per_series_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_series_rows)

    summary_rows: list[dict] = []
    for method_name in methods:
        rows = method_accum[method_name]
        if not rows:
            continue
        arr_p = np.array([r["precision"] for r in rows])
        arr_r = np.array([r["recall"] for r in rows])
        arr_f1 = np.array([r["f1"] for r in rows])
        arr_mae = np.array([r["mae"] for r in rows])
        valid_mae = arr_mae[arr_mae < nan_mae]
        mean_mae = float(np.mean(valid_mae)) if len(valid_mae) > 0 else nan_mae

        roc_vals = [r.get("roc_auc") for r in rows if r.get("roc_auc") is not None]
        pr_vals = [r.get("pr_auc") for r in rows if r.get("pr_auc") is not None]
        mean_roc = round(float(np.mean(roc_vals)), 4) if roc_vals else None
        mean_pr = round(float(np.mean(pr_vals)), 4) if pr_vals else None

        summary_rows.append({
            "method": method_name,
            "n_series": len(rows),
            "mean_precision": round(float(arr_p.mean()), 4),
            "mean_recall": round(float(arr_r.mean()), 4),
            "mean_f1": round(float(arr_f1.mean()), 4),
            "std_f1": round(float(arr_f1.std()), 4),
            "mean_mae": round(mean_mae, 2),
            "mean_roc_auc": mean_roc,
            "mean_pr_auc": mean_pr,
        })

    summary_path = output_dir / "summary.csv"
    if summary_rows:
        fieldnames = [
            "method", "n_series", "mean_precision", "mean_recall",
            "mean_f1", "std_f1", "mean_mae", "mean_roc_auc", "mean_pr_auc",
        ]
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

    print(f"Результаты сохранены в {output_dir}/")
    print(f"  per_series.csv : {len(per_series_rows)} строк")
    print(f"  summary.csv    : {len(summary_rows)} методов")
    print()

    if summary_rows:
        print(f"{'Метод':<20} {'P':>6} {'R':>6} {'F1':>6} {'±F1':>6} {'MAE':>9} {'ROC':>6} {'PR':>6}")
        print("-" * 75)
        for row in sorted(summary_rows, key=lambda r: -r["mean_f1"]):
            roc_str = f"{row['mean_roc_auc']:>6.3f}" if row["mean_roc_auc"] is not None else "   N/A"
            pr_str = f"{row['mean_pr_auc']:>6.3f}" if row["mean_pr_auc"] is not None else "   N/A"
            print(
                f"{row['method']:<20} "
                f"{row['mean_precision']:>6.3f} "
                f"{row['mean_recall']:>6.3f} "
                f"{row['mean_f1']:>6.3f} "
                f"{row['std_f1']:>6.3f} "
                f"{row['mean_mae']:>9.1f} "
                f"{roc_str} "
                f"{pr_str}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
