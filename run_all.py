import argparse
import csv
import importlib
import time
from pathlib import Path

import numpy as np

from data_source_policy import (
    PolicyError,
    generate_from_policy,
    load_config,
    save_json,
    validate_and_normalize_config,
    validate_meta,
)
from algoritms.evaluation import compute_metrics
from algoritms.Pettitt import pettitt
from algoritms.SNHT import snht
from algoritms.chow import chow
from algoritms.MDL import mdl
from algoritms.SWAB import swab
from algoritms.GP import gp
from algoritms.Nyblom_test_GOBACK import nyblom
from algoritms.cusum_adapter import run_cusum
from algoritms.lstm_adapter import run_lstm
from algoritms.gru_adapter import run_gru
from algoritms.catboost_adapter import run_catboost
from algoritms.transformer_adapter import run_transformer

_buishand = importlib.import_module("algoritms.Buishand's_tests")

REGISTRY = {
    "Pettitt": pettitt,
    "SNHT": snht,
    "BuishandQ": _buishand.buishand_q,
    "BuishandRange": _buishand.buishand_range,
    "BuishandLR": _buishand.buishand_likelihood_ratio,
    "BuishandU": _buishand.buishand_u,
    "Chow": chow,
    "MDL": mdl,
    "SWAB": swab,
    "GP": gp,
    "Nyblom": nyblom,
    "CUSUM": run_cusum,
    "LSTM": run_lstm,
    "GRU": run_gru,
    "CatBoost": run_catboost,
    "Transformer": run_transformer,
    "LSTM_v2": run_lstm,
    "GRU_v2": run_gru,
    "Transformer_v2": run_transformer,
}


def _resolve_methods(methods_cfg) -> list[str]:
    """
    Возвращает список методов для запуска из значения конфига methods.

    Если methods = "all", возвращает все ключи REGISTRY.
    Иначе фильтрует список по наличию в REGISTRY.

    :param methods_cfg: значение ключа methods из конфига
    :return: список имён методов
    """
    if methods_cfg == "all":
        return list(REGISTRY.keys())
    if isinstance(methods_cfg, list):
        unknown = [m for m in methods_cfg if m not in REGISTRY]
        if unknown:
            print(f"Предупреждение: неизвестные методы проигнорированы: {unknown}")
        return [m for m in methods_cfg if m in REGISTRY]
    raise PolicyError(f"methods должен быть 'all' или списком; получено: {methods_cfg!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()

    try:
        raw_config = load_config(args.config)
        normalized = validate_and_normalize_config(raw_config)
        x_values, cp_labels, meta = generate_from_policy(normalized)
        validate_meta(meta)
    except PolicyError as exc:
        print(f"Policy check failed: {exc}")
        return 2

    dataset_name = normalized["dataset_name"]
    output_dir = Path(raw_config.get("output_dir", "results"))
    seed = normalized["generation"]["seed"]

    try:
        methods_to_run = _resolve_methods(raw_config.get("methods", "all"))
    except PolicyError as exc:
        print(f"Config error: {exc}")
        return 2

    method_params = raw_config.get("method_params") or {}
    eval_cfg = raw_config.get("evaluation") or {}
    margin = int(eval_cfg.get("margin", 50))
    nan_mae = float(eval_cfg.get("nan_mae", 9999.0))

    true_cps = np.where(cp_labels == 1)[0].tolist()

    base_kwargs = {
        "dataset_source": meta["dataset_source"],
        "generation_params": meta["generation_params"],
        "source_path": meta["source_path"],
    }

    print(f"Dataset : {dataset_name}")
    print(f"Length  : {len(x_values)}")
    print(f"True CPs: {len(true_cps)} at {true_cps}")
    print(f"Methods : {methods_to_run}")
    print(f"Margin  : ±{margin}")
    print()

    summary_rows: list[dict] = []

    for method_name in methods_to_run:
        fn = REGISTRY[method_name]
        call_kwargs = {**base_kwargs, **method_params.get(method_name, {})}

        t_start = time.time()
        try:
            change_points, scores, method_meta = fn(x_values, seed, **call_kwargs)
        except Exception as exc:
            print(f"{method_name:20s}  ERROR: {exc}")
            continue
        elapsed = time.time() - t_start

        try:
            validate_meta(method_meta)
        except PolicyError as exc:
            print(f"{method_name:20s}  meta validation failed: {exc}")
            continue

        metrics = compute_metrics(true_cps, change_points, scores, len(x_values), margin, nan_mae)

        method_dir = output_dir / dataset_name / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        save_json(method_dir / "metrics.json", metrics)
        save_json(method_dir / "predicted_cps.json", {"change_points": change_points})
        np.save(
            str(method_dir / "scores.npy"),
            np.asarray(scores) if scores is not None else np.zeros(0),
        )

        print(
            f"{method_name:20s}  CP={len(change_points):3d}"
            f"  P={metrics['precision']:.3f}"
            f"  R={metrics['recall']:.3f}"
            f"  F1={metrics['f1']:.3f}"
            f"  MAE={metrics['mae']:8.1f}"
            f"  ({elapsed:.1f}s)"
        )

        summary_rows.append(
            {
                "method": method_name,
                "n_pred": len(change_points),
                "n_true": len(true_cps),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "mae": metrics["mae"],
                "time_s": round(elapsed, 2),
            }
        )

    if summary_rows:
        summary_path = output_dir / "summary.csv"
        fieldnames = ["method", "n_pred", "n_true", "precision", "recall", "f1", "mae", "time_s"]
        with summary_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({k: row[k] for k in fieldnames})
        print(f"\nResults saved to {output_dir}/")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
