"""
calibrate_threshold.py — калибровка порогов ML-методов на калибровочном наборе.

Генерирует N калибровочных рядов с теми же параметрами генерации, что и тест
(из конфига), запускает инференс с уже обученными весами (без переобучения),
перебирает пороги и находит оптимальный по F1. Обновляет threshold в конфиге.

Калибровочные ряды используют seeds из диапазона [cal_seed_start, cal_seed_start+N),
которые не пересекаются с тестовыми (по умолчанию 1000-1049) и обучающими
(seed+1, seed+2 от каждого метода).

Использование:
    python calibrate_threshold.py
    python calibrate_threshold.py --n-cal 30 --methods GRU_v2,LSTM_v2,Transformer_v2
    python calibrate_threshold.py --dry-run
"""

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from data_source_policy import load_config, build_meta, PolicyError, validate_and_normalize_config
from algoritms.data_utils import generate_dataset
from algoritms.evaluation import compute_metrics
from algoritms.lstm_cpd import nms_predictions
from run_all import REGISTRY

_ML_METHODS = {"LSTM", "GRU", "CatBoost", "LSTM_v2", "GRU_v2", "Transformer", "Transformer_v2"}
_THRESHOLD_GRID = np.round(np.linspace(0.05, 0.99, 95), 4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Калибровка порогов ML-методов")
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Путь к YAML-конфигу")
    parser.add_argument("--n-cal", type=int, default=20,
                        help="Число калибровочных рядов (по умолчанию 20)")
    parser.add_argument("--cal-seed-start", type=int, default=5000,
                        help="Первый сид калибровочных рядов (по умолчанию 5000)")
    parser.add_argument("--methods", default="all",
                        help="Методы через запятую или 'all' (только ML-методы)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Найти пороги, но не обновлять конфиг")
    return parser.parse_args()


def _resolve_methods(methods_arg: str, method_params: dict) -> list[str]:
    """Возвращает список ML-методов с существующими weights_path."""
    if methods_arg == "all":
        candidates = [m for m in REGISTRY if m in _ML_METHODS]
    else:
        candidates = [m.strip() for m in methods_arg.split(",") if m.strip()]

    result = []
    for name in candidates:
        if name not in REGISTRY:
            print(f"  Пропуск {name}: не в REGISTRY")
            continue
        cfg = method_params.get(name, {})
        wp = cfg.get("weights_path")
        if not wp:
            print(f"  Пропуск {name}: weights_path не задан в конфиге")
            continue
        if not Path(wp).exists():
            print(f"  Пропуск {name}: файл весов не найден: {wp}")
            continue
        result.append(name)
    return result


def _collect_scores(method_name: str, method_cfg: dict, cal_series: list,
                    cal_true_cps: list, base_kwargs: dict) -> list[np.ndarray] | None:
    """
    Запускает инференс метода на всех калибровочных рядах.

    Возвращает список массивов сырых скоров — по одному на ряд.
    Порог 0.001 гарантирует, что adapter возвращает все ненулевые скоры.
    """
    fn = REGISTRY[method_name]
    scores_list = []

    for idx, (series, true_cps) in enumerate(zip(cal_series, cal_true_cps)):
        call_kwargs = {
            **base_kwargs,
            **method_cfg,
            "threshold": 0.001,
        }
        try:
            _, scores_raw, _ = fn(series, 42, **call_kwargs)
        except Exception as exc:
            print(f"    [{idx + 1}] ERROR: {exc}")
            return None
        scores_list.append(scores_raw)
        print(f"    [{idx + 1}/{len(cal_series)}] scores OK, "
              f"true_cps={len(true_cps)}", flush=True)

    return scores_list


def _sweep_thresholds(
    scores_list: list[np.ndarray],
    true_cps_list: list[list[int]],
    nms_min_distance: int,
    margin: int,
    nan_mae: float,
) -> tuple[float, float, dict]:
    """
    Перебирает пороги на калибровочном наборе.

    Возвращает (best_threshold, best_mean_f1, {thr: mean_f1}).
    """
    best_thr = 0.5
    best_f1 = -1.0
    thr_to_f1 = {}

    for thr in _THRESHOLD_GRID:
        f1_values = []
        for scores_raw, true_cps in zip(scores_list, true_cps_list):
            scores_clean = np.nan_to_num(scores_raw, nan=0.0)
            preds = nms_predictions(scores_clean, threshold=float(thr),
                                    min_distance=nms_min_distance)
            pred_cps = np.where(preds == 1)[0].tolist()
            metrics = compute_metrics(true_cps, pred_cps, None,
                                      len(scores_raw), margin, nan_mae)
            f1_values.append(metrics["f1"])

        mean_f1 = float(np.mean(f1_values))
        thr_to_f1[float(thr)] = round(mean_f1, 4)

        if mean_f1 > best_f1:
            best_f1 = mean_f1
            best_thr = float(thr)

    return best_thr, best_f1, thr_to_f1


def _update_yaml_threshold(config_path: str, method_name: str, new_threshold: float) -> None:
    """
    Обновляет значение threshold для метода в YAML-файле.

    Ищет секцию method_name: и заменяет первую строку threshold: внутри неё.
    Не затрагивает структуру и комментарии остального файла.
    """
    lines = Path(config_path).read_text(encoding="utf-8").splitlines()
    new_lines = []
    in_section = False
    section_indent = None
    replaced = False

    for line in lines:
        stripped = line.lstrip()
        indent = len(line) - len(stripped)

        if not in_section and stripped.startswith(f"{method_name}:"):
            in_section = True
            section_indent = indent
            new_lines.append(line)
            continue

        if in_section:
            is_new_top_key = (
                stripped
                and not stripped.startswith("#")
                and indent <= section_indent
                and not stripped.startswith(f"{method_name}:")
            )
            if is_new_top_key:
                in_section = False
            elif not replaced and stripped.startswith("threshold:"):
                key_indent = " " * indent
                line = f"{key_indent}threshold: {new_threshold}"
                replaced = True

        new_lines.append(line)

    Path(config_path).write_text("\n".join(new_lines), encoding="utf-8")


def main() -> int:
    args = _parse_args()

    try:
        raw_config = load_config(args.config)
        normalized = validate_and_normalize_config(raw_config)
    except PolicyError as exc:
        print(f"Policy check failed: {exc}")
        return 2

    gen = normalized["generation"]
    method_params = raw_config.get("method_params") or {}
    eval_cfg = raw_config.get("evaluation") or {}
    margin = int(eval_cfg.get("margin", 50))
    nan_mae = float(eval_cfg.get("nan_mae", 9999.0))

    methods = _resolve_methods(args.methods, method_params)
    if not methods:
        print("Нет методов для калибровки (нет доступных весов).")
        return 0

    print(f"Методы     : {methods}")
    print(f"Cal рядов  : {args.n_cal}  (сиды {args.cal_seed_start}..{args.cal_seed_start + args.n_cal - 1})")
    print(f"Gen params : D={gen['D']}, dt={gen['dt']}, noise={gen['noise_type']}, length={gen['length']}")
    print(f"Margin     : ±{margin}")
    print(f"Dry-run    : {args.dry_run}")
    print()

    print("Генерация калибровочных рядов...", flush=True)
    cal_series = []
    cal_true_cps = []
    for i in range(args.n_cal):
        seed = args.cal_seed_start + i
        x, cp_labels = generate_dataset(
            length=gen["length"],
            dt=gen["dt"],
            D=gen["D"],
            noise_type=gen["noise_type"],
            seed=seed,
        )
        true_cps = np.where(cp_labels == 1)[0].tolist()
        cal_series.append(x)
        cal_true_cps.append(true_cps)
    print(f"  {args.n_cal} рядов готовы. "
          f"Среднее CP на ряд: {np.mean([len(c) for c in cal_true_cps]):.1f}\n",
          flush=True)

    base_kwargs = build_meta(normalized["dataset_source"], {**gen, "seed": 42})

    results = {}

    for method_name in methods:
        print(f"=== {method_name} ===", flush=True)
        method_cfg = dict(method_params.get(method_name, {}))
        nms_dist = int(method_cfg.get("nms_min_distance", 50))
        old_threshold = float(method_cfg.get("threshold", 0.5))

        t0 = time.time()
        scores_list = _collect_scores(
            method_name, method_cfg, cal_series, cal_true_cps, base_kwargs
        )
        if scores_list is None:
            print(f"  Пропуск: инференс завершился с ошибкой\n")
            continue

        print(f"  Инференс завершён за {time.time() - t0:.1f}s. Перебор порогов...", flush=True)
        best_thr, best_f1, thr_map = _sweep_thresholds(
            scores_list, cal_true_cps, nms_dist, margin, nan_mae
        )

        top5 = sorted(thr_map.items(), key=lambda kv: -kv[1])[:5]
        print(f"  Старый threshold : {old_threshold}")
        print(f"  Лучший threshold : {best_thr}  (mean F1 = {best_f1:.4f})")
        print(f"  Топ-5 порогов    : {[(t, f) for t, f in top5]}")

        results[method_name] = {"old": old_threshold, "new": best_thr, "best_f1": best_f1}

        if not args.dry_run:
            _update_yaml_threshold(args.config, method_name, best_thr)
            print(f"  Конфиг обновлён: {args.config}")
        print()

    print("=" * 55)
    print(f"{'Метод':<18} {'Старый':>8} {'Новый':>8} {'F1 cal':>8}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<18} {r['old']:>8.4f} {r['new']:>8.4f} {r['best_f1']:>8.4f}")

    if args.dry_run:
        print("\n[dry-run] Конфиг НЕ обновлён. Уберите --dry-run чтобы сохранить.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
