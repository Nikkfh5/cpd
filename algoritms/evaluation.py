"""
Модуль оценки качества обнаружения точек разладки (CPD).

Реализует жадное паросочетание истинных и предсказанных точек разладки
с допуском margin, а также вычисление стандартных метрик P/R/F1/MAE.
Дополнительно: pointwise ROC-AUC и PR-AUC по per-timestep скорам.
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score


def match_change_points(
    true_cps: list[int],
    pred_cps: list[int],
    margin: int,
) -> tuple[int, int, int, list[float]]:
    """
    Жадное паросочетание истинных и предсказанных точек разладки.

    Строит все пары (i, j, |dist|) с dist ≤ margin, сортирует по dist,
    жадно выбирает пары без повторения индексов.

    :param true_cps: список истинных точек разладки (0-based)
    :param pred_cps: список предсказанных точек разладки (0-based)
    :param margin: максимально допустимое расстояние для совпадения
    :return: (TP, FP, FN, ошибки локализации для TP-пар)
    """
    pairs = []
    for i, t in enumerate(true_cps):
        for j, p in enumerate(pred_cps):
            d = abs(t - p)
            if d <= margin:
                pairs.append((i, j, float(d)))

    pairs.sort(key=lambda x: x[2])

    used_true: set[int] = set()
    used_pred: set[int] = set()
    errors: list[float] = []

    for i, j, d in pairs:
        if i not in used_true and j not in used_pred:
            used_true.add(i)
            used_pred.add(j)
            errors.append(d)

    tp = len(used_true)
    fp = len(pred_cps) - len(used_pred)
    fn = len(true_cps) - len(used_true)
    return tp, fp, fn, errors


def compute_metrics(
    true_cps: list[int],
    pred_cps: list[int],
    scores,
    series_length: int,
    margin: int,
    nan_mae: float,
) -> dict:
    """
    Вычисляет метрики P / R / F1 / MAE для одного запуска алгоритма.

    При отсутствии предсказанных CP, если есть истинные, MAE = nan_mae.
    При отсутствии и истинных и предсказанных CP возвращает идеальные метрики.

    :param true_cps: истинные точки разладки (0-based)
    :param pred_cps: предсказанные точки разладки (0-based)
    :param scores: массив скоров (np.ndarray или None, не используется в метриках)
    :param series_length: длина временного ряда
    :param margin: допуск ±margin для совпадения CP
    :param nan_mae: MAE-заглушка когда нет ни одного предсказанного CP
    :return: словарь с ключами precision, recall, f1, mae, n_true, n_pred
    """
    if not true_cps and not pred_cps:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "mae": 0.0,
            "n_true": 0,
            "n_pred": 0,
        }

    tp, fp, fn, errors = match_change_points(true_cps, pred_cps, margin)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mae = float(np.mean(errors)) if errors else nan_mae

    result = {
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mae": round(mae, 4),
        "n_true": len(true_cps),
        "n_pred": len(pred_cps),
    }

    auc_metrics = compute_auc(true_cps, scores, series_length, margin)
    result.update(auc_metrics)

    return result


def compute_auc(
    true_cps: list[int],
    scores,
    series_length: int,
    margin: int,
) -> dict:
    """
    Вычисляет pointwise ROC-AUC и PR-AUC по per-timestep скорам.

    Строит бинарную разметку: позиции в пределах ±margin от любого
    истинного CP помечаются как 1, остальные — 0. Затем считает
    стандартные ROC-AUC и PR-AUC (average precision) по этой разметке
    и массиву скоров.

    :param true_cps: истинные точки разладки (0-based)
    :param scores: np.ndarray per-timestep скоров или None
    :param series_length: длина временного ряда
    :param margin: допуск ±margin для расширения зоны CP
    :return: словарь с ключами roc_auc и pr_auc (float или None)
    """
    if scores is None or not isinstance(scores, np.ndarray) or scores.size == 0:
        return {"roc_auc": None, "pr_auc": None}

    scores_flat = scores.ravel()
    if len(scores_flat) != series_length:
        return {"roc_auc": None, "pr_auc": None}

    valid_mask = np.isfinite(scores_flat)
    if valid_mask.sum() < 100:
        return {"roc_auc": None, "pr_auc": None}

    labels = np.zeros(series_length, dtype=int)
    for cp in true_cps:
        lo = max(0, cp - margin)
        hi = min(series_length, cp + margin + 1)
        labels[lo:hi] = 1

    labels_valid = labels[valid_mask]
    scores_valid = scores_flat[valid_mask]

    if labels_valid.sum() == 0 or labels_valid.sum() == len(labels_valid):
        return {"roc_auc": None, "pr_auc": None}

    try:
        roc = float(roc_auc_score(labels_valid, scores_valid))
        pr = float(average_precision_score(labels_valid, scores_valid))
    except ValueError:
        return {"roc_auc": None, "pr_auc": None}

    return {
        "roc_auc": round(roc, 4),
        "pr_auc": round(pr, 4),
    }
