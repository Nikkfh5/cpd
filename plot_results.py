"""
Генерация графиков для курсовой работы по CPD.

Читает results/eval_multi/summary.csv и per_series.csv,
строит четыре фигуры:
  1. f1_comparison.pdf  — сравнение F1-скоров всех методов
  2. precision_recall.pdf — диаграмма Precision vs Recall
  3. example_series.pdf — пример ряда с истинными и предсказанными CP
  4. mae_comparison.pdf — точность локализации CP
"""

import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from algoritms.data_utils import generate_dataset
from algoritms.evaluation import match_change_points


RESULTS_DIR = Path("results/eval_multi")
KAGGLE_DIR = Path("results/kaggle_output_v2")
FIGURES_DIR = Path("figures")
SYNTHETIC_BASE = Path("results/synthetic_base")
MARGIN = 50

SCENARIO_TAGS = ["D05_white", "D10_white", "D10_pink", "D15_white"]
SCENARIO_LABELS = {
    "D05_white": "D=0.5, white",
    "D10_white": "D=1.0, white",
    "D10_pink": "D=1.0, pink",
    "D15_white": "D=1.5, white",
}

COLOR_CLASSICAL = "#1f77b4"
COLOR_ML_V1 = "#d62728"
COLOR_ML_V2 = "#2ca02c"

ML_V1 = {"LSTM", "GRU", "CatBoost", "Transformer"}
ML_V2 = {"LSTM_v2", "GRU_v2", "Transformer_v2"}


def _method_color(name):
    """Цвет метода по категории."""
    if name in ML_V2:
        return COLOR_ML_V2
    if name in ML_V1:
        return COLOR_ML_V1
    return COLOR_CLASSICAL


def _legend_patches():
    """Стандартные legend-патчи для трёх категорий методов."""
    return [
        mpatches.Patch(color=COLOR_CLASSICAL, label="Классические"),
        mpatches.Patch(color=COLOR_ML_V1, label="ML v1"),
        mpatches.Patch(color=COLOR_ML_V2, label="ML v2"),
    ]


def _clean_spines(ax):
    """Убирает верхнюю и правую рамки графика."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_fig(fig, output_path):
    """Сохраняет фигуру и закрывает её."""
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {output_path}")


def load_summary(path=None):
    """Загрузка summary.csv со всеми метриками включая AUC."""
    if path is None:
        path = RESULTS_DIR / "summary.csv"
    rows = []
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            entry = {
                "method": row["method"],
                "n_series": int(row["n_series"]),
                "precision": float(row["mean_precision"]),
                "recall": float(row["mean_recall"]),
                "f1": float(row["mean_f1"]),
                "std_f1": float(row["std_f1"]),
                "mae": float(row["mean_mae"]),
            }
            roc_raw = row.get("mean_roc_auc", "")
            pr_raw = row.get("mean_pr_auc", "")
            entry["roc_auc"] = float(roc_raw) if roc_raw else None
            entry["pr_auc"] = float(pr_raw) if pr_raw else None
            rows.append(entry)
    return rows


def _n_series_label(rows):
    """Извлекает число рядов из данных для подписей осей."""
    if not rows:
        return ""
    n = rows[0]["n_series"]
    return f"{n} рядов"


def plot_f1_comparison(rows, output_path=None):
    """Горизонтальный барплот F1-скоров всех методов."""
    if output_path is None:
        output_path = FIGURES_DIR / "f1_comparison.pdf"

    sorted_rows = sorted(rows, key=lambda r: r["f1"])
    methods = [r["method"] for r in sorted_rows]
    f1_values = [r["f1"] for r in sorted_rows]
    std_values = [r["std_f1"] for r in sorted_rows]
    colors = [_method_color(m) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(methods))

    ax.barh(y_pos, f1_values, xerr=std_values, color=colors,
            edgecolor="white", linewidth=0.5, capsize=3, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel(f"F1-score (среднее ± std, {_n_series_label(rows)})", fontsize=11)
    ax.set_title("Сравнение методов CPD по F1-score", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1.0)
    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.3, linewidth=0.8)

    for i, (f1, std) in enumerate(zip(f1_values, std_values)):
        ax.text(f1 + std + 0.015, i, f"{f1:.3f}", va="center", fontsize=8)

    ax.legend(handles=_legend_patches(), loc="lower right", fontsize=9)
    _clean_spines(ax)
    _save_fig(fig, output_path)


def plot_precision_recall(rows, output_path=None):
    """Диаграмма Precision vs Recall для всех методов."""
    if output_path is None:
        output_path = FIGURES_DIR / "precision_recall.pdf"

    fig, ax = plt.subplots(figsize=(8, 6))

    for row in rows:
        color = _method_color(row["method"])
        marker = "s" if row["method"] in ML_V1 or row["method"] in ML_V2 else "o"
        size = 80 if row["method"] in ML_V2 else 60
        ax.scatter(row["recall"], row["precision"], c=color, s=size,
                   marker=marker, edgecolors="black", linewidth=0.5, zorder=3)
        offset_x = 0.01
        offset_y = 0.02
        if row["method"] == "CUSUM":
            offset_y = -0.04
        if row["method"] == "CatBoost":
            offset_y = -0.04
        ax.annotate(row["method"], (row["recall"] + offset_x, row["precision"] + offset_y),
                    fontsize=7.5, alpha=0.85)

    f1_levels = [0.1, 0.2, 0.3, 0.5, 0.7]
    recall_grid = np.linspace(0.01, 1.0, 200)
    for f1_val in f1_levels:
        precision_curve = f1_val * recall_grid / (2 * recall_grid - f1_val)
        valid = (precision_curve > 0) & (precision_curve <= 1.0)
        ax.plot(recall_grid[valid], precision_curve[valid],
                color="gray", linestyle="--", alpha=0.3, linewidth=0.8)
        label_idx = np.argmin(np.abs(precision_curve - 0.95))
        if valid[label_idx]:
            ax.text(recall_grid[label_idx], precision_curve[label_idx] + 0.02,
                    f"F1={f1_val}", fontsize=7, color="gray", alpha=0.6)

    n_label = _n_series_label(rows)
    ax.set_xlabel(f"Recall (среднее, {n_label})", fontsize=11)
    ax.set_ylabel(f"Precision (среднее, {n_label})", fontsize=11)
    ax.set_title("Precision–Recall: сравнение методов CPD", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 0.75)
    ax.set_ylim(0, 1.08)

    ax.legend(handles=_legend_patches(), loc="upper right", fontsize=9)
    _clean_spines(ax)
    _save_fig(fig, output_path)


def plot_example_series(output_path=None, seed=42):
    """Пример временного ряда SDE с истинными и предсказанными CP."""
    if output_path is None:
        output_path = FIGURES_DIR / "example_series.pdf"

    x_values, cp_labels = generate_dataset(
        length=2000, dt=1.0, D=1.0, noise_type="white", seed=seed,
    )
    true_cps = np.where(cp_labels == 1)[0].tolist()
    t = np.arange(len(x_values))

    show_methods = ["SNHT", "Chow", "CUSUM", "Transformer_v2"]
    base_dir = SYNTHETIC_BASE.parent / "synthetic_base_d10"
    method_predictions = {}
    for m in show_methods:
        cp_file = base_dir / m / "predicted_cps.json"
        if cp_file.exists():
            with open(cp_file) as fh:
                data = json.load(fh)
            method_predictions[m] = data["change_points"]

    n_panels = 1 + len(method_predictions)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 2.5 * n_panels),
                             sharex=True, gridspec_kw={"hspace": 0.08})
    if n_panels == 1:
        axes = [axes]

    x_over_pi = x_values / math.pi

    axes[0].plot(t, x_over_pi, color=COLOR_CLASSICAL, linewidth=0.6, alpha=0.8)
    for cp in true_cps:
        axes[0].axvline(x=cp, color="red", linestyle="--", alpha=0.5, linewidth=0.8)
    axes[0].set_ylabel("x(t) / π", fontsize=10)
    axes[0].set_title(
        f"Траектория SDE (D=1.0, white noise, seed={seed}), "
        f"истинные CP: {len(true_cps)} (красный пунктир)",
        fontsize=11, fontweight="bold",
    )
    pi_levels = np.arange(math.floor(x_over_pi.min()), math.ceil(x_over_pi.max()) + 1)
    for level in pi_levels:
        axes[0].axhline(y=level, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
    _clean_spines(axes[0])

    method_colors = {
        "SNHT": "#2ca02c",
        "Chow": "#ff7f0e",
        "LSTM": "#d62728",
        "CatBoost": "#9467bd",
    }

    for idx, (method_name, pred_cps) in enumerate(method_predictions.items()):
        ax = axes[idx + 1]
        ax.plot(t, x_over_pi, color=COLOR_CLASSICAL, linewidth=0.5, alpha=0.4)

        for cp in true_cps:
            ax.axvline(x=cp, color="red", linestyle="--", alpha=0.4, linewidth=0.8)

        pred_color = method_colors.get(method_name, "#333333")
        for cp in pred_cps:
            ax.axvline(x=cp, color=pred_color, linestyle="-", alpha=0.8, linewidth=1.5)

        tp, fp, fn, _ = match_change_points(true_cps, pred_cps, MARGIN)

        ax.set_ylabel("x(t) / π", fontsize=9)
        ax.text(
            0.01, 0.92,
            f"{method_name}: pred={len(pred_cps)}, TP={tp}, FP={fp}, FN={fn}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        for level in pi_levels:
            ax.axhline(y=level, color="gray", linestyle=":", alpha=0.3, linewidth=0.5)
        _clean_spines(ax)

    axes[-1].set_xlabel("Шаг времени t", fontsize=10)

    true_line = plt.Line2D([0], [0], color="red", linewidth=1.0,
                           linestyle="--", label="Истинные CP")
    pred_line = plt.Line2D([0], [0], color="gray", linewidth=1.5,
                           linestyle="-", label="Предсказанные CP")
    fig.legend(handles=[true_line, pred_line], loc="lower center",
               ncol=2, fontsize=10, bbox_to_anchor=(0.5, -0.02))

    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено: {output_path}")


def plot_mae_comparison(rows, output_path=None):
    """Барплот MAE (средняя ошибка локализации) для методов с MAE < 50."""
    if output_path is None:
        output_path = FIGURES_DIR / "mae_comparison.pdf"

    filtered = [r for r in rows if r["mae"] < 50]
    sorted_rows = sorted(filtered, key=lambda r: r["mae"])
    methods = [r["method"] for r in sorted_rows]
    mae_values = [r["mae"] for r in sorted_rows]
    colors = [_method_color(m) for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(methods))

    ax.barh(y_pos, mae_values, color=colors, edgecolor="white",
            linewidth=0.5, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(methods, fontsize=9)
    ax.set_xlabel("MAE (средняя ошибка локализации, шаги)", fontsize=11)
    ax.set_title("Точность локализации CP (MAE, ниже — лучше)", fontsize=13, fontweight="bold")

    for i, mae in enumerate(mae_values):
        ax.text(mae + 0.3, i, f"{mae:.1f}", va="center", fontsize=8)

    ax.legend(handles=_legend_patches(), loc="lower right", fontsize=9)
    _clean_spines(ax)
    _save_fig(fig, output_path)


def plot_f1_across_scenarios(output_path=None):
    """F1-скор ключевых методов по 4 сценариям (grouped bar chart)."""
    if output_path is None:
        output_path = FIGURES_DIR / "f1_scenarios.pdf"

    key_methods = ["SNHT", "Chow", "CUSUM", "CatBoost", "LSTM_v2", "GRU_v2", "Transformer_v2"]
    scenario_data = {}
    for tag in SCENARIO_TAGS:
        path = KAGGLE_DIR / f"eval_multi_{tag}" / "summary.csv"
        if path.exists():
            scenario_data[tag] = {r["method"]: r for r in load_summary(path)}

    if not scenario_data:
        print("Нет данных для f1_scenarios")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    n_methods = len(key_methods)
    n_scenarios = len(scenario_data)
    bar_width = 0.8 / n_scenarios
    x = np.arange(n_methods)

    colors_scenario = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]
    for i, tag in enumerate(SCENARIO_TAGS):
        if tag not in scenario_data:
            continue
        data = scenario_data[tag]
        f1_vals = [data.get(m, {}).get("f1", 0) for m in key_methods]
        offset = (i - n_scenarios / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, f1_vals, bar_width, label=SCENARIO_LABELS[tag],
                      color=colors_scenario[i], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, f1_vals):
            if val > 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(key_methods, fontsize=10)
    ax.set_ylabel("F1-score", fontsize=11)
    ax.set_title("F1 по сценариям: зависимость от D и типа шума", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.85)
    ax.legend(fontsize=9, loc="upper left")
    _clean_spines(ax)
    _save_fig(fig, output_path)


def plot_roc_auc_scenarios(output_path=None):
    """ROC-AUC ML-методов по сценариям."""
    if output_path is None:
        output_path = FIGURES_DIR / "roc_auc_scenarios.pdf"

    ml_methods = ["LSTM_v2", "GRU_v2", "Transformer_v2", "CatBoost"]
    scenario_data = {}
    for tag in SCENARIO_TAGS:
        path = KAGGLE_DIR / f"eval_multi_{tag}" / "summary.csv"
        if path.exists():
            scenario_data[tag] = {r["method"]: r for r in load_summary(path)}

    if not scenario_data:
        print("Нет данных для roc_auc_scenarios")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"LSTM_v2": "o", "GRU_v2": "s", "Transformer_v2": "^", "CatBoost": "D"}
    colors_ml = {"LSTM_v2": "#d62728", "GRU_v2": "#2ca02c", "Transformer_v2": "#1f77b4", "CatBoost": "#ff7f0e"}

    x_positions = np.arange(len(SCENARIO_TAGS))
    for method in ml_methods:
        roc_vals = []
        for tag in SCENARIO_TAGS:
            data = scenario_data.get(tag, {})
            roc = data.get(method, {}).get("roc_auc")
            roc_vals.append(roc if roc else 0)
        ax.plot(x_positions, roc_vals, marker=markers[method], color=colors_ml[method],
                label=method, linewidth=2, markersize=8)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, linewidth=0.8)
    ax.text(len(SCENARIO_TAGS) - 0.5, 0.51, "random", fontsize=8, color="gray", alpha=0.6)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([SCENARIO_LABELS[t] for t in SCENARIO_TAGS], fontsize=9)
    ax.set_ylabel("ROC-AUC", fontsize=11)
    ax.set_title("ROC-AUC ML-моделей по сценариям", fontsize=13, fontweight="bold")
    ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=9)
    _clean_spines(ax)
    _save_fig(fig, output_path)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    summary_d10 = None
    d10_path = KAGGLE_DIR / "eval_multi_D10_white" / "summary.csv"
    if d10_path.exists():
        summary_d10 = load_summary(d10_path)

    summary_d05 = load_summary()
    print(f"D=0.5: {len(summary_d05)} методов")
    if summary_d10:
        print(f"D=1.0: {len(summary_d10)} методов")
    print()

    plot_f1_comparison(summary_d05, FIGURES_DIR / "f1_comparison_D05.pdf")
    plot_precision_recall(summary_d05, FIGURES_DIR / "precision_recall_D05.pdf")

    if summary_d10:
        plot_f1_comparison(summary_d10, FIGURES_DIR / "f1_comparison_D10.pdf")
        plot_precision_recall(summary_d10, FIGURES_DIR / "precision_recall_D10.pdf")

    plot_example_series()
    plot_mae_comparison(summary_d10 if summary_d10 else summary_d05)
    plot_f1_across_scenarios()
    plot_roc_auc_scenarios()

    print()
    print("Все графики сохранены в figures/")


if __name__ == "__main__":
    main()
