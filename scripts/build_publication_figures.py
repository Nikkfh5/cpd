from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algoritms.data_utils import compute_levels, generate_dataset


TABLE_DIR = ROOT / "manuscript" / "assets" / "tables"
FIGURE_DIR = ROOT / "manuscript" / "assets" / "figures"
ROBUSTNESS_DIR = ROOT / "results" / "matching_margin_robustness_outputs"

PRIMARY_METHODS = ["SNHT", "Chow", "CUSUM", "CatBoost", "LSTM_v2", "GRU_v2", "Transformer_v2"]
DISPLAY_NAMES = {
    "LSTM_v2": "LSTM",
    "GRU_v2": "GRU",
    "Transformer_v2": "Transformer",
    "Transformer_v2_same_noise_white": "Transformer",
    "Transformer_v2_same_noise_pink": "Transformer",
    "Transformer_v2_same_noise_brownian": "Transformer",
    "Transformer_v2_same_noise_blue": "Transformer",
    "Transformer_v2_same_noise_violet": "Transformer",
}
SCENARIO_ORDER = ["D=0.5, white", "D=1, white", "D=1, pink", "D=1.5, white"]
SCENARIO_TAGS = {
    "D=0.5, white": ("D05_white", 0.5, "white"),
    "D=1, white": ("D10_white", 1.0, "white"),
    "D=1, pink": ("D10_pink", 1.0, "pink"),
    "D=1.5, white": ("D15_white", 1.5, "white"),
}
SCENARIO_LABELS = {
    "D=0.5, white": r"$D=0.5$, white",
    "D=1, white": r"$D=1.0$, white",
    "D=1, pink": r"$D=1.0$, pink",
    "D=1.5, white": r"$D=1.5$, white",
}
METHOD_COLORS = {
    "SNHT": "#303030",
    "Chow": "#777777",
    "CUSUM": "#a45c40",
    "CatBoost": "#8a8a8a",
    "LSTM_v2": "#547a9a",
    "GRU_v2": "#6f8fb0",
    "Transformer_v2": "#1f5d8c",
}
MARKERS = {
    "SNHT": "o",
    "Chow": "s",
    "CUSUM": "^",
    "CatBoost": "D",
    "LSTM_v2": "P",
    "GRU_v2": "X",
    "Transformer_v2": "*",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "grid.color": "#d0d0d0",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.45,
        }
    )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def display_method(method: str) -> str:
    return DISPLAY_NAMES.get(method, method)


def clean_scenario_label(scenario: str) -> str:
    return SCENARIO_LABELS.get(scenario, scenario)


def float_or_nan(value: str) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = FIGURE_DIR / f"{stem}.pdf"
    png_path = FIGURE_DIR / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {pdf_path.relative_to(ROOT)}")
    print(f"saved {png_path.relative_to(ROOT)}")


def cp_indices(cp_labels: np.ndarray) -> list[int]:
    return np.where(cp_labels == 1)[0].astype(int).tolist()


def select_representative_seed(D: float, noise_type: str) -> int:
    candidates = []
    for seed in range(1000, 1050):
        _, cp_labels = generate_dataset(length=2000, dt=1.0, D=D, noise_type=noise_type, seed=seed)
        cps = cp_indices(cp_labels)
        if not cps:
            continue
        early_penalty = 1 if min(cps) < 40 else 0
        cluster_penalty = sum(1 for a, b in zip(cps, cps[1:]) if b - a <= 5)
        candidates.append((seed, len(cps), early_penalty, cluster_penalty))
    counts = np.array([item[1] for item in candidates], dtype=float)
    target = float(np.median(counts))
    candidates.sort(key=lambda item: (item[2], abs(item[1] - target), item[3], item[0]))
    return candidates[0][0]


def plot_generated_regime_examples() -> dict[str, int]:
    seeds = {}
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8), sharex=True)
    axes = axes.ravel()

    for ax, scenario in zip(axes, SCENARIO_ORDER):
        _, D, noise_type = SCENARIO_TAGS[scenario]
        seed = select_representative_seed(D, noise_type)
        seeds[scenario] = seed
        x_values, cp_labels = generate_dataset(length=2000, dt=1.0, D=D, noise_type=noise_type, seed=seed)
        levels, _ = compute_levels(x_values)
        t = np.arange(len(x_values))
        x_pi = x_values / math.pi
        cps = cp_indices(cp_labels)

        ax.plot(t, x_pi, color="#303030", linewidth=0.65)
        ymin = float(np.nanmin(x_pi))
        ymax = float(np.nanmax(x_pi))
        tick_base = ymin + 0.04 * (ymax - ymin)
        tick_top = ymin + 0.13 * (ymax - ymin)
        ax.vlines(cps, tick_base, tick_top, color="#9a3d2f", linewidth=0.55, alpha=0.75)
        level_min = int(np.floor(ymin))
        level_max = int(np.ceil(ymax))
        if level_max - level_min <= 10:
            levels_to_draw = range(level_min, level_max + 1)
        else:
            levels_to_draw = np.linspace(level_min, level_max, 6)
        for level in levels_to_draw:
            ax.axhline(level, color="#c2c2c2", linestyle=":", linewidth=0.45, alpha=0.65)
        ax.set_title(f"{clean_scenario_label(scenario)}; seed {seed}; CPs={len(cps)}")
        ax.set_ylabel(r"$x_t/\pi$")
        ax.grid(axis="x")

    axes[2].set_xlabel("time step")
    axes[3].set_xlabel("time step")
    fig.suptitle("Representative generated trajectories and labeled level crossings", y=0.995, fontsize=11)
    save_figure(fig, "fig01_generated_regime_examples")
    return seeds


def plot_main_f1_benchmark() -> None:
    rows = read_csv_rows(TABLE_DIR / "main_benchmark_recomputed.csv")
    by_key = {(row["scenario"], row["method"]): float(row["mean_f1"]) for row in rows}
    matrix = np.array(
        [[by_key.get((scenario, method), np.nan) for scenario in SCENARIO_ORDER] for method in PRIMARY_METHODS],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    im = ax.imshow(matrix, aspect="auto", cmap="Greys", vmin=0.0, vmax=0.8)
    ax.set_xticks(range(len(SCENARIO_ORDER)))
    ax.set_xticklabels([clean_scenario_label(s) for s in SCENARIO_ORDER], rotation=20, ha="right")
    ax.set_yticks(range(len(PRIMARY_METHODS)))
    ax.set_yticklabels([display_method(m) for m in PRIMARY_METHODS])
    ax.set_title("Primary benchmark F1 at matching margin 25")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                color = "white" if value > 0.48 else "#202020"
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", color=color, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cbar.set_label("mean F1")
    save_figure(fig, "fig02_primary_f1_benchmark")


def plot_margin_robustness() -> None:
    rows = read_csv_rows(TABLE_DIR / "matching_margin_robustness_summary.csv")
    data = defaultdict(dict)
    for row in rows:
        method = row["method"]
        if method not in {"SNHT", "Chow", "CUSUM", "LSTM_v2", "Transformer_v2"}:
            continue
        data[(row["scenario"], method)][int(row["margin"])] = float(row["mean_f1"])

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8), sharex=True, sharey=True)
    axes = axes.ravel()
    margins = [25, 50, 100]

    for ax, scenario in zip(axes, SCENARIO_ORDER):
        for method in ["SNHT", "Chow", "CUSUM", "LSTM_v2", "Transformer_v2"]:
            values = [data.get((scenario, method), {}).get(m, np.nan) for m in margins]
            if not np.isfinite(values).any():
                continue
            ax.plot(
                margins,
                values,
                marker=MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method, "#444444"),
                linewidth=1.5,
                markersize=5,
                label=display_method(method),
            )
        ax.set_title(clean_scenario_label(scenario))
        ax.set_ylim(0, 0.84)
        ax.set_xticks(margins)
        ax.grid(True)

    axes[2].set_xlabel("matching margin")
    axes[3].set_xlabel("matching margin")
    axes[0].set_ylabel("mean F1")
    axes[2].set_ylabel("mean F1")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.015))
    fig.suptitle("Robustness of benchmark rankings to matching margin", y=0.995, fontsize=11)
    save_figure(fig, "fig03_matching_margin_robustness")


def plot_f1_mae_tradeoff() -> None:
    rows = [
        row
        for row in read_csv_rows(TABLE_DIR / "main_benchmark_recomputed.csv")
        if row["method"] in PRIMARY_METHODS and float_or_nan(row["mean_mae"]) < 100
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.8), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, scenario in zip(axes, SCENARIO_ORDER):
        scenario_rows = [row for row in rows if row["scenario"] == scenario]
        for row in scenario_rows:
            method = row["method"]
            f1 = float(row["mean_f1"])
            mae = float(row["mean_mae"])
            size = 90 if method == "Transformer_v2" else 52
            ax.scatter(
                f1,
                mae,
                s=size,
                color=METHOD_COLORS.get(method, "#555555"),
                marker=MARKERS.get(method, "o"),
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
            label_methods = {"CUSUM", "CatBoost", "Transformer_v2"}
            if scenario == "D=0.5, white":
                label_methods.update({"SNHT", "Chow"})
            if method in label_methods:
                offsets = {
                    ("D=0.5, white", "SNHT"): (5, 2),
                    ("D=0.5, white", "Chow"): (5, 2),
                    ("D=1, white", "CUSUM"): (5, 4),
                    ("D=1, white", "CatBoost"): (5, 3),
                    ("D=1, white", "Transformer_v2"): (5, 3),
                    ("D=1, pink", "CUSUM"): (5, 4),
                    ("D=1, pink", "CatBoost"): (5, -7),
                    ("D=1, pink", "Transformer_v2"): (5, 3),
                    ("D=1.5, white", "CUSUM"): (5, 4),
                    ("D=1.5, white", "CatBoost"): (5, -7),
                    ("D=1.5, white", "Transformer_v2"): (5, 3),
                }
                ax.annotate(
                    display_method(method),
                    (f1, mae),
                    xytext=offsets.get((scenario, method), (4, 3)),
                    textcoords="offset points",
                    fontsize=7,
                )
        ax.set_title(clean_scenario_label(scenario))
        ax.grid(True)

    axes[2].set_xlabel("mean F1")
    axes[3].set_xlabel("mean F1")
    axes[0].set_ylabel("MAE of matched detections")
    axes[2].set_ylabel("MAE of matched detections")
    axes[0].set_xlim(0, 0.82)
    axes[0].set_ylim(0, 13)
    fig.suptitle("Event-detection strength and localization error at margin 25", y=0.995, fontsize=11)
    save_figure(fig, "fig04_f1_mae_tradeoff")


def load_raw_predictions() -> list[dict]:
    path = ROBUSTNESS_DIR / "raw_predictions.jsonl"
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def find_prediction(raw_rows: list[dict], scenario_tag: str, method: str, seed: int) -> dict | None:
    for row in raw_rows:
        if row["scenario_tag"] == scenario_tag and row["method"] == method and int(row["series_seed"]) == seed:
            return row
    return None


def plot_detection_overlay() -> dict[str, int]:
    raw_rows = load_raw_predictions()
    cases = [
        ("D=0.5, white", 1000, 1350, 1800),
        ("D=1.5, white", 1020, 130, 330),
    ]
    fig, axes = plt.subplots(len(cases), 1, figsize=(7.1, 5.0), sharex=False)
    if len(cases) == 1:
        axes = [axes]
    y_offsets = {"SNHT": 0.0, "CUSUM": 0.16, "Transformer_v2": 0.32}
    methods = ["SNHT", "CUSUM", "Transformer_v2"]

    chosen = {}
    for idx, (ax, (scenario, seed, start, end)) in enumerate(zip(axes, cases)):
        tag, D, noise_type = SCENARIO_TAGS[scenario]
        chosen[scenario] = seed
        x_values, cp_labels = generate_dataset(length=2000, dt=1.0, D=D, noise_type=noise_type, seed=seed)
        t = np.arange(len(x_values))[start:end]
        x_pi = x_values[start:end] / math.pi
        cps = [cp for cp in cp_indices(cp_labels) if start <= cp < end]
        ax.plot(t, x_pi, color="#303030", linewidth=0.8, label="trajectory")
        ymin, ymax = ax.get_ylim()
        for cp in cps:
            ax.axvline(cp, color="#9a3d2f", linewidth=0.8, alpha=0.75)
        for method in methods:
            row = find_prediction(raw_rows, tag, method, seed)
            if row is None:
                continue
            y_level = ymin + (ymax - ymin) * (0.07 + y_offsets[method])
            window_preds = [pred for pred in row["pred_cps"] if start <= pred < end]
            for pred in window_preds:
                ax.plot(pred, y_level, marker=MARKERS[method], color=METHOD_COLORS[method], markersize=5)
            ax.text(
                0.01,
                0.92 - y_offsets[method] * 0.85,
                f"{display_method(method)}: {len(window_preds)} predictions in window",
                transform=ax.transAxes,
                fontsize=7,
                color=METHOD_COLORS[method],
                va="top",
            )
        ax.set_title(f"{clean_scenario_label(scenario)}; seed {seed}; window {start}-{end}; true CPs={len(cps)}")
        ax.set_ylabel(r"$x_t/\pi$")
        ax.set_xlim(start, end)
        ax.grid(axis="x")
        if idx < len(axes) - 1:
            ax.tick_params(labelbottom=False)

    axes[-1].set_xlabel("time step")
    true_handle = plt.Line2D([0], [0], color="#9a3d2f", linewidth=1.0, label="true CP")
    pred_handles = [
        plt.Line2D([0], [0], marker=MARKERS[m], color="none", markerfacecolor=METHOD_COLORS[m], label=display_method(m))
        for m in methods
    ]
    fig.legend([true_handle, *pred_handles], ["true CP", *[display_method(m) for m in methods]], loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(hspace=0.62, bottom=0.16, top=0.87)
    fig.suptitle("Representative detection windows from frozen inference predictions", y=0.98, fontsize=11)
    save_figure(fig, "fig05_detection_overlay")
    return chosen


def write_manifest(seeds: dict[str, int], overlay_cases: dict[str, int]) -> None:
    manifest = FIGURE_DIR / "publication_figures_manifest.md"
    text = [
        "# Publication Figures Manifest",
        "",
        "Generated by `scripts/build_publication_figures.py`.",
        "",
        "## Inputs",
        "",
        "- `manuscript/assets/tables/main_benchmark_recomputed.csv`",
        "- `manuscript/assets/tables/matching_margin_robustness_summary.csv`",
        "- `results/matching_margin_robustness_outputs/raw_predictions.jsonl`",
        "- `algoritms/data_utils.py` for regenerated trajectories.",
        "",
        "## Representative Seeds",
        "",
    ]
    for scenario, seed in seeds.items():
        text.append(f"- `{scenario}` regime example: seed `{seed}`")
    for scenario, seed in overlay_cases.items():
        text.append(f"- `{scenario}` detection overlay: seed `{seed}`")
    text.extend(
        [
            "",
            "## Outputs",
            "",
            "- `fig01_generated_regime_examples.{pdf,png}`",
            "- `fig02_primary_f1_benchmark.{pdf,png}`",
            "- `fig03_matching_margin_robustness.{pdf,png}`",
            "- `fig04_f1_mae_tradeoff.{pdf,png}`",
            "- `fig05_detection_overlay.{pdf,png}`",
        ]
    )
    manifest.write_text("\n".join(text) + "\n", encoding="utf-8")
    print(f"saved {manifest.relative_to(ROOT)}")


def main() -> None:
    setup_style()
    seeds = plot_generated_regime_examples()
    plot_main_f1_benchmark()
    plot_margin_robustness()
    plot_f1_mae_tradeoff()
    overlay_cases = plot_detection_overlay()
    write_manifest(seeds, overlay_cases)


if __name__ == "__main__":
    main()
