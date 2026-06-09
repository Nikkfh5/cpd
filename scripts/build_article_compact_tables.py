"""Build compact manuscript tables from publication CSV artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLES_DIR = ROOT / "manuscript" / "assets" / "tables"


SCENARIO_LATEX = {
    "D=0.5, white": r"\(D=0.5\), white",
    "D=1, white": r"\(D=1.0\), white",
    "D=1, pink": r"\(D=1.0\), pink",
    "D=1.5, white": r"\(D=1.5\), white",
}

SCENARIO_ORDER = ["D=0.5, white", "D=1, white", "D=1, pink", "D=1.5, white"]
MARGIN_ORDER = [25, 50, 100]
NOISE_ORDER = ["white", "pink", "brownian", "blue", "violet"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build compact manuscript-only tables")
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    return parser.parse_args()


def display_method(method: str) -> str:
    if method == "Transformer_v2":
        return "Transformer"
    if method == "LSTM_v2":
        return "LSTM"
    if method == "GRU_v2":
        return "GRU"
    return method


def write_latex_table(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    lines = [
        rf"\begin{{tabular}}{{{'l' * len(columns)}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_margin_leaders(tables_dir: Path) -> Path:
    summary = pd.read_csv(tables_dir / "matching_margin_robustness_summary.csv")
    rows: list[list[str]] = []
    for scenario in SCENARIO_ORDER:
        row = [SCENARIO_LATEX[scenario]]
        for margin in MARGIN_ORDER:
            subset = summary[
                (summary["scenario"] == scenario)
                & (summary["margin"].astype(int) == margin)
            ].copy()
            if subset.empty:
                row.append("")
                continue
            leader = subset.sort_values("mean_f1", ascending=False).iloc[0]
            row.append(f"{display_method(str(leader['method']))} {float(leader['mean_f1']):.3f}")
        rows.append(row)
    output = tables_dir / "matching_margin_robustness_leaders.tex"
    write_latex_table(output, ["Scenario", "Margin 25", "Margin 50", "Margin 100"], rows)
    return output


def build_same_noise_summary(tables_dir: Path) -> Path:
    summary = pd.read_csv(tables_dir / "same_noise_comparison_summary.csv")
    rows: list[list[str]] = []
    for noise_type in NOISE_ORDER:
        transformer = summary[
            (summary["noise_type"] == noise_type)
            & (summary["result_group"] == "transformer")
        ].copy()
        classical = summary[
            (summary["noise_type"] == noise_type)
            & (summary["result_group"] == "classical")
        ].copy()
        if transformer.empty or classical.empty:
            continue
        transformer_best = transformer.sort_values("mean_f1", ascending=False).iloc[0]
        classical_sorted = classical.sort_values("mean_f1", ascending=False)
        best_f1 = float(classical_sorted.iloc[0]["mean_f1"])
        tied = classical_sorted[
            (classical_sorted["mean_f1"].astype(float) - best_f1).abs() < 0.0005
        ]
        best_methods = "/".join(display_method(str(method)) for method in tied["method"].tolist())
        rows.append(
            [
                noise_type,
                f"{float(transformer_best['mean_f1']):.3f}",
                f"{best_f1:.3f}",
                best_methods,
            ]
        )
    output = tables_dir / "same_noise_best_f1_summary.tex"
    write_latex_table(
        output,
        ["Noise", "Transformer F1", "Classical F1", "Best classical method"],
        rows,
    )
    return output


def main() -> int:
    args = parse_args()
    tables_dir = Path(args.tables_dir)
    margin_path = build_margin_leaders(tables_dir)
    same_noise_path = build_same_noise_summary(tables_dir)
    print(f"Wrote {margin_path.relative_to(ROOT)}")
    print(f"Wrote {same_noise_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
