"""Сборка публикационных таблиц из замороженных артефактов benchmark.

Скрипт читает результаты `results/kaggle_output_v2/*`, независимо
пересчитывает агрегаты из `per_series.csv`, проверяет согласованность с
`summary.csv`, строит bootstrap-доверительные интервалы по тестовым рядам и
paired Wilcoxon-тесты для ключевых сравнений методов.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results" / "kaggle_output_v2"
DEFAULT_TABLES_DIR = ROOT / "manuscript" / "assets" / "tables"
EXPECTED_N_SERIES = 50
EXPECTED_SEED_START = 1000
NAN_MAE = 9999.0
NOISE_ORDER = {
    "white": 0,
    "pink": 1,
    "brownian": 2,
    "blue": 3,
    "violet": 4,
}


@dataclass(frozen=True)
class Scenario:
    """Описание одного frozen-сценария evaluation."""

    directory: Path
    scenario_id: str
    diffusion: float
    noise_type: str

    @property
    def label(self) -> str:
        """Возвращает короткую подпись сценария для таблиц."""
        return f"D={self.diffusion:g}, {self.noise_type}"


@dataclass(frozen=True)
class Comparison:
    """Описание paired-сравнения методов."""

    scenario_id: str
    method_a: str
    method_b: str
    alternative: str
    rationale: str


KEY_COMPARISONS = (
    Comparison("eval_multi_D05_white", "SNHT", "Chow", "greater", "low-noise classical winner vs runner-up"),
    Comparison("eval_multi_D05_white", "SNHT", "GRU_v2", "greater", "low-noise classical winner vs best ML"),
    Comparison("eval_multi_D10_white", "Transformer_v2", "CUSUM", "greater", "working-regime ML vs strongest classical baseline"),
    Comparison("eval_multi_D10_white", "Transformer_v2", "LSTM_v2", "two-sided", "top ML separation in working regime"),
    Comparison("eval_multi_D10_white", "Transformer_v2", "GRU_v2", "greater", "Transformer vs next recurrent model"),
    Comparison("eval_multi_D10_pink", "Transformer_v2", "CUSUM", "greater", "pink-noise transfer vs strongest classical baseline"),
    Comparison("eval_multi_D10_pink", "Transformer_v2", "GRU_v2", "two-sided", "top ML separation under pink noise"),
    Comparison("eval_multi_D15_white", "Transformer_v2", "CUSUM", "greater", "high-noise ML vs strongest classical baseline"),
    Comparison("eval_multi_D15_white", "Transformer_v2", "LSTM_v2", "two-sided", "top ML separation at high noise"),
)


def parse_args() -> argparse.Namespace:
    """Разбирает параметры запуска."""
    parser = argparse.ArgumentParser(description="Build publication benchmark tables")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260506)
    return parser.parse_args()


def discover_scenarios(results_dir: Path) -> list[Scenario]:
    """Находит frozen-сценарии `eval_multi_D*_noise`."""
    pattern = re.compile(r"^eval_multi_D(?P<d>\d+)_(?P<noise>[A-Za-z0-9_]+)$")
    scenarios: list[Scenario] = []
    for directory in sorted(results_dir.iterdir()):
        if not directory.is_dir():
            continue
        match = pattern.match(directory.name)
        if match is None:
            continue
        diffusion = int(match.group("d")) / 10.0
        scenarios.append(
            Scenario(
                directory=directory,
                scenario_id=directory.name,
                diffusion=diffusion,
                noise_type=match.group("noise"),
            )
        )
    return scenarios


def read_scenario_frames(scenario: Scenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Читает summary и per-series таблицы сценария."""
    summary_path = scenario.directory / "summary.csv"
    per_series_path = scenario.directory / "per_series.csv"
    summary = pd.read_csv(summary_path)
    per_series = pd.read_csv(per_series_path)
    return summary, per_series


def metric_mean(values: pd.Series, metric: str) -> float:
    """Считает среднее метрики с правилами eval_multi."""
    numeric = pd.to_numeric(values, errors="coerce")
    if metric == "mae":
        numeric = numeric[numeric < NAN_MAE]
    numeric = numeric.dropna()
    if len(numeric) == 0:
        return math.nan
    return float(numeric.mean())


def recompute_summary(per_series: pd.DataFrame) -> pd.DataFrame:
    """Пересчитывает summary.csv из per_series.csv."""
    rows: list[dict[str, object]] = []
    for method, group in per_series.groupby("method", sort=False):
        f1_values = pd.to_numeric(group["f1"], errors="coerce").dropna().to_numpy()
        rows.append(
            {
                "method": method,
                "n_series": int(group["series_seed"].nunique()),
                "mean_precision": round(metric_mean(group["precision"], "precision"), 4),
                "mean_recall": round(metric_mean(group["recall"], "recall"), 4),
                "mean_f1": round(metric_mean(group["f1"], "f1"), 4),
                "std_f1": round(float(np.std(f1_values)), 4),
                "mean_mae": round(metric_mean(group["mae"], "mae"), 2),
                "mean_roc_auc": round(metric_mean(group["roc_auc"], "roc_auc"), 4),
                "mean_pr_auc": round(metric_mean(group["pr_auc"], "pr_auc"), 4),
            }
        )
    return pd.DataFrame(rows)


def values_equal(expected: object, observed: object, tolerance: float) -> bool:
    """Сравнивает два числовых значения с поддержкой NaN."""
    expected_float = float(expected) if pd.notna(expected) else math.nan
    observed_float = float(observed) if pd.notna(observed) else math.nan
    if math.isnan(expected_float) and math.isnan(observed_float):
        return True
    return abs(expected_float - observed_float) <= tolerance


def validate_scenario(
    scenario: Scenario,
    summary: pd.DataFrame,
    per_series: pd.DataFrame,
    recomputed: pd.DataFrame,
) -> dict[str, object]:
    """Проверяет наличие rows, seeds и согласованность summary."""
    expected_seeds = set(range(EXPECTED_SEED_START, EXPECTED_SEED_START + EXPECTED_N_SERIES))
    issues: list[str] = []
    seed_issues: dict[str, dict[str, list[int]]] = {}

    for method, group in per_series.groupby("method"):
        seeds = set(int(seed) for seed in group["series_seed"].unique())
        missing = sorted(expected_seeds - seeds)
        extra = sorted(seeds - expected_seeds)
        if missing or extra:
            seed_issues[method] = {"missing": missing, "extra": extra}

    if seed_issues:
        issues.append("unexpected per-method seed coverage")

    summary_methods = set(str(method) for method in summary["method"])
    per_series_methods = set(str(method) for method in per_series["method"])
    if summary_methods != per_series_methods:
        issues.append("summary and per_series method sets differ")

    comparison_columns = {
        "n_series": 0.0,
        "mean_precision": 0.0005,
        "mean_recall": 0.0005,
        "mean_f1": 0.0005,
        "std_f1": 0.0005,
        "mean_mae": 0.005,
        "mean_roc_auc": 0.0005,
        "mean_pr_auc": 0.0005,
    }
    merged = summary.merge(recomputed, on="method", suffixes=("_file", "_recomputed"))
    mismatches: list[dict[str, object]] = []
    for _, row in merged.iterrows():
        for column, tolerance in comparison_columns.items():
            if not values_equal(row[f"{column}_file"], row[f"{column}_recomputed"], tolerance):
                mismatches.append(
                    {
                        "method": row["method"],
                        "column": column,
                        "summary": row[f"{column}_file"],
                        "recomputed": row[f"{column}_recomputed"],
                    }
                )

    if mismatches:
        issues.append("summary values differ from per_series recomputation")

    return {
        "scenario_id": scenario.scenario_id,
        "summary_path": str((scenario.directory / "summary.csv").relative_to(ROOT)),
        "per_series_path": str((scenario.directory / "per_series.csv").relative_to(ROOT)),
        "n_summary_methods": int(len(summary)),
        "n_per_series_rows": int(len(per_series)),
        "expected_n_series": EXPECTED_N_SERIES,
        "expected_seed_start": EXPECTED_SEED_START,
        "seed_issues": seed_issues,
        "summary_mismatches": mismatches,
        "issues": issues,
        "ok": not issues,
    }


def noise_sort_value(noise_type: str) -> int:
    """Возвращает порядок типа шума для публикационных таблиц."""
    return NOISE_ORDER.get(noise_type, 100)


def sort_publication_table(table: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    """Сортирует таблицу в порядке сценариев статьи."""
    sorted_table = table.copy()
    sorted_table["_noise_order"] = sorted_table["noise_type"].map(noise_sort_value)
    sorted_table = sorted_table.sort_values(
        ["D", "_noise_order", metric_column, "method"],
        ascending=[True, True, False, True],
    )
    return sorted_table.drop(columns=["_noise_order"])


def build_main_table(scenarios: Iterable[Scenario]) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Собирает основную таблицу benchmark и результаты валидации."""
    tables: list[pd.DataFrame] = []
    validations: list[dict[str, object]] = []
    for scenario in scenarios:
        summary, per_series = read_scenario_frames(scenario)
        recomputed = recompute_summary(per_series)
        validations.append(validate_scenario(scenario, summary, per_series, recomputed))
        recomputed.insert(0, "scenario", scenario.label)
        recomputed.insert(1, "scenario_id", scenario.scenario_id)
        recomputed.insert(2, "D", scenario.diffusion)
        recomputed.insert(3, "noise_type", scenario.noise_type)
        tables.append(recomputed)
    main = pd.concat(tables, ignore_index=True)
    return sort_publication_table(main, "mean_f1"), validations


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    """Считает percentile bootstrap CI для среднего."""
    clean = values[np.isfinite(values)]
    if len(clean) == 0:
        return math.nan, math.nan
    sample_indices = rng.integers(0, len(clean), size=(n_boot, len(clean)))
    boot_means = clean[sample_indices].mean(axis=1)
    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return float(lower), float(upper)


def build_bootstrap_table(
    scenarios: Iterable[Scenario],
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    """Собирает bootstrap CI по тестовым рядам."""
    metrics = ("precision", "recall", "f1", "mae", "roc_auc", "pr_auc")
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        _, per_series = read_scenario_frames(scenario)
        for method, group in per_series.groupby("method", sort=False):
            row: dict[str, object] = {
                "scenario": scenario.label,
                "scenario_id": scenario.scenario_id,
                "D": scenario.diffusion,
                "noise_type": scenario.noise_type,
                "method": method,
                "n_series": int(group["series_seed"].nunique()),
            }
            for metric in metrics:
                values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
                if metric == "mae":
                    values = values[values < NAN_MAE]
                values = values[np.isfinite(values)]
                mean_value = float(values.mean()) if len(values) else math.nan
                low, high = bootstrap_mean_ci(values, rng, n_boot)
                row[f"{metric}_mean"] = mean_value
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            rows.append(row)
    result = pd.DataFrame(rows)
    return sort_publication_table(result, "f1_mean")


def paired_values(per_series: pd.DataFrame, method_a: str, method_b: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает выровненные по seed значения двух методов."""
    subset = per_series[per_series["method"].isin([method_a, method_b])]
    pivot = subset.pivot(index="series_seed", columns="method", values=metric)
    pivot = pivot.dropna(subset=[method_a, method_b])
    return (
        pivot[method_a].to_numpy(dtype=float),
        pivot[method_b].to_numpy(dtype=float),
    )


def build_paired_tests_table(scenario_map: dict[str, Scenario]) -> pd.DataFrame:
    """Собирает paired Wilcoxon-тесты для ключевых сравнений."""
    rows: list[dict[str, object]] = []
    for comparison in KEY_COMPARISONS:
        scenario = scenario_map.get(comparison.scenario_id)
        if scenario is None:
            continue
        _, per_series = read_scenario_frames(scenario)
        values_a, values_b = paired_values(per_series, comparison.method_a, comparison.method_b, "f1")
        if len(values_a) == 0:
            continue
        result = wilcoxon(values_a, values_b, alternative=comparison.alternative, zero_method="wilcox")
        diff = values_a - values_b
        rows.append(
            {
                "scenario": scenario.label,
                "scenario_id": scenario.scenario_id,
                "D": scenario.diffusion,
                "noise_type": scenario.noise_type,
                "metric": "f1",
                "method_a": comparison.method_a,
                "method_b": comparison.method_b,
                "alternative": comparison.alternative,
                "mean_a": float(values_a.mean()),
                "mean_b": float(values_b.mean()),
                "mean_diff": float(diff.mean()),
                "median_diff": float(np.median(diff)),
                "n_pairs": int(len(values_a)),
                "wilcoxon_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
                "rationale": comparison.rationale,
            }
        )
    table = pd.DataFrame(rows)
    table["_noise_order"] = table["noise_type"].map(noise_sort_value)
    table = table.sort_values(["D", "_noise_order", "method_a", "method_b"], ascending=[True, True, True, True])
    return table.drop(columns=["_noise_order"])


def latex_escape(value: object) -> str:
    """Экранирует значение для LaTeX-таблицы."""
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def format_float(value: object, digits: int = 3) -> str:
    """Форматирует число для компактной таблицы."""
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def format_p_value(value: object) -> str:
    """Форматирует p-value для таблиц."""
    if pd.isna(value):
        return ""
    numeric = float(value)
    if numeric < 0.0001:
        return "<0.0001"
    return f"{numeric:.4f}"


def write_latex_table(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    """Записывает простую LaTeX-таблицу с booktabs."""
    column_spec = "l" * len(columns)
    lines = [
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        " & ".join(columns) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_main_latex(main: pd.DataFrame, path: Path) -> None:
    """Пишет compact LaTeX-таблицу top-5 методов по каждому сценарию."""
    rows: list[list[str]] = []
    for _, group in main.groupby("scenario", sort=False):
        top = group.sort_values("mean_f1", ascending=False).head(5)
        for _, row in top.iterrows():
            rows.append(
                [
                    latex_escape(row["scenario"]),
                    latex_escape(row["method"]),
                    format_float(row["mean_precision"]),
                    format_float(row["mean_recall"]),
                    format_float(row["mean_f1"]),
                    format_float(row["mean_mae"]),
                    format_float(row["mean_roc_auc"]),
                    format_float(row["mean_pr_auc"]),
                ]
            )
    write_latex_table(path, ["Scenario", "Method", "P", "R", "F1", "MAE", "ROC", "PR"], rows)


def write_bootstrap_latex(bootstrap: pd.DataFrame, path: Path) -> None:
    """Пишет compact LaTeX-таблицу F1 с 95% CI для top-5 методов."""
    rows: list[list[str]] = []
    for _, group in bootstrap.groupby("scenario", sort=False):
        top = group.sort_values("f1_mean", ascending=False).head(5)
        for _, row in top.iterrows():
            ci = f"{format_float(row['f1_mean'])} [{format_float(row['f1_ci_low'])}, {format_float(row['f1_ci_high'])}]"
            rows.append([latex_escape(row["scenario"]), latex_escape(row["method"]), ci])
    write_latex_table(path, ["Scenario", "Method", "Mean F1 [95\\% CI]"], rows)


def write_paired_latex(paired: pd.DataFrame, path: Path) -> None:
    """Пишет LaTeX-таблицу paired Wilcoxon-сравнений."""
    rows: list[list[str]] = []
    for _, row in paired.iterrows():
        rows.append(
            [
                latex_escape(row["scenario"]),
                latex_escape(row["method_a"]),
                latex_escape(row["method_b"]),
                latex_escape(row["alternative"]),
                format_float(row["mean_diff"]),
                format_p_value(row["p_value"]),
            ]
        )
    write_latex_table(path, ["Scenario", "A", "B", "Alt.", "Mean F1 diff", "p"], rows)


def write_validation_report(path: Path, validations: list[dict[str, object]], smoke_status: str) -> None:
    """Записывает короткий markdown-отчёт о P0-валидации."""
    lines = [
        "# P0 Artifact Validation",
        "",
        f"- Smoke test status: `{smoke_status}`.",
        f"- Expected test seeds: `{EXPECTED_SEED_START}-{EXPECTED_SEED_START + EXPECTED_N_SERIES - 1}`.",
        f"- Expected series per method: `{EXPECTED_N_SERIES}`.",
        "",
        "| Scenario | Status | Issues |",
        "| --- | --- | --- |",
    ]
    for validation in validations:
        status = "ok" if validation["ok"] else "check"
        issues = "; ".join(str(issue) for issue in validation["issues"]) if validation["issues"] else ""
        lines.append(f"| `{validation['scenario_id']}` | {status} | {issues} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Точка входа для сборки P0-таблиц."""
    args = parse_args()
    results_dir = Path(args.results_dir)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    scenarios = discover_scenarios(results_dir)
    if not scenarios:
        raise SystemExit(f"No scenarios found in {results_dir}")

    main_table, validations = build_main_table(scenarios)
    rng = np.random.default_rng(args.seed)
    bootstrap_table = build_bootstrap_table(scenarios, rng, args.bootstrap_samples)
    scenario_map = {scenario.scenario_id: scenario for scenario in scenarios}
    paired_table = build_paired_tests_table(scenario_map)

    main_path = tables_dir / "main_benchmark_recomputed.csv"
    bootstrap_path = tables_dir / "bootstrap_ci_by_series.csv"
    paired_path = tables_dir / "paired_wilcoxon_key_comparisons.csv"
    validation_json_path = tables_dir / "p0_artifact_validation.json"
    validation_md_path = tables_dir / "p0_artifact_validation.md"

    main_table.to_csv(main_path, index=False)
    bootstrap_table.to_csv(bootstrap_path, index=False)
    paired_table.to_csv(paired_path, index=False)
    validation_json_path.write_text(json.dumps(validations, ensure_ascii=False, indent=2), encoding="utf-8")

    write_main_latex(main_table, tables_dir / "main_benchmark_top5.tex")
    write_bootstrap_latex(bootstrap_table, tables_dir / "bootstrap_f1_ci_top5.tex")
    write_paired_latex(paired_table, tables_dir / "paired_wilcoxon_key_comparisons.tex")
    write_validation_report(validation_md_path, validations, "passed before table build")

    failed = [validation for validation in validations if not validation["ok"]]
    print(f"Scenarios: {len(scenarios)}")
    print(f"Main table: {main_path.relative_to(ROOT)}")
    print(f"Bootstrap CI table: {bootstrap_path.relative_to(ROOT)}")
    print(f"Paired tests table: {paired_path.relative_to(ROOT)}")
    print(f"Validation report: {validation_md_path.relative_to(ROOT)}")
    if failed:
        print(f"Validation failed for {len(failed)} scenario(s)")
        return 1
    print("P0 table build completed without validation issues.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
