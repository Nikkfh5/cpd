"""Сборка таблиц P1 same-noise Transformer pilots.

Скрипт читает импортированные результаты `results/publication_same_noise`,
проверяет покрытие тестовых seed, строит bootstrap-интервалы для средних
метрик и сравнивает same-noise Transformer с frozen universal Transformer там,
где есть сопоставимый сценарий.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = ROOT / "results" / "publication_same_noise"
DEFAULT_CLASSICAL_RESULTS_DIR = ROOT / "results" / "publication_same_noise_classical"
DEFAULT_FROZEN_DIR = ROOT / "results" / "kaggle_output_v2"
DEFAULT_TABLES_DIR = ROOT / "manuscript" / "assets" / "tables"
EXPECTED_SEEDS = set(range(1000, 1050))
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
    """Описание одного same-noise сценария."""

    directory: Path
    scenario_id: str
    diffusion: float
    noise_type: str

    @property
    def label(self) -> str:
        """Возвращает подпись сценария для таблиц."""
        return f"D={self.diffusion:g}, {self.noise_type}"


def parse_args() -> argparse.Namespace:
    """Разбирает параметры запуска."""

    parser = argparse.ArgumentParser(description="Build P1 same-noise publication tables")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--classical-results-dir", default=str(DEFAULT_CLASSICAL_RESULTS_DIR))
    parser.add_argument("--frozen-dir", default=str(DEFAULT_FROZEN_DIR))
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260507)
    return parser.parse_args()


def parse_scenario(directory: Path) -> Scenario | None:
    """Извлекает параметры сценария из имени каталога."""

    prefix = "same_noise_D"
    if not directory.name.startswith(prefix):
        return None
    remainder = directory.name[len(prefix):]
    diffusion_text, noise_type = remainder.split("_", 1)
    diffusion = int(diffusion_text) / 10.0
    return Scenario(directory, directory.name, diffusion, noise_type)


def discover_scenarios(results_dir: Path) -> list[Scenario]:
    """Находит same-noise сценарии с `summary.csv`."""

    scenarios: list[Scenario] = []
    for directory in sorted(results_dir.iterdir()):
        if not directory.is_dir():
            continue
        summary_path = directory / "eval_multi" / "summary.csv"
        per_series_path = directory / "eval_multi" / "per_series.csv"
        if not summary_path.exists() or not per_series_path.exists():
            continue
        scenario = parse_scenario(directory)
        if scenario is not None:
            scenarios.append(scenario)
    return sorted(scenarios, key=scenario_sort_key)


def scenario_sort_key(scenario: Scenario) -> tuple[float, int, str]:
    """Возвращает ключ сортировки сценариев."""

    return (
        scenario.diffusion,
        NOISE_ORDER.get(scenario.noise_type, 100),
        scenario.noise_type,
    )


def read_frames(scenario: Scenario) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Читает summary и per-series таблицы сценария."""

    summary = pd.read_csv(scenario.directory / "eval_multi" / "summary.csv")
    per_series = pd.read_csv(scenario.directory / "eval_multi" / "per_series.csv")
    return summary, per_series


def metric_values(group: pd.DataFrame, metric: str) -> np.ndarray:
    """Возвращает очищенные значения метрики."""

    values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
    if metric == "mae":
        values = values[values < NAN_MAE]
    return values[np.isfinite(values)]


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float]:
    """Считает percentile bootstrap CI для среднего."""

    if len(values) == 0:
        return math.nan, math.nan
    sample_indices = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[sample_indices].mean(axis=1)
    lower, upper = np.percentile(boot_means, [2.5, 97.5])
    return float(lower), float(upper)


def validate_scenario(
    scenario: Scenario,
    summary: pd.DataFrame,
    per_series: pd.DataFrame,
    result_group: str,
) -> dict[str, object]:
    """Проверяет тестовое покрытие и согласованность summary."""

    issues: list[str] = []
    method_seed_counts: list[int] = []
    seed_min: int | None = None
    seed_max: int | None = None
    for method, group in per_series.groupby("method"):
        seeds = set(int(seed) for seed in group["series_seed"].unique())
        method_seed_counts.append(len(seeds))
        if seeds:
            current_min = min(seeds)
            current_max = max(seeds)
            seed_min = current_min if seed_min is None else min(seed_min, current_min)
            seed_max = current_max if seed_max is None else max(seed_max, current_max)
        missing = sorted(EXPECTED_SEEDS - seeds)
        extra = sorted(seeds - EXPECTED_SEEDS)
        if len(group) != 50:
            issues.append(f"{method}:bad_n_rows:{len(group)}")
        if missing:
            issues.append(f"{method}:missing_seeds:{missing}")
        if extra:
            issues.append(f"{method}:extra_seeds:{extra}")
    summary_counts = {
        str(row["method"]): int(row["n_series"])
        for _, row in summary.iterrows()
    }
    for method, group in per_series.groupby("method"):
        if summary_counts.get(str(method)) != group["series_seed"].nunique():
            issues.append(f"{method}:summary_n_series_mismatch")
    return {
        "result_group": result_group,
        "scenario_id": scenario.scenario_id,
        "summary_path": str((scenario.directory / "eval_multi" / "summary.csv").relative_to(ROOT)),
        "per_series_path": str((scenario.directory / "eval_multi" / "per_series.csv").relative_to(ROOT)),
        "n_rows": int(len(per_series)),
        "n_methods": int(per_series["method"].nunique()),
        "n_unique_seeds_min": min(method_seed_counts) if method_seed_counts else 0,
        "seed_min": seed_min,
        "seed_max": seed_max,
        "issues": "; ".join(issues),
        "ok": not issues,
    }


def build_bootstrap_table(
    scenarios: list[Scenario],
    rng: np.random.Generator,
    n_boot: int,
    result_group: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Строит validation и bootstrap-таблицы."""

    validation_rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    metrics = ("precision", "recall", "f1", "mae", "roc_auc", "pr_auc")
    for scenario in scenarios:
        summary, per_series = read_frames(scenario)
        validation_rows.append(validate_scenario(scenario, summary, per_series, result_group))
        for method, group in per_series.groupby("method", sort=False):
            row: dict[str, object] = {
                "result_group": result_group,
                "scenario": scenario.label,
                "scenario_id": scenario.scenario_id,
                "D": scenario.diffusion,
                "noise_type": scenario.noise_type,
                "method": method,
                "n_series": int(group["series_seed"].nunique()),
            }
            for metric in metrics:
                values = metric_values(group, metric)
                low, high = bootstrap_mean_ci(values, rng, n_boot)
                row[f"{metric}_mean"] = float(values.mean()) if len(values) else math.nan
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            bootstrap_rows.append(row)
    return pd.DataFrame(validation_rows), pd.DataFrame(bootstrap_rows)


def build_summary_table(scenarios: list[Scenario], result_group: str) -> pd.DataFrame:
    """Собирает summary rows с параметрами сценария."""

    rows: list[pd.DataFrame] = []
    for scenario in scenarios:
        summary, _ = read_frames(scenario)
        current = summary.copy()
        current.insert(0, "result_group", result_group)
        current.insert(1, "scenario", scenario.label)
        current.insert(2, "scenario_id", scenario.scenario_id)
        current.insert(3, "D", scenario.diffusion)
        current.insert(4, "noise_type", scenario.noise_type)
        rows.append(current)
    if not rows:
        return pd.DataFrame()
    table = pd.concat(rows, ignore_index=True)
    return table.sort_values(["D", "noise_type", "mean_f1"], ascending=[True, True, False])


def frozen_scenario_id(scenario: Scenario) -> str | None:
    """Возвращает frozen-сценарий для paired comparison, если он есть."""

    if scenario.diffusion != 1.0:
        return None
    if scenario.noise_type == "white":
        return "eval_multi_D10_white"
    if scenario.noise_type == "pink":
        return "eval_multi_D10_pink"
    return None


def build_paired_table(scenarios: list[Scenario], frozen_dir: Path) -> pd.DataFrame:
    """Сравнивает same-noise и universal Transformer по тем же seed."""

    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        frozen_id = frozen_scenario_id(scenario)
        if frozen_id is None:
            continue
        _, same_per_series = read_frames(scenario)
        frozen_per_series_path = frozen_dir / frozen_id / "per_series.csv"
        if not frozen_per_series_path.exists():
            continue
        frozen_per_series = pd.read_csv(frozen_per_series_path)
        same = same_per_series[["series_seed", "f1"]].rename(columns={"f1": "same_noise_f1"})
        frozen = frozen_per_series[frozen_per_series["method"] == "Transformer_v2"]
        frozen = frozen[["series_seed", "f1"]].rename(columns={"f1": "universal_f1"})
        merged = same.merge(frozen, on="series_seed", how="inner").dropna()
        if len(merged) == 0:
            continue
        same_values = merged["same_noise_f1"].to_numpy(dtype=float)
        universal_values = merged["universal_f1"].to_numpy(dtype=float)
        result = wilcoxon(same_values, universal_values, alternative="two-sided", zero_method="wilcox")
        diff = same_values - universal_values
        rows.append(
            {
                "scenario": scenario.label,
                "scenario_id": scenario.scenario_id,
                "frozen_scenario_id": frozen_id,
                "metric": "f1",
                "method_a": "Transformer_v2_same_noise",
                "method_b": "Transformer_v2_universal",
                "mean_a": float(same_values.mean()),
                "mean_b": float(universal_values.mean()),
                "mean_diff": float(diff.mean()),
                "median_diff": float(np.median(diff)),
                "n_pairs": int(len(merged)),
                "wilcoxon_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    return pd.DataFrame(rows)


def build_transformer_vs_classical_paired(
    transformer_scenarios: list[Scenario],
    classical_scenarios: list[Scenario],
) -> pd.DataFrame:
    """Сравнивает Transformer same-noise с лучшим classical baseline."""

    classical_by_id = {scenario.scenario_id: scenario for scenario in classical_scenarios}
    rows: list[dict[str, object]] = []
    for scenario in transformer_scenarios:
        classical = classical_by_id.get(scenario.scenario_id)
        if classical is None:
            continue
        _, transformer_per_series = read_frames(scenario)
        classical_summary, classical_per_series = read_frames(classical)
        best_method = str(classical_summary.sort_values("mean_f1", ascending=False)["method"].iloc[0])
        transformer_method = str(transformer_per_series["method"].iloc[0])
        transformer_values = transformer_per_series[["series_seed", "f1"]].rename(columns={"f1": "transformer_f1"})
        classical_values = classical_per_series[classical_per_series["method"] == best_method]
        classical_values = classical_values[["series_seed", "f1"]].rename(columns={"f1": "classical_f1"})
        merged = transformer_values.merge(classical_values, on="series_seed", how="inner").dropna()
        if len(merged) == 0:
            continue
        values_a = merged["transformer_f1"].to_numpy(dtype=float)
        values_b = merged["classical_f1"].to_numpy(dtype=float)
        result = wilcoxon(values_a, values_b, alternative="two-sided", zero_method="wilcox")
        diff = values_a - values_b
        rows.append(
            {
                "scenario": scenario.label,
                "scenario_id": scenario.scenario_id,
                "metric": "f1",
                "method_a": transformer_method,
                "method_b": best_method,
                "mean_a": float(values_a.mean()),
                "mean_b": float(values_b.mean()),
                "mean_diff": float(diff.mean()),
                "median_diff": float(np.median(diff)),
                "n_pairs": int(len(merged)),
                "wilcoxon_statistic": float(result.statistic),
                "p_value": float(result.pvalue),
            }
        )
    return pd.DataFrame(rows)


def format_float(value: object, digits: int = 3) -> str:
    """Форматирует число для LaTeX."""

    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def latex_escape(value: object) -> str:
    """Экранирует строку для LaTeX."""

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


def write_latex_table(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    """Записывает простую LaTeX-таблицу."""

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


def write_f1_ci_latex(bootstrap: pd.DataFrame, path: Path) -> None:
    """Пишет compact F1 CI table."""

    rows: list[list[str]] = []
    sorted_table = bootstrap.sort_values(["D", "noise_type"])
    for _, row in sorted_table.iterrows():
        ci = (
            f"{format_float(row['f1_mean'])} "
            f"[{format_float(row['f1_ci_low'])}, {format_float(row['f1_ci_high'])}]"
        )
        rows.append([latex_escape(row["scenario"]), latex_escape(row["method"]), ci])
    write_latex_table(path, ["Scenario", "Method", "Mean F1 [95\\% CI]"], rows)


def write_paired_latex(paired: pd.DataFrame, path: Path) -> None:
    """Пишет compact paired comparison table."""

    rows: list[list[str]] = []
    for _, row in paired.iterrows():
        rows.append(
            [
                latex_escape(row["scenario"]),
                format_float(row["mean_a"]),
                format_float(row["mean_b"]),
                format_float(row["mean_diff"]),
                format_float(row["p_value"], 4),
            ]
        )
    write_latex_table(path, ["Scenario", "Same-noise F1", "Universal F1", "Diff", "p"], rows)


def write_transformer_vs_classical_latex(paired: pd.DataFrame, path: Path) -> None:
    """Пишет paired comparison table для Transformer vs classical."""

    rows: list[list[str]] = []
    for _, row in paired.iterrows():
        rows.append(
            [
                latex_escape(row["scenario"]),
                latex_escape(row["method_a"]),
                latex_escape(row["method_b"]),
                format_float(row["mean_a"]),
                format_float(row["mean_b"]),
                format_float(row["mean_diff"]),
                format_float(row["p_value"], 4),
            ]
        )
    write_latex_table(path, ["Scenario", "Transformer", "Classical", "A F1", "B F1", "Diff", "p"], rows)


def write_validation_report(validation: pd.DataFrame, path: Path) -> None:
    """Пишет markdown-отчет о валидации P1 artifacts."""

    lines = [
        "# P1 Same-Noise Artifact Validation",
        "",
        "- Expected test seeds: `1000-1049`.",
        "- Expected rows per scenario: `50`.",
        "",
        "| Group | Scenario | Status | Rows | Methods | Seeds | Issues |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for _, row in validation.iterrows():
        status = "ok" if bool(row["ok"]) else "check"
        seed_span = f"{row['seed_min']}-{row['seed_max']}"
        lines.append(
            f"| `{row['result_group']}` | `{row['scenario_id']}` | {status} | "
            f"{row['n_rows']} | {row['n_methods']} | {seed_span} | {row['issues']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Точка входа для сборки P1 таблиц."""

    args = parse_args()
    results_dir = Path(args.results_dir)
    classical_results_dir = Path(args.classical_results_dir)
    frozen_dir = Path(args.frozen_dir)
    tables_dir = Path(args.tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)

    transformer_scenarios = discover_scenarios(results_dir)
    classical_scenarios = discover_scenarios(classical_results_dir) if classical_results_dir.exists() else []
    if not transformer_scenarios:
        raise SystemExit(f"No same-noise scenarios found in {results_dir}")

    rng = np.random.default_rng(args.seed)
    transformer_validation, transformer_bootstrap = build_bootstrap_table(
        transformer_scenarios,
        rng,
        args.bootstrap_samples,
        "transformer",
    )
    paired = build_paired_table(transformer_scenarios, frozen_dir)
    summary_tables = [build_summary_table(transformer_scenarios, "transformer")]
    validation_tables = [transformer_validation]
    bootstrap_tables = [transformer_bootstrap]
    if classical_scenarios:
        classical_validation, classical_bootstrap = build_bootstrap_table(
            classical_scenarios,
            rng,
            args.bootstrap_samples,
            "classical",
        )
        validation_tables.append(classical_validation)
        bootstrap_tables.append(classical_bootstrap)
        summary_tables.append(build_summary_table(classical_scenarios, "classical"))
    validation = pd.concat(validation_tables, ignore_index=True)
    bootstrap = pd.concat(bootstrap_tables, ignore_index=True)
    comparison_summary = pd.concat(summary_tables, ignore_index=True)
    transformer_vs_classical = build_transformer_vs_classical_paired(
        transformer_scenarios,
        classical_scenarios,
    )

    transformer_validation.to_csv(tables_dir / "same_noise_transformer_validation.csv", index=False)
    transformer_bootstrap.to_csv(tables_dir / "same_noise_transformer_bootstrap_ci.csv", index=False)
    paired.to_csv(tables_dir / "same_noise_transformer_vs_universal_paired.csv", index=False)
    validation.to_csv(tables_dir / "same_noise_comparison_validation.csv", index=False)
    bootstrap.to_csv(tables_dir / "same_noise_comparison_bootstrap_ci.csv", index=False)
    comparison_summary.to_csv(tables_dir / "same_noise_comparison_summary.csv", index=False)
    transformer_vs_classical.to_csv(tables_dir / "same_noise_transformer_vs_classical_paired.csv", index=False)
    write_f1_ci_latex(transformer_bootstrap, tables_dir / "same_noise_transformer_f1_ci.tex")
    write_f1_ci_latex(bootstrap, tables_dir / "same_noise_comparison_f1_ci.tex")
    write_paired_latex(paired, tables_dir / "same_noise_transformer_vs_universal_paired.tex")
    write_transformer_vs_classical_latex(
        transformer_vs_classical,
        tables_dir / "same_noise_transformer_vs_classical_paired.tex",
    )
    write_validation_report(transformer_validation, tables_dir / "same_noise_transformer_validation.md")
    write_validation_report(validation, tables_dir / "same_noise_comparison_validation.md")

    print(
        "same_noise_tables "
        f"transformer_scenarios={len(transformer_scenarios)} "
        f"classical_scenarios={len(classical_scenarios)} "
        f"validation_ok={bool(validation['ok'].all())} "
        f"universal_paired_rows={len(paired)} "
        f"classical_paired_rows={len(transformer_vs_classical)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
