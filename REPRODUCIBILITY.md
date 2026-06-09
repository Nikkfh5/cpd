# Reproducibility Guide

This guide documents the commands used to verify the public repository and
rebuild the manuscript-facing artifacts from committed frozen outputs.

Run commands from the repository root unless a command says otherwise.

## 1. Environment

```bash
python -m pip install -r requirements.txt
```

Required Python packages are listed in `requirements.txt`. Building the
manuscript PDF additionally requires `pdflatex` and `bibtex`.

## 2. Quick Policy Check

```bash
python smoke_test.py
```

Expected output:

```text
Smoke test passed: policy checks and reproducibility are valid.
```

This check validates the public data-generation policy, mandatory metadata, and
deterministic generation.

## 3. Manuscript-Facing Frozen Configuration

The exact manuscript-facing frozen protocol is documented in:

```text
configs/fnl_primary.yaml
```

This file records the four frozen test scenarios, train/calibration/test seed
ranges, primary 25-step matching margin, matching-margin robustness margins,
series-level no-change F1 convention, and learned-detector post-processing used
in the manuscript table (`threshold 0.15; NMS 10` for LSTM_v2, GRU_v2, and
Transformer_v2).

`configs/base.yaml` remains a generic example/default rerun configuration. It
is intentionally not the frozen manuscript configuration.

## 4. Primary Benchmark Tables

```bash
python scripts/build_publication_tables.py
```

Inputs:

- `results/kaggle_output_v2/*/summary.csv`
- `results/kaggle_output_v2/*/per_series.csv`

Outputs:

- `manuscript/assets/tables/main_benchmark_recomputed.csv`
- `manuscript/assets/tables/main_benchmark_top5.tex`
- `manuscript/assets/tables/bootstrap_ci_by_series.csv`
- `manuscript/assets/tables/bootstrap_f1_ci_top5.tex`
- `manuscript/assets/tables/paired_wilcoxon_key_comparisons.csv`
- `manuscript/assets/tables/paired_wilcoxon_key_comparisons.tex`
- `manuscript/assets/tables/p0_artifact_validation.md`
- `manuscript/assets/tables/p0_artifact_validation.json`

## 5. Same-Noise Diagnostic Tables

```bash
python scripts/build_same_noise_tables.py
```

Inputs:

- `results/publication_same_noise/*/eval_multi/*.csv`
- `results/publication_same_noise_classical/*/eval_multi/*.csv`
- `results/kaggle_output_v2/*/per_series.csv`

Outputs:

- `manuscript/assets/tables/same_noise_comparison_summary.csv`
- `manuscript/assets/tables/same_noise_comparison_validation.md`
- `manuscript/assets/tables/same_noise_comparison_validation.csv`
- `manuscript/assets/tables/same_noise_transformer_validation.md`
- `manuscript/assets/tables/same_noise_transformer_validation.csv`
- `manuscript/assets/tables/same_noise_comparison_bootstrap_ci.csv`
- `manuscript/assets/tables/same_noise_comparison_f1_ci.tex`
- `manuscript/assets/tables/same_noise_transformer_vs_classical_paired.csv`
- `manuscript/assets/tables/same_noise_transformer_vs_classical_paired.tex`
- `manuscript/assets/tables/same_noise_transformer_vs_universal_paired.csv`
- `manuscript/assets/tables/same_noise_transformer_vs_universal_paired.tex`

## 6. Compact Manuscript Tables

```bash
python scripts/build_article_compact_tables.py
```

Inputs:

- `manuscript/assets/tables/matching_margin_robustness_summary.csv`
- `manuscript/assets/tables/same_noise_comparison_summary.csv`

Outputs:

- `manuscript/assets/tables/matching_margin_robustness_leaders.tex`
- `manuscript/assets/tables/same_noise_best_f1_summary.tex`

## 7. Figures

```bash
python scripts/build_publication_figures.py
```

Inputs:

- `manuscript/assets/tables/main_benchmark_recomputed.csv`
- `manuscript/assets/tables/matching_margin_robustness_summary.csv`
- `results/matching_margin_robustness_outputs/raw_predictions.jsonl`
- `algoritms/data_utils.py`

Outputs:

- `manuscript/assets/figures/fig01_generated_regime_examples.{pdf,png}`
- `manuscript/assets/figures/fig02_primary_f1_benchmark.{pdf,png}`
- `manuscript/assets/figures/fig03_matching_margin_robustness.{pdf,png}`
- `manuscript/assets/figures/fig04_f1_mae_tradeoff.{pdf,png}`
- `manuscript/assets/figures/fig05_detection_overlay.{pdf,png}`
- `manuscript/assets/figures/publication_figures_manifest.md`

## 8. Manuscript PDF

```bash
cd manuscript/src
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected output:

```text
manuscript/src/main.pdf
```

Optional supplementary material is provided as:

```text
manuscript/src/supplementary.tex
manuscript/src/supplementary.pdf
```

## 9. Expensive Reruns

The committed manuscript artifacts are rebuilt from frozen outputs. Full
evaluation reruns can be expensive and may depend on model weights and hardware.
Use them for independent re-evaluation, not for exact manuscript rebuilds.
`configs/base.yaml` is the generic example used by this command; for the exact
manuscript-facing frozen protocol, read `configs/fnl_primary.yaml`.

Example:

```bash
python eval_multi.py --config configs/base.yaml --n-series 50 --output-dir results/eval_multi_rerun
```
