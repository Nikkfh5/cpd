# Data Generation Contract

This document defines the public reproducibility contract for the generated
trajectory benchmark used in the manuscript.

## Scope

The benchmark uses simulated one-dimensional trajectories with level-crossing
labels. The generator is a discrete-time stochastic map motivated by nonlinear
stochastic dynamics:

```text
x[t + 1] = x[t] + sin(x[t]) * dt + sqrt(D) * error[t]
```

The benchmark does not rely on a strict Euler-Maruyama discretization of a
continuous-time stochastic differential equation. In particular, the committed
generator follows the project convention above and normalizes the innovation
sequence before simulation.

## Authoritative Generator

The executable source of truth is:

```text
algoritms/data_utils.py
```

Reportable runs should use `dataset_source: "data_utils"`.

## Allowed Configuration Values

Allowed `dataset_source` value:

- `data_utils`

Allowed `noise_type` values:

- `white`
- `pink`
- `brownian`
- `violet`
- `blue`

Every generation config must include:

- `length`
- `dt`
- `D`
- `noise_type`
- `seed`

## Output Schema

The generator returns:

- `x_values`: NumPy array with shape `(length + 1,)`
- `cp_labels`: binary NumPy array with shape `(length + 1,)`

Change-point labels mark transitions between generated state-space levels after
simulation. The benchmark treats these events as transition-time labels.

## Metadata Contract

Every reportable method output must include `meta` with:

- `dataset_source`
- `generation_params`
- `source_path`

`generation_params` must include `length`, `dt`, `D`, `noise_type`, and `seed`.

Entrypoints should fail if:

- `dataset_source` is missing or invalid;
- a required generation parameter is missing;
- `noise_type` is outside the allowed set;
- output metadata is missing required fields.

## Evaluation Indexing

- Internal time-series shape is `(T,)`.
- Legacy method adapters may accept `(T,)` or `(T, 1)`.
- Change-point indices are 0-based.
- No-change output is `[]` for change-point lists and an all-zero vector for
  label vectors.

## Public Verification Command

Run:

```bash
python smoke_test.py
```

Expected output:

```text
Smoke test passed: policy checks and reproducibility are valid.
```
