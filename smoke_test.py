from copy import deepcopy

import numpy as np

from data_source_policy import (
    PolicyError,
    generate_from_policy,
    validate_meta,
)


def _base_config() -> dict:
    return {
        "dataset_source": "data_utils",
        "generation": {
            "length": 1000,
            "dt": 1.0,
            "D": 0.5,
            "noise_type": "white",
            "seed": 42,
        },
        "dataset_name": "smoke_dataset",
        "output_dir": "results/smoke",
    }


def _expect_policy_error(fn, needle: str) -> None:
    try:
        fn()
    except PolicyError as exc:
        if needle not in str(exc):
            raise AssertionError(f"Unexpected PolicyError message: {exc}") from exc
        return
    raise AssertionError("Expected PolicyError was not raised")


def main() -> int:
    cfg = _base_config()
    x_values, cp_labels, meta = generate_from_policy(cfg)
    validate_meta(meta)

    assert len(x_values) == cfg["generation"]["length"] + 1
    assert len(cp_labels) == cfg["generation"]["length"] + 1

    invalid_source_cfg = deepcopy(cfg)
    invalid_source_cfg["dataset_source"] = "custom_random"
    _expect_policy_error(lambda: generate_from_policy(invalid_source_cfg), "dataset_source")

    missing_seed_cfg = deepcopy(cfg)
    del missing_seed_cfg["generation"]["seed"]
    _expect_policy_error(lambda: generate_from_policy(missing_seed_cfg), "generation")

    _expect_policy_error(lambda: validate_meta({}), "meta")

    x1, cp1, _ = generate_from_policy(cfg)
    x2, cp2, _ = generate_from_policy(cfg)
    if not np.array_equal(x1, x2):
        raise AssertionError("Reproducibility check failed for x_values")
    if not np.array_equal(cp1, cp2):
        raise AssertionError("Reproducibility check failed for cp_labels")

    print("Smoke test passed: policy checks and reproducibility are valid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
