from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from algoritms.data_utils import generate_dataset

ALLOWED_DATASET_SOURCES = {"data_utils"}
ALLOWED_NOISE_TYPES = {"white", "pink", "brownian", "violet", "blue"}
REQUIRED_GENERATION_KEYS = ("length", "dt", "D", "noise_type", "seed")


class PolicyError(ValueError):
    pass


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _source_path(dataset_source: str) -> str:
    root = _repo_root()
    mapping = {
        "data_utils": root / "algoritms" / "data_utils.py",
    }
    return str(mapping[dataset_source].resolve())


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise PolicyError(f"{context} must be an object")
    return value


def _normalize_generation(generation: dict[str, Any]) -> dict[str, Any]:
    missing = [key for key in REQUIRED_GENERATION_KEYS if key not in generation]
    if missing:
        raise PolicyError(f"generation is missing required keys: {missing}")

    length = int(generation["length"])
    dt = float(generation["dt"])
    diffusion = float(generation["D"])
    noise_type = str(generation["noise_type"])
    seed = int(generation["seed"])

    if length <= 0:
        raise PolicyError("generation.length must be > 0")
    if dt <= 0:
        raise PolicyError("generation.dt must be > 0")
    if diffusion < 0:
        raise PolicyError("generation.D must be >= 0")
    if noise_type not in ALLOWED_NOISE_TYPES:
        raise PolicyError(
            f"generation.noise_type must be one of {sorted(ALLOWED_NOISE_TYPES)}"
        )

    return {
        "length": length,
        "dt": dt,
        "D": diffusion,
        "noise_type": noise_type,
        "seed": seed,
    }


def load_config(config_path: str) -> dict[str, Any]:
    config_text = Path(config_path).read_text(encoding="utf-8")

    yaml_data = None
    try:
        import yaml  # type: ignore

        yaml_data = yaml.safe_load(config_text)
    except Exception:
        yaml_data = None

    if yaml_data is None:
        try:
            yaml_data = json.loads(config_text)
        except json.JSONDecodeError as exc:
            raise PolicyError(
                "Config parsing failed. Install PyYAML or use JSON-compatible YAML."
            ) from exc

    if not isinstance(yaml_data, dict):
        raise PolicyError("Config root must be an object")

    return yaml_data


def validate_and_normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    mapping = _require_mapping(config, "config")
    dataset_source = mapping.get("dataset_source")
    if not isinstance(dataset_source, str) or dataset_source not in ALLOWED_DATASET_SOURCES:
        raise PolicyError(
            f"dataset_source must be one of {sorted(ALLOWED_DATASET_SOURCES)}"
        )

    generation_raw = _require_mapping(mapping.get("generation"), "generation")
    generation = _normalize_generation(generation_raw)

    return {
        "dataset_source": dataset_source,
        "generation": generation,
        "dataset_name": str(mapping.get("dataset_name", "synthetic_default")),
        "output_dir": str(mapping.get("output_dir", "results/policy_preflight")),
    }


def build_meta(dataset_source: str, generation: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset_source": dataset_source,
        "generation_params": generation,
        "source_path": _source_path(dataset_source),
    }


def validate_meta(meta: dict[str, Any]) -> None:
    mapping = _require_mapping(meta, "meta")

    for key in ("dataset_source", "generation_params", "source_path"):
        if key not in mapping:
            raise PolicyError(f"meta is missing required key: {key}")

    dataset_source = mapping["dataset_source"]
    if dataset_source not in ALLOWED_DATASET_SOURCES:
        raise PolicyError(
            f"meta.dataset_source must be one of {sorted(ALLOWED_DATASET_SOURCES)}"
        )

    generation = _require_mapping(mapping["generation_params"], "meta.generation_params")
    for key in REQUIRED_GENERATION_KEYS:
        if key not in generation:
            raise PolicyError(f"meta.generation_params is missing required key: {key}")


def generate_from_policy(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    normalized = validate_and_normalize_config(config)
    generation = normalized["generation"]

    x_values, cp_labels = generate_dataset(
        length=generation["length"],
        dt=generation["dt"],
        D=generation["D"],
        noise_type=generation["noise_type"],
        seed=generation["seed"],
    )

    meta = build_meta(normalized["dataset_source"], generation)
    validate_meta(meta)
    return x_values, cp_labels, meta


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def summarize_dataset(x_values: np.ndarray, cp_labels: np.ndarray) -> dict[str, Any]:
    change_points = np.where(cp_labels == 1)[0].tolist()
    return {
        "series_length": int(len(x_values)),
        "cp_label_length": int(len(cp_labels)),
        "change_points_count": int(len(change_points)),
        "change_points": change_points,
    }
