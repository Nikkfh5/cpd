"""
SNHT (Standard Normal Homogeneity Test) — тест стандартной нормальной однородности.

Обнаружение точки изменения среднего во временном ряду путём сравнения
локальных средних слева и справа от каждой точки.

Алгоритм:
    1. Для каждой точки i вычисляются:
       - z_i = (x_i − x̄) / σ  — стандартизованные отклонения
       - T₀(τ) = τ · z̄₁² + (n − τ) · z̄₂²,
         где z̄₁ = среднее z_i за i = 1..τ,
             z̄₂ = среднее z_i за i = τ+1..n
    2. Точка изменения: τ* = argmax T₀(τ)
    3. Если T₀(τ*) превышает критическое значение → значимый разрыв.

Особенности:
    - Параметрический тест (предполагает нормальность).
    - Обнаруживает единственную точку изменения среднего.
    - Чувствителен к выбросам (существует робастная версия с M-оценками Хубера).
    - Не работает для первых и последних точек ряда.
"""

from pathlib import Path

import numpy as np
import pyhomogeneity as hg


_DEFAULT_SOURCE_PATH = str(Path(__file__).resolve().parent / "data_utils.py")
_ALPHA = 0.05


def _build_meta(kwargs: dict, result, seed: int) -> dict:
    """
    Формирует словарь метаданных запуска.

    Извлекает dataset_source, generation_params и source_path из kwargs;
    добавляет p_value из объекта результата pyhomogeneity.

    :param kwargs: словарь kwargs, переданный функции snht
    :param result: объект результата из pyhomogeneity
    :param seed: случайное зерно
    :return: словарь meta
    """
    return {
        "dataset_source": kwargs.get("dataset_source", "data_utils"),
        "generation_params": kwargs.get("generation_params", {}),
        "source_path": kwargs.get("source_path", _DEFAULT_SOURCE_PATH),
        "p_value": float(result.p),
        "seed": seed,
    }


def _cp_from_result(result, alpha: float) -> list[int]:
    """
    Извлекает 0-based индекс точки разладки из результата pyhomogeneity.

    Возвращает пустой список, если p-value превышает порог alpha
    или если cp равен None.

    :param result: объект результата из pyhomogeneity
    :param alpha: уровень значимости
    :return: list[int] — либо [cp_index], либо []
    """
    if result.p > alpha or result.cp is None:
        return []
    return [int(result.cp)]


def snht(
    series: np.ndarray,
    seed: int,
    alpha: float = _ALPHA,
    **kwargs,
) -> tuple[list[int], None, dict]:
    """
    SNHT для обнаружения единственной точки изменения среднего.

    Параметрический тест на основе статистики T₀(τ) = τ·z̄₁² + (n−τ)·z̄₂².

    :param series: одномерный временной ряд
    :param seed: случайное зерно (не используется тестом, сохраняется в meta)
    :param alpha: уровень значимости для принятия решения о разладке
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, None, meta)
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    result = hg.snht_test(series)
    change_points = _cp_from_result(result, alpha)
    meta = _build_meta(kwargs, result, seed)
    return change_points, None, meta


if __name__ == "__main__":
    from algoritms.data_utils import generate_dataset

    generation_params = {"length": 359, "dt": 1.0, "D": 0.5, "noise_type": "white", "seed": 42}
    series, _ = generate_dataset(**generation_params)

    run_kwargs = {
        "dataset_source": "data_utils",
        "generation_params": generation_params,
    }

    cps, _, meta = snht(series, seed=generation_params["seed"], **run_kwargs)
    print(f"snht  cp={cps}  p={meta['p_value']:.4f}")
