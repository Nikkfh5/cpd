"""
Тест Петтитта для обнаружения единственной точки изменения среднего.

Непараметрический ранговый тест, не требующий предположений о распределении.

Основная идея:
    Для каждой возможной точки разделения t вычисляется «ранговое превосходство»
    левой части ряда над правой.

Статистика:
    U_{t,T} = ΣΣ sign(X_i − X_j),  i = 1..t,  j = t+1..T
    sign(a) = −1 если a < 0,  0 если a = 0,  +1 если a > 0

Критерий:
    K_T = max |U_{t,T}| по всем t
    t_change = argmax |U_{t,T}|  — точка изменения

p-value ≈ 2 · exp(−6 · K_T² / (T³ + T²))

Если p-value < α → отвергаем H0 об однородности среднего.

Особенности:
    - Обнаруживает только одну точку изменения.
    - Лучше работает для изменений в центре ряда.
    - Устойчив к выбросам.
    - Чувствителен к длине ряда и уровню шума.
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

    :param kwargs: словарь kwargs, переданный функции pettitt
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


def pettitt(
    series: np.ndarray,
    seed: int,
    alpha: float = _ALPHA,
    **kwargs,
) -> tuple[list[int], None, dict]:
    """
    Тест Петтитта для обнаружения единственной точки изменения среднего.

    Непараметрический ранговый тест на основе статистики K_T = max|U_{t,T}|.

    :param series: одномерный временной ряд
    :param seed: случайное зерно (не используется тестом, сохраняется в meta)
    :param alpha: уровень значимости для принятия решения о разладке
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, None, meta)
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    result = hg.pettitt_test(series)
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

    cps, _, meta = pettitt(series, seed=generation_params["seed"], **run_kwargs)
    print(f"pettitt  cp={cps}  p={meta['p_value']:.4f}")
