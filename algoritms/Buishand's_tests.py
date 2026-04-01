"""
Тесты Буишанда для обнаружения одиночной точки разладки в среднем.

Дан ряд x₁,…,xₙ.
Обозначения:
    x̄ = (1/n) Σᵢ xᵢ
    σ² = (1/(n−1)) Σᵢ (xᵢ − x̄)²
    Sₖ = Σᵢ₌₁ᵏ (xᵢ − x̄),  k = 1,…,n   (кумулятивные отклонения от среднего)

При однородном среднем траектория Sₖ блуждает около 0;
при скачке среднего Sₖ после τ уходит примерно линейно вверх/вниз.

Тест Q (Буишанда):
    Q = max₁≤k≤n |Sₖ| / (σ√n)
    Точка разладки ≈ argmax |Sₖ|
    Измеряет максимум отклонения траектории от 0;
    разрывы близко к краям ряда ловит хуже.

Тест Range (R, Буишанда):
    R = (maxₖ Sₖ − minₖ Sₖ) / (σ√n)
    Учитывает весь размах Sₖ;
    особенно чувствителен к разрыву в середине ряда.

Тест Likelihood Ratio (V, Буишанда):
    V = max₁≤k≤n−1 |Sₖ| / (σ √(k·(n−k)))
    Штрафует точки близко к краям через √(k(n−k));
    реализует LR-подход «одно среднее» против «два разных средних».

Тест U (Буишанда):
    U = [1 / (n(n+1))] Σₖ₌₁^{n−1} (Sₖ/σ)²
    Использует всю траекторию Sₖ (сумма квадратов);
    устойчив к шуму, хорошо ловит разрыв в середине, но хуже локализует τ.
"""

from pathlib import Path

import numpy as np
import pyhomogeneity as hg


_DEFAULT_SOURCE_PATH = str(Path(__file__).resolve().parent / "data_utils.py")
_ALPHA = 0.05


def _compute_sk(series: np.ndarray) -> np.ndarray:
    """
    Вычисляет массив кумулятивных отклонений от среднего.

    Sₖ = Σᵢ₌₁ᵏ (xᵢ − x̄) для k = 1,…,n.

    :param series: одномерный массив длины n
    :return: np.ndarray длины n
    """
    mean = np.mean(series)
    return np.cumsum(series - mean)


def _build_meta(kwargs: dict, result) -> dict:
    """
    Формирует словарь метаданных запуска.

    Извлекает dataset_source, generation_params и source_path из kwargs;
    добавляет p_value из объекта результата pyhomogeneity.

    :param kwargs: словарь kwargs, переданный функции-алгоритму
    :param result: объект результата из pyhomogeneity
    :return: словарь meta
    """
    return {
        "dataset_source": kwargs.get("dataset_source", "data_utils"),
        "generation_params": kwargs.get("generation_params", {}),
        "source_path": kwargs.get("source_path", _DEFAULT_SOURCE_PATH),
        "p_value": float(result.p),
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


def buishand_q(
    series: np.ndarray,
    seed: int,
    alpha: float = _ALPHA,
    **kwargs,
) -> tuple[list[int], np.ndarray, dict]:
    """
    Тест Буишанда Q для обнаружения точки разладки в среднем.

    Статистика Q = max₁≤k≤n |Sₖ| / (σ√n).
    Точка разладки определяется как argmax |Sₖ|.

    :param series: одномерный временной ряд
    :param seed: случайное зерно (не используется тестом, сохраняется в meta)
    :param alpha: уровень значимости для принятия решения о разладке
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, scores, meta)
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    result = hg.buishand_q_test(series)
    change_points = _cp_from_result(result, alpha)
    scores = _compute_sk(series)
    meta = _build_meta(kwargs, result)
    meta["seed"] = int(seed)
    return change_points, scores, meta


def buishand_range(
    series: np.ndarray,
    seed: int,
    alpha: float = _ALPHA,
    **kwargs,
) -> tuple[list[int], np.ndarray, dict]:
    """
    Тест Буишанда Range (R) для обнаружения точки разладки в среднем.

    Статистика R = (maxₖ Sₖ − minₖ Sₖ) / (σ√n).
    Особенно чувствителен к разрыву в середине ряда.

    :param series: одномерный временной ряд
    :param seed: случайное зерно (не используется тестом, сохраняется в meta)
    :param alpha: уровень значимости для принятия решения о разладке
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, scores, meta)
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    result = hg.buishand_range_test(series)
    change_points = _cp_from_result(result, alpha)
    scores = _compute_sk(series)
    meta = _build_meta(kwargs, result)
    meta["seed"] = int(seed)
    return change_points, scores, meta


def buishand_likelihood_ratio(
    series: np.ndarray,
    seed: int,
    alpha: float = _ALPHA,
    **kwargs,
) -> tuple[list[int], np.ndarray, dict]:
    """
    Тест Буишанда Likelihood Ratio (V) для обнаружения точки разладки в среднем.

    Статистика V = max₁≤k≤n−1 |Sₖ| / (σ √(k·(n−k))).
    Штрафует точки близко к краям ряда; реализует подход LR.

    :param series: одномерный временной ряд
    :param seed: случайное зерно (не используется тестом, сохраняется в meta)
    :param alpha: уровень значимости для принятия решения о разладке
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, scores, meta)
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    result = hg.buishand_likelihood_ratio_test(series)
    change_points = _cp_from_result(result, alpha)
    scores = _compute_sk(series)
    meta = _build_meta(kwargs, result)
    meta["seed"] = int(seed)
    return change_points, scores, meta


def buishand_u(
    series: np.ndarray,
    seed: int,
    alpha: float = _ALPHA,
    **kwargs,
) -> tuple[list[int], np.ndarray, dict]:
    """
    Тест Буишанда U для обнаружения точки разладки в среднем.

    Статистика U = [1/(n(n+1))] Σₖ₌₁^{n−1} (Sₖ/σ)².
    Использует всю траекторию Sₖ; устойчив к шуму.

    :param series: одномерный временной ряд
    :param seed: случайное зерно (не используется тестом, сохраняется в meta)
    :param alpha: уровень значимости для принятия решения о разладке
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, scores, meta)
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    result = hg.buishand_u_test(series)
    change_points = _cp_from_result(result, alpha)
    scores = _compute_sk(series)
    meta = _build_meta(kwargs, result)
    meta["seed"] = int(seed)
    return change_points, scores, meta


if __name__ == "__main__":
    from algoritms.data_utils import generate_dataset

    generation_params = {"length": 359, "dt": 1.0, "D": 0.5, "noise_type": "white", "seed": 42}
    series, _ = generate_dataset(**generation_params)

    run_kwargs = {
        "dataset_source": "data_utils",
        "generation_params": generation_params,
    }

    for fn in [buishand_q, buishand_range, buishand_likelihood_ratio, buishand_u]:
        cps, scores, meta = fn(series, seed=generation_params["seed"], **run_kwargs)
        print(f"{fn.__name__:35s}  cp={cps}  p={meta['p_value']:.4f}")
