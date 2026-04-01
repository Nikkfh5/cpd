"""
Тест Chow для обнаружения структурного разрыва в известный или неизвестный момент.

Модель: y_t = β' x_t + ε_t,  t = 1..n.
Точка разрыва T0 может быть задана явно или определена автоматически (sweep).

H0: коэффициенты регрессии до и после T0 одинаковы (одна модель для всего ряда).
H1: хотя бы один коэффициент отличается (есть структурный разрыв в T0).

Идея:
1) Строим одну «общую» регрессию по всей выборке → RSS_pooled.
2) Строим две отдельные регрессии:
   – по данным до T0 → RSS1,
   – по данным после T0 → RSS2.
3) Оцениваем, насколько сильно уменьшилась сумма квадратов остатков,
   если позволить коэффициентам быть разными до/после разрыва.

Обозначения:
    k  = число коэффициентов в одной регрессии (включая константу),
    n1 = число наблюдений до разрыва,
    n2 = число наблюдений после разрыва.

F-статистика Chow:
    F = ((RSS_pooled − (RSS1 + RSS2)) / k) / ((RSS1 + RSS2) / (n1 + n2 − 2k))

При H0: F ~ F_{k, n1 + n2 − 2k}.
Если F велик (p-value < alpha) → отвергаем H0 → коэффициенты до и после T0 различаются.

Особенности:
- Классический Chow предполагает одинаковую дисперсию ошибок до/после (гомоскедастичность).
- Для неизвестного момента разрыва функция поддерживает режим sweep (supF):
  перебираются все возможные T0, возвращается argmax F-статистики.
- Для известного T0 используется один вызов библиотеки chow_test.
"""

from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


_DEFAULT_SOURCE_PATH = str(Path(__file__).resolve().parent / "data_utils.py")


def _rss(x: np.ndarray, y: np.ndarray) -> float:
    """Сумма квадратов остатков линейной регрессии y ~ x."""
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    residuals = y - model.predict(x.reshape(-1, 1))
    return float((residuals ** 2).sum())


def _chow_f_at(series: np.ndarray, t0: int) -> tuple[float, float]:
    """
    Вычисляет F-статистику и p-value теста Chow в точке t0.

    В качестве независимой переменной используется временной индекс np.arange(n).

    :param series: одномерный временной ряд длины n
    :param t0: индекс предполагаемой точки разрыва (0-based)
    :return: (f_stat, p_val)
    """
    n = len(series)
    x = np.arange(n, dtype=float)
    k = 2

    rss_pooled = _rss(x, series)
    rss1 = _rss(x[:t0], series[:t0])
    rss2 = _rss(x[t0:], series[t0:])

    numerator = (rss_pooled - (rss1 + rss2)) / k
    denominator = (rss1 + rss2) / (n - 2 * k)
    f_stat = numerator / denominator if denominator > 0 else 0.0
    p_val = float(1 - stats.f.cdf(f_stat, dfn=k, dfd=(n - 2 * k)))
    return f_stat, p_val


def _build_meta(
    kwargs: dict,
    t0: int,
    f_stat: float,
    p_val: float,
    seed: int,
) -> dict:
    """
    Формирует словарь метаданных запуска.

    :param kwargs: словарь kwargs, переданный функции chow
    :param t0: использованная точка разрыва
    :param f_stat: значение F-статистики
    :param p_val: p-value
    :param seed: случайное зерно
    :return: словарь meta
    """
    return {
        "dataset_source": kwargs.get("dataset_source", "data_utils"),
        "generation_params": kwargs.get("generation_params", {}),
        "source_path": kwargs.get("source_path", _DEFAULT_SOURCE_PATH),
        "t0": t0,
        "f_statistic": f_stat,
        "p_value": p_val,
        "seed": seed,
    }


def chow(
    series: np.ndarray,
    seed: int,
    t0: int | None = None,
    alpha: float = 0.05,
    min_segment: int = 10,
    **kwargs,
) -> tuple[list[int], np.ndarray | None, dict]:
    """
    Тест Chow для обнаружения структурного разрыва в среднем/тренде.

    Поддерживает два режима:
    - t0 задан явно: однократный вызов теста в указанной точке;
    - t0 = None (sweep): перебор всех позиций, возврат argmax F-статистики (supF).

    :param series: одномерный временной ряд
    :param seed: случайное зерно (не используется тестом, сохраняется в meta)
    :param t0: известная точка разрыва (0-based); если None — режим sweep
    :param alpha: уровень значимости для принятия решения о разладке
    :param min_segment: минимальная длина сегмента при sweep
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, scores, meta)
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    n = len(series)

    if t0 is not None:
        f_stat, p_val = _chow_f_at(series, int(t0))
        scores = None
        best_t0 = int(t0)
    else:
        scores = np.zeros(n, dtype=float)
        for candidate in range(min_segment, n - min_segment):
            f, _ = _chow_f_at(series, candidate)
            scores[candidate] = f
        best_t0 = int(np.argmax(scores))
        f_stat, p_val = _chow_f_at(series, best_t0)

    change_points = [best_t0] if p_val < alpha else []
    meta = _build_meta(kwargs, best_t0, f_stat, p_val, seed)
    return change_points, scores, meta


if __name__ == "__main__":
    from algoritms.data_utils import generate_dataset

    generation_params = {"length": 359, "dt": 1.0, "D": 0.5, "noise_type": "white", "seed": 42}
    series, _ = generate_dataset(**generation_params)

    run_kwargs = {
        "dataset_source": "data_utils",
        "generation_params": generation_params,
    }

    cps, scores, meta = chow(series, seed=42, **run_kwargs)
    print(f"sweep   cp={cps}  F={meta['f_statistic']:.3f}  p={meta['p_value']:.4f}")

    cps2, _, meta2 = chow(series, seed=42, t0=180, **run_kwargs)
    print(f"t0=180  cp={cps2}  F={meta2['f_statistic']:.3f}  p={meta2['p_value']:.4f}")
