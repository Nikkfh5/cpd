"""
Обнаружение точек разладки алгоритмом SWAB (Sliding Window And Bottom-up).

Алгоритм работает в два этапа:

Этап 1 — генерация кандидатов (swab_candidates):
    Скользящее окно шириной w проходит по ряду.
    Внутри каждого окна выполняется Bottom-Up сегментация:
    ряд дробится на начальные сегменты длины min_len,
    затем жадно объединяются соседние сегменты с наименьшим ростом SSE.
    Правая граница первого (оставшегося) сегмента окна — кандидат точки разладки.
    Окно сдвигается к следующей позиции после кандидата.

Этап 2 — валидация кандидатов (detect_cps_with_prob):
    Каждый кандидат проверяется тестом Chow в окне ±win.
    Применяется поправка FWER (Бонферрони): порог = 1 − α/m.
    Принимаются только кандидаты с prob ≥ порога.

F-статистика Chow:
    F = ((RSS_pooled − (RSS₁ + RSS₂)) / k) / ((RSS₁ + RSS₂) / (n₁ + n₂ − 2k))
    При H0: F ~ F_{k, n₁+n₂−2k}
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy.stats import f as f_dist
except Exception as e:
    raise ImportError("Нужен scipy (scipy.stats.f)") from e


_DEFAULT_SOURCE_PATH = str(Path(__file__).resolve().parent / "data_utils.py")

ALPHA_FWER = 0.05
PROB_THRESHOLD = None
MIN_CP_GAP = 100
TEST_WIN = 200


@dataclass
class Segment:
    """Линейный сегмент временного ряда с предвычисленными суммами для МНК."""

    t0: int
    t1: int
    n: int
    st: float
    st2: float
    sy: float
    sty: float
    sy2: float

    @staticmethod
    def from_arrays(t: np.ndarray, y: np.ndarray, i0: int, i1: int) -> "Segment":
        """Строит сегмент из массивов временных индексов t и значений y."""
        tt = t[i0:i1]
        yy = y[i0:i1]
        n = int(yy.size)
        st = float(tt.sum())
        st2 = float((tt * tt).sum())
        sy = float(yy.sum())
        sty = float((tt * yy).sum())
        sy2 = float((yy * yy).sum())
        return Segment(int(tt[0]), int(tt[-1]), n, st, st2, sy, sty, sy2)

    def merge(self, other: "Segment") -> "Segment":
        """Объединяет два соседних сегмента."""
        return Segment(
            self.t0,
            other.t1,
            self.n + other.n,
            self.st + other.st,
            self.st2 + other.st2,
            self.sy + other.sy,
            self.sty + other.sty,
            self.sy2 + other.sy2,
        )

    def fit_ab(self) -> tuple[float, float]:
        """Оценивает коэффициенты линейной регрессии y = a + b·t по МНК."""
        denom = self.n * self.st2 - self.st * self.st
        if denom == 0.0:
            b = 0.0
            a = self.sy / self.n
            return a, b
        b = (self.n * self.sty - self.st * self.sy) / denom
        a = (self.sy - b * self.st) / self.n
        return a, b

    def sse_linear(self) -> float:
        """Сумма квадратов остатков линейной регрессии y = a + b·t."""
        a, b = self.fit_ab()
        sse = (
            self.sy2
            - 2.0 * a * self.sy
            - 2.0 * b * self.sty
            + self.n * a * a
            + 2.0 * a * b * self.st
            + b * b * self.st2
        )
        if sse < 0.0:
            sse = 0.0
        return float(sse)

    def sse_const(self) -> float:
        """Сумма квадратов остатков модели константы y = μ."""
        if self.n <= 0:
            return 0.0
        mu = self.sy / self.n
        sse = self.sy2 - 2.0 * mu * self.sy + self.n * mu * mu
        if sse < 0.0:
            sse = 0.0
        return float(sse)


@dataclass
class CPEvent:
    """Событие точки разладки с F-статистикой и вероятностью."""

    cp: int
    prob: float
    p_value: float
    F: float
    df1: int
    df2: int


def initial_segments(
    t: np.ndarray,
    y: np.ndarray,
    min_len: int,
) -> list[Segment]:
    """
    Разбивает ряд на начальные сегменты длины min_len для Bottom-Up сегментации.

    Последний сегмент объединяется с предпоследним, если он короче min_len.

    :param t: массив временных индексов
    :param y: массив значений
    :param min_len: минимальная длина сегмента
    :return: список сегментов
    """
    n = int(y.size)
    if n <= 0:
        return []

    cuts = []
    i = 0
    while i < n:
        j = min(i + min_len, n)
        cuts.append((i, j))
        i = j

    if len(cuts) >= 2:
        a0, a1 = cuts[-2]
        b0, b1 = cuts[-1]
        if (b1 - b0) < min_len:
            cuts = cuts[:-2] + [(a0, b1)]

    return [Segment.from_arrays(t, y, i0, i1) for i0, i1 in cuts]


def bottom_up_in_buffer(
    t: np.ndarray,
    y: np.ndarray,
    min_len: int,
    target_segments: int = 6,
    max_merge_cost: float | None = None,
) -> list[Segment]:
    """
    Bottom-Up сегментация внутри одного буфера.

    Жадно объединяет соседние сегменты с наименьшим ростом SSE
    до достижения числа сегментов target_segments или превышения max_merge_cost.

    :param t: массив временных индексов буфера
    :param y: массив значений буфера
    :param min_len: минимальная длина начального сегмента
    :param target_segments: целевое число сегментов
    :param max_merge_cost: максимально допустимый рост SSE при слиянии
    :return: список сегментов после объединения
    """
    segs = initial_segments(t, y, min_len)
    if len(segs) <= 1:
        return segs

    target_segments = max(1, int(target_segments))

    while len(segs) > target_segments:
        best_i = -1
        best_cost = None
        best_merged = None

        for i in range(len(segs) - 1):
            merged = segs[i].merge(segs[i + 1])
            cost = merged.sse_linear() - segs[i].sse_linear() - segs[i + 1].sse_linear()
            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_i = i
                best_merged = merged

        if best_i < 0:
            break

        if max_merge_cost is not None and best_cost is not None and best_cost > max_merge_cost:
            break

        segs = segs[:best_i] + [best_merged] + segs[best_i + 2:]

    return segs


def swab_candidates(
    y: np.ndarray,
    w: int = 400,
    min_len: int = 50,
    target_segments: int = 6,
    max_merge_cost: float | None = None,
) -> list[int]:
    """
    Генерирует кандидатов точек разладки алгоритмом SWAB.

    Скользящее окно шириной w перемещается по ряду.
    Внутри окна выполняется Bottom-Up сегментация.
    Правая граница первого сегмента окна — кандидат.

    :param y: временной ряд
    :param w: ширина скользящего окна
    :param min_len: минимальная длина сегмента в Bottom-Up
    :param target_segments: целевое число сегментов в Bottom-Up
    :param max_merge_cost: максимально допустимый рост SSE при слиянии
    :return: отсортированный список 0-based индексов кандидатов
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.size)
    t = np.arange(n, dtype=int)

    cps = []
    left = 0

    while left < n:
        right = min(left + w, n)
        tb = t[left:right]
        yb = y[left:right]

        if yb.size < min_len:
            break

        segs = bottom_up_in_buffer(
            tb, yb,
            min_len=min_len,
            target_segments=target_segments,
            max_merge_cost=max_merge_cost,
        )
        if len(segs) == 0:
            break

        first = segs[0]
        cp = int(first.t1)
        if cp < n - 1:
            cps.append(cp)

        left = cp + 1

    return cps


def chow_f_test_window(
    y: np.ndarray,
    cp: int,
    win: int,
    model: str = "linear",
):
    """
    Тест Chow в окне ±win вокруг кандидата cp.

    Сравнивает единую модель для всего окна и две модели до/после cp.

    :param y: временной ряд
    :param cp: индекс кандидата (0-based)
    :param win: полуширина окна
    :param model: 'linear' (тренд) или 'const' (среднее)
    :return: (prob, p_value, F, df1, df2) или None если окно выходит за границы
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.size)

    l0 = cp - win + 1
    l1 = cp + 1
    r0 = cp + 1
    r1 = cp + 1 + win

    if l0 < 0 or r1 > n:
        return None

    t = np.arange(n, dtype=int)
    left = Segment.from_arrays(t, y, l0, l1)
    right = Segment.from_arrays(t, y, r0, r1)
    merged = left.merge(right)

    if model == "const":
        k = 1
        sse_l = left.sse_const()
        sse_r = right.sse_const()
        sse_m = merged.sse_const()
    else:
        k = 2
        sse_l = left.sse_linear()
        sse_r = right.sse_linear()
        sse_m = merged.sse_linear()

    nn = left.n + right.n
    df1 = int(k)
    df2 = int(nn - 2 * k)
    if df2 <= 0:
        return 0.0, 1.0, 0.0, df1, df2

    num = (sse_m - (sse_l + sse_r)) / k
    den = (sse_l + sse_r) / df2
    if den <= 0.0:
        return 0.0, 1.0, 0.0, df1, df2

    F = num / den
    if F < 0.0:
        F = 0.0

    p_value = float(f_dist.sf(F, df1, df2))
    prob = 1.0 - p_value
    return prob, p_value, float(F), df1, df2


def detect_cps_with_prob(
    y: np.ndarray,
    cps: list[int],
    win: int = TEST_WIN,
    model: str = "linear",
    prob_threshold: float | None = PROB_THRESHOLD,
    alpha_fwer: float = ALPHA_FWER,
    min_cp_gap: int = MIN_CP_GAP,
) -> tuple[float, list[CPEvent]]:
    """
    Фильтрует кандидатов по тесту Chow с поправкой FWER.

    Порог вероятности: prob_threshold = 1 − α/m, где m — число кандидатов.
    Принимаются кандидаты с prob ≥ порога и расстоянием ≥ min_cp_gap.

    :param y: временной ряд
    :param cps: список кандидатов
    :param win: полуширина окна теста Chow
    :param model: 'linear' или 'const'
    :param prob_threshold: явный порог (None — автоматически по FWER)
    :param alpha_fwer: уровень значимости FWER
    :param min_cp_gap: минимальное расстояние между принятыми точками
    :return: (prob_threshold, список CPEvent)
    """
    cps = sorted(int(x) for x in cps)

    if prob_threshold is None:
        m = max(1, len(cps))
        alpha_per = float(alpha_fwer) / m
        prob_threshold = 1.0 - alpha_per

    events: list[CPEvent] = []
    last_cp = None

    for cp in cps:
        if last_cp is not None and abs(cp - last_cp) < int(min_cp_gap):
            continue

        out = chow_f_test_window(y, cp, win=win, model=model)
        if out is None:
            continue

        prob, p_value, F, df1, df2 = out
        if prob >= float(prob_threshold):
            events.append(CPEvent(cp=cp, prob=prob, p_value=p_value, F=F, df1=df1, df2=df2))
            last_cp = cp

    return float(prob_threshold), events


def _build_meta(kwargs: dict, seed: int, prob_threshold: float) -> dict:
    """
    Формирует словарь метаданных запуска.

    :param kwargs: словарь kwargs, переданный функции swab
    :param seed: случайное зерно
    :param prob_threshold: использованный порог вероятности
    :return: словарь meta
    """
    return {
        "dataset_source": kwargs.get("dataset_source", "data_utils"),
        "generation_params": kwargs.get("generation_params", {}),
        "source_path": kwargs.get("source_path", _DEFAULT_SOURCE_PATH),
        "prob_threshold": prob_threshold,
        "seed": seed,
    }


def swab(
    series: np.ndarray,
    seed: int,
    w: int = 400,
    min_len: int = 50,
    target_segments: int = 6,
    win: int = TEST_WIN,
    model: str = "const",
    alpha_fwer: float = ALPHA_FWER,
    min_cp_gap: int = MIN_CP_GAP,
    **kwargs,
) -> tuple[list[int], None, dict]:
    """
    Обнаружение точек разладки алгоритмом SWAB — интерфейс пайплайна.

    Выполняет двухэтапный пайплайн:
    1. swab_candidates → кандидаты методом скользящего окна + Bottom-Up
    2. detect_cps_with_prob → фильтрация по тесту Chow с FWER

    :param series: одномерный временной ряд
    :param seed: случайное зерно (сохраняется в meta, алгоритм детерминирован)
    :param w: ширина скользящего окна SWAB
    :param min_len: минимальная длина сегмента в Bottom-Up
    :param target_segments: целевое число сегментов в Bottom-Up
    :param win: полуширина окна теста Chow
    :param model: модель теста Chow: 'const' или 'linear'
    :param alpha_fwer: уровень значимости FWER
    :param min_cp_gap: минимальное расстояние между финальными точками разладки
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, None, meta)
    """
    series = np.asarray(series, dtype=float).reshape(-1)

    cps = swab_candidates(series, w=w, min_len=min_len, target_segments=target_segments)
    prob_threshold, events = detect_cps_with_prob(
        series, cps, win=win, model=model, alpha_fwer=alpha_fwer, min_cp_gap=min_cp_gap,
    )

    change_points = [e.cp for e in events]
    meta = _build_meta(kwargs, seed, prob_threshold)
    return change_points, None, meta


if __name__ == "__main__":
    from algoritms.data_utils import generate_dataset

    generation_params = {"length": 359, "dt": 1.0, "D": 0.5, "noise_type": "white", "seed": 42}
    series, _ = generate_dataset(**generation_params)

    run_kwargs = {
        "dataset_source": "data_utils",
        "generation_params": generation_params,
    }

    cps, _, meta = swab(series, seed=generation_params["seed"], **run_kwargs)
    print(f"swab  cp={cps}  prob_threshold={meta['prob_threshold']:.4f}")
