"""
Обнаружение точек разладки методом минимальной длины описания (MDL).

Алгоритм работает в два этапа:

Этап 1 — генерация кандидатов (mdl_candidates):
    Для каждой позиции cp вычисляется MDL-скор в окне ±win:
        score(cp) = DL(всё окно) − DL(левая половина) − DL(правая половина)
    Длина описания DL(x) = n · H(q), где H(q) — энтропия квантованного сигнала.
    Локальные пики score(cp) с промежутком ≥ min_cp_gap — кандидаты разладки.

Этап 2 — валидация кандидатов (detect_cps_with_prob):
    Каждый кандидат проверяется тестом Chow в окне ±win.
    Применяется поправка FWER (Бонферрони): порог = 1 − α/m.
    Принимаются только кандидаты с prob ≥ порога.

Квантование (quantize_fixed):
    Равномерное квантование сигнала в диапазоне [lo, hi] на 2^b уровней.
    Используется для оценки энтропии через частоты символов.
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
TEST_WIN = 120

MDL_B = 6
MDL_STRIDE = 2
MDL_TOP_K = 25
MDL_MIN_CP_GAP = 120


def quantize_fixed(
    x: np.ndarray,
    lo: float,
    hi: float,
    b: int = 6,
) -> np.ndarray:
    """
    Равномерное квантование массива x на 2^b уровней в диапазоне [lo, hi].

    :param x: входной массив
    :param lo: нижняя граница диапазона
    :param hi: верхняя граница диапазона
    :param b: число бит квантования (число уровней = 2^b)
    :return: массив целочисленных кодов int16
    """
    x = np.asarray(x, dtype=float)
    card = 1 << int(b)
    if hi <= lo:
        return np.zeros_like(x, dtype=np.int16)
    y = (x - lo) / (hi - lo) * (card - 1)
    y = np.floor(y + 1e-12).astype(np.int32)
    y = np.clip(y, 0, card - 1).astype(np.int16)
    return y


def entropy_q(q: np.ndarray, card: int) -> float:
    """
    Вычисляет эмпирическую энтропию (бит) квантованного массива q.

    H(q) = −Σ p_i · log₂(p_i), где p_i — частота символа i.

    :param q: массив квантованных кодов
    :param card: мощность алфавита (число уровней)
    :return: энтропия в битах
    """
    q = np.asarray(q)
    m = int(q.size)
    if m == 0:
        return 0.0
    counts = np.bincount(q.astype(np.int64), minlength=int(card))
    counts = counts[counts > 0]
    p = counts / m
    return float(-(p * np.log2(p)).sum())


def dl_q(q: np.ndarray, card: int) -> float:
    """
    Длина описания квантованного сигнала: DL = n · H(q).

    :param q: массив квантованных кодов
    :param card: мощность алфавита
    :return: длина описания в битах
    """
    q = np.asarray(q)
    return float(q.size * entropy_q(q, card))


def mdl_score_window(
    y: np.ndarray,
    cp: int,
    win: int,
    b: int = 6,
) -> float | None:
    """
    Вычисляет MDL-скор разладки в точке cp для окна ±win.

    score = DL(всё окно) − DL(левая половина) − DL(правая половина)
    Положительный скор означает, что два отдельных описания короче общего.

    :param y: временной ряд
    :param cp: индекс кандидата (0-based)
    :param win: полуширина окна
    :param b: число бит квантования
    :return: MDL-скор ≥ 0 или None если окно выходит за границы
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.size)

    l0 = cp - win + 1
    l1 = cp + 1
    r0 = cp + 1
    r1 = cp + 1 + win

    if l0 < 0 or r1 > n:
        return None

    full = y[l0:r1]
    lo = float(full.min())
    hi = float(full.max())

    q_full = quantize_fixed(full, lo, hi, b=b)
    card = 1 << int(b)

    q_left = q_full[:win]
    q_right = q_full[win:]

    score = dl_q(q_full, card) - dl_q(q_left, card) - dl_q(q_right, card)
    if score < 0.0:
        score = 0.0
    return float(score)


def _by_score_desc(i: int, scores: np.ndarray) -> float:
    """Ключ сортировки по убыванию MDL-скора."""
    return -scores[i]


def mdl_candidates(
    y: np.ndarray,
    win: int = TEST_WIN,
    b: int = MDL_B,
    stride: int = MDL_STRIDE,
    min_cp_gap: int = MDL_MIN_CP_GAP,
    top_k: int = MDL_TOP_K,
) -> list[int]:
    """
    Генерирует кандидатов точек разладки по MDL-скору.

    Вычисляет MDL-скор в каждой позиции (с шагом stride).
    Локальные пики скора с промежутком ≥ min_cp_gap — кандидаты.
    Если пиков нет, берутся top_k позиций с наибольшим скором.

    :param y: временной ряд
    :param win: полуширина окна MDL-скора
    :param b: число бит квантования
    :param stride: шаг перебора позиций
    :param min_cp_gap: минимальное расстояние между кандидатами
    :param top_k: максимальное число возвращаемых кандидатов
    :return: отсортированный список 0-based индексов кандидатов
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.size)

    if n < 2 * int(win) + 2:
        return []

    idx = list(range(int(win) - 1, n - int(win) - 1, int(stride)))
    scores = np.zeros(len(idx), dtype=float)

    for k, cp in enumerate(idx):
        sc = mdl_score_window(y, int(cp), int(win), b=int(b))
        if sc is None:
            sc = 0.0
        scores[k] = sc

    peaks = []
    for i in range(1, scores.size - 1):
        if scores[i] > scores[i - 1] and scores[i] >= scores[i + 1] and scores[i] > 0.0:
            peaks.append(i)

    if len(peaks) == 0:
        peaks = list(np.argsort(-scores)[: max(1, min(int(top_k), scores.size))])

    def sort_key(i: int) -> float:
        return _by_score_desc(i, scores)

    peaks = sorted(peaks, key=sort_key)

    chosen = []
    for i in peaks:
        cp = int(idx[int(i)])
        ok = all(abs(cp - c) >= int(min_cp_gap) for c in chosen)
        if ok:
            chosen.append(cp)
            if len(chosen) >= int(top_k):
                break

    chosen.sort()
    return chosen


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

    :param kwargs: словарь kwargs, переданный функции mdl
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


def mdl(
    series: np.ndarray,
    seed: int,
    win: int = TEST_WIN,
    b: int = MDL_B,
    stride: int = MDL_STRIDE,
    min_cp_gap_cand: int = MDL_MIN_CP_GAP,
    top_k: int = MDL_TOP_K,
    model: str = "const",
    alpha_fwer: float = ALPHA_FWER,
    min_cp_gap: int = MIN_CP_GAP,
    **kwargs,
) -> tuple[list[int], None, dict]:
    """
    Обнаружение точек разладки методом MDL — интерфейс пайплайна.

    Выполняет двухэтапный пайплайн:
    1. mdl_candidates → кандидаты по локальным пикам MDL-скора
    2. detect_cps_with_prob → фильтрация по тесту Chow с FWER

    :param series: одномерный временной ряд
    :param seed: случайное зерно (сохраняется в meta, алгоритм детерминирован)
    :param win: полуширина окна MDL-скора и теста Chow
    :param b: число бит квантования MDL
    :param stride: шаг перебора позиций кандидатов
    :param min_cp_gap_cand: минимальное расстояние между кандидатами MDL
    :param top_k: максимальное число кандидатов MDL
    :param model: модель теста Chow: 'const' или 'linear'
    :param alpha_fwer: уровень значимости FWER
    :param min_cp_gap: минимальное расстояние между финальными точками разладки
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, None, meta)
    """
    series = np.asarray(series, dtype=float).reshape(-1)

    cps = mdl_candidates(
        series, win=win, b=b, stride=stride, min_cp_gap=min_cp_gap_cand, top_k=top_k,
    )
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

    cps, _, meta = mdl(series, seed=generation_params["seed"], **run_kwargs)
    print(f"mdl  cp={cps}  prob_threshold={meta['prob_threshold']:.4f}")
