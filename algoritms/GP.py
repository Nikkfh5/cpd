"""
Обнаружение точек разладки методом гауссовского процесса (GP).

Алгоритм работает в два этапа:

Этап 1 — генерация кандидатов (gp_change_candidates):
    Для каждого момента t предсказывается y_t по предыдущим m наблюдениям
    с помощью GP с RBF-ядром. Если наблюдение y_t значимо отличается
    от предсказания (двусторонний p-value < alpha), точка признаётся кандидатом.
    Кандидаты фильтруются по минимальному расстоянию min_cp_gap.

    RBF-ядро: K(t, t') = σ_f² · exp(−(t−t')² / (2ℓ²))
    GP-предсказание: μ* = k*^T (K + σ_n² I)^{-1} y,  var* = k** − k*^T (K + σ_n² I)^{-1} k*

Этап 2 — валидация кандидатов (detect_cps_with_prob):
    Каждый кандидат проверяется тестом Chow в окне ±win.
    Применяется поправка FWER (Бонферрони): порог вероятности = 1 − α/m.
    Оставляются только кандидаты с prob ≥ порога.

Вспомогательная функция collapse_cps_by_local_score:
    Группирует близкие кандидаты и выбирает наилучший в каждой группе
    по минимальному p-value теста Chow.
"""

from dataclasses import dataclass
from math import erf, sqrt
from pathlib import Path

import numpy as np

try:
    from scipy.stats import f as f_dist
except Exception as e:
    raise ImportError("Нужен scipy (scipy.stats.f)") from e


_DEFAULT_SOURCE_PATH = str(Path(__file__).resolve().parent / "data_utils.py")

ALPHA_FWER = 0.05
PROB_THRESHOLD = None
MIN_CP_GAP = 200
TEST_WIN = 200

GP_TRAIN_WIN = 180
GP_ALPHA_CAND = 1e-3
GP_MIN_CP_GAP = 60
GP_NOISE_RATIO = 0.12
GP_ELL_RATIO = 0.35


def norm_cdf(z: float) -> float:
    """Стандартная нормальная функция распределения через erf."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def two_sided_pvalue(y: float, mu: float, var: float) -> float:
    """
    Двусторонний p-value для нормального предсказания GP.

    :param y: наблюдаемое значение
    :param mu: предсказанное среднее
    :param var: предсказанная дисперсия
    :return: p-value ∈ [0, 1]
    """
    if var <= 0.0:
        return 1.0
    z = abs(y - mu) / sqrt(var)
    p = 2.0 * (1.0 - norm_cdf(z))
    return float(max(0.0, min(1.0, p)))


def rbf_kernel_from_d2(d2: np.ndarray, sigma_f2: float, ell: float) -> np.ndarray:
    """
    Вычисляет RBF-ядро по матрице квадратов расстояний.

    K(t, t') = σ_f² · exp(−d² / (2ℓ²))

    :param d2: матрица или вектор квадратов расстояний
    :param sigma_f2: дисперсия сигнала σ_f²
    :param ell: характерная длина ℓ
    :return: значения ядра той же формы, что d2
    """
    if ell <= 0.0:
        return sigma_f2 * (d2 == 0).astype(float)
    return sigma_f2 * np.exp(-0.5 * d2 / (ell * ell))


def build_gp_cache(
    m: int,
    sigma_f2: float,
    sigma_n2: float,
    ell: float,
):
    """
    Предвычисляет разложение Холецкого и вектор k* для GP-предсказания.

    Кэш действителен для всех обучающих окон одинаковой длины m,
    так как GP использует только относительные индексы.

    :param m: размер обучающего окна
    :param sigma_f2: дисперсия сигнала
    :param sigma_n2: дисперсия шума
    :param ell: характерная длина RBF-ядра
    :return: (L, k_star, base_var) или None при ошибке Холецкого
    """
    idx = np.arange(m, dtype=float)
    diff = idx[:, None] - idx[None, :]
    K = rbf_kernel_from_d2(diff * diff, sigma_f2, ell)
    K = K + (sigma_n2 + 1e-12) * np.eye(m, dtype=float)

    jitter = 1e-10 * (sigma_f2 + sigma_n2 + 1.0)
    for _ in range(6):
        try:
            L = np.linalg.cholesky(K + jitter * np.eye(m, dtype=float))
            break
        except np.linalg.LinAlgError:
            jitter *= 10.0
    else:
        return None

    kstar_d2 = (idx - float(m)) ** 2
    k_star = rbf_kernel_from_d2(kstar_d2, sigma_f2, ell).reshape(-1, 1)
    k_ss = float(sigma_f2 + sigma_n2)

    v = np.linalg.solve(L, k_star)
    proj = float((v * v).sum())
    base_var = max(1e-12, k_ss - proj)

    return L, k_star.reshape(-1), float(base_var)


def gp_predict_from_cache(
    L: np.ndarray,
    k_star: np.ndarray,
    base_var: float,
    y_train: np.ndarray,
) -> tuple[float, float]:
    """
    GP-предсказание следующего значения по предвычисленному кэшу.

    :param L: нижнетреугольный множитель Холецкого матрицы K
    :param k_star: вектор ковариаций тестовой точки с обучающими
    :param base_var: базовая дисперсия предсказания
    :param y_train: обучающие наблюдения, форма (m,)
    :return: (mu, var)
    """
    y_train = np.asarray(y_train, dtype=float).reshape(-1, 1)
    a = np.linalg.solve(L, y_train)
    alpha = np.linalg.solve(L.T, a)
    mu = float(k_star @ alpha.reshape(-1))
    return mu, float(base_var)


def _by_pvalue(item: tuple) -> float:
    """Ключ сортировки по p-value (второй элемент кортежа)."""
    return item[1]


def gp_change_candidates(
    y: np.ndarray,
    train_win: int = GP_TRAIN_WIN,
    alpha: float = GP_ALPHA_CAND,
    min_cp_gap: int = GP_MIN_CP_GAP,
    noise_ratio: float = GP_NOISE_RATIO,
    ell_ratio: float = GP_ELL_RATIO,
    keep_top: int | None = None,
) -> list[int]:
    """
    Генерирует кандидатов точек разладки методом GP-аномалий.

    Для каждого момента t предсказывается y_t по обучающему окну [t−m, t).
    Точки с p-value < alpha признаются аномальными кандидатами.
    Применяется жадная фильтрация по минимальному расстоянию min_cp_gap.

    :param y: временной ряд
    :param train_win: размер обучающего окна m
    :param alpha: порог p-value для признания точки кандидатом
    :param min_cp_gap: минимальное расстояние между кандидатами
    :param noise_ratio: σ_n = noise_ratio · σ (отношение шума к сигналу)
    :param ell_ratio: ℓ = ell_ratio · m (характерная длина в долях окна)
    :param keep_top: максимальное число возвращаемых кандидатов (None — без ограничения)
    :return: отсортированный список 0-based индексов кандидатов
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.size)
    m = int(train_win)
    if n <= m + 2:
        return []

    sigma = float(np.std(y[:max(m, 5)]))
    if sigma <= 1e-12:
        sigma = float(np.std(y))
    if sigma <= 1e-12:
        return []

    sigma_f2 = sigma * sigma
    sigma_n2 = (noise_ratio * sigma) ** 2
    ell = float(max(1.0, ell_ratio * m))

    cache = build_gp_cache(m, sigma_f2, sigma_n2, ell)
    if cache is None:
        return []
    L, k_star, base_var = cache

    cand = []
    for t in range(m, n):
        y_train = y[t - m: t]
        mu, var = gp_predict_from_cache(L, k_star, base_var, y_train)
        p = two_sided_pvalue(y[t], mu, var)
        if p < float(alpha):
            cand.append((int(t - 1), p))

    if len(cand) == 0:
        return []

    cand.sort(key=_by_pvalue)

    chosen = []
    for cp, p in cand:
        ok = all(abs(cp - c) >= int(min_cp_gap) for c in chosen)
        if ok:
            chosen.append(int(cp))
            if keep_top is not None and len(chosen) >= int(keep_top):
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
        return Segment(
            int(tt[0]), int(tt[-1]), n,
            float(tt.sum()), float((tt * tt).sum()),
            float(yy.sum()), float((tt * yy).sum()), float((yy * yy).sum()),
        )

    def merge(self, other: "Segment") -> "Segment":
        """Объединяет два соседних сегмента."""
        return Segment(
            self.t0, other.t1, self.n + other.n,
            self.st + other.st, self.st2 + other.st2,
            self.sy + other.sy, self.sty + other.sty, self.sy2 + other.sy2,
        )

    def fit_ab(self) -> tuple[float, float]:
        """Оценивает коэффициенты линейной регрессии y = a + b·t по МНК."""
        denom = self.n * self.st2 - self.st * self.st
        if denom == 0.0:
            return self.sy / self.n, 0.0
        b = (self.n * self.sty - self.st * self.sy) / denom
        a = (self.sy - b * self.st) / self.n
        return a, b

    def sse_linear(self) -> float:
        """Сумма квадратов остатков линейной регрессии y = a + b·t."""
        a, b = self.fit_ab()
        sse = (
            self.sy2 - 2.0 * a * self.sy - 2.0 * b * self.sty
            + self.n * a * a + 2.0 * a * b * self.st + b * b * self.st2
        )
        return float(max(0.0, sse))

    def sse_const(self) -> float:
        """Сумма квадратов остатков модели константы y = μ."""
        if self.n <= 0:
            return 0.0
        mu = self.sy / self.n
        sse = self.sy2 - 2.0 * mu * self.sy + self.n * mu * mu
        return float(max(0.0, sse))


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

    Сравнивает модель с единым набором коэффициентов и модель
    с разными коэффициентами до/после cp.

    :param y: временной ряд
    :param cp: индекс кандидата (0-based)
    :param win: полуширина окна
    :param model: 'linear' (тренд) или 'const' (среднее)
    :return: (prob, p_value, F, df1, df2) или None если окно выходит за границы
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = int(y.size)

    l0, l1 = cp - win + 1, cp + 1
    r0, r1 = cp + 1, cp + 1 + win

    if l0 < 0 or r1 > n:
        return None

    t = np.arange(n, dtype=int)
    left = Segment.from_arrays(t, y, l0, l1)
    right = Segment.from_arrays(t, y, r0, r1)
    merged = left.merge(right)

    if model == "const":
        k = 1
        sse_l, sse_r, sse_m = left.sse_const(), right.sse_const(), merged.sse_const()
    else:
        k = 2
        sse_l, sse_r, sse_m = left.sse_linear(), right.sse_linear(), merged.sse_linear()

    nn = left.n + right.n
    df1 = int(k)
    df2 = int(nn - 2 * k)
    if df2 <= 0:
        return 0.0, 1.0, 0.0, df1, df2

    den = (sse_l + sse_r) / df2
    if den <= 0.0:
        return 0.0, 1.0, 0.0, df1, df2

    F = max(0.0, (sse_m - (sse_l + sse_r)) / k / den)
    p_value = float(f_dist.sf(F, df1, df2))
    return 1.0 - p_value, p_value, float(F), df1, df2


def collapse_cps_by_local_score(
    y: np.ndarray,
    cps: list[int],
    win: int = TEST_WIN,
    model: str = "const",
    group_gap: int = 2 * TEST_WIN,
) -> list[int]:
    """
    Группирует близкие кандидаты и выбирает лучшего в каждой группе.

    Два кандидата попадают в одну группу, если расстояние между ними ≤ group_gap.
    Лучший выбирается по минимальному p-value теста Chow.

    :param y: временной ряд
    :param cps: список кандидатов
    :param win: полуширина окна теста Chow
    :param model: 'linear' или 'const'
    :param group_gap: максимальное расстояние для объединения в группу
    :return: отфильтрованный список кандидатов
    """
    cps = sorted(int(x) for x in cps)
    if len(cps) == 0:
        return []

    groups: list[list[int]] = []
    cur = [cps[0]]
    for cp in cps[1:]:
        if cp - cur[-1] <= int(group_gap):
            cur.append(cp)
        else:
            groups.append(cur)
            cur = [cp]
    groups.append(cur)

    keep = []
    for g in groups:
        best = None
        best_p = None
        for cp in g:
            out = chow_f_test_window(y, cp, win=win, model=model)
            if out is None:
                continue
            _, p_value, _, _, _ = out
            if best_p is None or p_value < best_p:
                best_p = p_value
                best = cp
        if best is not None:
            keep.append(int(best))

    keep.sort()
    return keep


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
    Кандидаты с prob ≥ порога и расстоянием ≥ min_cp_gap принимаются.

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
        prob_threshold = 1.0 - float(alpha_fwer) / m

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

    :param kwargs: словарь kwargs, переданный функции gp
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


def gp(
    series: np.ndarray,
    seed: int,
    train_win: int = GP_TRAIN_WIN,
    alpha_cand: float = GP_ALPHA_CAND,
    min_cp_gap_cand: int = GP_MIN_CP_GAP,
    keep_top: int | None = None,
    win: int = TEST_WIN,
    model: str = "const",
    alpha_fwer: float = ALPHA_FWER,
    min_cp_gap: int = MIN_CP_GAP,
    **kwargs,
) -> tuple[list[int], None, dict]:
    """
    Обнаружение точек разладки методом гауссовского процесса — интерфейс пайплайна.

    Выполняет полный двухэтапный пайплайн:
    1. gp_change_candidates → кандидаты по GP-аномалиям
    2. collapse_cps_by_local_score → дедупликация в группах
    3. detect_cps_with_prob → фильтрация по тесту Chow с FWER

    :param series: одномерный временной ряд
    :param seed: случайное зерно (сохраняется в meta, алгоритм детерминирован)
    :param train_win: размер обучающего окна GP
    :param alpha_cand: порог p-value для кандидатов GP
    :param min_cp_gap_cand: минимальное расстояние между кандидатами GP
    :param keep_top: максимальное число кандидатов GP (None — без ограничения)
    :param win: полуширина окна теста Chow
    :param model: модель теста Chow: 'const' или 'linear'
    :param alpha_fwer: уровень значимости FWER
    :param min_cp_gap: минимальное расстояние между финальными точками разладки
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, None, meta)
    """
    series = np.asarray(series, dtype=float).reshape(-1)

    cps = gp_change_candidates(
        series,
        train_win=train_win,
        alpha=alpha_cand,
        min_cp_gap=min_cp_gap_cand,
        keep_top=keep_top,
    )
    cps = collapse_cps_by_local_score(series, cps, win=win, model=model)
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

    cps, _, meta = gp(series, seed=generation_params["seed"], **run_kwargs)
    print(f"gp  cp={cps}  prob_threshold={meta['prob_threshold']:.4f}")
