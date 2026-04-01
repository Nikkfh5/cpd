"""
Тест Ниблома для проверки стабильности параметров линейной модели.

Модель под H0: y_t = x_t^T β + ε_t,  β = const (параметры не меняются).
Если параметры изменяются в некоторый момент τ, тест обнаруживает это.

Определения:
    β̂ = (X^T X)^{-1} X^T y  — МНК-оценка коэффициентов
    ε̂_t = y_t − x_t^T β̂     — остатки
    f_t = x_t · ε̂_t           — скор-вклад наблюдения t
    S_t = Σ_{j≤t} f_j          — накопленный скор (кумулятивный процесс)
    V = (1/n) X^T X
    L = (1/(n² σ̂²)) Σ_{t=1..n} S_t^T V^{-1} S_t

При H0: L → ∫₀¹ ‖B_k(λ)‖² dλ,
где B_k(λ) = W_k(λ) − λ W_k(1) — k-мерный броуновский мост.

Точка изменения локализуется как argmax q_t, где q_t = S_t^T V^{-1} S_t.

Особенности:
    - Обнаруживает как резкую смену (step), так и постепенный дрейф параметров.
    - При X = None тестирует только стабильность среднего (k = 1).
    - Критическое значение и p-value оцениваются через симуляцию броуновского моста.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


_DEFAULT_SOURCE_PATH = str(Path(__file__).resolve().parent / "data_utils.py")


@dataclass
class NyblomResult:
    """Результат теста Ниблома."""

    stat: float
    p_value: float
    cv: float
    reject: bool
    cp_index0: int
    score_q: np.ndarray
    beta_hat: np.ndarray
    resid: np.ndarray


def nyblom_stat(y: np.ndarray, X: np.ndarray | None = None):
    """
    Вычисляет статистику Ниблома L и вспомогательные массивы.

    Модель под H0: y_t = x_t^T β + ε_t, β = const.
    β̂ = (X^T X)^{-1} X^T y — МНК-оценка.
    S_t = Σ_{j≤t} x_j · ε̂_j — кумулятивный скор.
    L = (1/(n² σ̂²)) Σ_t S_t^T V^{-1} S_t, где V = (1/n) X^T X.

    :param y: целевой ряд, форма (n,) или (n, 1)
    :param X: матрица регрессоров, форма (n, k); если None — только константа
    :return: (L, q, beta_hat, resid)
        L — скаляр статистики
        q — np.ndarray формы (n,): q_t = S_t^T V^{-1} S_t
        beta_hat — np.ndarray оценок коэффициентов
        resid — np.ndarray остатков
    """
    y = np.asarray(y, dtype=float).reshape(-1)
    n = y.shape[0]
    if X is None:
        X = np.ones((n, 1), dtype=float)
    else:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != n:
            raise ValueError("X и y должны иметь одинаковое число строк (наблюдений)")

    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
    resid = y - X @ beta_hat

    sigma2_hat = float(resid @ resid) / n
    V = X.T @ X / n
    k = X.shape[1]

    S = np.zeros(k, dtype=float)
    q = np.empty(n, dtype=float)

    try:
        R = np.linalg.cholesky(V)
        use_chol = True
    except Exception:
        use_chol = False

    for t in range(n):
        S = S + X[t] * resid[t]
        if use_chol:
            z = np.linalg.solve(R, S)
            q[t] = float(z @ z)
        else:
            z = np.linalg.lstsq(V, S, rcond=None)[0]
            q[t] = float(S @ z)

    if sigma2_hat <= 0.0:
        L = np.nan
    else:
        L = float(np.sum(q)) / (n * n * sigma2_hat)

    return L, q, beta_hat, resid


def nyblom_cv_and_pvalue(
    L: float,
    k: int,
    alpha: float = 0.05,
    m: int = 8000,
    grid: int = 2048,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Оценка критического значения и p-value через симуляцию предельного распределения.

    Под H0: L → ∫₀¹ ‖B_k(λ)‖² dλ, где B_k — k-мерный броуновский мост.
    Интеграл дискретизируется на сетке из grid точек.

    :param L: наблюдаемое значение статистики
    :param k: число параметров (столбцов X)
    :param alpha: уровень значимости для критического значения
    :param m: число симуляций
    :param grid: число точек дискретизации броуновского моста
    :param seed: зерно для генератора симуляций
    :return: (cv, p_value)
    """
    rng = np.random.default_rng(seed)
    dt = 1.0 / grid
    sims = np.empty(m, dtype=float)

    for i in range(m):
        inc = rng.normal(0.0, np.sqrt(dt), size=(grid, k))
        W = np.cumsum(inc, axis=0)
        W1 = W[-1]
        lam = (np.arange(1, grid + 1, dtype=float) / grid).reshape(-1, 1)
        B = W - lam * W1.reshape(1, -1)
        sims[i] = float(np.sum(B * B)) * dt

    cv = float(np.quantile(sims, 1.0 - alpha))
    p_value = float((np.sum(sims >= L) + 1.0) / (m + 1.0))
    return cv, p_value


def nyblom_test(
    y: np.ndarray,
    X: np.ndarray | None = None,
    alpha: float = 0.05,
    m: int = 8000,
    grid: int = 2048,
    seed: int = 0,
) -> NyblomResult:
    """
    Полный тест Ниблома: вычисляет L, критическое значение, p-value и точку изменения.

    Точка изменения определяется как argmax q_t (эвристика локализации).

    :param y: целевой ряд
    :param X: матрица регрессоров; если None — тест стабильности среднего (k=1)
    :param alpha: уровень значимости
    :param m: число симуляций для оценки критического значения
    :param grid: число точек дискретизации броуновского моста
    :param seed: зерно симуляций критического значения
    :return: NyblomResult
    """
    L, q, beta_hat, resid = nyblom_stat(y, X)

    if X is None:
        k = 1
    else:
        X_ = np.asarray(X)
        k = 1 if X_.ndim == 1 else X_.shape[1]

    cv, p_value = nyblom_cv_and_pvalue(L, k, alpha=alpha, m=m, grid=grid, seed=seed)
    reject = bool(L > cv)
    cp_index0 = int(np.argmax(q))

    return NyblomResult(
        stat=L,
        p_value=p_value,
        cv=cv,
        reject=reject,
        cp_index0=cp_index0,
        score_q=q,
        beta_hat=beta_hat,
        resid=resid,
    )


def _build_meta(
    kwargs: dict,
    result: NyblomResult,
    seed: int,
) -> dict:
    """
    Формирует словарь метаданных запуска.

    :param kwargs: словарь kwargs, переданный функции nyblom
    :param result: объект NyblomResult
    :param seed: случайное зерно
    :return: словарь meta
    """
    return {
        "dataset_source": kwargs.get("dataset_source", "data_utils"),
        "generation_params": kwargs.get("generation_params", {}),
        "source_path": kwargs.get("source_path", _DEFAULT_SOURCE_PATH),
        "p_value": result.p_value,
        "stat": result.stat,
        "cv": result.cv,
        "reject": result.reject,
        "seed": seed,
    }


def nyblom(
    series: np.ndarray,
    seed: int,
    X: np.ndarray | None = None,
    alpha: float = 0.05,
    m: int = 8000,
    grid: int = 2048,
    **kwargs,
) -> tuple[list[int], np.ndarray, dict]:
    """
    Тест Ниблома — унифицированный интерфейс пайплайна.

    Обёртка над nyblom_test; возвращает change_points, scores и meta.
    scores = score_q (q_t = S_t^T V^{-1} S_t для каждого момента t).
    Точка изменения включается в change_points только при reject=True.

    :param series: одномерный временной ряд (зависимая переменная y)
    :param seed: случайное зерно (передаётся в симуляцию критического значения)
    :param X: матрица регрессоров, форма (n, k); если None — тест стабильности среднего
    :param alpha: уровень значимости
    :param m: число симуляций для оценки критического значения
    :param grid: число точек дискретизации броуновского моста
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, scores, meta)
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    result = nyblom_test(series, X=X, alpha=alpha, m=m, grid=grid, seed=seed)
    change_points = [result.cp_index0] if result.reject else []
    meta = _build_meta(kwargs, result, seed)
    return change_points, result.score_q, meta


if __name__ == "__main__":
    from algoritms.data_utils import generate_dataset

    generation_params = {"length": 359, "dt": 1.0, "D": 0.5, "noise_type": "white", "seed": 42}
    series, _ = generate_dataset(**generation_params)

    run_kwargs = {
        "dataset_source": "data_utils",
        "generation_params": generation_params,
    }

    cps, scores, meta = nyblom(series, seed=generation_params["seed"], **run_kwargs)
    print(f"nyblom  cp={cps}  stat={meta['stat']:.4f}  p={meta['p_value']:.4f}  reject={meta['reject']}")
