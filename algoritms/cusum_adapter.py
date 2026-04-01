"""
Адаптер потокового CUSUM-детектора изменения среднего.

Реализует страничный CUSUM (Page's CUSUM) для обнаружения сдвигов среднего
в потоковом режиме: predict_next(y) возвращает (confidence, is_cp).

Алгоритм:
    Фаза прогрева (warmup шагов): оценивает μ и σ по первым наблюдениям.
    Основная фаза:
        S⁺ = max(0, S⁺ + (y − μ − k·σ))   — детектор роста среднего
        S⁻ = max(0, S⁻ − (y − μ + k·σ))   — детектор падения среднего
        Сигнал при max(S⁺, S⁻) ≥ h·σ
        Confidence = max(S⁺, S⁻) / (h·σ)
    После сигнала: сброс S⁺, S⁻ и пропуск t_warmup шагов.
"""

from pathlib import Path

import numpy as np


_DEFAULT_SOURCE_PATH = str(Path(__file__).resolve().parent / "data_utils.py")


class CusumMeanDetector:
    """
    Потоковый CUSUM-детектор изменения среднего.

    Параметры:
        k — слаковый параметр (обычно 0.5 · ожидаемый сдвиг в σ-единицах)
        h — порог обнаружения в σ-единицах
        warmup — число начальных шагов для оценки μ и σ
    """

    def __init__(self, k: float = 0.5, h: float = 5.0, warmup: int = 30):
        self._k = float(k)
        self._h = float(h)
        self._warmup = int(warmup)
        self._buf: list[float] = []
        self._mu = 0.0
        self._sigma = 1.0
        self._sp = 0.0
        self._sm = 0.0
        self._ready = False

    def predict_next(self, y: float) -> tuple[float, bool]:
        """
        Обрабатывает одно новое наблюдение и возвращает (confidence, is_cp).

        В фазе прогрева всегда возвращает (0.0, False).
        В основной фазе confidence = max(S⁺, S⁻) / (h·σ).
        is_cp = True когда confidence ≥ 1.0.

        :param y: новое наблюдение
        :return: (confidence, is_cp)
        """
        if not self._ready:
            self._buf.append(float(y))
            if len(self._buf) >= self._warmup:
                arr = np.array(self._buf, dtype=float)
                self._mu = float(np.mean(arr))
                self._sigma = max(float(np.std(arr)), 1e-12)
                self._ready = True
            return 0.0, False

        slack = self._k * self._sigma
        threshold = self._h * self._sigma

        self._sp = max(0.0, self._sp + (float(y) - self._mu - slack))
        self._sm = max(0.0, self._sm - (float(y) - self._mu + slack))

        confidence = max(self._sp, self._sm) / threshold
        is_cp = confidence >= 1.0

        if is_cp:
            self._sp = 0.0
            self._sm = 0.0

        return float(confidence), is_cp

    def reset(self) -> None:
        """Сбрасывает накопленные суммы S⁺ и S⁻."""
        self._sp = 0.0
        self._sm = 0.0


def run_cusum(
    series: np.ndarray,
    seed: int,
    t_warmup: int = 30,
    k: float = 0.5,
    h: float = 5.0,
    p_limit: float = 0.01,
    **kwargs,
) -> tuple[list[int], np.ndarray, dict]:
    """
    Прогоняет CusumMeanDetector по ряду, собирает точки разладки и скоры.

    После каждого обнаруженного CP детектор сбрасывается и пропускает
    следующие t_warmup шагов (чтобы избежать каскадных срабатываний).

    :param series: одномерный временной ряд
    :param seed: случайное зерно (сохраняется в meta; CUSUM детерминирован)
    :param t_warmup: размер окна прогрева и число пропускаемых шагов после CP
    :param k: слаковый параметр CUSUM
    :param h: порог обнаружения в σ-единицах
    :param p_limit: параметр чувствительности (сохраняется в meta)
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, scores, meta)
    """
    series = np.asarray(series, dtype=float).reshape(-1)
    n = int(series.size)

    detector = CusumMeanDetector(k=k, h=h, warmup=t_warmup)
    scores = np.zeros(n, dtype=float)
    change_points: list[int] = []
    skip_until = 0

    for t in range(n):
        y = float(series[t])
        confidence, is_cp = detector.predict_next(y)
        scores[t] = confidence

        if is_cp and t >= skip_until:
            change_points.append(t)
            skip_until = t + t_warmup

    meta = {
        "dataset_source": kwargs.get("dataset_source", "data_utils"),
        "generation_params": kwargs.get("generation_params", {}),
        "source_path": kwargs.get("source_path", _DEFAULT_SOURCE_PATH),
        "seed": int(seed),
        "k": k,
        "h": h,
        "p_limit": p_limit,
    }

    return change_points, scores, meta
