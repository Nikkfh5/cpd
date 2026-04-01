"""
CatBoost-модель для обнаружения точек разладки (change point detection) во временных рядах.

Скользящие окна описываются 14 статистическими признаками; бинарная классификация
выполняется через CatBoostClassifier. Нормировка каждого окна перед извлечением
признаков сохраняется (аналогично LSTM/GRU).

Признаки вычисляются батчево (векторизованно) для ускорения на больших датасетах.
"""

import numpy as np
from scipy import stats
from scipy.ndimage import binary_dilation


def extract_window_features(window: np.ndarray) -> np.ndarray:
    """
    Извлекает 14 статистических признаков из одного окна.

    Используется в predict_catboost для единичных окон.
    Для батчевой обработки использовать _extract_features_batch.

    Признаки:
        0  — mean
        1  — std
        2  — min
        3  — max
        4  — range (max - min)
        5  — skewness
        6  — kurtosis
        7  — mean(first_half) - mean(second_half)
        8  — std(first_half) - std(second_half)
        9  — линейный наклон (polyfit degree=1)
        10 — autocorr lag-1
        11 — mean(|diff(window)|)
        12 — std(diff(window))
        13 — median

    :param window: 1-мерный массив значений окна
    :return: вектор из 14 признаков
    """
    half = len(window) // 2
    first_half = window[:half]
    second_half = window[half:]

    diff = np.diff(window)
    autocorr = float(np.corrcoef(window[:-1], window[1:])[0, 1]) if len(window) > 2 else 0.0
    if not np.isfinite(autocorr):
        autocorr = 0.0

    slope = float(np.polyfit(np.arange(len(window)), window, 1)[0])

    features = np.array([
        float(np.mean(window)),
        float(np.std(window)),
        float(np.min(window)),
        float(np.max(window)),
        float(np.max(window) - np.min(window)),
        float(stats.skew(window)),
        float(stats.kurtosis(window)),
        float(np.mean(first_half) - np.mean(second_half)),
        float(np.std(first_half) - np.std(second_half)),
        slope,
        autocorr,
        float(np.mean(np.abs(diff))),
        float(np.std(diff)),
        float(np.median(window)),
    ], dtype=np.float32)

    features = np.where(np.isfinite(features), features, 0.0)
    return features


def _extract_features_batch(windows: np.ndarray) -> np.ndarray:
    """
    Вычисляет 14 признаков для батча окон векторизованно.

    :param windows: массив формы (N, window_size) — нормированные окна
    :return: матрица признаков формы (N, 14)
    """
    N, W = windows.shape
    half = W // 2

    first_half = windows[:, :half]
    second_half = windows[:, half:]
    diff = np.diff(windows, axis=1)

    t = np.arange(W, dtype=np.float64)
    t_c = t - t.mean()
    t_var = (t_c ** 2).sum()
    w_mean = windows.mean(axis=1, keepdims=True)
    slope = ((windows - w_mean) * t_c).sum(axis=1) / t_var

    x_ac = windows[:, :-1]
    y_ac = windows[:, 1:]
    xm = x_ac.mean(axis=1, keepdims=True)
    ym = y_ac.mean(axis=1, keepdims=True)
    num = ((x_ac - xm) * (y_ac - ym)).sum(axis=1)
    den = np.sqrt(((x_ac - xm) ** 2).sum(axis=1) * ((y_ac - ym) ** 2).sum(axis=1))
    autocorr = np.where(den > 1e-8, num / den, 0.0)

    features = np.column_stack([
        windows.mean(axis=1),
        windows.std(axis=1),
        windows.min(axis=1),
        windows.max(axis=1),
        windows.max(axis=1) - windows.min(axis=1),
        stats.skew(windows, axis=1),
        stats.kurtosis(windows, axis=1),
        first_half.mean(axis=1) - second_half.mean(axis=1),
        first_half.std(axis=1) - second_half.std(axis=1),
        slope,
        autocorr,
        np.abs(diff).mean(axis=1),
        diff.std(axis=1),
        np.median(windows, axis=1),
    ]).astype(np.float32)

    features = np.where(np.isfinite(features), features, 0.0)
    return features


def _make_normalized_windows(x_values: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Создаёт матрицу нормированных скользящих окон через stride_tricks.

    :param x_values: временной ряд
    :param window_size: размер окна
    :return: (windows, indices) — матрица (N_windows, window_size) и массив центральных индексов
    """
    x = np.asarray(x_values, dtype=np.float64)
    half = window_size // 2
    indices = np.arange(half, len(x_values) - half)
    n = len(indices)
    windows_raw = np.lib.stride_tricks.sliding_window_view(x, window_size)[:n]

    means = windows_raw.mean(axis=1, keepdims=True)
    stds = windows_raw.std(axis=1, keepdims=True)
    stds = np.where(stds < 1e-8, 1.0, stds)
    windows_norm = (windows_raw - means) / stds

    return windows_norm.astype(np.float32), indices


def build_feature_matrix(
    x_values: np.ndarray,
    cp_labels: np.ndarray,
    window_size: int,
    margin: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Строит матрицу признаков и вектор меток для скользящих окон (векторизованно).

    Каждое окно нормируется перед извлечением признаков.
    Метка = 1, если в пределах margin от центра окна есть точка разладки.

    :param x_values: временной ряд
    :param cp_labels: бинарные метки точек разладки
    :param window_size: размер скользящего окна (чётное)
    :param margin: допуск для метки
    :return: (X, y) — матрица признаков и вектор меток
    """
    windows_norm, indices = _make_normalized_windows(x_values, window_size)
    X = _extract_features_batch(windows_norm)

    struct = np.ones(2 * margin + 1, dtype=bool)
    dilated = binary_dilation(cp_labels == 1, structure=struct)
    y = dilated[indices].astype(np.int32)

    return X, y


def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    iterations: int = 500,
    depth: int = 6,
    learning_rate: float = 0.05,
    early_stopping_rounds: int = 30,
    random_seed: int = 42,
    **cb_params,
):
    """
    Обучает CatBoostClassifier с early stopping по валидационной выборке.

    :param X_train: матрица признаков обучающей выборки
    :param y_train: метки обучающей выборки
    :param X_val: матрица признаков валидационной выборки
    :param y_val: метки валидационной выборки
    :param iterations: максимальное число итераций (деревьев)
    :param depth: глубина деревьев
    :param learning_rate: скорость обучения
    :param early_stopping_rounds: терпение ранней остановки
    :param random_seed: случайное зерно
    :param cb_params: дополнительные параметры CatBoostClassifier
    :return: обученная модель CatBoostClassifier
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError("catboost обязателен для CatBoost-адаптера: pip install catboost") from exc

    model = CatBoostClassifier(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        random_seed=random_seed,
        verbose=100,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="F1",
        **cb_params,
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
    )
    return model


def predict_catboost(
    model,
    x_values: np.ndarray,
    window_size: int,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Применяет модель ко всему временному ряду через скользящее окно (векторизованно).

    :param model: обученная модель CatBoostClassifier
    :param x_values: временной ряд
    :param window_size: размер скользящего окна
    :param threshold: порог бинаризации
    :return: (scores, preds) — вероятности и бинарные предсказания для каждого шага
    """
    scores = np.full(len(x_values), np.nan)

    windows_norm, indices = _make_normalized_windows(x_values, window_size)
    if len(windows_norm) == 0:
        return scores, np.zeros(len(x_values), dtype=int)

    X = _extract_features_batch(windows_norm)
    probs = model.predict_proba(X)[:, 1]

    scores[indices] = probs

    preds = np.zeros(len(x_values), dtype=int)
    valid = ~np.isnan(scores)
    preds[valid] = (scores[valid] >= threshold).astype(int)

    return scores, preds
