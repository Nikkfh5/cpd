"""
Модуль генерации синтетических данных для задачи обнаружения точек разладки (CPD).

Извлечено из ploting_graphs (1).ipynb (ячейки 2, 3, 22-27).
Генерация траекторий SDE: dx = sin(x)*dt + sqrt(D)*noise
с различными типами шума (white, pink, brownian, violet, blue).

Определение точек разладки: момент, когда floor(x_t / π) изменяется
(процесс переходит через границу кратную π).

Публичный API:
    generate_tc_dataframe  — аналог ячеек 22-27 ноутбука; возвращает DataFrame
                             с колонками data_pi, levels, indic.
    load_ecg_dataframe     — загружает ECG-данные из sample_29.csv и строит
                             аналогичный DataFrame (если файл доступен).
"""

import math
import numpy as np


# ============================================================
# 1. Функции генерации шума (из ноутбука, ячейка 2)
# ============================================================

def _noise_psd(N, psd=lambda f: 1):
    """Генерация шума с заданной спектральной плотностью мощности (PSD)."""
    X_white = np.fft.rfft(np.random.randn(N))
    S = psd(np.fft.rfftfreq(N))
    S = S / np.sqrt(np.mean(S**2))
    X_shaped = X_white * S
    return np.fft.irfft(X_shaped)


def white_noise(N):
    """Белый шум — равномерная спектральная плотность."""
    return _noise_psd(N, lambda f: 1)


def pink_noise(N):
    """Розовый шум — PSD ~ 1/sqrt(f)."""
    return _noise_psd(N, lambda f: 1 / np.where(f == 0, float('inf'), np.sqrt(f)))


def brownian_noise(N):
    """Броуновский (красный) шум — PSD ~ 1/f."""
    return _noise_psd(N, lambda f: 1 / np.where(f == 0, float('inf'), f))


def violet_noise(N):
    """Фиолетовый шум — PSD ~ f."""
    return _noise_psd(N, lambda f: f)


def blue_noise(N):
    """Голубой шум — PSD ~ sqrt(f)."""
    return _noise_psd(N, lambda f: np.sqrt(f))


# Словарь для доступа по строковому имени
_NOISE_FUNCTIONS = {
    "white": white_noise,
    "pink": pink_noise,
    "brownian": brownian_noise,
    "violet": violet_noise,
    "blue": blue_noise,
}


def get_noise(N, noise_type="white"):
    """
    Возвращает массив шума длины N заданного типа.

    :param N: длина массива
    :param noise_type: тип шума — "white", "pink", "brownian", "violet", "blue"
    :return: np.array длины N
    """
    if noise_type not in _NOISE_FUNCTIONS:
        raise ValueError(
            f"Неизвестный тип шума: {noise_type!r}. "
            f"Допустимые: {list(_NOISE_FUNCTIONS.keys())}"
        )
    return _NOISE_FUNCTIONS[noise_type](N)


# ============================================================
# 2. Генерация траектории SDE (из get_data, ветка no_error=False)
# ============================================================

def generate_sde_trajectory(length, dt=1.0, D=0.5, noise_type="white", seed=42):
    """
    Генерирует траекторию SDE: x_{t+1} = x_t + sin(x_t)*dt + sqrt(D)*errors[i]

    Шум (errors) генерируется функцией get_noise, затем нормализуется
    (центрируется и делится на std) — как в Binary_Telegraph_Process.get_data().

    :param length: количество шагов (не считая начального)
    :param dt: шаг по времени
    :param D: коэффициент диффузии
    :param noise_type: тип шума — "white", "pink", "brownian", "violet", "blue"
    :param seed: случайное зерно
    :return: np.array длины length+1 (значения процесса x_t)
    """
    np.random.seed(seed)

    # Генерируем и нормализуем шум (как в ноутбуке)
    errors = get_noise(length, noise_type)
    errors -= np.mean(errors)
    std = np.std(errors)
    if std > 1e-12:
        errors /= std

    x_t = math.pi / 1e6
    data = [x_t]

    for i in range(length):
        x_t = x_t + math.sin(x_t) * dt + math.sqrt(D) * errors[i]
        data.append(x_t)

    return np.array(data)


# ============================================================
# 3. Определение уровней и точек разладки (из data_to_pi)
# ============================================================

def compute_levels(x_values):
    """
    Определяет уровни floor(x/π) и точки разладки для траектории.

    Логика 1:1 из Binary_Telegraph_Process.data_to_pi() в ноутбуке:
    - pi_list[i] = x_values[i] / π
    - level_list отслеживает текущий «уровень» (целое число)
    - Точка разладки — момент, когда level_list[i] != level_list[i-1]

    :param x_values: np.array значений процесса
    :return: (levels, cp_labels)
        levels — np.array[int] длины len(x_values)
        cp_labels — np.array[int] длины len(x_values), 1 в точках разладки
    """
    n = len(x_values)
    pi_list = np.zeros(n)
    level_list = np.zeros(n, dtype=int)

    pi_list[0] = 0  # начальное x ≈ 0, x/π ≈ 0
    level_list[0] = 0

    for i in range(1, n):
        temp = x_values[i] / math.pi
        pi_list[i] = temp
        prev_level = level_list[i - 1]
        prev_pi = pi_list[i - 1]

        # Проверка пересечения нуля
        if (prev_pi > 0 and temp < 0) or (prev_pi < 0 and temp > 0):
            if abs(temp - prev_level) >= 1:
                if temp > 0:
                    level_list[i] = prev_level + math.floor(abs(temp - prev_level))
                else:
                    level_list[i] = prev_level - math.floor(abs(temp - prev_level))
            else:
                level_list[i] = prev_level
        else:
            diff = abs(temp) - abs(prev_level)
            if diff >= 1:
                if temp > 0:
                    level_list[i] = prev_level + math.floor(diff)
                else:
                    level_list[i] = prev_level - math.floor(diff)
            elif diff <= -1:
                if temp > 0:
                    level_list[i] = prev_level - math.floor(abs(diff))
                else:
                    level_list[i] = prev_level + math.floor(abs(diff))
            else:
                level_list[i] = prev_level

    # Точки разладки — там, где уровень меняется
    cp_labels = np.zeros(n, dtype=int)
    for i in range(1, n):
        if level_list[i] != level_list[i - 1]:
            cp_labels[i] = 1

    return level_list, cp_labels


# ============================================================
# 4. Удобные обёртки
# ============================================================

def generate_dataset(length, dt=1.0, D=0.5, noise_type="white", seed=42):
    """
    Генерирует траекторию SDE + определяет точки разладки.

    :param length: количество шагов траектории
    :param dt: шаг по времени
    :param D: коэффициент диффузии
    :param noise_type: тип шума — "white", "pink", "brownian", "violet", "blue"
    :param seed: случайное зерно
    :return: (x_values, cp_labels)
        x_values — np.array длины length+1
        cp_labels — np.array длины length+1, 1 в точках разладки
    """
    x_values = generate_sde_trajectory(length, dt, D, noise_type, seed)
    _, cp_labels = compute_levels(x_values)
    return x_values, cp_labels


def generate_multi_dataset(n_seq, seq_length, dt=1.0, D_range=(0.3, 2.0),
                           noise_types=None, seed=42, dt_values=None,
                           t4_fraction=0.0):
    """
    Генерирует несколько траекторий SDE с разными D, dt и шумом для обучения.
    Объединяет их в одну длинную последовательность.

    :param n_seq: количество последовательностей
    :param seq_length: длина каждой последовательности (шагов)
    :param dt: шаг по времени (используется если dt_values=None)
    :param D_range: диапазон (min, max) для случайного выбора D
    :param noise_types: список типов шума для случайного выбора,
                        по умолчанию ["white"]
    :param seed: базовое случайное зерно
    :param dt_values: список значений dt для случайного выбора;
                      None = использовать фиксированный dt
    :param t4_fraction: доля последовательностей с T4-динамикой (sticky π-переходы);
                        0.0 = только стандартная SDE, 1.0 = только T4
    :return: (x_values, cp_labels) — конкатенация всех последовательностей
    """
    if noise_types is None:
        noise_types = ["white"]

    rng = np.random.RandomState(seed)
    all_x = []
    all_cp = []

    for _ in range(n_seq):
        D = rng.uniform(D_range[0], D_range[1])
        seq_seed = rng.randint(0, 100000)
        noise_type = noise_types[rng.randint(0, len(noise_types))]
        chosen_dt = dt_values[rng.randint(0, len(dt_values))] if dt_values else dt
        use_t4 = rng.random() < t4_fraction
        if use_t4:
            x, cp = generate_dataset_t4(seq_length, chosen_dt, D, noise_type, seq_seed)
        else:
            x, cp = generate_dataset(seq_length, chosen_dt, D, noise_type, seq_seed)
        all_x.append(x)
        all_cp.append(cp)

    return np.concatenate(all_x), np.concatenate(all_cp)


# ============================================================
# 6. T4-генератор и clusters-разметка (из ноутбука)
# ============================================================

def generate_sde_trajectory_t4(length, dt=1.0, D=0.5, noise_type="white", seed=42):
    """
    T4-вариант траектории SDE: та же динамика dx = sin(x)*dt + sqrt(D)*noise,
    но с «залипанием» вблизи кратных π.

    Когда следующее значение попадает в пределах 5% от кратного π
    (т.е. |x/π - round(x/π)| < 0.05), sin(x) обнуляется на следующем шаге.
    Это делает π-переходы более резкими и устойчивыми.

    Логика извлечена из Binary_Telegraph_Process.get_data_t4() ноутбука.

    :param length: количество шагов (не считая начального)
    :param dt: шаг по времени
    :param D: коэффициент диффузии
    :param noise_type: тип шума — "white", "pink", "brownian", "violet", "blue"
    :param seed: случайное зерно
    :return: np.array длины length+1
    """
    np.random.seed(seed)

    errors = get_noise(length, noise_type)
    errors -= np.mean(errors)
    std = np.std(errors)
    if std > 1e-12:
        errors /= std

    x_t = math.pi / 1e6
    sin_t = math.sin(x_t)
    data = [x_t]

    for i in range(length):
        temp = x_t + sin_t * dt + math.sqrt(D) * errors[i]
        ratio = temp / math.pi
        if abs(ratio - round(ratio)) < 0.05:
            x_t = temp
            sin_t = 0.0
        else:
            x_t = temp
            sin_t = math.sin(x_t)
        data.append(x_t)

    return np.array(data)


def compute_levels_clusters(x_values):
    """
    Альтернативная разметка точек разладки: округление до ближайшего кратного π.

    Уровень = round(x/π), точка разладки — момент смены уровня.
    В отличие от compute_levels (floor-логика), здесь граница между уровнями
    проходит ровно посередине между соседними кратными π.

    Логика извлечена из Binary_Telegraph_Process.clusters() ноутбука.

    :param x_values: np.array значений процесса
    :return: (levels, cp_labels)
        levels — np.array[int] длины len(x_values)
        cp_labels — np.array[int] длины len(x_values), 1 в точках разладки
    """
    n = len(x_values)
    levels = np.array([round(v / math.pi) for v in x_values], dtype=int)
    cp_labels = np.zeros(n, dtype=int)
    for i in range(1, n):
        if levels[i] != levels[i - 1]:
            cp_labels[i] = 1
    return levels, cp_labels


def generate_dataset_t4(length, dt=1.0, D=0.5, noise_type="white", seed=42):
    """
    Генерирует T4-траекторию SDE + определяет точки разладки (floor-разметка).

    Использует T4-динамику (sticky π-переходы) с той же floor(x/π)-разметкой,
    что и стандартный generate_dataset. Это даёт более «чёткие» переходы
    при тех же критериях разладки — полезно для разнообразия обучающих данных.

    :param length: количество шагов
    :param dt: шаг по времени
    :param D: коэффициент диффузии
    :param noise_type: тип шума
    :param seed: случайное зерно
    :return: (x_values, cp_labels)
    """
    x_values = generate_sde_trajectory_t4(length, dt, D, noise_type, seed)
    _, cp_labels = compute_levels(x_values)
    return x_values, cp_labels


def generate_dataset_clusters(length, dt=1.0, D=0.5, noise_type="white", seed=42):
    """
    Генерирует стандартную траекторию SDE + определяет точки разладки (round-разметка).

    Использует стандартную SDE-динамику с clusters-разметкой (round(x/π) вместо
    floor(x/π)). Граница смены уровня находится посередине между кратными π —
    альтернативное определение точки разладки для анализа.

    :param length: количество шагов
    :param dt: шаг по времени
    :param D: коэффициент диффузии
    :param noise_type: тип шума
    :param seed: случайное зерно
    :return: (x_values, cp_labels)
    """
    x_values = generate_sde_trajectory(length, dt, D, noise_type, seed)
    _, cp_labels = compute_levels_clusters(x_values)
    return x_values, cp_labels


# ============================================================
# 5. DataFrame-обёртки (аналог ячеек 22-27 ноутбука)
# ============================================================

def generate_tc_dataframe(length, dt=1.0, D=0.5, noise_type="white", seed=42):
    """
    Генерирует t_c DataFrame — аналог ячеек 22-27 ноутбука.

    Столбцы:
        data_pi — x_t / π (вещественный уровень процесса)
        levels  — целочисленный уровень (floor-аппроксимация data_to_pi)
        indic   — 1 в моменты смены уровня (точки разладки), иначе 0

    Логика 1:1 из Binary_Telegraph_Process.data_to_pi() + индикаторный столбец
    из ячейки 27 ноутбука (t_c['indic']).

    :param length: количество шагов SDE
    :param dt: шаг по времени
    :param D: коэффициент диффузии
    :param noise_type: тип шума — "white", "pink", "brownian", "violet", "blue"
    :param seed: случайное зерно
    :return: pandas.DataFrame с колонками data_pi, levels, indic
    """
    import pandas as pd

    x_values = generate_sde_trajectory(length, dt, D, noise_type, seed)
    levels, cp_labels = compute_levels(x_values)

    return pd.DataFrame(
        {
            "data_pi": x_values / math.pi,
            "levels": levels,
            "indic": cp_labels,
        }
    )


def load_ecg_dataframe(csv_path: str):
    """
    Загружает ECG-данные из CSV и строит t_c DataFrame.

    Ожидает CSV с колонкой «X» (значения ЭКГ-сигнала), аналогично
    строке ``errors = pd.read_csv('sample_29.csv')["X"].values``
    из ноутбука.

    Возвращает DataFrame с теми же колонками, что generate_tc_dataframe:
        data_pi, levels, indic

    :param csv_path: путь к CSV-файлу с колонкой «X»
    :return: pandas.DataFrame с колонками data_pi, levels, indic
    :raises FileNotFoundError: если файл не найден
    :raises KeyError: если в CSV нет колонки «X»
    """
    import pandas as pd
    from pathlib import Path

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"ECG-файл не найден: {csv_path}. "
            "Поместите sample_29.csv в корень репозитория."
        )

    raw = pd.read_csv(str(path))
    if "X" not in raw.columns:
        raise KeyError(f"В файле {csv_path} нет колонки 'X'. Доступные: {list(raw.columns)}")

    x_values = raw["X"].to_numpy(dtype=float)
    levels, cp_labels = compute_levels(x_values)

    return pd.DataFrame(
        {
            "data_pi": x_values / math.pi,
            "levels": levels,
            "indic": cp_labels,
        }
    )
