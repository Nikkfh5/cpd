"""
LSTM-модель для обнаружения точек разладки (change point detection) во временных рядах.

Используется бинарная классификация на скользящих окнах:
  - Вход: окно из window_size значений x_t
  - Выход: вероятность наличия точки разладки в центре окна
  - Обучение на синтетических данных от SDE: dx = sin(x)dt + sqrt(D)*noise

Точка разладки определяется как момент, когда floor(x_t / π) изменяется
(процесс переходит через границу кратную π).

Поддерживаемые источники данных (--data-source):
    synthetic — generate_multi_dataset со всеми типами шума (белый, розовый,
                броуновский, фиолетовый, голубой); диапазон D подбирается случайно.
    notebook  — Binary_Telegraph_Process-эквивалент через generate_dataset;
                тип шума и параметры задаются явно через --noise-type, --D, --dt.
    file      — внешний CSV с колонками x (или data_pi) и indic;
                обучение остаётся синтетическим, тест берётся из файла.
"""

import argparse
import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from algoritms.data_utils import (
    generate_dataset,
    generate_multi_dataset,
    compute_levels,
)


# ============================================================
# 1. Источники данных
# ============================================================

_ALL_NOISE_TYPES = ["white", "pink", "brownian", "violet", "blue"]


def load_notebook_data(
    dt: float = 1.0,
    D: float = 0.5,
    length: int = 10000,
    noise_type: str = "white",
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Загружает тестовые данные через полноценный генератор SDE.

    Точный аналог Binary_Telegraph_Process.get_data() + data_to_pi() из ноутбука:
    траектория dx = sin(x)dt + sqrt(D)*noise с нормированным шумом,
    метки разладки — моменты смены floor(x/π)-уровня.

    :param dt: шаг по времени
    :param D: коэффициент диффузии
    :param length: число шагов траектории (возвращается length+1 точка)
    :param noise_type: тип шума — "white", "pink", "brownian", "violet", "blue"
    :param seed: случайное зерно для воспроизводимости
    :return: (x_values, cp_labels) — np.ndarray длины length+1
    """
    return generate_dataset(length, dt=dt, D=D, noise_type=noise_type, seed=seed)


def load_file_data(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Загружает данные из CSV-файла.

    Ожидаемые форматы:
      - колонки «x» и «indic»: x_values напрямую, cp_labels из indic
      - колонки «data_pi» и «indic»: x_values = data_pi * π, cp_labels из indic
      - колонка «x» без «indic»: cp_labels вычисляются через compute_levels
      - колонка «X» (ECG): x_values из X, cp_labels вычисляются

    :param csv_path: путь к CSV-файлу
    :return: (x_values, cp_labels) — np.ndarray
    :raises FileNotFoundError: если файл не найден
    :raises ValueError: если нет ни одной из ожидаемых колонок
    """
    import pandas as pd
    from pathlib import Path

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    df = pd.read_csv(str(path))
    cols = set(df.columns)

    if "x" in cols:
        x_values = df["x"].to_numpy(dtype=float)
    elif "data_pi" in cols:
        x_values = df["data_pi"].to_numpy(dtype=float) * math.pi
    elif "X" in cols:
        x_values = df["X"].to_numpy(dtype=float)
    else:
        raise ValueError(
            f"CSV должен содержать колонку 'x', 'data_pi' или 'X'. "
            f"Найдено: {sorted(cols)}"
        )

    if "indic" in cols:
        cp_labels = df["indic"].to_numpy(dtype=int)
    else:
        _, cp_labels = compute_levels(x_values)

    return x_values, cp_labels


# ============================================================
# 2. Подготовка датасета (скользящие окна)
# ============================================================

class SlidingWindowDataset(Dataset):
    """
    Датасет скользящих окон для бинарной классификации.
    Метка = 1, если в пределах margin от центра окна есть точка разладки.
    """

    def __init__(self, x_values, cp_labels, window_size=50, margin=5):
        self.window_size = window_size
        self.margin = margin

        half = window_size // 2
        self.windows = []
        self.labels = []

        for i in range(half, len(x_values) - half):
            window = x_values[i - half:i + half]
            left = max(0, i - margin)
            right = min(len(cp_labels), i + margin + 1)
            label = 1 if np.any(cp_labels[left:right] == 1) else 0
            self.windows.append(window)
            self.labels.append(label)

        self.windows = np.array(self.windows, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        w = self.windows[idx]
        std = w.std()
        if std > 1e-8:
            w = (w - w.mean()) / std
        else:
            w = w - w.mean()
        x = torch.tensor(w, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


# ============================================================
# 3. Модель LSTM
# ============================================================

class LSTMChangePointDetector(nn.Module):
    """
    LSTM бинарный классификатор для обнаружения точек разладки.

    Архитектура:
        LSTM(input_size=1, hidden_size, num_layers, dropout) → Linear → Sigmoid
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        logit = self.fc(last_hidden).squeeze(-1)
        return logit


# ============================================================
# 4. Обучение
# ============================================================

def train_model(
    model,
    train_loader,
    val_loader,
    epochs=30,
    lr=1e-3,
    pos_weight=None,
    device="cpu",
    patience=5,
    weight_decay=1e-4,
):
    """
    Обучение модели с ранней остановкой по F1 на валидации.

    :param model: модель LSTMChangePointDetector
    :param train_loader: DataLoader для обучения
    :param val_loader: DataLoader для валидации
    :param epochs: максимальное число эпох
    :param lr: скорость обучения
    :param pos_weight: вес положительного класса для BCE
    :param device: устройство (cpu/cuda)
    :param patience: терпение для ранней остановки
    :param weight_decay: L2-регуляризация в Adam (защита от переобучения)
    :return: обученная модель
    """
    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.to(device)

    best_f1 = 0.0
    best_state = None
    no_improve = 0

    t_train_start = time.time()
    for epoch in range(epochs):
        t_epoch_start = time.time()
        model.train()
        total_loss = 0
        n_batches = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        val_f1, val_prec, val_rec = evaluate_batch(model, val_loader, device)
        epoch_sec = time.time() - t_epoch_start
        elapsed_total = time.time() - t_train_start

        print(
            f"Epoch {epoch + 1:3d}/{epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Val P: {val_prec:.3f} R: {val_rec:.3f} F1: {val_f1:.3f} | "
            f"{epoch_sec:.1f}s/ep  elapsed {elapsed_total/60:.1f}m",
            flush=True,
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    return model


def evaluate_batch(model, loader, device="cpu", threshold=0.5):
    """Вычисление Precision, Recall, F1 по батчам."""
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            logits = model(x_batch)
            preds = (torch.sigmoid(logits) >= threshold).float()
            tp += ((preds == 1) & (y_batch == 1)).sum().item()
            fp += ((preds == 1) & (y_batch == 0)).sum().item()
            fn += ((preds == 0) & (y_batch == 1)).sum().item()

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return f1, precision, recall


# ============================================================
# 5. Оценка на временном ряде (поточечная)
# ============================================================

def predict_on_series(model, x_values, window_size=50, device="cpu", threshold=0.5):
    """
    Применяет модель ко всему временному ряду через скользящее окно.

    :return: scores — np.array вероятностей для каждого временного шага,
             preds — np.array бинарных предсказаний
    """
    model.eval()
    half = window_size // 2
    scores = np.full(len(x_values), np.nan)

    windows = []
    indices = []
    for i in range(half, len(x_values) - half):
        window = x_values[i - half:i + half]
        windows.append(window)
        indices.append(i)

    if not windows:
        return scores, np.zeros(len(x_values), dtype=int)

    windows_arr = np.array(windows, dtype=np.float32)
    means = windows_arr.mean(axis=1, keepdims=True)
    stds = windows_arr.std(axis=1, keepdims=True)
    stds[stds < 1e-8] = 1.0
    windows_arr = (windows_arr - means) / stds
    windows_tensor = torch.tensor(windows_arr).unsqueeze(-1).to(device)

    with torch.no_grad():
        batch_size = 1024
        all_probs = []
        for start in range(0, len(windows_tensor), batch_size):
            batch = windows_tensor[start:start + batch_size]
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
        all_probs = np.concatenate(all_probs)

    for idx, prob in zip(indices, all_probs):
        scores[idx] = prob

    preds = np.zeros(len(x_values), dtype=int)
    preds[~np.isnan(scores)] = (scores[~np.isnan(scores)] >= threshold).astype(int)

    return scores, preds


def evaluate_cpd(true_labels, pred_labels, margin=5):
    """
    Оценка качества CPD с допуском δ (margin).

    Для каждой истинной точки разладки ищется ближайшая предсказанная
    в пределах ±margin. Метрики: Precision, Recall, F1, MAE.

    :return: dict с метриками
    """
    true_cps = np.where(true_labels == 1)[0]
    pred_cps = np.where(pred_labels == 1)[0]

    if len(true_cps) == 0 and len(pred_cps) == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "mae": 0.0,
                "true_count": 0, "pred_count": 0}

    if len(pred_cps) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mae": float("inf"),
                "true_count": len(true_cps), "pred_count": 0}

    if len(true_cps) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mae": float("inf"),
                "true_count": 0, "pred_count": len(pred_cps)}

    matched_true = set()
    matched_pred = set()
    distances = []

    for t_idx, t_cp in enumerate(true_cps):
        dists = np.abs(pred_cps.astype(float) - t_cp)
        nearest_idx = np.argmin(dists)
        if dists[nearest_idx] <= margin:
            matched_true.add(t_idx)
            matched_pred.add(nearest_idx)
            distances.append(dists[nearest_idx])

    tp = len(matched_true)
    fp = len(pred_cps) - len(matched_pred)
    fn = len(true_cps) - len(matched_true)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    mae = np.mean(distances) if distances else float("inf")

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mae": mae,
        "true_count": len(true_cps),
        "pred_count": len(pred_cps),
    }


# ============================================================
# 6. Визуализация
# ============================================================

def plot_results(x_values, true_labels, scores, preds, title="LSTM CPD", save_path=None):
    """Визуализация результатов: временной ряд, вероятности и предсказания."""
    import matplotlib.pyplot as plt

    true_cps = np.where(true_labels == 1)[0]
    pred_cps = np.where(preds == 1)[0]

    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    ax1 = axes[0]
    ax1.plot(x_values, linewidth=0.5, color="steelblue", label="x(t)")
    ax1.scatter(true_cps, x_values[true_cps], color="green", marker="^", s=60,
                zorder=5, label=f"Истинные CP ({len(true_cps)})")
    ax1.scatter(pred_cps, x_values[pred_cps], color="red", marker="v", s=40,
                zorder=5, label=f"Предсказанные CP ({len(pred_cps)})")
    ax1.set_ylabel("x(t)")
    ax1.set_title(title)
    ax1.legend(loc="upper right")

    ax2 = axes[1]
    valid_mask = ~np.isnan(scores)
    ax2.plot(np.where(valid_mask)[0], scores[valid_mask], linewidth=0.5, color="orange",
             label="P(change point)")
    ax2.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8)
    for cp in true_cps:
        ax2.axvline(x=cp, color="green", alpha=0.3, linewidth=0.5)
    ax2.set_ylabel("Вероятность")
    ax2.set_xlabel("Время (шаг)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"График сохранён: {save_path}")
    plt.show()


# ============================================================
# 7. NMS
# ============================================================

def nms_predictions(scores, threshold=0.5, min_distance=10):
    """
    Non-maximum suppression: из кластера предсказаний оставляет только пик.
    Предотвращает множественные срабатывания около одной точки разладки.

    :param scores: np.array вероятностей
    :param threshold: порог срабатывания
    :param min_distance: минимальное расстояние между предсказаниями
    :return: np.array бинарных предсказаний после NMS
    """
    preds = np.zeros(len(scores), dtype=int)
    candidates = np.where(scores >= threshold)[0]
    if len(candidates) == 0:
        return preds

    order = candidates[np.argsort(-scores[candidates])]
    suppressed = set()

    for idx in order:
        if idx in suppressed:
            continue
        preds[idx] = 1
        for j in range(max(0, idx - min_distance), min(len(scores), idx + min_distance + 1)):
            if j != idx:
                suppressed.add(j)

    return preds


# ============================================================
# 8. Главный блок с argparse
# ============================================================

def _build_parser() -> argparse.ArgumentParser:
    """
    Создаёт ArgumentParser для запуска обучения/инференса LSTM CPD.

    Группы аргументов:
        Источник данных: --data-source, --file-path
        Параметры SDE:   --dt, --D, --noise-type, --length, --seed
        Обучение:        --n-train-seq, --seq-length, --n-val-seq,
                         --epochs, --batch-size, --lr, --patience
        Инференс:        --window-size, --threshold
        Прочее:          --no-plot
    """
    parser = argparse.ArgumentParser(
        description="LSTM CPD: обучение и оценка детектора точек разладки"
    )

    parser.add_argument(
        "--data-source",
        choices=["synthetic", "notebook", "file"],
        default="synthetic",
        help=(
            "synthetic — все типы шума, случайный D; "
            "notebook  — один тип шума, фиксированный D (как в ноутбуке); "
            "file      — тестовые данные из CSV (--file-path)"
        ),
    )
    parser.add_argument(
        "--file-path",
        default=None,
        metavar="PATH",
        help="Путь к CSV для --data-source file (колонки: x или data_pi, indic)",
    )

    parser.add_argument("--dt", type=float, default=1.0, help="Шаг по времени (default: 1.0)")
    parser.add_argument("--D", type=float, default=0.5, help="Коэффициент диффузии (default: 0.5)")
    parser.add_argument(
        "--noise-type",
        default="white",
        choices=_ALL_NOISE_TYPES,
        help="Тип шума для notebook/test (default: white)",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=10000,
        help="Длина тестового ряда (default: 10000)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно (default: 42)")

    parser.add_argument(
        "--n-train-seq", type=int, default=20, help="Число обучающих последовательностей (default: 20)"
    )
    parser.add_argument(
        "--seq-length", type=int, default=5000, help="Длина каждой обучающей последовательности (default: 5000)"
    )
    parser.add_argument(
        "--n-val-seq", type=int, default=5, help="Число валидационных последовательностей (default: 5)"
    )
    parser.add_argument("--epochs", type=int, default=30, help="Число эпох (default: 30)")
    parser.add_argument("--batch-size", type=int, default=256, help="Размер батча (default: 256)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Скорость обучения (default: 1e-3)")
    parser.add_argument(
        "--patience", type=int, default=7, help="Терпение ранней остановки (default: 7)"
    )
    parser.add_argument(
        "--window-size", type=int, default=50,
        help="Размер скользящего окна, чётное >= 2 (default: 50)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Порог бинаризации (default: 0.5)"
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="Не строить график результатов"
    )

    return parser


def main():
    """
    Точка входа: обучение LSTM и оценка на тестовых данных.

    Режимы --data-source:
        synthetic — обучение на всех типах шума с разными D; тест на white/D=0.5
        notebook  — обучение на одном типе шума; тест с теми же параметрами
        file      — обучение синтетическое; тест из CSV-файла
    """
    parser = _build_parser()
    args = parser.parse_args()

    if args.window_size < 2 or args.window_size % 2 != 0:
        parser.error(
            f"--window-size должен быть чётным целым >= 2, получено: {args.window_size}"
        )

    MAX_POS_WEIGHT = 30.0
    MARGIN = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Устройство: {device}")
    print(f"Источник данных: {args.data_source}")

    # --- Тестовые данные ---
    if args.data_source == "file":
        if args.file_path is None:
            parser.error("--file-path обязателен при --data-source file")
        print(f"Загрузка тестовых данных из {args.file_path} ...")
        x_test, cp_test = load_file_data(args.file_path)
        print(f"  Тест: {len(x_test)} точек, {int(cp_test.sum())} точек разладки")
    else:
        print("Генерация тестовых данных ...")
        x_test, cp_test = load_notebook_data(
            dt=args.dt, D=args.D, length=args.length,
            noise_type=args.noise_type, seed=args.seed + 1000,
        )
        print(f"  Тест: {len(x_test)} точек, {int(cp_test.sum())} точек разладки")

    # --- Обучающие данные ---
    if args.data_source == "synthetic":
        print("Генерация обучающих данных (все типы шума, случайный D) ...")
        x_train, cp_train = generate_multi_dataset(
            args.n_train_seq, args.seq_length,
            dt=args.dt, D_range=(0.3, 2.0),
            noise_types=_ALL_NOISE_TYPES,
            seed=args.seed,
        )
        print(f"  Обучение: {len(x_train)} точек, {int(cp_train.sum())} точек разладки")

        print("Генерация валидационных данных ...")
        x_val, cp_val = generate_multi_dataset(
            args.n_val_seq, args.seq_length,
            dt=args.dt, D_range=(0.3, 2.0),
            noise_types=_ALL_NOISE_TYPES,
            seed=args.seed + 777,
        )
        print(f"  Валидация: {len(x_val)} точек, {int(cp_val.sum())} точек разладки")

    else:
        print(f"Генерация обучающих данных (noise={args.noise_type}, D={args.D}) ...")
        x_train, cp_train = generate_multi_dataset(
            args.n_train_seq, args.seq_length,
            dt=args.dt, D_range=(max(0.1, args.D * 0.6), args.D * 1.4),
            noise_types=[args.noise_type],
            seed=args.seed,
        )
        print(f"  Обучение: {len(x_train)} точек, {int(cp_train.sum())} точек разладки")

        print("Генерация валидационных данных ...")
        x_val, cp_val = generate_multi_dataset(
            args.n_val_seq, args.seq_length,
            dt=args.dt, D_range=(max(0.1, args.D * 0.6), args.D * 1.4),
            noise_types=[args.noise_type],
            seed=args.seed + 777,
        )
        print(f"  Валидация: {len(x_val)} точек, {int(cp_val.sum())} точек разладки")

    # --- Датасеты и загрузчики ---
    train_dataset = SlidingWindowDataset(x_train, cp_train, args.window_size, MARGIN)
    val_dataset = SlidingWindowDataset(x_val, cp_val, args.window_size, MARGIN)

    n_pos = float(train_dataset.labels.sum())
    n_neg = float(len(train_dataset.labels)) - n_pos
    pos_weight = min(n_neg / max(n_pos, 1.0), MAX_POS_WEIGHT)
    print(
        f"Баланс классов: {n_pos:.0f} pos / {n_neg:.0f} neg, "
        f"pos_weight={pos_weight:.1f} (макс {MAX_POS_WEIGHT})"
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Модель ---
    model = LSTMChangePointDetector(input_size=1, hidden_size=64, num_layers=2, dropout=0.2)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Модель: {total_params} параметров")

    # --- Обучение ---
    print("\n--- Обучение ---")
    model = train_model(
        model, train_loader, val_loader,
        epochs=args.epochs, lr=args.lr,
        pos_weight=pos_weight, device=device, patience=args.patience,
    )

    # --- Оценка на тесте ---
    print("\n--- Оценка на тестовых данных ---")
    scores, _ = predict_on_series(model, x_test, args.window_size, device, args.threshold)
    scores_clean = np.nan_to_num(scores, nan=0.0)
    preds = nms_predictions(scores_clean, threshold=args.threshold, min_distance=MARGIN * 2)

    metrics = evaluate_cpd(cp_test, preds, margin=MARGIN)
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  MAE:       {metrics['mae']:.2f}")
    print(f"  Истинных CP:       {metrics['true_count']}")
    print(f"  Предсказанных CP:  {metrics['pred_count']}")

    # --- Визуализация ---
    if not args.no_plot:
        plot_results(
            x_test, cp_test, scores, preds,
            title=f"LSTM CPD | source={args.data_source} | F1={metrics['f1']:.3f}",
            save_path="algoritms/lstm_cpd_result.png",
        )


if __name__ == "__main__":
    main()
