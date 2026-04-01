"""
Адаптер GRU-пайплайна для обнаружения точек разладки.

Оборачивает algoritms/gru_cpd.py в унифицированный интерфейс CPD-пайплайна.

Обучение на синтетических данных генерируется с seed+1 (train) и seed+2 (val),
чтобы тестовая серия series не участвовала в обучении.
"""

import random
from pathlib import Path

import numpy as np


_DEFAULT_SOURCE_PATH = str(Path(__file__).resolve().parent / "data_utils.py")


def run_gru(
    series: np.ndarray,
    seed: int,
    window_size: int = 50,
    threshold: float = 0.7,
    nms_min_distance: int = 50,
    n_train_seq: int = 100,
    seq_length: int = 5000,
    n_val_seq: int = 15,
    epochs: int = 100,
    batch_size: int = 512,
    lr: float = 1e-3,
    patience: int = 15,
    noise_types: list | None = None,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
    weight_decay: float = 1e-4,
    train_d_range: list | None = None,
    train_dt_values: list | None = None,
    weights_path: str | None = None,
    **kwargs,
) -> tuple[list[int], np.ndarray, dict]:
    """
    Обучает GRU на синтетических данных и делает инференс на series.

    Обучающие данные генерируются с seed+1 (n_train_seq последовательностей
    длины seq_length), валидационные — с seed+2 (n_val_seq последовательностей).
    Тестовая серия series не используется при обучении.

    :param series: тестовый временной ряд (инференс)
    :param seed: случайное зерно; train = seed+1, val = seed+2
    :param window_size: чётное целое >= 2 — размер скользящего окна GRU
    :param threshold: порог бинаризации вероятностей
    :param nms_min_distance: минимальное расстояние между CP после NMS
    :param n_train_seq: число обучающих последовательностей
    :param seq_length: длина каждой обучающей последовательности
    :param n_val_seq: число валидационных последовательностей
    :param epochs: максимальное число эпох обучения
    :param batch_size: размер мини-батча
    :param lr: скорость обучения Adam
    :param patience: терпение ранней остановки по F1 на валидации
    :param noise_types: типы шума для обучающих данных; None = все пять типов
    :param hidden_size: размер скрытого состояния GRU
    :param num_layers: число слоёв GRU
    :param dropout: вероятность dropout между слоями
    :param weight_decay: L2-регуляризация в оптимизаторе Adam
    :param train_d_range: диапазон [min, max] коэффициента диффузии для обучающих данных;
        None = широкий диапазон (0.05, 5.0)
    :param train_dt_values: список значений dt для обучающих данных;
        None = [0.1, 0.5, 1.0, 2.0, 5.0]
    :param weights_path: путь к .pt файлу с сохранёнными весами; если задан и файл
        существует — модель загружается без обучения
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, scores, meta)
    """
    if not isinstance(window_size, int) or window_size < 2 or window_size % 2 != 0:
        raise ValueError(
            f"window_size должен быть чётным целым >= 2, получено: {window_size!r}"
        )

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:
        raise ImportError("torch обязателен для GRU-адаптера") from exc

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    from algoritms.gru_cpd import (
        GRUChangePointDetector,
        SlidingWindowDataset,
        nms_predictions,
        predict_on_series,
        train_model,
    )
    from algoritms.data_utils import generate_multi_dataset

    _all_noise_types = ["white", "pink", "brownian", "violet", "blue"]
    train_noise_types = noise_types if noise_types is not None else _all_noise_types
    _train_dt_values = list(train_dt_values) if train_dt_values is not None else [0.1, 0.5, 1.0, 2.0, 5.0]
    _train_D_range = tuple(train_d_range) if train_d_range is not None else (0.05, 5.0)

    series = np.asarray(series, dtype=float).reshape(-1)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = GRUChangePointDetector(input_size=1, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)

    if weights_path and Path(weights_path).exists():
        checkpoint = torch.load(weights_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        print(f"GRU: loaded weights from {weights_path}", flush=True)
    else:
        print(f"GRU: device={device}, generating {n_train_seq} train seqs x {seq_length}...", flush=True)
        x_train, cp_train = generate_multi_dataset(
            n_train_seq, seq_length, seed=seed + 1,
            noise_types=train_noise_types,
            D_range=_train_D_range,
            dt_values=_train_dt_values,
        )
        print(f"GRU: train data ready ({len(x_train)} pts), generating {n_val_seq} val seqs...", flush=True)
        x_val, cp_val = generate_multi_dataset(
            n_val_seq, seq_length, seed=seed + 2,
            noise_types=train_noise_types,
            D_range=_train_D_range,
            dt_values=_train_dt_values,
        )
        print(f"GRU: val data ready ({len(x_val)} pts), building datasets...", flush=True)

        train_dataset = SlidingWindowDataset(x_train, cp_train, window_size)
        val_dataset = SlidingWindowDataset(x_val, cp_val, window_size)
        print(f"GRU: {len(train_dataset)} train windows, {len(val_dataset)} val windows. Starting training.", flush=True)

        n_pos = float(train_dataset.labels.sum())
        n_neg = float(len(train_dataset.labels)) - n_pos
        pos_weight = min(n_neg / max(n_pos, 1.0), 30.0)

        g = torch.Generator()
        g.manual_seed(seed)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            generator=g, num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=0,
        )

        model = train_model(
            model,
            train_loader,
            val_loader,
            epochs=epochs,
            lr=lr,
            pos_weight=pos_weight,
            device=device,
            patience=patience,
            weight_decay=weight_decay,
        )

        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        noise_tag = "_".join(train_noise_types) if len(train_noise_types) <= 2 else "all_noise"
        model_filename = (
            f"gru_h{hidden_size}_l{num_layers}_seq{seq_length}"
            f"_n{n_train_seq}_{noise_tag}_seed{seed}.pt"
        )
        model_path = models_dir / model_filename
        torch.save({
            "state_dict": model.state_dict(),
            "config": {
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "dropout": dropout,
                "window_size": window_size,
                "seq_length": seq_length,
                "n_train_seq": n_train_seq,
                "seed": seed,
            },
        }, str(model_path))
        print(f"GRU: model saved to {model_path}", flush=True)

    scores_raw, _ = predict_on_series(model, series, window_size, device, threshold)
    scores_clean = np.nan_to_num(scores_raw, nan=0.0)
    preds = nms_predictions(scores_clean, threshold=threshold, min_distance=nms_min_distance)

    change_points = np.where(preds == 1)[0].tolist()

    meta = {
        "dataset_source": kwargs.get("dataset_source", "data_utils"),
        "generation_params": kwargs.get("generation_params", {}),
        "source_path": kwargs.get("source_path", _DEFAULT_SOURCE_PATH),
        "seed": int(seed),
    }

    return change_points, scores_raw, meta
