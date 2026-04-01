"""
Адаптер CatBoost-пайплайна для обнаружения точек разладки.

Оборачивает algoritms/catboost_cpd.py в унифицированный интерфейс CPD-пайплайна.

Обучение на синтетических данных генерируется с seed+1 (train) и seed+2 (val),
чтобы тестовая серия series не участвовала в обучении.
"""

from pathlib import Path

import numpy as np


_DEFAULT_SOURCE_PATH = str(Path(__file__).resolve().parent / "data_utils.py")


def run_catboost(
    series: np.ndarray,
    seed: int,
    window_size: int = 50,
    threshold: float = 0.5,
    nms_min_distance: int = 50,
    n_train_seq: int = 50,
    seq_length: int = 5000,
    n_val_seq: int = 10,
    iterations: int = 500,
    depth: int = 6,
    learning_rate: float = 0.05,
    weights_path: str | None = None,
    **kwargs,
) -> tuple[list[int], np.ndarray, dict]:
    """
    Обучает CatBoost на синтетических данных и делает инференс на series.

    Обучающие данные генерируются с seed+1 (n_train_seq последовательностей
    длины seq_length), валидационные — с seed+2 (n_val_seq последовательностей).
    Тестовая серия series не используется при обучении.

    :param series: тестовый временной ряд (инференс)
    :param seed: случайное зерно; train = seed+1, val = seed+2
    :param window_size: размер скользящего окна для извлечения признаков
    :param threshold: порог бинаризации вероятностей
    :param nms_min_distance: минимальное расстояние между CP после NMS
    :param n_train_seq: число обучающих последовательностей
    :param seq_length: длина каждой обучающей последовательности
    :param n_val_seq: число валидационных последовательностей
    :param iterations: максимальное число деревьев CatBoost
    :param depth: глубина деревьев
    :param learning_rate: скорость обучения
    :param weights_path: путь к .cbm файлу с сохранённой моделью; если задан и файл
        существует — модель загружается без обучения
    :param kwargs: generation_params, dataset_source, source_path
    :return: (change_points, scores, meta)
    """
    from catboost import CatBoostClassifier
    from algoritms.catboost_cpd import (
        build_feature_matrix,
        predict_catboost,
        train_catboost,
    )
    from algoritms.data_utils import generate_multi_dataset
    from algoritms.lstm_cpd import nms_predictions

    _all_noise_types = ["white", "pink", "brownian", "violet", "blue"]
    _train_dt_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    _train_D_range = (0.05, 5.0)

    series = np.asarray(series, dtype=float).reshape(-1)

    if weights_path and Path(weights_path).exists():
        model = CatBoostClassifier()
        model.load_model(weights_path)
        print(f"CatBoost: loaded model from {weights_path}", flush=True)
    else:
        print(f"CatBoost: generating {n_train_seq} train seqs x {seq_length}...", flush=True)

        x_train, cp_train = generate_multi_dataset(
            n_train_seq, seq_length, seed=seed + 1,
            noise_types=_all_noise_types,
            D_range=_train_D_range,
            dt_values=_train_dt_values,
        )
        print(f"CatBoost: train data ready ({len(x_train)} pts), generating {n_val_seq} val seqs...", flush=True)
        x_val, cp_val = generate_multi_dataset(
            n_val_seq, seq_length, seed=seed + 2,
            noise_types=_all_noise_types,
            D_range=_train_D_range,
            dt_values=_train_dt_values,
        )
        print(f"CatBoost: val data ready ({len(x_val)} pts), building features...", flush=True)

        X_train, y_train = build_feature_matrix(x_train, cp_train, window_size)
        X_val, y_val = build_feature_matrix(x_val, cp_val, window_size)
        print(
            f"CatBoost: {len(X_train)} train samples "
            f"(pos={y_train.sum()}, neg={(y_train == 0).sum()}), "
            f"{len(X_val)} val samples. Starting training.",
            flush=True,
        )

        model = train_catboost(
            X_train, y_train, X_val, y_val,
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_seed=seed,
        )

        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        model_filename = f"catboost_seq{seq_length}_n{n_train_seq}_seed{seed}.cbm"
        model_path = models_dir / model_filename
        model.save_model(str(model_path))
        print(f"CatBoost: model saved to {model_path}", flush=True)

    scores_raw, _ = predict_catboost(model, series, window_size, threshold)
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
