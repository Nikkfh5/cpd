"""
Transformer-модель для обнаружения точек разладки (change point detection).

Архитектура: window-based бинарный классификатор на основе TransformerEncoder.
  - Вход: окно (batch, W, 1) — нормированное скользящее окно
  - Проекция: Linear(1 → d_model)
  - Позиционное кодирование: синусоидальное (Vaswani et al. 2017)
  - TransformerEncoder: num_layers слоёв с multi-head attention
  - Pooling: среднее по временной оси (mean pooling)
  - Выход: Linear(d_model → 1), логит для BCEWithLogitsLoss

Вспомогательные функции (SlidingWindowDataset, train_model, evaluate_batch,
predict_on_series, nms_predictions) импортируются из lstm_cpd без копирования.
"""

import math

import torch
import torch.nn as nn

from algoritms.lstm_cpd import (  # noqa: F401  (реэкспорт для удобства)
    SlidingWindowDataset,
    evaluate_batch,
    nms_predictions,
    predict_on_series,
    train_model,
)


class PositionalEncoding(nn.Module):
    """
    Синусоидальное позиционное кодирование (Vaswani et al. 2017).

    Добавляет к входному тензору фиксированные позиционные признаки:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    :param d_model: размерность модели
    :param dropout: вероятность dropout после добавления кодирования
    :param max_len: максимальная длина последовательности
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerChangePointDetector(nn.Module):
    """
    Transformer бинарный классификатор для обнаружения точек разладки.

    Архитектура:
        Linear(input_size → d_model)
        → PositionalEncoding(d_model, dropout)
        → TransformerEncoder(num_layers × TransformerEncoderLayer)
        → mean pooling по временной оси
        → Linear(d_model → 1)

    Параметры по умолчанию соответствуют конфигурации «очень большой»:
        d_model=256, nhead=8, num_layers=6, dim_feedforward=512 → ~3.2M параметров.

    :param input_size: размерность входного признака (1 для одномерного ряда)
    :param d_model: размерность пространства представлений
    :param nhead: число голов multi-head attention (d_model должно делиться на nhead)
    :param num_layers: число слоёв TransformerEncoderLayer
    :param dim_feedforward: размерность скрытого слоя FFN внутри энкодера
    :param dropout: вероятность dropout в attention и FFN
    """

    def __init__(
        self,
        input_size: int = 1,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.fc(x).squeeze(-1)
