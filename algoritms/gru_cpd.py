"""
GRU-модель для обнаружения точек разладки (change point detection) во временных рядах.

Архитектура идентична LSTM, но использует GRU вместо LSTM.
Вспомогательные функции (SlidingWindowDataset, train_model, evaluate_batch,
predict_on_series, nms_predictions) импортируются из lstm_cpd без копирования.
"""

import torch.nn as nn

from algoritms.lstm_cpd import (  # noqa: F401  (реэкспорт для удобства)
    SlidingWindowDataset,
    evaluate_batch,
    nms_predictions,
    predict_on_series,
    train_model,
)


class GRUChangePointDetector(nn.Module):
    """
    GRU бинарный классификатор для обнаружения точек разладки.

    Архитектура:
        GRU(input_size=1, hidden_size, num_layers, dropout) → Linear → Sigmoid
    """

    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        logit = self.fc(out[:, -1, :]).squeeze(-1)
        return logit
