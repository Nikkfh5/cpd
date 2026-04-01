# Сравнительный анализ методов обнаружения точек разладки в стохастических дифференциальных уравнениях

Курсовая работа посвящена сравнению классических статистических и ML-методов обнаружения точек разладки (Change Point Detection, CPD) на синтетических траекториях стохастического дифференциального уравнения вида:

```
dx = sin(x) * dt + sqrt(D) * dW
```

Траектории генерируются с различными параметрами шума (`D`), шага дискретизации (`dt`) и типами шума (`white`, `colored`). Точки разладки определяются как моменты перехода решения между впадинами потенциала (пересечение границ, кратных pi).

## Установка

```bash
pip install -r requirements.txt
```

Основные зависимости: `numpy`, `pandas`, `torch`, `scipy`, `scikit-learn`, `catboost`, `pyhomogeneity`, `matplotlib`.

## Запуск

Полная оценка всех методов на 50 синтетических рядах:

```bash
python eval_multi.py --n-series 50
```

Запуск конкретных методов:

```bash
python eval_multi.py --methods LSTM_v2,Transformer_v2
```

Запуск через единый конфиг:

```bash
python run_all.py --config configs/base.yaml
```

Анализ статистики переходов:

```bash
python analyze_transitions.py
```

Построение графиков:

```bash
python plot_results.py
```

Быстрый smoke-тест:

```bash
python smoke_test.py
```

Калибровка порогов ML-моделей:

```bash
python calibrate_threshold.py --n-cal 20
```

## Структура проекта

```
cursovaya/
    algoritms/                  -- реализации методов CPD
        data_utils.py           -- генерация синтетических данных (SDE)
        evaluation.py           -- метрики (F1, MAE, ROC-AUC, PR-AUC)
        Pettitt.py              -- критерий Петтитта
        SNHT.py                 -- Standard Normal Homogeneity Test
        Buishand's_tests.py     -- тесты Буишанда (Q, Range, LR, U)
        chow.py                 -- критерий Чоу
        CUMSUM.py               -- CUSUM (ядро)
        cusum_adapter.py        -- адаптер CUSUM
        MDL.py                  -- Minimum Description Length
        SWAB.py                 -- Sliding Window and Bottom-up
        GP.py                   -- Gaussian Process
        Nyblom_test_GOBACK.py   -- критерий Найблома
        lstm_cpd.py             -- LSTM-детектор
        lstm_adapter.py         -- адаптер LSTM / LSTM_v2
        gru_cpd.py              -- GRU-детектор
        gru_adapter.py          -- адаптер GRU / GRU_v2
        transformer_cpd.py      -- Transformer-детектор
        transformer_adapter.py  -- адаптер Transformer_v2
        catboost_cpd.py         -- CatBoost-детектор
        catboost_adapter.py     -- адаптер CatBoost
    configs/
        base.yaml               -- конфигурация параметров генерации и методов
    models/                     -- обученные веса (.pt, .cbm)
    notebooks/
        ploting_graphs (1).ipynb    -- научный источник генерации данных
        kaggle_eval_multi.ipynb     -- Kaggle-ноутбук для воспроизведения
    results/
        eval_multi/summary.csv  -- сводная таблица метрик
    figures/                    -- графики (PDF)
    docs/
        document.tex            -- исходник отчёта
    run_all.py                  -- единый запуск через конфиг
    eval_multi.py               -- многорядовая оценка методов
    calibrate_threshold.py      -- калибровка порогов ML-моделей
    analyze_transitions.py      -- анализ статистики переходов SDE
    plot_results.py             -- построение сравнительных графиков
    smoke_test.py               -- быстрая проверка работоспособности
    data_source_policy.py       -- политика источников данных
```

## Реализованные методы

### Классические статистические (12 методов)

| Метод | Описание |
|-------|----------|
| Pettitt | Непараметрический критерий Петтитта |
| SNHT | Standard Normal Homogeneity Test |
| BuishandQ | Q-тест Буишанда |
| BuishandRange | Range-тест Буишанда |
| BuishandLR | Likelihood Ratio тест Буишанда |
| BuishandU | U-тест Буишанда |
| Chow | Критерий Чоу (структурный разрыв) |
| MDL | Minimum Description Length |
| SWAB | Sliding Window and Bottom-up |
| GP | Gaussian Process |
| Nyblom | Критерий Найблома |
| CUSUM | Cumulative Sum Control Chart |

### ML-методы (6 методов)

| Метод | Описание |
|-------|----------|
| LSTM | LSTM-сеть (скользящее окно) |
| LSTM_v2 | LSTM, обученная на реалистичных D и dt |
| GRU | GRU-сеть |
| GRU_v2 | GRU, обученная на реалистичных D и dt |
| CatBoost | Градиентный бустинг на признаках окна |
| Transformer_v2 | Transformer (d_model=256, 8 голов, 6 слоёв) |

Версии `_v2` обучены на диапазоне `D=[0.2, 1.0]`, `dt=[0.5, 1.0, 2.0]`, что соответствует реалистичным условиям тестирования и устраняет разрыв между обучающей и тестовой статистикой CP.

## Результаты

Сводка по 50 синтетическим рядам при D=1.0, white noise (рабочий режим):

| Метод | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|-----------|--------|----|---------|--------|
| **Transformer_v2** | **0.77** | **0.64** | **0.698** | **0.815** | **0.799** |
| **LSTM_v2** | **0.84** | **0.57** | **0.675** | **0.813** | **0.811** |
| **GRU_v2** | **0.91** | **0.51** | **0.650** | **0.847** | **0.850** |
| CUSUM | 0.46 | 0.65 | 0.534 | 0.483 | 0.467 |
| CatBoost | 0.92 | 0.33 | 0.479 | 0.733 | 0.788 |

При малом шуме (D=0.5, ~6 CP на ряд) лидируют классические тесты SNHT (F1=0.725) и Chow (F1=0.645). При рабочем уровне шума (D=1.0, ~200 CP на ряд) ML-методы v2 значительно превосходят классику.

## Лицензия

Учебный проект. НИУ ВШЭ, 2025--2026.
