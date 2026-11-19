"""Скачивает датасет Bike Sharing, проверяет его и готовит признаки.

Запускается один раз при сборке Docker-образа.
Результат: data/bike_days.csv (данные с признаками и колонкой split).
"""
import hashlib
import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
SHA256 = "b70182d0d0508e9abbb79306ce5c0cec34869000f8220175ac83d11dbe845401"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Разбиение по времени: модель не должна видеть будущее.
TRAIN_END = "2012-03-31"
VAL_END = "2012-06-30"


def download() -> pd.DataFrame:
    """Скачивает архив, сверяет контрольную сумму и читает day.csv."""
    print(f"Скачиваю {URL}")
    with urllib.request.urlopen(URL, timeout=120) as response:
        archive = response.read()
    if hashlib.sha256(archive).hexdigest() != SHA256:
        raise ValueError("Контрольная сумма архива не совпала: файл изменился")
    with zipfile.ZipFile(io.BytesIO(archive)) as zf:
        with zf.open("day.csv") as f:
            return pd.read_csv(f, parse_dates=["dteday"])


def check(df: pd.DataFrame) -> None:
    """Останавливает сборку, если данные не соответствуют ожиданиям."""
    if df.isna().any().any():
        raise ValueError("В данных есть пропуски")
    if df["dteday"].duplicated().any():
        raise ValueError("В данных есть повторяющиеся даты")
    all_days = pd.date_range(df["dteday"].min(), df["dteday"].max())
    if len(all_days) != len(df):
        raise ValueError("В календаре есть пропущенные дни")
    if not (df["casual"] + df["registered"] == df["cnt"]).all():
        raise ValueError("cnt не равен casual + registered")


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признаки, известные агенту утром, до начала аренды."""
    df = df.sort_values("dteday").reset_index(drop=True)

    # Спрос прошлых дней. shift(1) важен: спрос текущего дня агент не знает.
    df["demand_lag1"] = df["cnt"].shift(1)
    df["demand_avg7"] = df["cnt"].shift(1).rolling(7).mean()

    # Время года как точка на окружности: декабрь оказывается рядом с январём.
    day_of_year = df["dteday"].dt.dayofyear
    df["year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Погода как отдельные флаги: 1 — ясно, 2 — облачно, 3 — дождь или снег.
    for code in (1, 2, 3):
        df[f"weather_{code}"] = (df["weathersit"] == code).astype(int)

    # Первые 7 дней не имеют полной истории спроса.
    df = df.dropna(subset=["demand_avg7"]).reset_index(drop=True)

    df["split"] = np.select(
        [df["dteday"] <= TRAIN_END, df["dteday"] <= VAL_END],
        ["train", "val"],
        default="test",
    )
    return df


def main() -> None:
    raw = download()
    check(raw)
    df = add_features(raw)

    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(DATA_DIR / "bike_days.csv", index=False)

    print(f"Строк в исходных данных: {len(raw)}, после обработки: {len(df)}")
    print(df.groupby("split")["dteday"].agg(["min", "max", "count"]))


if __name__ == "__main__":
    main()
