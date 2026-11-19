"""Скачивает Fashion-MNIST, проверяет файлы и сохраняет data/fashion.npz.

Запускается один раз при сборке Docker-образа.
"""
import gzip
import hashlib
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

# Контрольные суммы опубликованы авторами датасета.
FILES = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
VAL_SIZE = 5000  # столько картинок обучающей части уходит в валидацию
SEED = 0


def download(name: str) -> bytes:
    """Скачивает файл, сверяет контрольную сумму и возвращает распакованные байты."""
    raw = urllib.request.urlopen(BASE_URL + name, timeout=120).read()
    if hashlib.md5(raw).hexdigest() != FILES[name]:
        raise ValueError(f"Контрольная сумма файла {name} не совпала")
    return gzip.decompress(raw)


def read_images(name: str) -> np.ndarray:
    """Формат IDX: 16 байт заголовка, затем пиксели по одному байту."""
    return np.frombuffer(download(name), dtype=np.uint8, offset=16).reshape(-1, 28, 28)


def read_labels(name: str) -> np.ndarray:
    """Формат IDX: 8 байт заголовка, затем метки по одному байту."""
    return np.frombuffer(download(name), dtype=np.uint8, offset=8)


def stratified_split(labels: np.ndarray):
    """Делит индексы на обучение и валидацию с равной долей каждого класса."""
    rng = np.random.default_rng(SEED)
    train_idx, val_idx = [], []
    for label in range(10):
        idx = rng.permutation(np.flatnonzero(labels == label))
        n_val = VAL_SIZE // 10
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return np.concatenate(train_idx), np.concatenate(val_idx)


def main():
    images = read_images("train-images-idx3-ubyte.gz")
    labels = read_labels("train-labels-idx1-ubyte.gz")
    test_images = read_images("t10k-images-idx3-ubyte.gz")
    test_labels = read_labels("t10k-labels-idx1-ubyte.gz")

    assert images.shape == (60000, 28, 28) and labels.shape == (60000,)
    assert test_images.shape == (10000, 28, 28) and test_labels.shape == (10000,)
    assert set(labels.tolist()) == set(range(10)) == set(test_labels.tolist())

    train_idx, val_idx = stratified_split(labels)
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(
        DATA / "fashion.npz",
        train_images=images[train_idx], train_labels=labels[train_idx],
        val_images=images[val_idx], val_labels=labels[val_idx],
        test_images=test_images, test_labels=test_labels,
    )
    print(f"Готово: обучение {len(train_idx)}, валидация {len(val_idx)}, тест {len(test_labels)}")


if __name__ == "__main__":
    main()
