"""Обучает диффузионную модель: сеть учится угадывать шум, добавленный к картинке."""
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from diffusion import T, UNet, add_noise, sample

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EMA_DECAY = 0.999    # насколько медленно обновляются усреднённые веса
BASE_CHANNELS = 32
SEED = 0
PROGRESS_EPOCHS = (1, 2, 5, 10, 15)  # после этих эпох сохраняются примеры генерации


def load_split(name):
    # Нормализация данных датасета пиксели переводятся из (0..255) в (-1..1)
    data = np.load(ROOT / "data" / "fashion.npz")
    images = torch.tensor(data[f"{name}_images"], dtype=torch.float32)[:, None] / 127.5 - 1
    labels = torch.tensor(data[f"{name}_labels"].astype(np.int64))
    return images, labels


def noise_loss(model, x0, labels, generator=None):
    """Один проход: случайный шаг → зашумление → ошибка в оценке шума (MSE)."""
    # выбираем уровень шума
    t = torch.randint(0, T, (len(x0),), generator=generator)
    # создаем шум из нормального распределения mean - 0, std - 1
    noise = torch.randn(x0.shape, generator=generator)
    # добавляем шум
    noisy = add_noise(x0, t, noise)
    # возвращаем модели для обучения
    return ((model(noisy, t, labels) - noise) ** 2).mean()


@torch.no_grad()
def validation_loss(model, images, labels):
    """Ошибка на валидации. Сид постоянный, поэтому шум одинаков во все эпохи."""
    model.eval()
    generator = torch.Generator().manual_seed(SEED)
    total = 0.0
    for start in range(0, len(images), 500):
        x0, y = images[start:start + 500], labels[start:start + 500]
        total += noise_loss(model, x0, y, generator).item() * len(x0)
    return total / len(images)


def main():
    OUTPUTS.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    train_images, train_labels = load_split("train")
    val_images, val_labels = load_split("val")

    model = UNet(BASE_CHANNELS)
    ema = copy.deepcopy(model)  # усреднённая копия весов: для валидации и генерации
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    total_steps = EPOCHS * (len(train_images) // BATCH_SIZE) # общее количество шагов обучения U-Net
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, total_steps)

    history, progress, start_time = [], {}, time.time()
    progress_labels = torch.arange(10).repeat_interleave(2) # для анализа прогресса
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = torch.randperm(len(train_images))
        losses = []
        for start in range(0, len(order) - BATCH_SIZE + 1, BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            loss = noise_loss(model, train_images[batch], train_labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for avg, current in zip(ema.parameters(), model.parameters()):
                    avg.lerp_(current, 1 - EMA_DECAY)
            losses.append(loss.item())

        val_loss = validation_loss(ema, val_images, val_labels)
        history.append({"epoch": epoch, "train_loss": np.mean(losses), "val_loss": val_loss,
                        "minutes": (time.time() - start_time) / 60})
        print(f"эпоха {epoch}/{EPOCHS}: обучение {np.mean(losses):.4f}, "
              f"валидация {val_loss:.4f} ({history[-1]['minutes']:.1f} мин)", flush=True)
        if epoch in PROGRESS_EPOCHS:
            progress[f"epoch_{epoch}"] = sample(ema, progress_labels, steps=50, seed=SEED).numpy()

    pd.DataFrame(history).to_csv(OUTPUTS / "history.csv", index=False)
    np.savez_compressed(OUTPUTS / "progress.npz", **progress)
    # Модель сохраняется последней: наличие model.pt значит, что обучение завершено.
    torch.save({"weights": ema.state_dict(), "base": BASE_CHANNELS}, OUTPUTS / "model.pt")


if __name__ == "__main__":
    main()
