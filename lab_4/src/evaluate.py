"""Проверяет модель: сравнивает число шагов, базовую модель и реальные данные.

Что считается:
1. Опыт с числом шагов на валидации: качество и время при 5…1000 шагах.
2. Итог на тесте при выбранном числе шагов: модель, базовая модель, реальные картинки.
3. Проверка, что модель не копирует обучающие картинки.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import torch

import classifier as clf
from diffusion import T, UNet, add_noise, sample
from train import OUTPUTS, SEED, load_split

STEP_OPTIONS = [5, 10, 20, 50, 100, 1000]  # отчёт сравнивает варианты 20, 50, 100 и выбранный
STEPS_IMAGES = 1000   # картинок на каждый вариант числа шагов
FINAL_IMAGES = 5000   # картинок в итоговом сравнении (столько же в валидации)
ALLOWED_WORSE = 1.10  # берём наименьшее число шагов, где FD не более чем на 10% хуже лучшего
TSTR_EPOCHS = 10
TSTR_SEEDS = (1, 2, 3)


def load_model():
    checkpoint = torch.load(OUTPUTS / "model.pt", weights_only=True)
    model = UNet(checkpoint["base"])
    model.load_state_dict(checkpoint["weights"])
    return model.eval()


def generate(model, count, steps, seed=SEED, batch_size=500):
    """Генерирует count картинок, поровну от каждого класса."""
    labels = torch.arange(count) % 10
    parts = [sample(model, labels[i:i + batch_size], steps, seed=seed + i).clamp(-1, 1)
             for i in range(0, count, batch_size)]
    return torch.cat(parts), labels


def gaussian_baseline(images, labels, count, seed=SEED):
    """Базовая модель: многомерное нормальное распределение для каждого класса."""
    rng = np.random.default_rng(seed)
    flat = images.flatten(1).numpy().astype(np.float64)
    per_class = count // 10
    result = []
    for label in range(10):
        part = flat[labels.numpy() == label]
        drawn = rng.multivariate_normal(part.mean(0), np.cov(part, rowvar=False), per_class,
                                        method="svd", check_valid="ignore")
        result.append(drawn)
    out = np.concatenate(result).clip(-1, 1).reshape(-1, 1, 28, 28)
    return torch.tensor(out, dtype=torch.float32), torch.arange(10).repeat_interleave(per_class)


def frechet_distance(a, b):
    # Расстояние Фреше между двумя наборами векторов. Чем меньше, тем наборы более похожи.
    mean_a, mean_b = a.mean(0), b.mean(0)
    cov_a, cov_b = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    root = scipy.linalg.sqrtm(cov_a @ cov_b).real
    return float(((mean_a - mean_b) ** 2).sum() + np.trace(cov_a + cov_b - 2 * root))


def tstr(images, labels, test):
    """Train on Synthetic, Test on Real: учимся на этих картинках, проверяем на реальном тесте."""
    scores = [clf.accuracy(clf.fit(images, labels, TSTR_EPOCHS, seed), *test) for seed in TSTR_SEEDS]
    return float(np.mean(scores)), float(np.std(scores))


def steps_experiment(model, judge, val_features, train):
    """Качество и время при разном числе шагов. Сравнение идёт с валидацией, не с тестом."""
    rows = []
    for steps in STEP_OPTIONS:
        start = time.time()
        images, labels = generate(model, STEPS_IMAGES, steps)
        seconds = time.time() - start
        rows.append({
            "steps": steps,
            "method": "DDPM" if steps == T else "DDIM",
            "fd": frechet_distance(clf.features(judge, images), val_features),
            "accuracy": clf.accuracy(judge, images, labels),
            "seconds_per_100": seconds / STEPS_IMAGES * 100,
        })
        print(f"шагов {steps}: FD {rows[-1]['fd']:.1f}, точность {rows[-1]['accuracy']:.3f}, "
              f"{seconds:.0f} с", flush=True)
    table = pd.DataFrame(rows)
    best = table["fd"].min()
    chosen = int(table[table["fd"] <= best * ALLOWED_WORSE]["steps"].min())

    # Ориентир: столько же настоящих картинок из обучающей части (поровну от класса).
    rng = np.random.default_rng(SEED)
    picked = np.concatenate([rng.choice(np.flatnonzero(train[1].numpy() == c), STEPS_IMAGES // 10, replace=False)
                             for c in range(10)])
    real_fd = frechet_distance(clf.features(judge, train[0][picked]), val_features)
    return table, chosen, real_fd


def final_comparison(judge, sets, train, test):
    """Три источника картинок на реальном тесте: одинаковое число и одинаковые метрики."""
    test_features = clf.features(judge, test[0])
    rows = []
    for name, (images, labels) in sets.items():
        mean, std = tstr(images, labels, test)
        rows.append({
            "source": name,
            "fd": frechet_distance(clf.features(judge, images), test_features),
            "accuracy": clf.accuracy(judge, images, labels),
            "tstr": mean,
            "tstr_std": std,
        })
        print(rows[-1], flush=True)
    return pd.DataFrame(rows)


def per_class(judge, images, labels, test):
    """Матрица ошибок и точность по классам: на реальном тесте и на сгенерированных картинках."""
    predictions = clf.predict(judge, images)
    confusion = np.zeros((10, 10), dtype=int)
    for true, pred in zip(labels.tolist(), predictions.tolist()):
        confusion[true, pred] += 1
    test_predictions = clf.predict(judge, test[0])
    real = [(test_predictions[test[1] == c] == c).float().mean().item() for c in range(10)]
    generated = [(predictions[labels == c] == c).float().mean().item() for c in range(10)]
    return confusion, pd.DataFrame({"label": range(10), "real": real, "generated": generated})


def nearest_neighbours(images, train_images, batch_size=200):
    """Для каждой картинки ищет ближайшую в обучающей части (пиксели в шкале 0…1)."""
    flat_train = (train_images.flatten(1) + 1) / 2
    flat = (images.flatten(1) + 1) / 2
    distances, indices = [], []
    for start in range(0, len(flat), batch_size):
        d = torch.cdist(flat[start:start + batch_size], flat_train)
        value, index = d.min(1)
        distances.append(value)
        indices.append(index)
    return torch.cat(distances).numpy(), torch.cat(indices).numpy()


@torch.no_grad()
def loss_by_time(model, images, labels, bins=10, repeats=3):
    """Как ошибка оценки шума зависит от шага t: разбиваем шаги на равные интервалы."""
    generator = torch.Generator().manual_seed(SEED)
    rows = []
    for b in range(bins):
        low, high = b * T // bins, (b + 1) * T // bins
        total, count = 0.0, 0
        for _ in range(repeats):
            for start in range(0, len(images), 500):
                x0, y = images[start:start + 500], labels[start:start + 500]
                t = torch.randint(low, high, (len(x0),), generator=generator)
                noise = torch.randn(x0.shape, generator=generator)
                total += ((model(add_noise(x0, t, noise), t, y) - noise) ** 2).mean().item() * len(x0)
                count += len(x0)
        rows.append({"t_low": low, "t_high": high - 1, "mse": total / count})
    return pd.DataFrame(rows)


def to_uint8(images):
    return ((images.clamp(-1, 1) + 1) * 127.5).round().numpy().astype(np.uint8)[:, 0]


def main():
    torch.manual_seed(SEED)
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    model = load_model()

    print("Обучение классификатора-судьи…", flush=True)
    judge = clf.fit(*train, epochs=5, seed=SEED, val=val)
    judge_test = clf.accuracy(judge, *test)
    print(f"Точность судьи на реальном тесте: {judge_test:.3f}", flush=True)

    print("Опыт с числом шагов…", flush=True)
    val_features = clf.features(judge, val[0])
    steps_table, chosen, real_fd = steps_experiment(model, judge, val_features, train)
    steps_table.to_csv(OUTPUTS / "steps.csv", index=False)
    print(f"Выбрано шагов: {chosen}", flush=True)

    print("Итоговое сравнение на тесте…", flush=True)
    generated = generate(model, FINAL_IMAGES, chosen)
    baseline = gaussian_baseline(*train, FINAL_IMAGES)
    sets = {"Диффузионная модель": generated, "Базовая модель (нормальное распределение)": baseline,
            "Реальные картинки": val}
    comparison = final_comparison(judge, sets, train, test)
    comparison.to_csv(OUTPUTS / "comparison.csv", index=False)

    confusion, classes = per_class(judge, *generated, test)
    pd.DataFrame(confusion).to_csv(OUTPUTS / "confusion.csv", index=False)
    classes.to_csv(OUTPUTS / "per_class.csv", index=False)

    print("Проверка на копирование и ошибка по шагам…", flush=True)
    # По 1000 картинок поровну от класса. Метки сгенерированных идут по кругу 0…9,
    # а в базовой модели и валидации классы идут подряд, поэтому берём через равный шаг.
    first_1000 = generated[0][:1000]
    evenly = torch.arange(1000) * (FINAL_IMAGES // 1000)
    generated_dist, generated_nn = nearest_neighbours(first_1000, train[0])
    val_dist, _ = nearest_neighbours(val[0][torch.arange(1000) * (len(val[0]) // 1000)], train[0])
    baseline_dist, _ = nearest_neighbours(baseline[0][evenly], train[0])
    loss_by_time(model, *val).to_csv(OUTPUTS / "loss_by_t.csv", index=False)

    print("Картинки для отчёта…", flush=True)
    labels = torch.arange(10)
    _, history = sample(model, labels, steps=T, seed=SEED, keep_every=10)
    strips = {f"steps_{s}": to_uint8(sample(model, labels, s, seed=SEED).clamp(-1, 1)) for s in STEP_OPTIONS}
    np.savez_compressed(
        OUTPUTS / "samples.npz",
        generated=to_uint8(generated[0]), generated_labels=generated[1].numpy(),
        baseline=to_uint8(baseline[0][evenly]), baseline_labels=baseline[1][evenly].numpy(),
        chain=np.stack([to_uint8(h) for h in history]),
        nn_generated=to_uint8(first_1000), nn_train=to_uint8(train[0][generated_nn]),
        generated_dist=generated_dist, val_dist=val_dist, baseline_dist=baseline_dist,
        **strips,
    )

    summary = {
        "chosen_steps": chosen,
        "real_fd_steps": real_fd,
        "judge_test_accuracy": judge_test,
        "final_images": FINAL_IMAGES,
        "steps_images": STEPS_IMAGES,
        "allowed_worse": ALLOWED_WORSE,
        "tstr_epochs": TSTR_EPOCHS,
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (OUTPUTS / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
