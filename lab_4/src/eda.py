"""Разведочный анализ Fashion-MNIST: графики и статистика для отчёта."""
import hashlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

CLASS_NAMES = ["Футболка", "Брюки", "Свитер", "Платье", "Пальто",
               "Сандалия", "Рубашка", "Кроссовок", "Сумка", "Ботинок"]
CLASS_NAMES_EN = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                  "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#898781"
INK, INK_2 = "#0b0b0b", "#52514e"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"
BLUES = LinearSegmentedColormap.from_list("blues", ["#f2f6fc", "#a8c8ef", BLUE, "#123e75"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "text.color": INK, "axes.labelcolor": INK_2,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.major.size": 0, "ytick.major.size": 0,
    "lines.linewidth": 2, "legend.frameon": False, "savefig.dpi": 150, "savefig.bbox": "tight",
})


def save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)


def image_grid(rows, row_labels=None, column_labels=None, cell=0.72, label_width=1.1, cmap="gray_r"):
    """Рисует таблицу картинок. rows — список рядов, ряд — список картинок 28×28."""
    n_rows, n_cols = len(rows), len(rows[0])
    left = label_width if row_labels else 0.1
    top = 0.3 if column_labels else 0.05
    width, height = left + n_cols * cell, top + n_rows * cell
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width, height), squeeze=False)
    fig.subplots_adjust(left=left / width, right=1, bottom=0, top=1 - top / height, wspace=0.04, hspace=0.04)
    for r, row in enumerate(rows):
        for c, image in enumerate(row):
            ax = axes[r, c]
            ax.imshow(image, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(GRID)
            if column_labels and r == 0:
                ax.set_title(column_labels[c], fontsize=9, fontweight="normal", loc="center", color=INK_2)
            if row_labels and c == 0:
                ax.set_ylabel(row_labels[r], rotation=0, ha="right", va="center", fontsize=9, color=INK)
    return fig


def save_grid(fig, folder, name):
    """У таблицы картинок свои отступы, поэтому tight_layout здесь не нужен."""
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)


def plot_classes(images, labels, folder):
    rng = np.random.default_rng(0)
    rows = [[images[i] for i in rng.choice(np.flatnonzero(labels == c), 12, replace=False)] for c in range(10)]
    save_grid(image_grid(rows, CLASS_NAMES), folder, "eda_classes")


def plot_means(images, labels, folder):
    """Средняя картинка и разброс пикселей по классам."""
    means = [images[labels == c].mean(0) for c in range(10)]
    stds = [images[labels == c].std(0) for c in range(10)]
    scale = 255 / max(s.max() for s in stds)
    rows = [means, [s * scale for s in stds]]
    fig = image_grid(rows, ["Среднее", "Разброс"], CLASS_NAMES, cell=0.86, label_width=0.8)
    save_grid(fig, folder, "eda_means")


def plot_pixels(images, folder):
    """Распределение яркости всех пикселей. Ось Y логарифмическая: нулей гораздо больше."""
    counts = np.bincount(images.ravel(), minlength=256)
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(np.arange(256), counts, width=1.0, color=BLUE)
    ax.set_yscale("log")
    ax.set_xlim(-2, 257)
    ax.set_xlabel("Яркость пикселя (0 — фон, 255 — максимум)")
    ax.set_ylabel("Число пикселей (лог. шкала)")
    ax.grid(axis="x", visible=False)
    save(fig, folder, "eda_pixels")


def plot_filled(images, labels, folder):
    """Какая доля картинки занята предметом, по классам."""
    filled = (images > 0).mean(axis=(1, 2)) * 100
    order = np.argsort([np.median(filled[labels == c]) for c in range(10)])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.boxplot([filled[labels == c] for c in order], vert=False, widths=0.55, patch_artist=True,
               showfliers=False, whis=(5, 95),
               boxprops={"facecolor": "#a8c8ef", "edgecolor": BLUE},
               medianprops={"color": "#123e75", "linewidth": 2},
               whiskerprops={"color": GRAY}, capprops={"color": GRAY})
    ax.set_yticks(np.arange(1, 11), [CLASS_NAMES[c] for c in order])
    ax.set_xlabel("Доля ненулевых пикселей, % (ящик — 25–75%, усы — 5–95%)")
    ax.grid(axis="y", visible=False)
    save(fig, folder, "eda_filled")


def plot_class_distance(images, labels, folder):
    """Расстояние между средними картинками классов: близкие классы будет сложнее различать."""
    means = np.stack([images[labels == c].mean(0).ravel() / 255 for c in range(10)])
    distance = np.linalg.norm(means[:, None] - means[None], axis=2)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    shown = ax.imshow(distance, cmap=BLUES)
    ax.set_xticks(range(10), CLASS_NAMES, rotation=40, ha="right")
    ax.set_yticks(range(10), CLASS_NAMES)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    middle = distance.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{distance[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if distance[i, j] > middle else INK)
    fig.colorbar(shown, ax=ax, shrink=0.8, label="Расстояние между средними картинками")
    save(fig, folder, "eda_distance")
    pairs = [(distance[i, j], CLASS_NAMES[i], CLASS_NAMES[j]) for i in range(10) for j in range(i + 1, 10)]
    return sorted(pairs)[:3]


def count_duplicates(images):
    hashes = [hashlib.md5(image.tobytes()).hexdigest() for image in images]
    return hashes, len(hashes) - len(set(hashes))


def run_eda(data, folder):
    """Строит графики и возвращает таблицы для отчёта."""
    train, train_y = data["train_images"], data["train_labels"]
    val, val_y = data["val_images"], data["val_labels"]
    test, test_y = data["test_images"], data["test_labels"]
    all_train = np.concatenate([train, val])
    all_train_y = np.concatenate([train_y, val_y])

    plot_classes(all_train, all_train_y, folder)
    plot_means(all_train, all_train_y, folder)
    plot_pixels(all_train, folder)
    plot_filled(all_train, all_train_y, folder)
    closest = plot_class_distance(all_train, all_train_y, folder)

    filled = (all_train > 0).mean(axis=(1, 2)) * 100
    classes = pd.DataFrame({
        "label": range(10),
        "en": CLASS_NAMES_EN,
        "ru": CLASS_NAMES,
        "train": np.bincount(train_y, minlength=10),
        "val": np.bincount(val_y, minlength=10),
        "test": np.bincount(test_y, minlength=10),
        "brightness": [all_train[all_train_y == c].mean() for c in range(10)],
        "filled": [filled[all_train_y == c].mean() for c in range(10)],
    })
    splits = pd.DataFrame([
        {"name": name, "images": len(x), "per_class": len(x) // 10, "mean": x.mean(), "std": x.std(),
         "zeros": (x == 0).mean() * 100}
        for name, x in (("Обучение", train), ("Валидация", val), ("Тест", test))
    ])
    train_hashes, train_dups = count_duplicates(all_train)
    test_hashes, test_dups = count_duplicates(test)
    overlap = len(set(train_hashes) & set(test_hashes))
    return {
        "classes": classes,
        "splits": splits,
        "closest": closest,
        "train_duplicates": train_dups,
        "test_duplicates": test_dups,
        "overlap": overlap,
        "zero_share": (all_train == 0).mean() * 100,
        "max_share": (all_train == 255).mean() * 100,
        "mean": all_train.mean(),
    }
