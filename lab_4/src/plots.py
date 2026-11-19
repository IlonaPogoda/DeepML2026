"""Графики и картинки для отчёта: метод, обучение, результаты."""
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from diffusion import T, add_noise, alpha_bar
from eda import BLUE, BLUES, CLASS_NAMES, GRAY, INK, ORANGE, image_grid, save, save_grid

MUTED = "#c3c2b7"
FORWARD_STEPS = [0, 50, 100, 200, 300, 500, 700, 999]
REVERSE_FRAMES = [0, 30, 50, 65, 75, 85, 90, 95, 100]  # номера кадров: один кадр — 10 шагов


def to_gray(tensor):
    return ((tensor.clamp(-1, 1) + 1) * 127.5).numpy()


def plot_forward(images, labels, folder):
    """Прямой процесс: одна и та же картинка на разных шагах t."""
    torch.manual_seed(0)
    rows = []
    for c in (0, 7, 8, 9):
        x0 = images[labels == c][:1]
        noise = torch.randn_like(x0)
        rows.append([to_gray(add_noise(x0, torch.tensor([t]), noise))[0, 0] for t in FORWARD_STEPS])
    fig = image_grid(rows, [CLASS_NAMES[c] for c in (0, 7, 8, 9)], [f"t = {t}" for t in FORWARD_STEPS], cell=0.95)
    save_grid(fig, folder, "forward")


def plot_schedule(folder):
    t = np.arange(T)
    signal = alpha_bar.sqrt().numpy()
    noise = (1 - alpha_bar).sqrt().numpy()
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(t, signal, color=BLUE, label="Коэффициент картинки √ᾱ")
    ax.plot(t, noise, color=ORANGE, label="Коэффициент шума √(1−ᾱ)")
    ax.legend(loc="center right")
    ax.set_xlabel("Шаг t")
    ax.set_ylabel("Значение коэффициента")
    ax.set_xlim(0, T - 1)
    ax.set_ylim(0, 1.02)
    save(fig, folder, "schedule")


def plot_learning(history, folder):
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(history["epoch"], history["train_loss"], color=GRAY, label="Обучение", marker="o", markersize=5)
    ax.plot(history["epoch"], history["val_loss"], color=BLUE, label="Валидация (усреднённые веса)",
            marker="o", markersize=5)
    ax.legend()
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("MSE между шумом и его оценкой")
    ax.set_xticks(history["epoch"])
    save(fig, folder, "learning")


def plot_loss_by_t(table, folder):
    middle = (table["t_low"] + table["t_high"]) / 2
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.bar(middle, table["mse"], width=(table["t_high"] - table["t_low"]) * 0.85, color=BLUE)
    ax.set_xlabel("Шаг t (интервалы по 100 шагов)")
    ax.set_ylabel("MSE на валидации")
    ax.grid(axis="x", visible=False)
    save(fig, folder, "loss_by_t")


def plot_progress(progress, epochs, folder):
    """Примеры генерации после разных эпох: один и тот же начальный шум."""
    rows = [list(to_gray(torch.tensor(progress[f"epoch_{e}"][::2]))[:, 0]) for e in epochs]
    fig = image_grid(rows, [f"Эпоха {e}" for e in epochs], CLASS_NAMES, cell=0.86, label_width=0.8)
    save_grid(fig, folder, "progress")


def class_rows(images, labels, per_row=10):
    return [list(images[labels == c][:per_row]) for c in range(10)]


def plot_generated(samples, data, folder):
    rows = class_rows(samples["generated"], samples["generated_labels"], 12)
    save_grid(image_grid(rows, CLASS_NAMES), folder, "generated")
    # Реальные и сгенерированные одинаковым форматом, чтобы сравнивать рядом.
    fake = class_rows(samples["generated"], samples["generated_labels"], 6)
    real = class_rows(data["test_images"], data["test_labels"], 6)
    save_grid(image_grid(real, CLASS_NAMES, cell=0.62, label_width=0.95), folder, "compare_real")
    save_grid(image_grid(fake, CLASS_NAMES, cell=0.62, label_width=0.95), folder, "compare_fake")
    base = class_rows(samples["baseline"], samples["baseline_labels"], 6)
    save_grid(image_grid(base, CLASS_NAMES, cell=0.62, label_width=0.95), folder, "compare_base")


def plot_reverse(chain, folder):
    """Обратный процесс: от шума к картинке, столбцы — номера шагов."""
    last = len(chain) - 1
    titles = ["t = 999" if f == 0 else ("готово" if f == last else f"t = {T - 1 - f * 10}") for f in REVERSE_FRAMES]
    rows = [[chain[f, c] for f in REVERSE_FRAMES] for c in range(10)]
    save_grid(image_grid(rows, CLASS_NAMES, titles, cell=0.72), folder, "reverse")


def save_gif(chain, folder):
    """Анимация генерации: 10 картинок превращаются из шума в одежду."""
    scale, bar = 4, 6
    frames = []
    for index, frame in enumerate(chain):
        top = np.concatenate(frame[:5], axis=1)
        bottom = np.concatenate(frame[5:], axis=1)
        tile = 255 - np.concatenate([top, bottom], axis=0)
        tile = np.kron(tile, np.ones((scale, scale), dtype=np.uint8))
        progress = np.full((bar, tile.shape[1]), 225, dtype=np.uint8)
        progress[:, :int(tile.shape[1] * index / (len(chain) - 1))] = 40
        frames.append(Image.fromarray(np.concatenate([tile, progress], axis=0), mode="L"))
    durations = [70] * (len(frames) - 1) + [2500]
    frames[0].save(folder / "reverse.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)


def plot_steps_strip(samples, steps_list, folder):
    rows = [list(samples[f"steps_{s}"]) for s in steps_list]
    labels = [f"{s} шагов DDPM" if s == T else f"{s} шагов DDIM" for s in steps_list]
    fig = image_grid(rows, labels, CLASS_NAMES, cell=0.86, label_width=1.25)
    save_grid(fig, folder, "steps_strip")


def plot_steps_metrics(table, chosen, real_fd, folder):
    """Три отдельных графика с общей осью X: у метрик разные шкалы."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.1), sharex=True)
    specs = [("fd", "FD: чем меньше, тем лучше"),
             ("accuracy", "Точность судьи"),
             ("seconds_per_100", "Секунд на 100 картинок")]
    for ax, (column, title) in zip(axes, specs):
        values = table[column] * (100 if column == "accuracy" else 1)
        ax.plot(table["steps"], values, color=BLUE, marker="o", markersize=6)
        picked = table["steps"] == chosen
        ax.plot(table["steps"][picked], values[picked], color=ORANGE, marker="o", markersize=10, linestyle="none",
                label=f"Выбрано: {chosen}")
        ax.set_xscale("log")
        ax.set_xticks(table["steps"], table["steps"].astype(str))
        ax.minorticks_off()
        ax.set_title(title)
        ax.set_xlabel("Число шагов (лог. шкала)")
    axes[0].axhline(real_fd, color=GRAY, linewidth=1, label=f"Реальные картинки: {real_fd:.1f}")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("%")
    axes[2].set_yscale("log")
    save(fig, folder, "steps_metrics")


def plot_comparison(table, folder):
    """Три источника картинок и три метрики: цветом выделена наша модель."""
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.5), sharey=True)
    names = ["Реальные картинки", "Диффузионная модель", "Базовая модель"]
    order = [table[table["source"].str.startswith(n)].iloc[0] for n in names]
    colors = [MUTED, BLUE, MUTED]
    specs = [("fd", "FD (меньше — лучше)", "{:.1f}", 1),
             ("accuracy", "Точность судьи, %", "{:.1f}", 100),
             ("tstr", "Точность TSTR на тесте, %", "{:.1f}", 100)]
    positions = np.arange(3)[::-1]
    for ax, (column, title, fmt, scale) in zip(axes, specs):
        values = [row[column] * scale for row in order]
        ax.barh(positions, values, color=colors, height=0.55)
        for y, value in zip(positions, values):
            ax.text(value + max(values) * 0.02, y, fmt.format(value), va="center", fontsize=9, color=INK)
        ax.set_xlim(0, max(values) * 1.2)
        ax.set_title(title)
        ax.grid(False)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    axes[0].set_yticks(positions, names)
    save(fig, folder, "comparison")


def plot_confusion(confusion, folder):
    share = confusion / confusion.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    shown = ax.imshow(share, cmap=BLUES, vmin=0, vmax=100)
    ax.set_xticks(range(10), CLASS_NAMES, rotation=40, ha="right")
    ax.set_yticks(range(10), CLASS_NAMES)
    ax.set_xlabel("Что увидел судья")
    ax.set_ylabel("Что просили сгенерировать")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(10):
        for j in range(10):
            if share[i, j] >= 0.5:
                ax.text(j, i, f"{share[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if share[i, j] > 45 else INK)
    fig.colorbar(shown, ax=ax, shrink=0.8, label="% картинок класса")
    save(fig, folder, "confusion")


def plot_per_class(table, folder):
    """Точность судьи по классам: реальный тест и сгенерированные картинки."""
    ordered = table.sort_values("generated")
    y = np.arange(10)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.hlines(y, ordered["real"] * 100, ordered["generated"] * 100, color=MUTED, linewidth=2)
    ax.plot(ordered["real"] * 100, y, "o", color=GRAY, markersize=8, label="Реальный тест")
    ax.plot(ordered["generated"] * 100, y, "o", color=BLUE, markersize=8, label="Сгенерированные")
    ax.set_yticks(y, [CLASS_NAMES[c] for c in ordered["label"]])
    ax.set_xlabel("Точность судьи, %")
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, folder, "per_class")


def plot_copies(samples, folder):
    """Похожа ли генерация на обучающие картинки: пары и распределение расстояний."""
    # Первые 10 сгенерированных картинок — по одной от каждого класса.
    pair_rows = [list(samples["nn_generated"][:10]), list(samples["nn_train"][:10])]
    fig = image_grid(pair_rows, ["Сгенерировано", "Ближайшая\nв обучении"], CLASS_NAMES, cell=0.86, label_width=1.15)
    save_grid(fig, folder, "copies_pairs")

    fig, ax = plt.subplots(figsize=(10, 3.2))
    bins = np.linspace(0, max(samples["baseline_dist"].max(), samples["val_dist"].max()), 45)
    series = (("val_dist", GRAY, "Реальные картинки из валидации"),
              ("generated_dist", BLUE, "Диффузионная модель"),
              ("baseline_dist", ORANGE, "Базовая модель"))
    for key, color, label in series:
        ax.hist(samples[key], bins=bins, histtype="step", linewidth=2, color=color, label=label)
    ax.legend()
    ax.set_xlabel("Расстояние до ближайшей обучающей картинки")
    ax.set_ylabel("Число картинок из 1000")
    save(fig, folder, "copies_hist")
