"""Разведочный анализ данных: графики и таблицы для отчёта."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#898781"
INK, INK_2 = "#0b0b0b", "#52514e"
SURFACE, GRID, AXIS = "#fcfcfb", "#e1e0d9", "#c3c2b7"

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

SPLIT_NAMES = {"train": "Обучение", "val": "Валидация", "test": "Тест"}
SPLIT_COLORS = {"train": BLUE, "val": ORANGE, "test": AQUA}
MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def save(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / f"{name}.png")
    plt.close(fig)


def plot_demand(data, folder):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for split, part in data.groupby("split", sort=False):
        ax.plot(part["dteday"], part["cnt"], color=SPLIT_COLORS[split],
                linewidth=1.2, label=SPLIT_NAMES[split])
    low = data.loc[data["cnt"].idxmin()]
    ax.annotate(f"Минимум: {int(low['cnt'])} аренды\n{low['dteday']:%d.%m.%Y}",
                xy=(low["dteday"], low["cnt"]), xytext=(-125, 8),
                textcoords="offset points", color=INK_2, fontsize=9,
                arrowprops={"arrowstyle": "-", "color": GRAY, "linewidth": 0.8})
    ax.set_title("Спрос растёт от года к году и сильно зависит от сезона")
    ax.set_ylabel("Аренд в день")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left", ncol=3)
    save(fig, folder, "eda_demand")


def plot_distribution(data, folder):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(data["cnt"], bins=25, color=BLUE, edgecolor=SURFACE, linewidth=1.5)
    ax.axvline(data["cnt"].median(), color=INK_2, linewidth=1)
    ax.annotate(f"Медиана: {data['cnt'].median():.0f}", xy=(data["cnt"].median(), 1),
                xycoords=("data", "axes fraction"), xytext=(5, -12),
                textcoords="offset points", color=INK_2, fontsize=9)
    ax.set_title("Распределение дневного спроса")
    ax.set_xlabel("Аренд в день")
    ax.set_ylabel("Число дней")
    ax.grid(axis="x", visible=False)
    save(fig, folder, "eda_distribution")


def plot_monthly(data, folder):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    monthly = data.groupby(["yr", "mnth"])["cnt"].mean().unstack(0)
    for year, label, color in ((0, "2011", BLUE), (1, "2012", ORANGE)):
        ax.plot(monthly.index, monthly[year], color=color, label=label, marker="o", markersize=5)
    ax.set_xticks(range(1, 13), MONTHS)
    ax.set_title("Средний спрос по месяцам")
    ax.set_ylabel("Аренд в день")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower center", ncol=2)
    save(fig, folder, "eda_monthly")


def plot_categories(data, folder):
    groups = [
        ("Сезон", "season", {1: "Зима", 2: "Весна", 3: "Лето", 4: "Осень"}),
        ("Погода", "weathersit", {1: "Ясно", 2: "Облачно", 3: "Дождь, снег"}),
        ("День", "workingday", {0: "Выходной", 1: "Рабочий"}),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharey=True)
    for ax, (title, column, names) in zip(axes, groups):
        means = data.groupby(column)["cnt"].mean()
        bars = ax.bar([names[i] for i in means.index], means.values, color=BLUE, width=0.55)
        ax.bar_label(bars, fmt="%.0f", padding=3, color=INK_2, fontsize=9)
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Средний спрос, аренд в день")
    save(fig, folder, "eda_categories")


def plot_temperature(data, folder):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.scatter(data["temp"] * 41, data["cnt"], s=14, color=BLUE, alpha=0.5, linewidths=0)
    ax.set_title("Чем теплее, тем больше аренд, но до предела")
    ax.set_xlabel("Температура, °C")
    ax.set_ylabel("Аренд в день")
    save(fig, folder, "eda_temperature")


def plot_correlation(data, folder):
    labels = {"cnt": "Спрос", "demand_lag1": "Спрос вчера", "demand_avg7": "Среднее за 7 дней",
              "temp": "Температура", "hum": "Влажность", "windspeed": "Ветер",
              "workingday": "Рабочий день", "holiday": "Праздник"}
    corr = data[list(labels)].rename(columns=labels).corr()
    cmap = LinearSegmentedColormap.from_list("blue_red", ["#e34948", "#f0efec", "#2a78d6"])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    ax.spines[:].set_visible(False)
    for row in range(len(corr)):
        for col in range(len(corr)):
            value = corr.iloc[row, col]
            color = "#ffffff" if abs(value) > 0.75 else INK
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_title("Корреляция признаков")
    save(fig, folder, "eda_correlation")


def statistics_table(data) -> pd.DataFrame:
    """Описательная статистика в понятных единицах (в датасете значения нормированы)."""
    values = pd.DataFrame({
        "Спрос, аренд в день": data["cnt"],
        "Температура, °C": data["temp"] * 41,
        "Влажность, %": data["hum"] * 100,
        "Ветер, км/ч": data["windspeed"] * 67,
    })
    table = values.describe().T[["mean", "std", "min", "25%", "50%", "75%", "max"]]
    table.columns = ["Среднее", "Ст. откл.", "Мин", "25%", "Медиана", "75%", "Макс"]
    return table


def split_table(data) -> pd.DataFrame:
    table = data.groupby("split", sort=False).agg(
        start=("dteday", "min"), end=("dteday", "max"), days=("cnt", "size"),
        mean=("cnt", "mean"), maximum=("cnt", "max"),
    ).reset_index()
    table["split"] = table["split"].map(SPLIT_NAMES)
    return table


def run_eda(data, folder) -> dict:
    """Строит все графики EDA и возвращает значения, которые нужны в тексте отчёта."""
    plot_demand(data, folder)
    plot_distribution(data, folder)
    plot_monthly(data, folder)
    plot_categories(data, folder)
    plot_temperature(data, folder)
    plot_correlation(data, folder)
    low = data.loc[data["cnt"].idxmin()]
    return {
        "days": len(data),
        "statistics": statistics_table(data),
        "splits": split_table(data),
        "growth": data[data["yr"] == 1]["cnt"].mean() / data[data["yr"] == 0]["cnt"].mean(),
        "corr_temp": data["cnt"].corr(data["temp"]),
        "corr_lag1": data["cnt"].corr(data["demand_lag1"]),
        "low_value": int(low["cnt"]),
        "low_date": low["dteday"],
        "zero_humidity_days": int((data["hum"] == 0).sum()),
    }
