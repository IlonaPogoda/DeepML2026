"""Собирает outputs/report.html из результатов обучения.

Отдельный запуск пересоздаёт отчёт без обучения: python src/report.py
"""
import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda import AQUA, BLUE, GRAY, INK_2, ORANGE, run_eda, save
from env import ACTIONS, BIKE_COST, MAX_FLEET, MOVE_COST, PRICE, STATE_SIZE

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

GAMMA_COLORS = {0.0: BLUE, 0.9: ORANGE, 0.99: AQUA}


def plot_learning(history, best_gamma, rule_val, rule_name):
    """Два графика рядом: потери на обучении и прибыль на валидации."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.6))
    for gamma, part in history.groupby("gamma"):
        by_update = part.groupby("update")
        color = GAMMA_COLORS[gamma]
        for ax, column in ((left, "train_reward"), (right, "val_profit")):
            mean = by_update[column].mean()
            ax.plot(mean.index, mean.values, color=color, label=f"gamma = {gamma}")
            if gamma == best_gamma:
                ax.fill_between(mean.index, by_update[column].min(), by_update[column].max(),
                                color=color, alpha=0.15, linewidth=0)
    right.axhline(rule_val, color=GRAY, linewidth=1)
    right.text(0.98, 0.04, f"Серая линия — лучшее правило на валидации:\n{rule_name}, {rule_val:.0f}",
               transform=right.transAxes, ha="right", color=INK_2, fontsize=9)
    # Первые шаги обучения намного ниже: без этого предела важная часть графика слипается.
    right.set_ylim(rule_val - 30, rule_val + 8)
    left.set_title("Потери в учебном эпизоде")
    left.set_ylabel("Сумма наград (ближе к 0 лучше)")
    right.set_title("Прибыль на валидации (ось обрезана снизу)")
    right.set_ylabel("Прибыль, тыс. у. е.")
    for ax in (left, right):
        ax.set_xlabel("Шаг обучения")
    left.legend(loc="lower right")
    save(fig, FIGURES, "learning")


def plot_comparison(results, upper_bound):
    """Прибыль стратегий на тесте. Цветом выделен агент, остальные серые."""
    table = results.sort_values("profit")
    colors = [BLUE if s == "REINFORCE" else "#c3c2b7" for s in table["strategy"]]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.barh(table["strategy"], table["profit"], color=colors, height=0.55)
    # Подписи внутри столбцов: снаружи они пересекались бы с линией предела.
    for row in table.itertuples():
        label = f"{row.profit:.1f}" + (f" ± {row.profit_std:.1f}" if pd.notna(row.profit_std) else "")
        ax.text(row.profit - 8, row.strategy, label, ha="right", va="center", fontsize=9,
                color="#ffffff" if row.strategy == "REINFORCE" else "#0b0b0b")
    ax.axvline(upper_bound, color=INK_2, linewidth=1)
    ax.annotate(f"Предел: {upper_bound:.0f}", xy=(upper_bound, 1), xycoords=("data", "axes fraction"),
                xytext=(0, 3), textcoords="offset points", ha="center", color=INK_2, fontsize=9)
    ax.set_xlim(0, upper_bound * 1.08)
    ax.set_xlabel("Прибыль за тестовый период, тыс. у. е.")
    ax.grid(axis="y", visible=False)
    save(fig, FIGURES, "comparison")


def plot_days(agent_log, rule_log, rule_name):
    """Спрос и размер парка по дням на тесте: агент и лучшее простое правило."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), sharex=True, sharey=True)
    for ax, log, title in ((axes[0], agent_log, "REINFORCE (агент с медианной прибылью)"),
                           (axes[1], rule_log, rule_name)):
        dates = pd.to_datetime(log["date"])
        ax.plot(dates, log["demand"], color=GRAY, linewidth=1.2, label="Спрос")
        ax.plot(dates, log["fleet"], color=BLUE, linewidth=2, label="Парк на улицах")
        ax.set_title(title)
        ax.set_ylabel("Велосипеды, аренды")
        ax.set_ylim(bottom=0)
    axes[0].legend(loc="lower left", ncol=2)
    save(fig, FIGURES, "days")


def plot_actions(agent_log, rule_log, rule_name):
    """Как часто выбирается каждое действие на тесте."""
    positions = np.arange(len(ACTIONS))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 3))
    for shift, log, label, color in ((-width / 2, agent_log, "REINFORCE", BLUE),
                                     (width / 2, rule_log, rule_name, ORANGE)):
        counts = [(log["action"] == a).sum() for a in ACTIONS]
        bars = ax.bar(positions + shift, counts, width=width - 0.02, color=color, label=label)
        ax.bar_label(bars, padding=2, color=INK_2, fontsize=9)
    ax.set_xticks(positions, [f"{a:+d}" if a else "0" for a in ACTIONS])
    ax.set_xlabel("Изменение парка за утро, велосипедов")
    ax.set_ylabel("Число дней")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=2)
    save(fig, FIGURES, "actions")


def image(name, alt):
    encoded = base64.b64encode((FIGURES / f"{name}.png").read_bytes()).decode()
    return f'<img src="data:image/png;base64,{encoded}" alt="{alt}">'


def html_table(rows, headers, numeric_from=1):
    """Простая HTML-таблица. Колонки начиная с numeric_from выравниваются вправо."""
    head = "".join(f'<th class="{"num" if i >= numeric_from else ""}">{h}</th>'
                   for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f'<td class="{"num" if i >= numeric_from else ""}">{v}</td>'
                         for i, v in enumerate(row)) + "</tr>"
        for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def verdict_text(agent, rule, seeds, rule_name):
    """Вывод зависит от реальных чисел, поэтому не может разойтись с таблицами."""
    wins = int((seeds["profit"] > rule["profit"]).sum())
    total = len(seeds)
    diff = agent["profit"] - rule["profit"]
    percent = 100 * diff / rule["profit"]
    std = agent["profit_std"]
    if wins == total:
        main = "агент обогнал лучшее простое правило во всех запусках"
    elif abs(diff) <= std:
        main = "агент и лучшее простое правило показали сопоставимый результат: разница меньше разброса между запусками"
    elif diff > 0:
        main = "в среднем агент обогнал лучшее простое правило, но не во всех запусках"
    else:
        main = "в среднем агент уступил лучшему простому правилу"
    return (f"На тесте {main}. Средняя прибыль агента {agent['profit']:.1f} ± {std:.1f} тыс. у. е., "
            f"правила «{rule_name}» {rule['profit']:.1f} тыс. у. е. Разница {diff:+.1f} тыс. у. е. "
            f"({percent:+.1f}%). Правило обогнали {wins} из {total} запусков агента.")


def gamma_text(best_gamma, gamma_table):
    spread = gamma_table["mean"].max() - gamma_table["mean"].min()
    largest_std = gamma_table["std"].max()
    closeness = ("Разница между вариантами не больше разброса между запусками, поэтому "
                 "уверенно считать один из них лучшим нельзя."
                 if spread <= largest_std else
                 "Разница между вариантами больше разброса между запусками.")
    meaning = ("При gamma = 0 агент учитывает только награду текущего дня. Размер парка "
               "всё равно входит в состояние, поэтому текущий парк влияет на решение, но "
               "эффект действия на следующие дни при обучении не учитывается."
               if best_gamma == 0 else
               f"При gamma = {best_gamma} агент учитывает и награды следующих дней.")
    return f"По валидации выбрано значение gamma = {best_gamma}. {closeness} {meaning}"


def behaviour_text(agent, rule, rule_name):
    def compare(a, b, more, less):
        return more if a > b else less

    return (f"Агент теряет {agent['lost_share']:.1f}% спроса, правило «{rule_name}» "
            f"{rule['lost_share']:.1f}%. Простаивает в среднем {agent['idle_share']:.1f}% парка "
            f"у агента и {rule['idle_share']:.1f}% у правила. За период агент перевёз "
            f"{agent['moved']:.1f} тыс. велосипедов, правило {rule['moved']:.1f} тыс., то есть агент "
            + compare(agent["moved"], rule["moved"], "тратит на перевозки больше.", "тратит на перевозки меньше."))


CSS = """
:root { color-scheme: light; --ink:#0b0b0b; --ink2:#52514e; --line:#e1e0d9; --page:#f9f9f7;
        --card:#fcfcfb; --blue:#2a78d6; --tint:#eef4fc; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink); line-height:1.6;
       font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:960px; margin:0 auto; padding:40px 16px 64px; }
h1 { font-size:32px; line-height:1.2; margin:0 0 8px; }
h2 { font-size:22px; margin:44px 0 12px; padding-top:20px; border-top:1px solid var(--line); }
h3 { font-size:17px; margin:24px 0 8px; }
p, li { color:#2b2a28; } .lead { font-size:18px; color:var(--ink2); margin:0 0 24px; }
nav { display:flex; flex-wrap:wrap; gap:4px 16px; font-size:14px; margin-bottom:24px; }
a { color:#1c5cab; }
img { display:block; width:100%; height:auto; margin:12px 0; border:1px solid var(--line); border-radius:6px; }
.pair { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; }
.caption { font-size:14px; color:var(--ink2); margin:-4px 0 16px; }
.verdict { background:var(--tint); border-left:3px solid var(--blue); padding:14px 16px; border-radius:4px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:16px 0; }
.tile { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:14px; }
.tile b { display:block; font-size:26px; font-weight:600; line-height:1.2; }
.tile span { font-size:14px; color:var(--ink2); }
.table-wrap { overflow-x:auto; margin:12px 0 16px; }
table { border-collapse:collapse; width:100%; background:var(--card); font-size:14px; }
th, td { padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
th { color:var(--ink2); font-weight:600; }
.num { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
.flow { display:flex; flex-wrap:wrap; align-items:stretch; gap:8px; margin:12px 0; }
.step { flex:1 1 150px; background:var(--card); border:1px solid var(--line); border-radius:6px;
        padding:10px 12px; font-size:14px; }
.step b { display:block; margin-bottom:2px; } .step code { font-size:13px; }
.arrow { align-self:center; color:var(--ink2); }
code { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.92em; }
pre { background:#f0efec; padding:12px; border-radius:6px; overflow-x:auto; font-size:13px; }
"""


def flow(steps):
    boxes = [f'<div class="step"><b>{title}</b>{text}</div>' for title, text in steps]
    return '<div class="flow">' + '<span class="arrow">→</span>'.join(boxes) + "</div>"


def build():
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(ROOT / "data" / "bike_days.csv", parse_dates=["dteday"])
    results = pd.read_csv(OUTPUTS / "results.csv")
    seeds = pd.read_csv(OUTPUTS / "seeds.csv")
    gamma_table = pd.read_csv(OUTPUTS / "gamma_search.csv")
    history = pd.read_csv(OUTPUTS / "history.csv")
    agent_log = pd.read_csv(OUTPUTS / "test_days_reinforce.csv")
    rule_log = pd.read_csv(OUTPUTS / "test_days_best_rule.csv")
    summary = json.loads((OUTPUTS / "summary.json").read_text())

    best_gamma, rule_name = summary["best_gamma"], summary["best_rule"]
    upper_bound, settings = summary["upper_bound"], summary["settings"]
    agent = results[results["strategy"] == "REINFORCE"].iloc[0]
    rule = results[results["strategy"] == rule_name].iloc[0]
    rules_only = results[results["strategy"] != "REINFORCE"]
    val_rule = rules_only.sort_values("val_profit", ascending=False).iloc[0]

    eda = run_eda(data, FIGURES)
    plot_learning(history, best_gamma, val_rule["val_profit"], val_rule["strategy"])
    plot_comparison(results, upper_bound)
    plot_days(agent_log, rule_log, rule_name)
    plot_actions(agent_log, rule_log, rule_name)

    statistics = html_table(
        [[name, *[f"{v:.1f}" for v in row]] for name, row in eda["statistics"].iterrows()],
        ["Показатель", *eda["statistics"].columns],
    )
    splits = html_table(
        [[r.split, f"{r.start:%d.%m.%Y}", f"{r.end:%d.%m.%Y}", r.days, f"{r.mean:.0f}", r.maximum]
         for r in eda["splits"].itertuples()],
        ["Часть", "Начало", "Конец", "Дней", "Средний спрос", "Макс. спрос"],
    )
    gamma_rows = html_table(
        [[("<b>{}</b>" if r.gamma == best_gamma else "{}").format(r.gamma),
          f"{r.mean:.1f}", f"{r.std:.1f}", f"{r.min:.1f}", f"{r.max:.1f}", f"{r.test_mean:.1f}"]
         for r in gamma_table.itertuples()],
        ["gamma", "Средняя прибыль на валидации", "Ст. откл.", "Минимум", "Максимум",
         "На тесте (для справки)"],
    )
    result_rows = html_table(
        [[("<b>{}</b>" if r.strategy == "REINFORCE" else "{}").format(r.strategy),
          f"{r.profit:.1f}" + (f" ± {r.profit_std:.1f}" if pd.notna(r.profit_std) else ""),
          f"{100 * r.profit / upper_bound:.1f}%", f"{r.lost_share:.1f}%", f"{r.idle_share:.1f}%",
          f"{r.moved:.1f}", f"{r.val_profit:.1f}"]
         for r in results.sort_values("profit", ascending=False).itertuples()],
        ["Стратегия", "Прибыль на тесте", "От предела", "Потерянный спрос", "Простой парка",
         "Перевезено, тыс.", "Прибыль на валидации"],
    )
    seed_rows = html_table(
        [[int(r.seed), f"{r.val:.1f}", f"{r.profit:.1f}", f"{r.lost_share:.1f}%",
          f"{r.idle_share:.1f}%", f"{r.moved:.1f}"] for r in seeds.itertuples()],
        ["Сид", "Прибыль на валидации", "Прибыль на тесте", "Потерянный спрос",
         "Простой парка", "Перевезено, тыс."],
    )
    action_list = ", ".join(f"{a:+d}" if a else "0" for a in ACTIONS)
    parameters = STATE_SIZE * 64 + 64 + 64 * len(ACTIONS) + len(ACTIONS)
    verdict = verdict_text(agent, rule, seeds, rule_name)

    page = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Отчёт REINFORCE</title><style>{CSS}</style></head><body><main>
<h1>Управление парком велосипедов с помощью REINFORCE</h1>
<p class="lead">Лабораторная работа по обучению с подкреплением. Агент каждое утро решает,
сколько велосипедов держать на улицах, а спрос берётся из реальных данных проката.</p>
<nav><a href="#result">Итог</a><a href="#task">Задача</a><a href="#data">Данные и EDA</a>
<a href="#rl">Постановка RL</a><a href="#arch">Архитектура</a><a href="#work">Что делалось</a>
<a href="#results">Результаты</a><a href="#end">Выводы</a></nav>

<section id="result"><h2>Итог</h2>
<p class="verdict">{verdict}</p>
<div class="tiles">
<div class="tile"><b>{agent['profit']:.1f}</b><span>средняя прибыль агента на тесте, тыс. у. е.</span></div>
<div class="tile"><b>{rule['profit']:.1f}</b><span>лучшее простое правило на тесте, тыс. у. е.</span></div>
<div class="tile"><b>{upper_bound:.1f}</b><span>недостижимый предел, тыс. у. е.</span></div>
<div class="tile"><b>{len(data)}</b><span>дней в обработанных данных</span></div>
</div></section>

<section id="task"><h2>Задача</h2>
<p>Если велосипедов мало, клиенты уходят и сервис теряет доход. Если их много, лишние
велосипеды стоят без дела, но за них всё равно приходится платить. Перевозка тоже стоит денег, поэтому
менять парк каждый день невыгодно. Завтрашний спрос заранее неизвестен.</p>
<p><b>Цель:</b> обучить агента REINFORCE управлять размером парка и проверить, приносит ли он
больше прибыли, чем простые правила.</p></section>

<section id="data"><h2>Данные и EDA</h2>
<p>Используется файл <code>day.csv</code> из набора
<a href="https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset">Bike Sharing (UCI)</a>:
дневная статистика системы проката Capital Bikeshare (Вашингтон) за 2011–2012 годы.
Целевая величина спроса — <code>cnt</code>, число аренд за день.</p>
<h3>Обработка при сборке образа</h3>
<ul><li>Сверка контрольной суммы архива SHA-256.</li>
<li>Проверки: нет пропусков, повторов дат и пропущенных дней, <code>cnt = casual + registered</code>.</li>
<li>Признаки: спрос вчера, средний спрос за 7 дней, время года, флаги погоды.</li>
<li>Первые 7 дней удалены, потому что у них нет полной истории спроса.</li>
<li>Колонки <code>casual</code> и <code>registered</code> агент не получает: они раскрывают сегодняшний спрос.</li></ul>
<h3>Статистика</h3>{statistics}
{image('eda_demand', 'Дневной спрос по времени с разбиением на части')}
<p>Средний спрос в 2012 году в {eda['growth']:.2f} раза выше, чем в 2011-м. Поэтому данные разделены по
времени, а агент видит недавний спрос, чтобы учитывать новый уровень спроса.
Минимум {eda['low_value']} аренды приходится на {eda['low_date']:%d.%m.%Y}. Это день урагана Сэнди.
Такой реальный выброс оставлен в тесте: заранее исключать его было бы нечестно.
Влажность равна 0% в {eda['zero_humidity_days']} дн. Такое значение похоже на ошибку записи, но оно
единичное и находится в обучающей части, поэтому строка оставлена без изменений.</p>
<div class="pair">{image('eda_distribution', 'Гистограмма дневного спроса')}{image('eda_monthly', 'Средний спрос по месяцам')}</div>
{image('eda_categories', 'Средний спрос по сезону, погоде и типу дня')}
<div class="pair">{image('eda_temperature', 'Зависимость спроса от температуры')}{image('eda_correlation', 'Матрица корреляций')}</div>
<p>Корреляция спроса с температурой {eda['corr_temp']:.2f}, со вчерашним спросом {eda['corr_lag1']:.2f}.
Поэтому в состояние агента включены и погода, и недавний спрос.</p>
<h3>Разбиение</h3>{splits}
<p>Тестовый период идёт после обучения и валидации, как и в реальной работе сервиса.
Средний спрос на тесте выше, чем на обучении, и это усложняет задачу.</p></section>

<section id="rl"><h2>Постановка RL</h2>
{html_table([
    ["Шаг", "Один день"],
    ["Состояние", f"{STATE_SIZE} чисел: парк, разница парка со вчерашним и средним спросом, вчерашний и средний спрос, температура, влажность, ветер, рабочий день, праздник, 3 флага погоды, 2 числа времени года"],
    ["Действие", f"Изменить парк: {action_list} велосипедов (от 0 до {MAX_FLEET})"],
    ["Спрос", "Реальное значение cnt за этот день. Агент узнаёт его только после решения"],
    ["Прибыль", f"{PRICE:g} × обслуженные аренды − {BIKE_COST:g} × парк − {MOVE_COST:g} × перевезённые велосипеды"],
    ["Награда", f"(прибыль − идеальная прибыль) / 1000, где идеальная прибыль = {PRICE - BIKE_COST:g} × спрос"],
    ["Эпизод при обучении", f"{settings['episode_days']} дней со случайной датой начала и случайным парком"],
    ["Эпизод при проверке", "Все дни периода, стартовый парк равен вчерашнему спросу"],
], ["Элемент", "Описание"], numeric_from=9)}
<p>Цены и затраты условные: в датасете их нет, поэтому прибыль показывает качество решений
в рамках модели, а не реальные деньги. Для простоты один велосипед обслуживает одну аренду в день.</p>
<h3>Как учится REINFORCE</h3>
<pre>G_t = r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ...
A_t = (G_t - среднее(G)) / стандартное_отклонение(G)
loss = -среднее(log p(a_t | s_t) * A_t) - 0.01 * энтропия</pre>
<p>Если результат действия лучше среднего (A &gt; 0), его вероятность растёт. Если хуже, снижается.</p></section>

<section id="arch"><h2>Архитектура</h2>
<h3>Весь проект</h3>
{flow([("Сборка образа", "<code>prepare_data.py</code>: скачивание, проверка, признаки"),
       ("Обучение", "<code>train.py</code>, <code>env.py</code>, <code>agent.py</code>"),
       ("Выбор gamma", "только по валидации"),
       ("Тест", "сравнение с правилами"),
       ("Отчёт", "<code>eda.py</code>, <code>report.py</code>")])}
<h3>Нейросеть</h3>
{flow([("Вход", f"{STATE_SIZE} чисел"),
       ("Скрытый слой", "64 нейрона, Tanh"),
       ("Выход", f"{len(ACTIONS)} значений, softmax"),
       ("Результат", "вероятности действий")])}
<p>Всего {parameters} обучаемых параметров. Оптимизатор Adam, скорость обучения 0.003.</p>
<h3>Цикл обучения</h3>
{flow([("Состояние", "агент видит утренние данные"),
       ("Действие", "выбор по вероятностям"),
       ("Среда", "спрос и прибыль за день"),
       ("Награда", "сохраняется до конца эпизода"),
       ("Обновление", f"после {settings['episodes_per_update']} эпизодов")])}</section>

<section id="work"><h2>Что делалось</h2>
<ol><li>Подготовлены данные и проведён EDA.</li>
<li>Написана среда, в которой сегодняшний парк переходит на завтра.</li>
<li>Написан агент REINFORCE и добавлены простые правила для сравнения.</li>
<li><b>Первый прогон</b> с прибылью в качестве награды: агент получил на тесте 680.1 ± 28.5,
а лучшее правило 697.6 тыс. у. е.</li>
<li><b>Доработка.</b> Прибыль сильно зависит от сезона: летом она высокая почти при любых действиях.
Награда заменена на потери относительно идеального дня, добавлен признак «парк минус средний спрос»,
увеличены скорость обучения и число шагов.</li>
<li>Итоговая конфигурация запущена заново: {len(settings['gammas'])} значения gamma, по {len(settings['seeds'])} сидов,
{settings['updates']} шагов обучения.</li></ol>
<p>Настройки при доработке сравнивались только на валидации (награда уже новая, gamma = 0.9, по 5 сидов).
Эти числа получены разовым экспериментом и в обычный запуск не входят:</p>
{html_table([
    ["300 шагов, скорость 0.001", "367.1", "3.5"],
    ["600 шагов, скорость 0.001", "367.2", "3.6"],
    ["600 шагов, скорость 0.003", "368.5", "2.2"],
    ["600 шагов, скорость 0.003, новый признак", "370.4", "1.7"],
], ["Настройки", "Средняя лучшая прибыль на валидации", "Ст. откл."])}
<p><b>Важное ограничение:</b> тест уже был показан в первом прогоне, до доработки. Выбор настроек опирался на
валидацию, но автор работы знал первый результат на тесте, поэтому проверка не полностью независима.</p></section>

<section id="results"><h2>Результаты</h2>
<h3>Обучение</h3>
{image('learning', 'Кривые обучения для разных значений gamma')}
<p class="caption">Линии показывают среднее по сидам. Закрашенная область — от минимума до максимума для выбранного gamma.</p>
<h3>Выбор gamma на валидации</h3>{gamma_rows}
<p>{gamma_text(best_gamma, gamma_table)}</p>
<h3>Сравнение на тесте</h3>
{image('comparison', 'Прибыль стратегий на тесте')}
{result_rows}
<p class="caption">Предел — прибыль, если парк всегда равен спросу, а перевозки бесплатны. Он недостижим и нужен только как ориентир.
Постоянный парк подобран на валидации. Лучшим правилом считается лучшее на тесте, поэтому сравнение для агента строгое.</p>
<h3>Решения по дням</h3>
{image('days', 'Спрос и размер парка по дням на тесте')}
<p>{behaviour_text(agent, rule, rule_name)}</p>
<h3>Частота действий на тесте</h3>
{image('actions', 'Частота действий агента и правила на тесте')}
<h3>Результаты по сидам</h3>{seed_rows}</section>

<section id="end"><h2>Выводы</h2>
<p>{verdict}</p>
<p>Агент получает {100 * agent['profit'] / upper_bound:.1f}% предельной прибыли,
случайная стратегия {100 * results[results['strategy'] == 'Случайные действия'].iloc[0]['profit'] / upper_bound:.1f}%.</p>
<h3>Ограничения</h3>
<ul><li><b>Идеальный прогноз погоды.</b> Агент утром видит фактическую погоду этого дня, а простые правила погоду не
используют. Часть преимущества агента может объясняться этим допущением, а в реальности прогноз ошибается.</li>
<li>Цены и затраты условные, поэтому при других значениях результат может измениться.</li>
<li>Тест один и охватывает только второе полугодие 2012 года.</li>
<li>Лучшие веса выбираются по валидации, поэтому прибыль на валидации завышена. Для оценки важнее тест.</li>
<li>REINFORCE шумный алгоритм, поэтому показаны все запуски, а не только лучший.</li></ul>
<h3>Что можно улучшить</h3>
<ul><li>Перейти к Actor-Critic, чтобы снизить разброс оценок.</li>
<li>Использовать почасовые данные <code>hour.csv</code> и более реалистичную среду.</li></ul>
</section>
</main></body></html>"""
    (OUTPUTS / "report.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    build()
