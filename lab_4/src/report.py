"""Собирает outputs/report.html из результатов обучения и оценки.

Отдельный запуск пересоздаёт отчёт без обучения: python src/report.py
"""
import base64
import json

import numpy as np
import pandas as pd

import plots
import train
from diffusion import BETA_END, BETA_START, T, alpha_bar
from eda import CLASS_NAMES, run_eda
from train import OUTPUTS, ROOT, load_split

FIGURES = OUTPUTS / "figures"

CSS = """
:root { color-scheme: light; --ink:#0b0b0b; --ink2:#52514e; --line:#e1e0d9; --page:#f9f9f7;
        --card:#fcfcfb; --blue:#2a78d6; --tint:#eef4fc; }
* { box-sizing: border-box; }
body { margin:0; background:var(--page); color:var(--ink); font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif; }
main { max-width:960px; margin:0 auto; padding:40px 16px 64px; }
h1 { font-size:32px; line-height:1.2; margin:0 0 8px; }
h2 { font-size:22px; margin:44px 0 12px; padding-top:20px; border-top:1px solid var(--line); }
h3 { font-size:17px; margin:24px 0 8px; }
p, li { color:#2b2a28; } .lead { font-size:18px; color:var(--ink2); margin:0 0 24px; }
nav { display:flex; flex-wrap:wrap; gap:4px 16px; font-size:14px; margin-bottom:24px; }
a { color:#1c5cab; }
img { display:block; width:100%; height:auto; margin:12px 0; border:1px solid var(--line); border-radius:6px; }
img.gif { max-width:568px; image-rendering:pixelated; }
.pair { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; }
.pair h3 { margin:8px 0 0; }
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
.flow { display:flex; flex-wrap:wrap; align-items:stretch; gap:8px; margin:12px 0; }
.step { flex:1 1 150px; background:var(--card); border:1px solid var(--line); border-radius:6px;
        padding:10px 12px; font-size:14px; }
.step b { display:block; margin-bottom:2px; }
.arrow { align-self:center; color:var(--ink2); }
code { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:.92em; }
pre { background:#f0efec; padding:12px; border-radius:6px; overflow-x:auto; font-size:13px; }
"""


def picture(name, alt, extension="png", css_class=""):
    raw = (FIGURES / f"{name}.{extension}").read_bytes()
    mime = "image/gif" if extension == "gif" else "image/png"
    encoded = base64.b64encode(raw).decode()
    return f'<img class="{css_class}" src="data:{mime};base64,{encoded}" alt="{alt}">'


def html_table(rows, headers):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def flow(steps):
    boxes = [f'<div class="step"><b>{title}</b>{text}</div>' for title, text in steps]
    return '<div class="flow">' + '<span class="arrow">→</span>'.join(boxes) + "</div>"


def tile(value, text):
    return f'<div class="tile"><b>{value}</b><span>{text}</span></div>'


def percent(value):
    return f"{value * 100:.1f}%"


def number(value):
    return f"{value:,}".replace(",", " ")


def make_figures(data, samples, history, steps, comparison, confusion, classes, loss_t, summary):
    FIGURES.mkdir(parents=True, exist_ok=True)
    eda = run_eda(data, FIGURES)
    images, labels = load_split("test")
    plots.plot_forward(images, labels, FIGURES)
    plots.plot_schedule(FIGURES)
    plots.plot_learning(history, FIGURES)
    plots.plot_loss_by_t(loss_t, FIGURES)
    progress = np.load(OUTPUTS / "progress.npz")
    epochs = [e for e in train.PROGRESS_EPOCHS if f"epoch_{e}" in progress]
    plots.plot_progress(progress, epochs, FIGURES)
    plots.plot_generated(samples, data, FIGURES)
    plots.plot_reverse(samples["chain"], FIGURES)
    plots.save_gif(samples["chain"], FIGURES)
    plots.plot_steps_strip(samples, steps["steps"].tolist(), FIGURES)
    plots.plot_steps_metrics(steps, summary["chosen_steps"], summary["real_fd_steps"], FIGURES)
    plots.plot_comparison(comparison, FIGURES)
    plots.plot_confusion(confusion, FIGURES)
    plots.plot_per_class(classes, FIGURES)
    plots.plot_copies(samples, FIGURES)
    return eda


def dataset_section(eda):
    class_rows = [[r.label, r.en, r.ru, number(r.train), r.val, number(r.test), f"{r.brightness:.0f}", f"{r.filled:.0f}%"]
                  for r in eda["classes"].itertuples()]
    split_rows = [[r.name, number(r.images), number(r.per_class), f"{r.mean:.1f}", f"{r.std:.1f}", f"{r.zeros:.1f}%"]
                  for r in eda["splits"].itertuples()]
    closest = ", ".join(f"{a} и {b} ({d:.1f})" for d, a, b in eda["closest"])
    return f"""
<h2 id="data">Датасет: Fashion-MNIST</h2>
<p><a href="https://github.com/zalandoresearch/fashion-mnist">Fashion-MNIST</a> — набор изображений
одежды и обуви, опубликованный компанией Zalando в 2017 году (Xiao, Rasul, Vollgraf). Картинки сделаны из
фотографий товаров интернет-магазина: их перевели в оттенки серого и уменьшили до 28×28 пикселей.
Датасет создан как более сложная замена рукописным цифрам MNIST и имеет такой же формат.</p>

<div class="tiles">
{tile("70 000", "картинок всего: 60 000 в обучающем наборе и 10 000 в тестовом")}
{tile("28 × 28", "пикселей, 1 канал (оттенки серого)")}
{tile("10", "классов, в каждом одинаковое число картинок")}
{tile(f"{eda['zero_share']:.0f}%", "пикселей равны нулю, то есть это чистый фон")}
</div>

<h3>Из чего состоит</h3>
<p>Одна запись датасета — это пара «картинка и метка». Картинка состоит из 784 чисел от 0 до 255: 0 означает
фон, 255 — самый яркий пиксель. Метка — номер класса от 0 до 9. Других полей нет: ни текста, ни размера или цены товара.</p>
<p>Датасет распространяется четырьмя файлами в двоичном формате IDX: картинки и метки отдельно для обучающего
и тестового наборов. Скрипт <code>prepare_data.py</code> скачивает их при сборке Docker-образа и сверяет контрольные суммы MD5.</p>
{html_table(class_rows, ["Метка", "Класс", "Перевод", "Обучение", "Валидация", "Тест", "Средняя яркость", "Занято предметом"])}
<p class="caption">Средняя яркость (шкала 0–255) и доля ненулевых пикселей посчитаны по 60 000 картинок обучающего набора.</p>

<h3>Как выглядят картинки</h3>
{picture("eda_classes", "По 12 случайных картинок каждого из 10 классов")}
<p class="caption">По 12 случайных картинок каждого класса. Во всём отчёте цвета показаны инвертированно: фон белый,
а яркие пиксели тёмные, так картинки выглядят привычнее. На вход модели подаются исходные значения.</p>

<h3>Как устроены классы</h3>
{picture("eda_means", "Средняя картинка и карта разброса пикселей для каждого класса")}
<p class="caption">Верхний ряд — средняя картинка класса. Нижний — разброс: тёмные места сильнее всего меняются от картинки к картинке.
Предметы отцентрованы, а обувь повёрнута в одну сторону, поэтому средние картинки узнаваемы. Сандалии выглядят
самым размытым средним: их форма разнообразнее всего.</p>
{picture("eda_filled", "Доля ненулевых пикселей по классам")}
<p class="caption">Верхняя одежда и сумки занимают больше половины кадра, а кроссовки, сандалии и брюки примерно треть.</p>
{picture("eda_pixels", "Гистограмма яркости всех пикселей")}
<p class="caption">Распределение пикселей далеко от нормального: {eda['zero_share']:.0f}% значений равны нулю, остальные распределены
широко, а максимум 255 встречается у {eda['max_share']:.1f}% пикселей. Именно это распределение модели предстоит получить из нормального шума.</p>
<div class="pair"><div>{picture("eda_distance", "Матрица расстояний между средними картинками классов")}</div>
<div><h3>Какие классы похожи</h3><p>Матрица показывает евклидово расстояние между средними картинками классов.
Ближе всего: {closest}. Можно ожидать, что эти классы труднее всего различать и генерировать так, чтобы
их не путали. Эту догадку проверим в разделе результатов.</p></div></div>

<h3>Проверка качества и подготовка</h3>
<ul>
<li>Размеры массивов и набор меток проверяются при сборке образа. Контрольные суммы всех файлов совпадают.</li>
<li>Побайтово одинаковых картинок: {eda['train_duplicates']} в обучающем наборе, {eda['test_duplicates']} в тестовом,
{eda['overlap']} общих между ними. Значит, утечки теста через дубликаты нет.</li>
<li>Пропусков быть не может: это массив чисел фиксированного размера.</li>
<li>Для обучения пиксели переводятся из 0…255 в −1…1 по формуле <code>x / 127.5 − 1</code>. Так данные становятся одного масштаба с шумом, у которого среднее 0 и дисперсия 1.</li>
<li>Аугментация не используется: модель должна воспроизводить датасет как есть, а лишние повороты появились бы и в сгенерированных картинках.</li>
</ul>

<h3>Разделение</h3>
{html_table(split_rows, ["Часть", "Картинок", "На класс", "Средняя яркость", "Ст. откл.", "Доля нулей"])}
<p>Валидация взята из обучающего набора поровну от каждого класса и нужна для выбора числа шагов и контроля обучения.
Тестовый набор не участвует в обучении и в выборе настроек: он нужен для итогового сравнения и иллюстраций.</p>
"""


def method_section(summary):
    a500 = alpha_bar[500].item()
    a999 = alpha_bar[999].item()
    a100 = alpha_bar[100].item()
    layers = [
        ["Вход", "Шумная картинка, номер шага t, класс", "1 × 28 × 28"],
        ["Условие", "Вектор шага и вектор класса складываются и подаются в каждый блок", "128 чисел"],
        ["Первая свёртка и блок", "Находят мелкие черты: края, текстуру", "32 × 28 × 28"],
        ["Сжатие и блок", "Свёртка с шагом 2 уменьшает картинку вдвое", "64 × 14 × 14"],
        ["Сжатие и 2 блока", "Самый маленький размер: видна форма предмета целиком", "128 × 7 × 7"],
        ["Расширение и блок", "Увеличивает размер, получает skip-соединение с пути вниз", "64 × 14 × 14"],
        ["Расширение и блок", "То же самое, возвращает мелкие детали", "32 × 28 × 28"],
        ["Выход", "Оценка шума: своё число на каждый пиксель", "1 × 28 × 28"],
    ]
    settings = [
        ["Число шагов шума T", T, "Стандартное значение из статьи DDPM"],
        ["β: от и до", f"{BETA_START} … {BETA_END}", "Линейное расписание из статьи DDPM"],
        ["Базовое число каналов", train.BASE_CHANNELS, "Даёт каналы 32, 64, 128"],
        ["Параметров сети", number(summary["parameters"]), "Такая сеть обучается на процессоре"],
        ["Эпох", train.EPOCHS, f"Ограничено временем: около {summary['train_minutes']:.0f} мин"],
        ["Размер батча", train.BATCH_SIZE, ""],
        ["Оптимизатор", f"Adam, lr = {train.LEARNING_RATE}", "lr плавно падает до нуля (косинус)"],
        ["EMA", train.EMA_DECAY, "Коэффициент усреднения весов"],
    ]
    return f"""
<h2 id="method">Метод: как работает диффузия</h2>
<p><b>Главная идея.</b> Сразу нарисовать картинку из ничего трудно. Убрать из картинки немного шума намного легче.
Поэтому модель учат убирать шум понемногу, а затем запускают этот процесс от чистого шума: шаг за шагом в нём
проступает предмет. Метод называется DDPM (Denoising Diffusion Probabilistic Models).</p>
{flow([("1. Прямой процесс", "К картинке постепенно добавляется шум."),
       ("2. Обучение", "Сеть учится угадывать, какой шум был добавлен."),
       ("3. Обратный процесс", "Из чистого шума сеть шаг за шагом получает новую картинку.")])}

<h3>Шаг 1. Прямой процесс: добавляем шум</h3>
<p>Шаг (step) — одна маленькая порция шума. Всего шагов T = {T}. На шаге с номером t доля шума равна β<sub>t</sub>,
она линейно растёт от {BETA_START} до {BETA_END}. Проходить все шаги не нужно: картинку на любом шаге можно получить сразу:</p>
<pre>x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε</pre>
<ul>
<li><code>x_0</code> — исходная картинка, <code>ε</code> — случайный шум с нормальным распределением;</li>
<li><code>ᾱ_t</code> (альфа с чертой) — произведение всех (1 − β) до шага t: сколько картинки осталось.</li>
</ul>
<p><b>Пример для одного пикселя.</b> Пусть пиксель равен 0.8, а шум равен −1.2.
На шаге 500 ᾱ = {a500:.3f}, поэтому получаем
√{a500:.3f} · 0.8 + √{1 - a500:.3f} · (−1.2) = {a500 ** 0.5 * 0.8 + (1 - a500) ** 0.5 * -1.2:.2f}.
На шаге 100 ᾱ = {a100:.3f}: картинка ещё хорошо видна. На шаге 999 ᾱ = {a999:.5f}: от картинки не осталось почти ничего.</p>
{picture("forward", "Четыре картинки на разных шагах прямого процесса")}
<p class="caption">Одна и та же картинка из теста при разных t: чем больше номер шага, тем больше шума.</p>
{picture("schedule", "Как с номером шага меняются коэффициенты картинки и шума")}
<p class="caption">Расписание шума. Вклад картинки плавно падает к нулю, а вклад шума растёт к единице.</p>

<h3>Шаг 2. Обучение: сеть угадывает шум</h3>
<p>Сеть получает зашумлённую картинку и должна назвать шум, который в ней содержится. Один шаг обучения:</p>
<ol>
<li>Берём батч картинок <code>x_0</code> вместе с их классами.</li>
<li>Для каждой картинки случайно выбираем шаг <code>t</code> от 0 до {T - 1} и шум <code>ε</code>.</li>
<li>По формуле получаем зашумлённую картинку <code>x_t</code>.</li>
<li>Сеть выдаёт оценку шума: <code>ε̂ = сеть(x_t, t, класс)</code>.</li>
<li>Ошибка — среднеквадратичная: <code>loss = среднее((ε − ε̂)²)</code>. Шаг Adam уменьшает её.</li>
</ol>
<p>Правильный ответ известен, потому что шум добавили мы сами.</p>

<h3>Шаг 3. Обратный процесс: генерация</h3>
<p>Начинаем со случайного шума, задаём нужный класс и идём от шага {T - 1} к шагу 0. На каждом шаге сеть оценивает шум,
и часть этого шума вычитается. Сравниваем два способа:</p>
{html_table([
    ["<b>DDPM</b>", f"Все {T} шагов", "На каждом шаге добавляется немного нового случайного шума", f"<code>x_(t−1) = (x_t − β_t / √(1−ᾱ_t) · ε̂) / √(1−β_t) + √β_t · z</code>"],
    ["<b>DDIM</b>", f"Любое число шагов, например 50", "Случайность есть только в начальном шуме, дальше всё детерминировано. Оценка x̂_0 обрезается до диапазона −1…1", "<code>x̂_0 = (x_t − √(1−ᾱ_t) · ε̂) / √ᾱ_t</code>, затем <code>x_prev = √ᾱ_prev · x̂_0 + √(1−ᾱ_prev) · ε̂</code>"],
], ["Способ", "Шаги", "Особенность", "Формула"])}
<p>Обе формулы используют одну и ту же обученную сеть: повторно обучать её для другого числа шагов не нужно.
Чем меньше шагов, тем быстрее генерация, потому что на каждый шаг нужен один проход через сеть. Цена такого ускорения измеряется в разделе результатов.</p>

<h3>Архитектура сети: U-Net</h3>
<p>Сеть должна выдать результат того же размера, что и вход. U-Net сначала сжимает картинку, чтобы увидеть форму
предмета целиком, а затем разжимает её обратно. Skip-соединения передают мелкие детали с пути вниз на путь вверх напрямую.</p>
{html_table(layers, ["Часть", "Что делает", "Размер"])}
<p>Каждый блок (<code>ResBlock</code>) — две свёртки с нормализацией и ReLU и обходным соединением. Сеть узнаёт условие так:
номер шага превращается в вектор из синусов и косинусов, класс — в обучаемый вектор, а их сумма добавляется к признакам
внутри каждого блока. Поэтому одна сеть работает на всех шагах и для всех классов.</p>

<h3>Настройки</h3>
{html_table(settings, ["Параметр", "Значение", "Комментарий"])}
<p><b>EMA</b> (скользящее среднее весов): рядом с обучаемой сетью хранится её сглаженная копия. После каждого шага
она сдвигается к текущим весам на 0.1%. Для проверки и генерации используется эта копия: она меньше зависит от последних батчей.</p>
"""


def metrics_section(summary):
    return f"""
<h2 id="metrics">Как измерялось качество</h2>
<p>«Красиво или нет» не является метрикой, поэтому результат проверяется числами. Для этого отдельно обучен обычный
классификатор, он нигде не участвует в обучении диффузии. Его точность на реальном тесте — {percent(summary['judge_test_accuracy'])}.</p>
{html_table([
    ["<b>Точность судьи</b>", "Просим сгенерировать сумку и смотрим, видит ли классификатор сумку", "Соответствует ли картинка заданному классу", "Больше"],
    ["<b>FD</b> (Fréchet Distance)", "Классификатор превращает каждую картинку в 64 числа. Для реальных и сгенерированных наборов считаются средние и ковариации, затем расстояние Фреше между ними", "Насколько набор похож на реальный в целом, включая разнообразие", "Меньше"],
    ["<b>TSTR</b> (Train on Synthetic, Test on Real)", f"Новый классификатор учится только на {number(summary['final_images'])} картинках из проверяемого набора ({summary['tstr_epochs']} эпох, 3 сида), а проверяется на реальном тесте", "Можно ли использовать данные вместо настоящих", "Больше"],
], ["Метрика", "Как считается", "Что показывает", "Лучше, когда"])}
<p>FD устроен как известная метрика FID, но вместо сети Inception используется наш классификатор. Поэтому значения FD можно
сравнивать только внутри этой работы, с опубликованными значениями FID их сравнивать нельзя.</p>
<p>С чем сравниваем:</p>
<ul>
<li><b>Реальные картинки</b>: {number(summary['final_images'])} картинок из валидации. Это ориентир, лучшего результата ждать не стоит.</li>
<li><b>Базовая модель</b>: для каждого класса считается среднее и ковариация пикселей, затем картинки берутся из многомерного нормального распределения. Это простой генератор без нейросети: сложная модель должна быть лучше него.</li>
</ul>
"""


def steps_name(steps):
    return f"{steps} шагов DDPM" if steps == T else f"{steps} шагов DDIM"


def verdict(comparison):
    model = comparison[comparison["source"].str.startswith("Диффузионная")].iloc[0]
    base = comparison[comparison["source"].str.startswith("Базовая")].iloc[0]
    real = comparison[comparison["source"].str.startswith("Реальные")].iloc[0]
    better_fd = base["fd"] / model["fd"]
    return model, base, real, better_fd


def results_section(steps, comparison, classes, confusion, history, samples, summary):
    chosen = summary["chosen_steps"]
    chosen_row = steps[steps["steps"] == chosen].iloc[0]
    chosen_name = steps_name(chosen)
    fast = steps[steps["steps"] == 50].iloc[0]
    close = steps[steps["steps"] == 100].iloc[0]
    fewest = steps.iloc[0]
    final_minutes = chosen_row["seconds_per_100"] * summary["final_images"] / 100 / 60

    steps_rows = [[("<b>{}</b>" if r.steps == chosen else "{}").format(r.steps), r.method, f"{r.fd:.1f}",
                   percent(r.accuracy), f"{r.seconds_per_100:.1f} с"] for r in steps.itertuples()]
    comparison_rows = [[r.source, f"{r.fd:.1f}", percent(r.accuracy), f"{percent(r.tstr)} ± {r.tstr_std * 100:.1f}"]
                       for r in comparison.itertuples()]

    # Самые частые ошибки судьи на сгенерированных картинках.
    off = confusion.copy()
    np.fill_diagonal(off, 0)
    top = np.argsort(off.ravel())[::-1][:3]
    mistakes = ", ".join(f"{CLASS_NAMES[i // 10]} → {CLASS_NAMES[i % 10]} ({off.ravel()[i] / confusion[i // 10].sum() * 100:.0f}%)" for i in top)
    worst = classes.sort_values("generated").iloc[:3]
    worst_text = ", ".join(f"{CLASS_NAMES[int(r.label)]} ({percent(r.generated)})" for r in worst.itertuples())

    gen_d, val_d, base_d = samples["generated_dist"], samples["val_dist"], samples["baseline_dist"]
    copy_text = (f"Медиана расстояния до ближайшей обучающей картинки: {np.median(gen_d):.2f} у сгенерированных, "
                 f"{np.median(val_d):.2f} у реальных картинок из валидации (их сеть не видела) и "
                 f"{np.median(base_d):.2f} у базовой модели. Минимум у сгенерированных {gen_d.min():.2f}, у валидации {val_d.min():.2f}.")
    copy_verdict = ("Сгенерированные картинки находятся от обучающих примерно так же далеко, как настоящие новые картинки "
                    "или дальше. Признаков заучивания нет."
                    if np.median(gen_d) >= np.median(val_d) * 0.9 else
                    "Сгенерированные картинки заметно ближе к обучающим, чем новые реальные. Это может означать частичное заучивание.")

    last = history.iloc[-1]
    return f"""
<h2 id="training">Обучение</h2>
<p>Модель обучалась {train.EPOCHS} эпох, {summary['train_minutes']:.0f} минут на процессоре. Итоговая ошибка на обучении {last['train_loss']:.4f}, на валидации {last['val_loss']:.4f}.</p>
{picture("learning", "Ошибка на обучении и валидации по эпохам")}
<p class="caption">Ошибка на обучении считается по текущим весам, а на валидации — по усреднённым (EMA). Сначала усреднённые веса ещё помнят случайное начало, поэтому
валидация выше обучения, затем кривые сходятся. Сближение кривых говорит, что явного переобучения нет.</p>
{picture("loss_by_t", "Ошибка оценки шума в зависимости от шага")}
<p class="caption">Ошибка сильно зависит от шага. Когда шума мало (малые t), его трудно отделить от деталей картинки. Когда шума много, картинка почти не мешает его оценить.</p>
{picture("progress", "Примеры генерации после разных эпох")}
<p class="caption">Генерация из одного и того же начального шума после разных эпох, 50 шагов DDIM.</p>

<h2 id="results">Результаты</h2>
<h3>Что получилось</h3>
{picture("generated", "Сгенерированные картинки всех классов")}
<p class="caption">Сгенерированные картинки, по 12 на класс, без отбора: это первые картинки каждого класса в наборе, {chosen_name}. Проверка на копирование из датасета находится ниже.</p>
<div class="pair">
<div><h3>Реальные (тест)</h3>{picture("compare_real", "Реальные картинки")}</div>
<div><h3>Диффузионная модель</h3>{picture("compare_fake", "Сгенерированные картинки")}</div>
<div><h3>Базовая модель</h3>{picture("compare_base", "Картинки базовой модели")}</div>
</div>
<p class="caption">Базовая модель передаёт общую форму, но её картинки зернистые и похожи на наложение нескольких предметов: она знает только средние и ковариации пикселей.</p>

<h3>Как шум превращается в картинку</h3>
{picture("reverse", "Генерация картинок всех классов по шагам", "gif", "gif")}
<p class="caption">Анимация: все {T} шагов DDPM, по одному кадру на 10 шагов. Полоса снизу показывает прогресс.</p>
{picture("reverse", "Промежуточные картинки обратного процесса")}
<p class="caption">Те же картинки в отдельных кадрах. Форма предмета начинает проступать примерно с t ≈ 350, а на более ранних шагах глазом почти ничего не видно.</p>

<h3>Опыт: сколько шагов нужно для генерации</h3>
<p>Одна и та же обученная сеть запускалась с разным числом шагов. На каждый вариант генерировалось {number(summary['steps_images'])} картинок, поровну от класса.
Для честности сравнение шло с <b>валидацией</b>, а не с тестом. Правило выбора задано заранее: наименьшее число шагов, при котором FD не более чем на {(summary['allowed_worse'] - 1) * 100:.0f}% хуже лучшего значения.</p>
{picture("steps_strip", "Картинки при разном числе шагов из одного начального шума")}
{picture("steps_metrics", "FD, точность и время при разном числе шагов")}
{html_table(steps_rows, ["Шагов", "Способ", "FD на валидации", "Точность судьи", "Время на 100 картинок"])}
<p class="verdict"><b>По правилу выбрано: {chosen_name}.</b> Это самый медленный вариант: {chosen_row['seconds_per_100']:.0f} с на 100 картинок, зато лучший FD: {chosen_row['fd']:.1f}.
У {steps_name(100)} FD равен {close['fd']:.1f}, то есть в {close['fd'] / chosen_row['fd']:.1f} раза хуже, но генерация в {chosen_row['seconds_per_100'] / close['seconds_per_100']:.0f} раз быстрее.
Ориентир: FD настоящих {number(summary['steps_images'])} картинок из обучающей части относительно валидации равен {summary['real_fd_steps']:.1f}. Ноль здесь недостижим из-за конечного размера выборки.</p>
<p><b>Что видно из опыта:</b></p>
<ul>
<li><b>Качество быстро выходит на плато.</b> С 5 до 20 шагов DDIM FD падает с {fewest['fd']:.1f} до {steps[steps['steps'] == 20].iloc[0]['fd']:.1f}, а с 50 до 100 шагов только с {fast['fd']:.1f} до {close['fd']:.1f}.</li>
<li><b>Время растёт пропорционально числу шагов</b>, потому что каждый шаг — один проход через сеть.</li>
<li><b>DDIM из одного шума даёт почти одну и ту же картинку</b>, начиная примерно с 20 шагов: он детерминирован, а шаги лишь уточняют её. DDPM даёт другой предмет, потому что на каждом шаге добавляет новый случайный шум.</li>
<li><b>Точность судьи ведёт себя иначе, чем FD.</b> Она наибольшая у самого плохого по FD варианта: {percent(fewest['accuracy'])} при 5 шагах. Вероятное объяснение: при малом числе шагов картинки размытые и похожи на типичный предмет класса, поэтому их легко распознать. Отдельно это не проверялось.</li>
</ul>
<p>Если важна скорость, разумный компромисс — {steps_name(50)}: FD {fast['fd']:.1f} и {fast['seconds_per_100']:.1f} с на 100 картинок. Поэтому такое значение стоит по умолчанию в <code>generate.py</code>.
Для итогового сравнения выбранный вариант применён, как и задано правилом: {number(summary['final_images'])} картинок генерировались около {final_minutes:.0f} минут.</p>

<h3>Итоговое сравнение на тесте</h3>
<p>На каждый источник {number(summary['final_images'])} картинок, поровну от класса. Диффузионная модель использует {chosen_name}.</p>
{picture("comparison", "Сравнение трёх источников картинок по трём метрикам")}
{html_table(comparison_rows, ["Источник", "FD с тестом", "Точность судьи", "TSTR на тесте"])}
<p>Разброс TSTR показан по 3 сидам классификатора. Набор картинок и веса диффузионной модели при этом не меняются.</p>

<h3>Какие классы получаются хуже</h3>
<div class="pair"><div>{picture("confusion", "Матрица ошибок судьи на сгенерированных картинках")}</div>
<div><p>Строка — класс, который просили сгенерировать, столбец — что увидел судья. Хуже всего по точности: {worst_text}. Самые частые ошибки: {mistakes}.</p>
<p>Как и ожидалось по матрице расстояний из раздела о датасете, путаются свитер, пальто, рубашка и футболка. На настоящих картинках судья путает их тоже: на графике ниже рядом с каждой синей точкой стоит серая. Поэтому часть ошибок относится не к генератору, а к самому судье и сходству классов. Однако у свитера, пальто и рубашки точность на сгенерированных заметно ниже, чем на настоящих.</p></div></div>
{picture("per_class", "Точность судьи по классам на реальных и сгенерированных картинках")}

<h3>Не копирует ли модель обучающие картинки</h3>
<p>Модель могла бы просто заучить датасет: тогда метрики были бы хорошими, но новых картинок она бы не создавала. Для проверки у каждой картинки ищется ближайшая в обучающей части: считается евклидово расстояние между пикселями в шкале 0…1.</p>
{picture("copies_pairs", "Сгенерированные картинки и их ближайшие соседи из обучающей части")}
{picture("copies_hist", "Распределение расстояний до ближайшей обучающей картинки")}
<p>{copy_text} {copy_verdict}</p>
"""


def conclusions(comparison, summary, steps):
    model, base, real, better_fd = verdict(comparison)
    tstr_gap = (real["tstr"] - model["tstr"]) * 100
    speed = steps.set_index("steps")
    chosen = speed.loc[summary["chosen_steps"]]
    return f"""
<h2 id="conclusions">Выводы</h2>
<ul>
<li><b>Модель решает задачу.</b> Она создаёт новые узнаваемые картинки заданного класса: судья относит их к этому классу в {percent(model['accuracy'])} случаев, а настоящие картинки из валидации — в {percent(real['accuracy'])}.</li>
<li><b>По FD и TSTR модель лучше базовой.</b> FD: {model['fd']:.1f} против {base['fd']:.1f}, в {better_fd:.0f} раз меньше. TSTR: {percent(model['tstr'])} против {percent(base['tstr'])}. До настоящих данных она не дотягивает: у них FD {real['fd']:.1f} и TSTR {percent(real['tstr'])}, разница по TSTR {tstr_gap:.1f} п. п.</li>
<li><b>По точности судьи базовая модель выше:</b> {percent(base['accuracy'])} против {percent(model['accuracy'])}, хотя выглядит заметно хуже. Точность показывает только то, что класс узнаётся, но не то, что картинка реалистична. Поэтому нужны несколько метрик.</li>
<li><b>Число шагов задаёт выбор между скоростью и качеством.</b> Лучший FD у {steps_name(summary['chosen_steps'])} ({chosen['fd']:.1f}), но и время наибольшее. У {steps_name(50)} FD {speed.loc[50, 'fd']:.1f}, зато генерация в {chosen['seconds_per_100'] / speed.loc[50, 'seconds_per_100']:.0f} раз быстрее, а обучать сеть заново не нужно.</li>
<li><b>Признаков копирования обучающих картинок нет:</b> расстояния до ближайшей обучающей картинки такие же, как у настоящих новых картинок, или больше.</li>
</ul>
<h3>Ограничения</h3>
<ul>
<li><b>Один запуск обучения.</b> Модель обучена один раз (сид {train.SEED}), поэтому разброс между запусками обучения не измерен.</li>
<li><b>Метрики зависят от судьи.</b> Он обучен на этом же датасете и хорошо видит то, что помогает различать классы, но может не замечать других дефектов.</li>
<li><b>Опыт с шагами смешивает два фактора.</b> 1000 шагов проверялись только в DDPM, а меньшие значения только в DDIM. Поэтому нельзя сказать, что именно улучшило FD: большее число шагов или добавление случайности на каждом шаге.</li>
<li><b>Небольшая выборка в опыте с шагами.</b> На вариант приходится {number(summary['steps_images'])} картинок, поэтому малые различия FD могут быть случайными.</li>
<li><b>Простые данные.</b> Картинки маленькие, серые и отцентрованные. Для больших цветных изображений нужны сеть крупнее и намного больше вычислений.</li>
</ul>

<h2 id="run">Как запустить</h2>
<pre>docker build -t fashion-ddpm .
docker run --rm --user "$(id -u):$(id -g)" -v "$PWD/outputs:/app/outputs" fashion-ddpm</pre>
{flow([("prepare_data.py", "скачивает датасет при сборке образа"),
       ("train.py", "обучает модель: <code>model.pt</code>"),
       ("evaluate.py", "считает метрики и картинки"),
       ("report.py", "собирает этот отчёт")])}
<p>Подробности и команда для генерации своих картинок есть в README.md.</p>
"""


def build():
    data = dict(np.load(ROOT / "data" / "fashion.npz"))
    samples = dict(np.load(OUTPUTS / "samples.npz"))
    history = pd.read_csv(OUTPUTS / "history.csv")
    steps = pd.read_csv(OUTPUTS / "steps.csv")
    comparison = pd.read_csv(OUTPUTS / "comparison.csv")
    classes = pd.read_csv(OUTPUTS / "per_class.csv")
    confusion = pd.read_csv(OUTPUTS / "confusion.csv").to_numpy()
    loss_t = pd.read_csv(OUTPUTS / "loss_by_t.csv")
    summary = json.loads((OUTPUTS / "summary.json").read_text())
    summary["train_minutes"] = history["minutes"].iloc[-1]

    eda = make_figures(data, samples, history, steps, comparison, confusion, classes, loss_t, summary)
    model, base, real, _ = verdict(comparison)
    speed = steps.set_index("steps")["seconds_per_100"]
    close_ratio = speed[summary["chosen_steps"]] / speed[100]

    page = f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Диффузионная модель</title><style>{CSS}</style></head><body><main>
<h1>Генерация изображений одежды диффузионной моделью</h1>
<p class="lead">Лабораторная работа по генеративным моделям. Обучается DDPM на Fashion-MNIST: модель создаёт новые
картинки заданного класса, постепенно убирая шум. Исследуется, сколько шагов нужно для качественной генерации.</p>
<nav><a href="#task">Задача</a><a href="#data">Датасет</a><a href="#method">Метод</a><a href="#metrics">Метрики</a>
<a href="#training">Обучение</a><a href="#results">Результаты</a><a href="#conclusions">Выводы</a><a href="#run">Запуск</a></nav>

<div class="tiles">
{tile(percent(model['accuracy']), f"картинок классификатор относит к заданному классу (реальные: {percent(real['accuracy'])})")}
{tile(f"{model['fd']:.1f}", f"FD с тестом (базовая модель: {base['fd']:.1f}, реальные: {real['fd']:.1f})")}
{tile(percent(model['tstr']), f"точность на реальном тесте после обучения на сгенерированных (на реальных: {percent(real['tstr'])})")}
{tile(summary['chosen_steps'], f"шагов дали лучший FD. У 100 шагов DDIM он хуже, но генерация в {close_ratio:.0f} раз быстрее")}
</div>

<h2 id="task">Задача</h2>
<p><b>Какую задачу решаем.</b> Нужно обучить модель, которая по названию класса создаёт новое изображение этого класса,
например «сумка». Картинка должна выглядеть как настоящая, но её не должно быть в датасете. Это задача условной генерации: выучить распределение картинок p(x | класс) и уметь брать из него новые примеры.</p>
{flow([("Вход", "Класс (например, 8 — сумка) и случайный шум 28×28"),
       ("Модель", "U-Net несколько раз оценивает и убирает шум"),
       ("Выход", "Новая картинка 28×28 этого класса")])}
<p><b>Зачем это нужно.</b> Такие модели создают изображения по описанию и дополняют обучающие данные, когда настоящих примеров мало.
Диффузия лежит в основе большинства современных генераторов изображений. Здесь она показана на маленьких картинках, чтобы обучение шло на обычном процессоре, а каждый шаг можно было разобрать.</p>
<p><b>На какие вопросы отвечает работа:</b></p>
<ol>
<li>Создаёт ли модель картинки нужного класса, похожие на настоящие и не скопированные из датасета?</li>
<li>Лучше ли она простого статистического генератора?</li>
<li>Как число шагов генерации влияет на качество и скорость?</li>
</ol>
{dataset_section(eda)}
{method_section(summary)}
{metrics_section(summary)}
{results_section(steps, comparison, classes, confusion, history, samples, summary)}
{conclusions(comparison, summary, steps)}
</main></body></html>"""
    (OUTPUTS / "report.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    build()
