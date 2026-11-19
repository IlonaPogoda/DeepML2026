"""Minimal multimodal Cross-Attention + multi-task lab, Python stdlib HTTP service."""
import argparse
import csv
import html
import json
import logging
import math
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

EMOTIONS = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
SENTIMENTS = ["negative", "neutral", "positive"]
TOKEN = re.compile(r"[a-z']+|[!?]", re.I)


@dataclass(frozen=True)
class Config:
    mode: str = "serve"
    epochs: int = 6
    batch_size: int = 32
    learning_rate: float = 0.0005
    max_length: int = 128
    vocab_size: int = 6000
    hidden_size: int = 256
    audio_steps: int = 64
    n_mfcc: int = 20
    train_limit: int = 0
    dev_limit: int = 0
    test_limit: int = 0
    seed: int = 42
    port: int = 8000
    data_dir: str = "data"
    results_dir: str = "results"
    # Путь к предобученному текстовому энкодеру (скачивается при docker build).
    # Пустая строка — исходный вариант с nn.Embedding, обучаемым с нуля.
    text_encoder: str = "models/roberta-base"
    encoder_learning_rate: float = 0.00002
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    dropout: float = 0.1
    # Сколько предыдущих реплик диалога подаётся энкодеру как контекст (0 — без контекста).
    context_turns: int = 0


class Log:
    def __init__(self):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        self._logger = logging.getLogger("meld-mtl")

    def info(self, message):
        self._logger.info(message)


def audio_dim(config: Config) -> int:
    # prepare_data.py: [MFCC + log-энергия] + Δ + ΔΔ
    return 3 * (config.n_mfcc + 1)


class TextData:
    """Тексты, аудио и метки -> тензоры.
    tensors() возвращает (ids, attention_mask, query_mask, audio, emotion, sentiment):
    attention_mask — все реальные токены (реплика + контекст) для энкодера,
    query_mask — только токены текущей реплики, они служат Q в cross-attention и усредняются."""

    def __init__(self, config: Config):
        self.config = config
        self.vocab = {"<pad>": 0, "<unk>": 1}
        self.tokenizer = None
        if config.text_encoder:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)

    def read(self, split):
        with (Path(self.config.data_dir) / f"{split}.csv").open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        audio = np.load(Path(self.config.data_dir) / f"{split}_audio.npy")
        assert len(rows) == len(audio), "Text/audio alignment broken"
        expected = (self.config.audio_steps, audio_dim(self.config))
        if audio.ndim != 3 or audio.shape[1:] != expected:
            raise ValueError(f"Invalid audio feature shape: {audio.shape}; expected (N, {expected[0]}, {expected[1]})")
        return rows, audio

    def fit(self, rows):
        if self.tokenizer is not None:
            return  # словарь предобученного токенизатора уже готов
        count = Counter(word for row in rows for word in TOKEN.findall(row["text"].lower()))
        for word, _ in count.most_common(self.config.vocab_size - 2):
            self.vocab[word] = len(self.vocab)

    def context(self, row):
        turns = [t for t in row.get("context", "").split(" \n ") if t]
        return " ".join(turns[-self.config.context_turns:]) if self.config.context_turns > 0 else ""

    def token_ids(self, rows):
        if self.tokenizer is None:
            ids = np.zeros((len(rows), self.config.max_length), dtype=np.int64)
            for i, row in enumerate(rows):
                for j, word in enumerate(TOKEN.findall(row["text"].lower())[:self.config.max_length]):
                    ids[i, j] = self.vocab.get(word, 1)
            mask = ids != 0
            return torch.from_numpy(ids), torch.from_numpy(mask), torch.from_numpy(mask)
        texts = [row["text"] for row in rows]
        contexts = [self.context(row) for row in rows]
        if any(contexts):
            # Пара (реплика, контекст): при переполнении обрезается только контекст.
            encoded = self.tokenizer(texts, [c or " " for c in contexts], max_length=self.config.max_length,
                                     truncation="only_second", padding="max_length", return_tensors="pt")
        else:
            encoded = self.tokenizer(texts, max_length=self.config.max_length, truncation=True,
                                     padding="max_length", return_tensors="pt")
        query = torch.tensor([[sid == 0 for sid in encoded.sequence_ids(i)] for i in range(len(rows))])
        return encoded["input_ids"], encoded["attention_mask"].bool(), query

    def tokens(self, ids, query_mask):
        """Подписи токенов реплики (для оси карты внимания в отчёте)."""
        ids = ids[query_mask].tolist()
        if self.tokenizer is not None:
            return [t.replace("Ġ", "") or "␣" for t in self.tokenizer.convert_ids_to_tokens(ids)]
        inverse = {i: w for w, i in self.vocab.items()}
        return [inverse.get(i, "<unk>") for i in ids]

    def tensors(self, rows, audio):
        ids, attention_mask, query_mask = self.token_ids(rows)
        emotion = np.array([EMOTIONS.index(r["emotion"]) for r in rows])
        sentiment = np.array([SENTIMENTS.index(r["sentiment"]) for r in rows])
        return (ids, attention_mask, query_mask, torch.from_numpy(audio).float(),
                torch.from_numpy(emotion), torch.from_numpy(sentiment))


class CrossMTL(nn.Module):
    def __init__(self, config: Config, vocab_size: int):
        super().__init__()
        h = config.hidden_size
        self.encoder = None
        if config.text_encoder:
            from transformers import AutoModel
            # Контекстные эмбеддинги предобученного RoBERTa, проецированные в размерность h.
            self.encoder = AutoModel.from_pretrained(config.text_encoder)
            self.embedding = nn.Linear(self.encoder.config.hidden_size, h)
        else:
            self.embedding = nn.Embedding(vocab_size, h, padding_idx=0)
        self.audio = nn.Sequential(nn.Linear(audio_dim(config), h), nn.ReLU(), nn.Dropout(config.dropout))
        # Обучаемые позиции аудиокадров: без них K/V для внимания — неупорядоченное множество.
        self.audio_position = nn.Parameter(torch.zeros(1, config.audio_steps, h))
        self.cross = nn.MultiheadAttention(h, num_heads=4, dropout=config.dropout, batch_first=True)
        self.norm = nn.LayerNorm(h)
        self.shared = nn.Sequential(nn.Dropout(config.dropout), nn.Linear(h, h), nn.ReLU(), nn.Dropout(config.dropout))
        self.emotion = nn.Linear(h, len(EMOTIONS))
        self.sentiment = nn.Linear(h, len(SENTIMENTS))

    def forward(self, tokens, attention_mask, query_mask, audio, return_attention=False):
        if self.encoder is not None:
            hidden = self.encoder(input_ids=tokens, attention_mask=attention_mask.long()).last_hidden_state
            text = self.embedding(hidden)
        else:
            text = self.embedding(tokens)
        sound = self.audio(audio) + self.audio_position
        # CROSS-attention: Q=text, K=audio, V=audio
        attended, weights = self.cross(text, sound, sound, need_weights=return_attention)
        fused = self.norm(text + attended)
        mask = query_mask.unsqueeze(-1).to(fused.dtype)
        merged = (fused * mask).sum(1) / mask.sum(1).clamp(min=1)
        shared = self.shared(merged)
        result = self.emotion(shared), self.sentiment(shared)
        return (*result, weights) if return_attention else result


class Trainer:
    def __init__(self, config: Config, data: TextData, log: Log):
        self.config, self.data, self.log = config, data, log
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def batches(self, split, order=None):
        order = torch.arange(len(split[0])) if order is None else order
        for batch in order.split(self.config.batch_size):
            yield tuple(t[batch].to(self.device, non_blocking=True) for t in split)

    def predict(self, model, split):
        """Предсказания по всей выборке батчами + attention первого примера + средний loss."""
        model.eval()
        ce = nn.CrossEntropyLoss(reduction="sum")
        pe_all, ps_all, attention, loss = [], [], None, 0.0
        with torch.no_grad(), torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
            for ids, am, qm, a, e, s in self.batches(split):
                pe, ps, weights = model(ids, am, qm, a, return_attention=True)
                loss += (ce(pe.float(), e) + ce(ps.float(), s)).item()
                if attention is None:
                    attention = weights[0].float().cpu().numpy()
                pe_all.append(pe.argmax(1).cpu())
                ps_all.append(ps.argmax(1).cpu())
        return [torch.cat(pe_all).numpy(), torch.cat(ps_all).numpy()], attention, loss / len(split[0])

    @staticmethod
    def metrics(split, predictions):
        labels = [split[4].numpy(), split[5].numpy()]
        metrics = {}
        for name, true, pred, classes in zip(("emotion", "sentiment"), labels, predictions, (EMOTIONS, SENTIMENTS)):
            ids = list(range(len(classes)))
            metrics[name] = {
                "accuracy": round(float(accuracy_score(true, pred)), 4),
                "macro_f1": round(float(f1_score(true, pred, labels=ids, average="macro", zero_division=0)), 4),
                "weighted_f1": round(float(f1_score(true, pred, labels=ids, average="weighted", zero_division=0)), 4),
                "confusion": confusion_matrix(true, pred, labels=ids).tolist(),
                "classes": classes,
            }
        return metrics

    def run(self):
        random.seed(self.config.seed)
        np.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)
        device_name = torch.cuda.get_device_name() if self.device.type == "cuda" else "CPU"
        self.log.info(f"Device: {self.device}" + (f" ({device_name})" if self.device.type == "cuda" else
                                                  " — GPU не найден, запустите контейнер с --gpus all"))
        started = time.perf_counter()
        train_rows, train_audio = self.data.read("train")
        self.data.fit(train_rows)
        splits = {"train": self.data.tensors(train_rows, train_audio)}
        test_rows = None
        for name in ("dev", "test"):
            rows, audio = self.data.read(name)
            test_rows = rows if name == "test" else test_rows
            splits[name] = self.data.tensors(rows, audio)
        self.log.info("Sizes: " + ", ".join(f"{k}={len(v[0])}" for k, v in splits.items()))
        # Normalize using TRAIN only; apply the same parameters to dev/test.
        train_mfcc = splits["train"][3]
        mean = train_mfcc.mean(dim=(0, 1), keepdim=True)
        std = train_mfcc.std(dim=(0, 1), keepdim=True).clamp_min(1e-5)
        for split_name, (x, am, qm, audio, emotion, sentiment) in splits.items():
            splits[split_name] = (x, am, qm, (audio - mean) / std, emotion, sentiment)
        model = CrossMTL(self.config, len(self.data.vocab)).to(self.device)
        # Предобученный энкодер дообучается с маленьким LR, новые слои (cross-attention, головы) — с обычным.
        encoder_params = list(model.encoder.parameters()) if model.encoder is not None else []
        encoder_ids = {id(p) for p in encoder_params}
        head_params = [p for p in model.parameters() if id(p) not in encoder_ids]
        groups = [{"params": head_params, "lr": self.config.learning_rate}]
        if encoder_params:
            groups.append({"params": encoder_params, "lr": self.config.encoder_learning_rate})
        optim = torch.optim.AdamW(groups, weight_decay=self.config.weight_decay)
        steps_per_epoch = math.ceil(len(splits["train"][0]) / self.config.batch_size)
        total_steps = steps_per_epoch * self.config.epochs
        warmup = int(total_steps * self.config.warmup_ratio)
        # Linear warmup + linear decay — стандартное расписание для дообучения трансформеров.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda step: min((step + 1) / max(1, warmup), max(0.0, (total_steps - step) / max(1, total_steps - warmup))))
        scaler = torch.amp.GradScaler(enabled=self.device.type == "cuda")
        ce = nn.CrossEntropyLoss()
        # history: по эпохам — train-лоссы, dev-лосс, dev-метрики и время; всё уходит в отчёт.
        history, best_score, best_state, best_epoch = [], -1.0, None, 0
        for epoch in range(self.config.epochs):
            epoch_start = time.perf_counter()
            model.train()
            order = torch.randperm(len(splits["train"][0]))
            totals = np.zeros(3)
            for ids, am, qm, a, e, s in self.batches(splits["train"], order):
                with torch.autocast(self.device.type, enabled=self.device.type == "cuda"):
                    pe, ps = model(ids, am, qm, a)
                    le, ls = ce(pe.float(), e), ce(ps.float(), s)
                    loss = le + ls
                optim.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optim)
                scaler.update()
                scheduler.step()
                totals += [loss.item(), le.item(), ls.item()]
            totals /= steps_per_epoch
            # Валидация на dev: по ней выбирается лучшая эпоха (test в выборе не участвует).
            dev_predictions, _, dev_loss = self.predict(model, splits["dev"])
            dev = self.metrics(splits["dev"], dev_predictions)
            score = (dev["emotion"]["weighted_f1"] + dev["sentiment"]["weighted_f1"]) / 2
            history.append({"epoch": epoch + 1, "train_total": totals[0], "train_emotion": totals[1],
                            "train_sentiment": totals[2], "dev_loss": dev_loss,
                            "dev": {k: {m: v[m] for m in ("accuracy", "macro_f1", "weighted_f1")} for k, v in dev.items()},
                            "dev_score": score, "seconds": time.perf_counter() - epoch_start})
            self.log.info(f"Epoch {epoch + 1}/{self.config.epochs}: total={totals[0]:.4f}, emotion={totals[1]:.4f}, "
                          f"sentiment={totals[2]:.4f} | dev loss={dev_loss:.4f}, "
                          f"emotion acc={dev['emotion']['accuracy']:.4f} wF1={dev['emotion']['weighted_f1']:.4f}, "
                          f"sentiment acc={dev['sentiment']['accuracy']:.4f} wF1={dev['sentiment']['weighted_f1']:.4f}")
            if score > best_score:
                best_score, best_epoch = score, epoch + 1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        self.log.info(f"Best dev epoch: {best_epoch} (mean weighted F1={best_score:.4f})")
        predictions, attention, _ = self.predict(model, splits["test"])
        metrics = self.metrics(splits["test"], predictions)
        # Всё, что нужно для подробного отчёта на localhost.
        run_info = {
            "device": device_name,
            "seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
            "best_epoch": best_epoch,
            "sizes": {k: len(v[0]) for k, v in splits.items()},
            "distribution": {k: {"emotion": np.bincount(v[4].numpy(), minlength=len(EMOTIONS)).tolist(),
                                 "sentiment": np.bincount(v[5].numpy(), minlength=len(SENTIMENTS)).tolist()}
                             for k, v in splits.items()},
            "example": {"text": test_rows[0]["text"],
                        "emotion": test_rows[0]["emotion"], "sentiment": test_rows[0]["sentiment"],
                        "pred_emotion": EMOTIONS[predictions[0][0]], "pred_sentiment": SENTIMENTS[predictions[1][0]],
                        "tokens": self.data.tokens(splits["test"][0][0], splits["test"][2][0]),
                        "query_mask": splits["test"][2][0].numpy(),
                        "audio": splits["test"][3][0].numpy(), "attention": attention},
        }
        Report(self.config).write(history, metrics, run_info)
        return metrics


def metrics_from_confusion(confusion, classes):
    """Accuracy, macro/weighted F1 и метрики по классам из матрицы ошибок.
    Одна функция для новой модели и для сохранённых результатов прошлой версии,
    чтобы сравнение было посчитано одинаково."""
    cm = np.asarray(confusion, dtype=float)
    support = cm.sum(1)
    tp = np.diag(cm)
    precision = np.divide(tp, cm.sum(0), out=np.zeros_like(tp), where=cm.sum(0) > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=precision + recall > 0)
    return {"accuracy": tp.sum() / cm.sum(), "macro_f1": f1.mean(),
            "weighted_f1": (f1 * support).sum() / support.sum(),
            "per_class": [{"class": c, "precision": p, "recall": r, "f1": f, "support": int(n)}
                          for c, p, r, f, n in zip(classes, precision, recall, f1, support)]}


class Report:
    """Подробный HTML-отчёт (results/report.html) + графики PNG, которые отдаёт WebService."""

    # Цвета: категориальная палитра в фиксированном порядке; «до» — нейтральный серый.
    COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    BASELINE_COLOR = "#a3a29b"
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    TARGET = 0.6

    def __init__(self, config: Config):
        self.config = config
        self.out = Path(config.results_dir)

    def style(self, ax, title, xlabel="", ylabel="", grid="both"):
        ax.set_title(title, loc="left", fontsize=11, color=self.INK)
        ax.set_xlabel(xlabel, color=self.MUTED)
        ax.set_ylabel(ylabel, color=self.MUTED)
        ax.tick_params(colors=self.MUTED, labelsize=9)
        ax.grid(True, axis=grid, color=self.GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(self.GRID)

    def save(self, fig, name):
        fig.tight_layout()
        fig.savefig(self.out / name, dpi=130, facecolor="white")
        plt.close(fig)

    def loss_chart(self, history):
        epochs = [h["epoch"] for h in history]
        fig, ax = plt.subplots(figsize=(8, 3.8))
        series = [("Train total", "train_total", "-"), ("Train emotion", "train_emotion", "-"),
                  ("Train sentiment", "train_sentiment", "-"), ("Dev total", "dev_loss", "--")]
        for color, (label, key, line) in zip(self.COLORS, series):
            ax.plot(epochs, [h[key] for h in history], line, color=color, linewidth=2, marker="o", markersize=6, label=label)
        ax.axvline(self.best_epoch, color=self.MUTED, linestyle=":", linewidth=1.5, label="Лучшая эпоха (dev)")
        self.style(ax, "Функция потерь по эпохам", "Эпоха", "Cross-entropy")
        ax.set_xticks(epochs)
        ax.legend(frameon=False, fontsize=9, ncol=3)
        self.save(fig, "loss.png")

    def dev_chart(self, history):
        epochs = [h["epoch"] for h in history]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for ax, task in zip(axes, ("emotion", "sentiment")):
            for color, (label, key) in zip(self.COLORS, (("Accuracy", "accuracy"), ("Weighted-F1", "weighted_f1"), ("Macro-F1", "macro_f1"))):
                ax.plot(epochs, [h["dev"][task][key] for h in history], color=color, linewidth=2, marker="o", markersize=6, label=label)
            ax.axhline(self.TARGET, color=self.MUTED, linestyle="--", linewidth=1, label="Цель 0.6")
            ax.axvline(self.best_epoch, color=self.MUTED, linestyle=":", linewidth=1.5)
            self.style(ax, f"Dev: {'эмоции' if task == 'emotion' else 'тональность'}", "Эпоха", "Значение метрики")
            ax.set_xticks(epochs)
            ax.set_ylim(0, 1)
        axes[0].legend(frameon=False, fontsize=9, loc="lower right")
        self.save(fig, "dev_metrics.png")

    def comparison_chart(self, baseline, current):
        labels, before, after = [], [], []
        for task, name in (("emotion", "Эмоции"), ("sentiment", "Тональность")):
            for key, metric in (("accuracy", "Accuracy"), ("weighted_f1", "Weighted-F1"), ("macro_f1", "Macro-F1")):
                labels.append(f"{name}\n{metric}")
                before.append(baseline[task][key])
                after.append(current[task][key])
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10, 3.8))
        width = 0.38
        bars_before = ax.bar(x - width / 2 - 0.01, before, width, color=self.BASELINE_COLOR, label="Было (исходная версия)")
        bars_after = ax.bar(x + width / 2 + 0.01, after, width, color=self.COLORS[0], label="Стало")
        for bars in (bars_before, bars_after):
            ax.bar_label(bars, fmt="%.2f", fontsize=8, color=self.INK, padding=2)
        ax.axhline(self.TARGET, color=self.MUTED, linestyle="--", linewidth=1, label="Цель 0.6")
        self.style(ax, "Метрики на тесте: было и стало", grid="y")
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
        self.save(fig, "comparison.png")

    def per_class_chart(self, baseline, current):
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [7, 3]})
        for ax, task, title in zip(axes, ("emotion", "sentiment"), ("F1 по эмоциям", "F1 по тональности")):
            classes = [c["class"] for c in current[task]["per_class"]]
            y = np.arange(len(classes))
            ax.barh(y - 0.2, [c["f1"] for c in baseline[task]["per_class"]], 0.36, color=self.BASELINE_COLOR, label="Было")
            ax.barh(y + 0.2, [c["f1"] for c in current[task]["per_class"]], 0.36, color=self.COLORS[0], label="Стало")
            self.style(ax, title, "F1", grid="x")
            ax.set_yticks(y, classes)
            ax.invert_yaxis()
            ax.set_xlim(0, 1)
        axes[0].legend(frameon=False, fontsize=9, loc="lower right")
        self.save(fig, "per_class.png")

    def confusion_chart(self, metrics):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [7, 4]})
        for ax, task, title in zip(axes, ("emotion", "sentiment"), ("Эмоции", "Тональность")):
            cm = np.asarray(metrics[task]["confusion"], dtype=float)
            # Нормировка по строкам: доля примеров истинного класса, попавших в каждый предсказанный.
            share = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
            classes = metrics[task]["classes"]
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(j, i, f"{share[i, j]:.2f}\n({int(cm[i, j])})", ha="center", va="center", fontsize=7,
                            color="white" if share[i, j] > 0.55 else self.INK)
            ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(classes)), classes, fontsize=8)
            ax.set_xlabel("Предсказанный класс", color=self.MUTED)
            ax.set_ylabel("Истинный класс", color=self.MUTED)
            ax.set_title(title, loc="left", fontsize=11)
        self.save(fig, "confusion.png")

    def attention_chart(self, example):
        rows = np.flatnonzero(example["query_mask"])
        tokens = example["tokens"]
        fig, ax = plt.subplots(figsize=(10, max(2.5, 0.28 * len(rows) + 1.2)))
        im = ax.imshow(example["attention"][rows], aspect="auto", interpolation="nearest", cmap="Blues")
        ax.set_yticks(range(len(rows)), tokens, fontsize=8)
        ax.set_xlabel("Аудиокадр (шаг по времени)", color=self.MUTED)
        ax.set_ylabel("Токен текста (Q)", color=self.MUTED)
        ax.set_title("Cross-attention: текст Q → аудио K/V (среднее по 4 головам)", loc="left", fontsize=11)
        fig.colorbar(im, ax=ax, label="Вес внимания")
        self.save(fig, "attention.png")

    def audio_chart(self, example):
        n = self.config.n_mfcc
        features = example["audio"].T
        fig, ax = plt.subplots(figsize=(10, 4))
        # Расходящаяся шкала: признаки нормированы (0 = среднее по train).
        im = ax.imshow(features, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
        for edge in (n + 0.5, 2 * (n + 1) - 0.5):
            ax.axhline(edge, color=self.INK, linewidth=1)
        ax.set_yticks([n / 2, n + 1 + n / 2, 2 * (n + 1) + n / 2], ["MFCC + энергия", "Δ (скорость)", "ΔΔ (ускорение)"], fontsize=9)
        ax.set_xlabel("Временной шаг (после обрезки тишины и усреднения по сегментам)", color=self.MUTED)
        ax.set_title("Аудиопризнаки одного клипа (нормированы по статистике train)", loc="left", fontsize=11)
        fig.colorbar(im, ax=ax, label="z-score")
        self.save(fig, "audio_features.png")

    @staticmethod
    def pct(value):
        return f"{value:.3f}"

    def delta(self, before, after):
        diff = after - before
        cls = "up" if diff > 0 else "down"
        return f'<span class="{cls}">{"+" if diff > 0 else ""}{diff:.3f}</span>'

    def table(self, header, rows):
        head = "".join(f"<th>{h}</th>" for h in header)
        body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
        return f'<div class="table"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'

    def write(self, history, metrics, info):
        self.out.mkdir(parents=True, exist_ok=True)
        self.best_epoch = info["best_epoch"]
        classes = {"emotion": EMOTIONS, "sentiment": SENTIMENTS}
        current = {t: metrics_from_confusion(metrics[t]["confusion"], classes[t]) for t in classes}
        baseline_path = Path("baseline_metrics.json")
        baseline_raw = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
        baseline = ({t: metrics_from_confusion(baseline_raw[t]["confusion"], classes[t]) for t in classes}
                    if baseline_raw else None)

        self.loss_chart(history)
        self.dev_chart(history)
        self.confusion_chart(metrics)
        self.attention_chart(info["example"])
        self.audio_chart(info["example"])
        if baseline:
            self.comparison_chart(baseline, current)
            self.per_class_chart(baseline, current)

        names = {"emotion": "Эмоции (7 классов)", "sentiment": "Тональность (3 класса)"}
        # Итоговые плитки
        tiles = ""
        for task in classes:
            acc, wf1 = current[task]["accuracy"], current[task]["weighted_f1"]
            ok = acc >= self.TARGET
            badge = (f'<span class="badge {"good" if ok else "warn"}">{"✓ цель 0.6 достигнута" if ok else "✗ ниже цели 0.6"}</span>')
            change = f'<div class="sub">было {baseline[task]["accuracy"]:.3f} → {self.delta(baseline[task]["accuracy"], acc)}</div>' if baseline else ""
            tiles += (f'<div class="tile"><div class="label">{names[task]} · accuracy</div><div class="value">{acc:.3f}</div>'
                      f'{change}<div class="sub">weighted-F1 {wf1:.3f} · macro-F1 {current[task]["macro_f1"]:.3f}</div>{badge}</div>')

        # Сравнение с прошлой версией
        if baseline:
            rows = []
            for task in classes:
                for key, label in (("accuracy", "Accuracy"), ("weighted_f1", "Weighted-F1"), ("macro_f1", "Macro-F1")):
                    rows.append([names[task], label, self.pct(baseline[task][key]), self.pct(current[task][key]),
                                 self.delta(baseline[task][key], current[task][key])])
            comparison = (self.table(["Задача", "Метрика", "Было", "Стало", "Изменение"], rows)
                          + '<img src="comparison.png" alt="Сравнение метрик было/стало">'
                          + '<img src="per_class.png" alt="F1 по классам было/стало">'
                          + f'<p class="note">«Было» — сохранённый результат исходной версии (<code>baseline_metrics.json</code>): '
                            f'{sum(map(sum, baseline_raw["emotion"]["confusion"]))} тестовых примеров (первые клипы архива). '
                            f'«Стало» — полный тест MELD ({info["sizes"]["test"]} примеров), поэтому новая оценка ещё и надёжнее. '
                            'Weighted-F1 и F1 по классам для старой версии восстановлены из её матрицы ошибок.</p>')
        else:
            comparison = "<p>Файл baseline_metrics.json не найден — сравнение недоступно.</p>"

        # Данные
        dist_rows = []
        for i, name in enumerate(EMOTIONS):
            dist_rows.append([name] + [f'{info["distribution"][s]["emotion"][i]} ({info["distribution"][s]["emotion"][i] / info["sizes"][s]:.1%})'
                                       for s in ("train", "dev", "test")])
        sent_rows = []
        for i, name in enumerate(SENTIMENTS):
            sent_rows.append([name] + [f'{info["distribution"][s]["sentiment"][i]} ({info["distribution"][s]["sentiment"][i] / info["sizes"][s]:.1%})'
                                       for s in ("train", "dev", "test")])
        sizes = info["sizes"]
        data_html = (self.table(["Выборка", "Было", "Стало"], [["Train", 1200, sizes["train"]], ["Dev", 250, sizes["dev"]], ["Test", 250, sizes["test"]]])
                     + "<h3>Распределение эмоций</h3>" + self.table(["Класс", "Train", "Dev", "Test"], dist_rows)
                     + "<h3>Распределение тональности</h3>" + self.table(["Класс", "Train", "Dev", "Test"], sent_rows)
                     + f'<p class="note">Мажоритарный класс neutral — {info["distribution"]["test"]["emotion"][0] / sizes["test"]:.1%} теста: '
                       'модель «всегда neutral» имела бы такую accuracy при macro-F1 ≈ 0.09, поэтому важно смотреть и на F1.</p>')

        # Обучение
        c = self.config
        hyper = [["Устройство", info["device"]], ["Текстовый энкодер", c.text_encoder or "nn.Embedding (с нуля)"],
                 ["Эпох (лучшая по dev)", f'{len(history)} ({info["best_epoch"]})'], ["Batch size", c.batch_size],
                 ["LR новых слоёв / RoBERTa", f"{c.learning_rate} / {c.encoder_learning_rate}"],
                 ["Оптимизатор", f"AdamW, weight decay {c.weight_decay}"], ["Расписание LR", f"warmup {c.warmup_ratio:.0%} + линейное затухание"],
                 ["Dropout", c.dropout], ["Hidden size / головы внимания", f"{c.hidden_size} / 4"],
                 ["Макс. длина текста", c.max_length], ["Контекст диалога (реплик)", c.context_turns],
                 ["Аудио: шаги × признаки", f"{c.audio_steps} × {audio_dim(c)}"],
                 ["Параметров модели", f'{info["parameters"] / 1e6:.1f} млн'],
                 ["Общее время (данные + обучение + тест)", f'{info["seconds"] / 60:.1f} мин']]
        epoch_rows = [[h["epoch"] if h["epoch"] != info["best_epoch"] else f'<b>{h["epoch"]} ★</b>',
                       f'{h["train_total"]:.4f}', f'{h["dev_loss"]:.4f}',
                       f'{h["dev"]["emotion"]["accuracy"]:.3f}', f'{h["dev"]["emotion"]["weighted_f1"]:.3f}',
                       f'{h["dev"]["sentiment"]["accuracy"]:.3f}', f'{h["dev"]["sentiment"]["weighted_f1"]:.3f}',
                       f'{h["seconds"]:.0f} с'] for h in history]
        training = (self.table(["Параметр", "Значение"], hyper)
                    + '<img src="loss.png" alt="Кривые потерь">'
                    + '<p class="note">Сплошные линии — средний train-loss за эпоху (на нём идёт оптимизация), пунктир — loss на dev. '
                      'Рост dev-loss при падающем train-loss — признак переобучения; поэтому для теста берутся веса лучшей по dev эпохи.</p>'
                    + '<img src="dev_metrics.png" alt="Метрики на dev по эпохам">'
                    + "<h3>По эпохам</h3>"
                    + self.table(["Эпоха", "Train loss", "Dev loss", "Dev эмоции acc", "Dev эмоции wF1",
                                  "Dev тональность acc", "Dev тональность wF1", "Время"], epoch_rows)
                    + '<p class="note">★ — эпоха с лучшим средним weighted-F1 на dev; её веса используются для теста.</p>')

        # Результаты на тесте по классам
        results = ""
        for task in classes:
            rows = [[p["class"], f'{p["precision"]:.3f}', f'{p["recall"]:.3f}', f'{p["f1"]:.3f}', p["support"]]
                    for p in current[task]["per_class"]]
            results += f"<h3>{names[task]}</h3>" + self.table(["Класс", "Precision", "Recall", "F1", "Примеров"], rows)
        results += ('<img src="confusion.png" alt="Матрицы ошибок">'
                    '<p class="note">Строки — истинные классы, столбцы — предсказанные; в ячейке доля строки и (число примеров).</p>')

        ex = info["example"]
        example = (f'<p>Пример из теста: <q>{html.escape(ex["text"])}</q> — истинные метки <b>{ex["emotion"]}</b> / <b>{ex["sentiment"]}</b>, '
                   f'предсказание <b>{ex["pred_emotion"]}</b> / <b>{ex["pred_sentiment"]}</b>.</p>')

        template = Path("report.html").read_text(encoding="utf-8")
        page = (template.replace("{{TILES}}", tiles).replace("{{COMPARISON}}", comparison)
                .replace("{{DATA}}", data_html).replace("{{TRAINING}}", training)
                .replace("{{RESULTS}}", results).replace("{{EXAMPLE}}", example)
                .replace("{{EPOCHS}}", f'{len(history)} (лучшая по dev: {info["best_epoch"]})')
                .replace("{{DEVICE}}", html.escape(info["device"])))
        (self.out / "report.html").write_text(page, encoding="utf-8")
        (self.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (self.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


class WebService:
    def __init__(self, config: Config):
        self.config = config

    def serve(self):
        root = Path(self.config.results_dir).resolve()

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                paths = {"/": ("report.html", "text/html; charset=utf-8"),
                         "/report.html": ("report.html", "text/html; charset=utf-8"),
                         "/metrics": ("metrics.json", "application/json"),
                         "/history": ("history.json", "application/json")}
                # Все графики отчёта (*.png) из папки results; имя без путей — без выхода за её пределы.
                if re.fullmatch(r"/[\w.-]+\.png", self.path):
                    paths[self.path] = (self.path[1:], "image/png")
                if self.path not in paths or not (root / paths[self.path][0]).is_file():
                    self.send_error(404)
                    return
                filename, content_type = paths[self.path]
                payload = (root / filename).read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer(("0.0.0.0", self.config.port), Handler)
        print(f"Report: http://localhost:{self.config.port}", flush=True)
        server.serve_forever()


def load_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--mode", choices=["train", "serve"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--encoder-learning-rate", type=float)
    parser.add_argument("--context-turns", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    values = json.loads(Path(args.config).read_text())
    for key in ("mode", "epochs", "batch_size", "learning_rate", "encoder_learning_rate", "context_turns", "seed", "port"):
        val = getattr(args, key)
        if val is not None:
            values[key] = val
    return Config(**values)


if __name__ == "__main__":
    cfg = load_config()
    logger = Log()
    result = Trainer(cfg, TextData(cfg), logger).run()
    logger.info(f"Test metrics: {result}")
    if cfg.mode == "serve":
        WebService(cfg).serve()
