import argparse
import csv
import json
import subprocess
import tarfile
import tempfile
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.fft import dct
from scipy.signal import stft

SAMPLE_RATE = 16000
MAX_CONTEXT_TURNS = 3
# Типографские символы MELD (’ “ ” — …) приводим к ASCII, чтобы токенизатор
TEXT_FIXES = {"’": "'", "‘": "'", "“": '"', "”": '"', "—": " - ", "–": " - ", "…": "...", "\x92": "'", "\x85": "..."}


def clean_text(text: str) -> str:
    for bad, good in TEXT_FIXES.items():
        text = text.replace(bad, good)
    return " ".join(text.split())


def mel_filterbank(sample_rate: int = 16000, n_fft: int = 512, n_mels: int = 40) -> np.ndarray:
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_points = np.linspace(hz_to_mel(0), hz_to_mel(sample_rate / 2), n_mels + 2)
    frequencies = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    edges = mel_to_hz(mel_points)
    bank = np.zeros((n_mels, len(frequencies)), dtype=np.float32)
    for i in range(n_mels):
        left, center, right = edges[i:i + 3]
        up = (frequencies - left) / max(center - left, 1e-8)
        down = (right - frequencies) / max(right - center, 1e-8)
        bank[i] = np.maximum(0.0, np.minimum(up, down))
    return bank


def deltas(features: np.ndarray, width: int = 2) -> np.ndarray:
    """Регрессионные дельты (как в HTK/librosa) по оси времени: скорость изменения признаков."""
    padded = np.pad(features, ((width, width), (0, 0)), mode="edge")
    frames = len(features)
    out = sum(w * padded[width + w:width + w + frames] for w in range(-width, width + 1) if w)
    return (out / (2 * sum(w * w for w in range(1, width + 1)))).astype(np.float32)


def trim_silence(frames_db: np.ndarray, top_db: float = 35.0) -> slice:
    """Границы речи: отбрасываем начальные/конечные кадры тише максимума на top_db дБ."""
    voiced = np.flatnonzero(frames_db > frames_db.max() - top_db)
    if voiced.size == 0:
        return slice(0, len(frames_db))
    return slice(voiced[0], voiced[-1] + 1)


def pool_steps(matrix: np.ndarray, steps: int) -> np.ndarray:
    """Приводим переменное число кадров к фиксированному steps.
    Длинные записи усредняем по равным сегментам (в отличие от интерполяции,
    учитываются все кадры, а не только соседние к узлам). Короткие интерполируем."""
    if len(matrix) >= steps:
        return np.stack([chunk.mean(0) for chunk in np.array_split(matrix, steps)])
    old_positions = np.linspace(0.0, 1.0, len(matrix))
    new_positions = np.linspace(0.0, 1.0, steps)
    return np.stack([np.interp(new_positions, old_positions, matrix[:, i])
                     for i in range(matrix.shape[1])], axis=1)


def mfcc_from_wave(wave: np.ndarray, steps: int, n_mfcc: int = 20, sample_rate: int = 16000) -> np.ndarray:
    """Кадровые признаки (steps, 3 * (n_mfcc + 1)): [MFCC, log-энергия] + Δ + ΔΔ."""
    if steps <= 0 or not 1 <= n_mfcc <= 40:
        raise ValueError("steps must be > 0 and n_mfcc must be between 1 and 40")
    wave = np.asarray(wave, dtype=np.float32)
    if not wave.size:
        raise ValueError("empty audio")
    # Удаляем DC-смещение и выравниваем громкость клипа (пиковая нормализация).
    wave = wave - wave.mean()
    wave = wave / max(np.abs(wave).max(), 1e-6)
    # Pre-emphasis: усиливает высокие частоты, где у речи меньше энергии.
    wave = np.append(wave[0], wave[1:] - 0.97 * wave[:-1])
    win_length = int(sample_rate * 0.025)
    hop_length = int(sample_rate * 0.010)
    if wave.size < win_length:
        wave = np.pad(wave, (0, win_length - wave.size))
    _, _, z = stft(wave, fs=sample_rate, window="hann", nperseg=win_length,
                   noverlap=win_length - hop_length, nfft=512, boundary=None, padded=False)
    power = (np.abs(z) ** 2).astype(np.float32)
    # Обрезаем тишину/шум по краям, чтобы временные шаги приходились на речь.
    energy_db = 10.0 * np.log10(np.maximum(power.sum(0), 1e-10))
    span = trim_silence(energy_db)
    power, energy_db = power[:, span], energy_db[span]
    mel_energies = mel_filterbank(sample_rate) @ power
    log_mel = np.log(np.maximum(mel_energies, 1e-10))
    mfcc = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc].T
    # Log-энергия кадра несёт громкость/интенсивность — важный признак эмоций (злость, радость).
    base = np.concatenate([mfcc, energy_db[:, None] / 10.0], axis=1).astype(np.float32)
    full = np.concatenate([base, deltas(base), deltas(deltas(base))], axis=1)
    return pool_steps(full, steps).astype(np.float32)


def features(mp4: bytes, steps: int, n_mfcc: int = 20) -> np.ndarray:
    with tempfile.TemporaryDirectory() as folder:
        video = Path(folder) / "clip.mp4"
        video.write_bytes(mp4)
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video), "-vn",
               "-f", "f32le", "-ac", "1", "-ar", str(SAMPLE_RATE), "pipe:1"]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise ValueError(f"ffmpeg: {result.stderr.decode(errors='replace')[:160]}")
    wave = np.frombuffer(result.stdout, dtype="<f4")
    return mfcc_from_wave(wave, steps=steps, n_mfcc=n_mfcc)


def safe_features(key: str, mp4: bytes, steps: int, n_mfcc: int):
    """Обёртка для пула процессов: ошибку декодирования возвращаем, а не бросаем."""
    try:
        return key, features(mp4, steps, n_mfcc), None
    except (ValueError, subprocess.CalledProcessError) as error:
        return key, None, str(error)


def dialogue_context(records: list, turns: int) -> dict:
    """Для каждой реплики — до `turns` предыдущих реплик того же диалога ("Speaker: text").
    Берётся только текст из аннотаций (без меток), поэтому утечки ответа нет."""
    context, history = {}, {}
    for r in sorted(records, key=lambda r: (int(r["Dialogue_ID"]), int(r["Utterance_ID"]))):
        past = history.setdefault(r["Dialogue_ID"], [])
        context[(r["Dialogue_ID"], r["Utterance_ID"])] = " \n ".join(past[-turns:]) if turns else ""
        past.append(f"{r['Speaker']}: {clean_text(r['Utterance'])}")
    return context


def run(archive: Path, annotations: Path, output: Path, config: dict, workers: int):
    output.mkdir(parents=True, exist_ok=True)
    # limit <= 0 весь сплит (по умолчанию используем полный MELD)
    targets = {s: config.get(f"{s}_limit", 0) for s in ("train", "dev", "test")}
    with tarfile.open(archive, "r:gz") as outer:
        names = {Path(m.name).name: m for m in outer if m.isfile()}
        for split, limit in targets.items():
            with (annotations / f"{split}_sent_emo.csv").open(encoding="utf-8-sig", newline="") as csv_file:
                records = list(csv.DictReader(csv_file))
            wanted = {f"dia{r['Dialogue_ID']}_utt{r['Utterance_ID']}.mp4": r for r in records}
            # Храним до MAX_CONTEXT_TURNS предыдущих реплик; сколько из них использовать,
            # решает main.py (context_turns), без пересборки образа.
            context = dialogue_context(records, MAX_CONTEXT_TURNS)
            inner_name = f"{split}.tar.gz"
            inner_member = names.get(inner_name)
            if inner_member is None:
                raise FileNotFoundError(f"Audio archive {inner_name} not found; available: {list(names)[:20]}")
            with tempfile.TemporaryFile() as temp:
                source = outer.extractfile(inner_member)
                if source is None:
                    raise RuntimeError(f"Missing {inner_name}")
                while chunk := source.read(4 * 1024 * 1024):
                    temp.write(chunk)
                temp.seek(0)
                results = {}
                failures, seen, example_error = 0, 0, None

                def collect(future):
                    nonlocal failures, example_error
                    key, matrix, error = future.result()
                    if matrix is None:
                        failures += 1
                        example_error = example_error or error
                    else:
                        results[key] = matrix

                # Декодирование ffmpeg + MFCC — узкое место; распараллеливаем по ядрам CPU
                # (во время docker build GPU недоступен). Очередь ограничена, чтобы не держать
                # в памяти все mp4 сразу.
                with ProcessPoolExecutor(max_workers=workers) as pool, \
                        tarfile.open(fileobj=temp, mode="r:gz") as clips:
                    pending = deque()
                    for member in clips:
                        key = Path(member.name).name
                        if key not in wanted or not member.isfile() or key in results:
                            continue
                        stream = clips.extractfile(member)
                        if stream is None:
                            continue
                        seen += 1
                        pending.append(pool.submit(safe_features, key, stream.read(),
                                                   config["audio_steps"], config["n_mfcc"]))
                        while len(pending) > workers * 4:
                            collect(pending.popleft())
                        if 0 < limit <= seen:
                            break
                    while pending:
                        collect(pending.popleft())
            # Сохраняем в порядке аннотаций (диалог, реплика), а не в порядке архива.
            rows, audio = [], []
            for key, label in wanted.items():
                if key not in results:
                    continue
                rows.append({"text": clean_text(label["Utterance"]), "emotion": label["Emotion"].lower(),
                             "sentiment": label["Sentiment"].lower(), "clip": key,
                             "context": context[(label["Dialogue_ID"], label["Utterance_ID"])]})
                audio.append(results[key])
            print(f"{split}: valid={len(rows)}/{len(wanted)}, matched={seen}, "
                  f"decode_errors={failures}; example_error={example_error}", flush=True)
            if not rows:
                raise RuntimeError(f"No matching audio clips in {split}")
            if len(rows) < 30:
                raise RuntimeError(f"Too few usable {split} examples: {len(rows)}; check clip paths and ffmpeg diagnostics")
            with (output / f"{split}.csv").open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["text", "emotion", "sentiment", "clip", "context"])
                writer.writeheader()
                writer.writerows(rows)
            np.save(output / f"{split}_audio.npy", np.stack(audio))
            print(f"{split}: {len(rows)} real text+audio pairs cached, audio shape {np.stack(audio).shape}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for name in ("archive", "output", "annotations", "config"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run(args.archive, args.annotations, args.output, json.loads(args.config.read_text()), args.workers)
