FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates ffmpeg gawk tar && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# CUDA-сборка PyTorch: обучение идёт на GPU (docker run --gpus all), без GPU работает и на CPU.
# Wheel (~780 МБ) качается curl'ом с докачкой (-C -) в кэш BuildKit: прямое скачивание pip'ом
# с download.pytorch.org часто обрывается по таймауту. Установка CUDA-зависимостей (nvidia-*)
# повторяется: уже скачанные wheel'ы остаются в кэше pip, и каждая попытка продвигается дальше.
ARG TORCH_WHEEL=torch-2.5.1%2Bcu121-cp311-cp311-linux_x86_64.whl
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/wheel-cache \
    until pip install --timeout 120 --retries 10 -r requirements.txt; do sleep 5; done \
    && until python -c "import zipfile; zipfile.ZipFile('/wheel-cache/torch.partial')" 2>/dev/null; do \
         curl -fL --retry 10 --retry-all-errors -C - \
           "https://download-r2.pytorch.org/whl/cu121/${TORCH_WHEEL}" -o /wheel-cache/torch.partial \
         || { echo "torch download interrupted, resuming"; sleep 5; }; \
       done \
    && cp /wheel-cache/torch.partial /tmp/torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl \
    && for attempt in 1 2 3 4 5 6 7 8 9 10; do \
         pip install --timeout 120 --retries 10 /tmp/torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl && break; \
         echo "pip attempt $attempt failed, retrying (downloaded wheels stay in cache)"; sleep 5; \
       done \
    && python -c "import torch; print('torch', torch.__version__)" \
    && rm /tmp/torch-2.5.1+cu121-cp311-cp311-linux_x86_64.whl
# Предобученный текстовый энкодер скачивается при сборке, чтобы запуск не требовал интернета.
# Файлы качаются в кэш BuildKit (недокачанный файл докачивается при повторе), шаг повторяется
# при обрывах сети, затем снапшот копируется в models/roberta-base.
RUN --mount=type=cache,target=/root/.cache/huggingface \
    for attempt in 1 2 3 4 5 6 7 8 9 10; do \
      HF_HUB_OFFLINE=0 HF_HUB_DOWNLOAD_TIMEOUT=60 HF_HUB_ETAG_TIMEOUT=60 python -c "import shutil; \
from huggingface_hub import snapshot_download; \
path = snapshot_download('FacebookAI/roberta-base', allow_patterns=['config.json', 'model.safetensors', \
'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt']); \
shutil.copytree(path, 'models/roberta-base', dirs_exist_ok=True)" && break; \
      echo "roberta download attempt $attempt failed, retrying"; sleep 5; \
    done \
    && test -s models/roberta-base/model.safetensors
COPY config.json prepare_data.py ./
ARG MELD_URL=https://huggingface.co/datasets/declare-lab/MELD/resolve/main/MELD.Raw.tar.gz
ARG MELD_MIRROR=http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz
# BuildKit cache persists the large archive across subsequent build attempts.
# Архив (~10 ГБ) докачивается с места обрыва (curl -C -): недокачанный .partial лежит в кэше
# BuildKit и переживает даже упавшую сборку. При ошибке источники чередуются (HF / зеркало umich).
# Встроенный --retry curl не используем: он откатывает файл к размеру на старте и теряет прогресс,
# поэтому повтор делает внешний цикл, и каждый новый curl -C - продолжает с текущего размера.
RUN --mount=type=cache,target=/download-cache \
    mkdir -p data results annotations \
    && attempt=0; \
    while [ ! -s /download-cache/MELD.Raw.tar.gz ]; do \
      attempt=$((attempt + 1)); \
      if [ $((attempt % 2)) -eq 1 ]; then url="$MELD_URL"; else url="$MELD_MIRROR"; fi; \
      echo "MELD download attempt $attempt from $url"; \
      if curl -fL --speed-limit 50000 --speed-time 120 -C - \
           "$url" -o /download-cache/MELD.Raw.tar.gz.partial; then \
        if gzip -t /download-cache/MELD.Raw.tar.gz.partial; then \
          mv /download-cache/MELD.Raw.tar.gz.partial /download-cache/MELD.Raw.tar.gz; \
        else echo "archive corrupted, restarting download"; rm -f /download-cache/MELD.Raw.tar.gz.partial; fi; \
      elif [ "$attempt" -ge 100 ]; then echo "MELD download failed"; exit 1; \
      else sleep 10; fi; \
    done \
    && for split in train dev test; do \
         curl -fLsS --retry 3 \
           "https://raw.githubusercontent.com/declare-lab/MELD/master/data/MELD/${split}_sent_emo.csv" \
           -o "annotations/${split}_sent_emo.csv"; \
       done \
    && python prepare_data.py --archive /download-cache/MELD.Raw.tar.gz \
        --annotations annotations --output data --config config.json --workers "$(nproc)" \
    && awk 'END {print "Cached CSV rows (including header):", NR}' data/train.csv \
    && rm -rf annotations
COPY main.py report.html baseline_metrics.json ./
EXPOSE 8000
CMD ["python", "main.py"]
