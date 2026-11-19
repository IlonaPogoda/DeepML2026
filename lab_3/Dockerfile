FROM python:3.11-slim
WORKDIR /app
ENV MPLCONFIGDIR=/tmp
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
RUN python src/prepare_data.py
CMD ["python", "src/train.py"]
