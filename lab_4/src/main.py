"""Запускает весь конвейер: обучение → оценка → отчёт.

Готовые этапы пропускаются: обучение занимает больше получаса, а отчёт можно
пересобрать за минуту. Чтобы пройти этап заново, удалите его файл:
outputs/model.pt — обучение, outputs/summary.json — оценка.
"""
import evaluate
import report
import train
from train import OUTPUTS


def main():
    OUTPUTS.mkdir(exist_ok=True)
    if (OUTPUTS / "model.pt").exists():
        print("Найден outputs/model.pt: обучение пропущено.", flush=True)
    else:
        train.main()

    if (OUTPUTS / "summary.json").exists():
        print("Найден outputs/summary.json: оценка пропущена.", flush=True)
    else:
        evaluate.main()

    report.build()
    print("Готово: outputs/report.html", flush=True)


if __name__ == "__main__":
    main()
