"""Показ работы модели: генерирует картинки нужного класса и сохраняет их в PNG.

Пример: python src/generate.py --label 9 --count 16 --steps 50 --seed 7
Классы: 0 футболка, 1 брюки, 2 свитер, 3 платье, 4 пальто,
        5 сандалия, 6 рубашка, 7 кроссовок, 8 сумка, 9 ботинок.
"""
import argparse

import numpy as np
import torch
from PIL import Image

from diffusion import T, UNet, sample
from train import OUTPUTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", type=int, default=9, choices=range(10), help="класс от 0 до 9")
    parser.add_argument("--count", type=int, default=16, help="сколько картинок")
    parser.add_argument("--steps", type=int, default=50, help=f"число шагов от 1 до {T}")
    parser.add_argument("--seed", type=int, default=0, help="другой сид даёт другие картинки")
    args = parser.parse_args()

    checkpoint = torch.load(OUTPUTS / "model.pt", weights_only=True)
    model = UNet(checkpoint["base"])
    model.load_state_dict(checkpoint["weights"])

    labels = torch.full((args.count,), args.label)
    images = sample(model, labels, args.steps, seed=args.seed).clamp(-1, 1)
    pixels = 255 - ((images[:, 0] + 1) * 127.5).round().numpy().astype(np.uint8)

    # Складываем картинки в таблицу и увеличиваем в 4 раза, иначе 28×28 почти не видно.
    columns = int(np.ceil(np.sqrt(args.count)))
    rows = int(np.ceil(args.count / columns))
    sheet = np.full((rows * 28, columns * 28), 255, dtype=np.uint8)
    for i, image in enumerate(pixels):
        sheet[(i // columns) * 28:(i // columns + 1) * 28, (i % columns) * 28:(i % columns + 1) * 28] = image
    path = OUTPUTS / f"generated_label{args.label}_steps{args.steps}_seed{args.seed}.png"
    Image.fromarray(sheet).resize((columns * 112, rows * 112), Image.NEAREST).save(path)
    print(f"Сохранено: {path}")


if __name__ == "__main__":
    main()
