"""Простая свёрточная сеть-классификатор. Нужна только для измерения качества генерации."""
import copy

import torch
from torch import nn

FEATURE_SIZE = 64


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14×14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 7×7
            nn.Flatten(), nn.Linear(64 * 7 * 7, FEATURE_SIZE), nn.ReLU(),
        )
        self.head = nn.Linear(FEATURE_SIZE, 10)

    def forward(self, x):
        return self.head(self.body(x))


@torch.no_grad()
def predict(model, images, batch_size=1000):
    model.eval()
    return torch.cat([model(images[i:i + batch_size]).argmax(1) for i in range(0, len(images), batch_size)])


@torch.no_grad()
def features(model, images, batch_size=1000):
    """Вектор из 64 чисел на картинку: что сеть «видит» перед самым выбором класса."""
    model.eval()
    return torch.cat([model.body(images[i:i + batch_size]) for i in range(0, len(images), batch_size)]).numpy()


def accuracy(model, images, labels):
    return (predict(model, images) == labels).float().mean().item()


def fit(images, labels, epochs, seed, val=None):
    """Обучает классификатор. Если передана валидация, возвращает лучшие на ней веса."""
    torch.manual_seed(seed)
    model = Classifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_acc, best_weights = -1.0, None
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(images))
        for start in range(0, len(order), 128):
            batch = order[start:start + 128]
            loss = nn.functional.cross_entropy(model(images[batch]), labels[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if val is not None:
            acc = accuracy(model, *val)
            if acc > best_acc:
                best_acc, best_weights = acc, copy.deepcopy(model.state_dict())
    if val is not None:
        model.load_state_dict(best_weights)
    return model
