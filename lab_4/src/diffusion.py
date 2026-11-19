"""Диффузионная модель: сеть U-Net и формулы зашумления и генерации."""
import math

import torch
from torch import nn

T = 1000           # сколько шагов шума в прямом процессе
BETA_START = 1e-4  # доля шума на первом шаге
BETA_END = 0.02    # доля шума на последнем шаге
NUM_CLASSES = 10

# Расписание шума. alpha_bar[t] — какая доля исходной картинки (по дисперсии) осталась к шагу t.
betas = torch.linspace(BETA_START, BETA_END, T)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)


def add_noise(x0, t, noise):
    a = alpha_bar[t].view(-1, 1, 1, 1)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise


def time_embedding(t, size):
    """Превращает номер шага в вектор из синусов и косинусов разной частоты."""
    half = size // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half) / half)
    angles = t[:, None].float() * freqs[None]
    return torch.cat([angles.sin(), angles.cos()], dim=1)


class ResBlock(nn.Module):
    """Две свёртки, между ними добавляется информация о шаге и классе."""

    def __init__(self, in_channels, out_channels, cond_size):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.cond = nn.Linear(cond_size, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(torch.relu(self.norm1(x)))
        h = h + self.cond(cond)[:, :, None, None]
        h = self.conv2(torch.relu(self.norm2(h)))
        return h + self.skip(x)


class UNet(nn.Module):
    """Получает шумную картинку, шаг t и класс, возвращает оценку добавленного шума.

    Размеры: 28×28 → 14×14 → 7×7 → 14×14 → 28×28.
    Skip-соединения передают мелкие детали с пути вниз на путь вверх.
    """

    def __init__(self, base=32):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        cond_size = base * 4
        self.base = base
        self.time_mlp = nn.Sequential(nn.Linear(base, cond_size), nn.ReLU(), nn.Linear(cond_size, cond_size))
        self.class_embedding = nn.Embedding(NUM_CLASSES, cond_size)

        self.first = nn.Conv2d(1, c1, 3, padding=1)
        self.down1 = ResBlock(c1, c1, cond_size)
        self.pool1 = nn.Conv2d(c1, c2, 3, stride=2, padding=1)
        self.down2 = ResBlock(c2, c2, cond_size)
        self.pool2 = nn.Conv2d(c2, c3, 3, stride=2, padding=1)
        self.middle1 = ResBlock(c3, c3, cond_size)
        self.middle2 = ResBlock(c3, c3, cond_size)
        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.dec2 = ResBlock(c2 * 2, c2, cond_size)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.dec1 = ResBlock(c1 * 2, c1, cond_size)
        self.last = nn.Sequential(nn.GroupNorm(8, c1), nn.ReLU(), nn.Conv2d(c1, 1, 3, padding=1))

    def forward(self, x, t, y):
        cond = self.time_mlp(time_embedding(t, self.base)) + self.class_embedding(y)
        h1 = self.down1(self.first(x), cond)                      # 28×28
        h2 = self.down2(self.pool1(h1), cond)                     # 14×14
        h = self.middle2(self.middle1(self.pool2(h2), cond), cond)  # 7×7
        h = self.dec2(torch.cat([self.up2(h), h2], dim=1), cond)  # 14×14
        h = self.dec1(torch.cat([self.up1(h), h1], dim=1), cond)  # 28×28
        return self.last(h)


def timesteps_for(steps):
    # Выбирает равномерно steps шагов из T, от большего к меньшему
    return torch.linspace(T - 1, 0, steps).round().long()


@torch.no_grad()
def sample(model, labels, steps=50, seed=0, keep_every=None):
    """Генерация: начинает с чистого шума и убирает его за steps шагов.

    steps == T — классический DDPM: на каждом шаге добавляется немного случайности.
    steps < T  — DDIM: те же веса, но шаги длиннее, а случайность не добавляется.
    keep_every — если задано, вернёт ещё и промежуточные картинки.
    Возвращает тензор со значениями примерно от -1 до 1.
    """
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(len(labels), 1, 28, 28, generator=generator)
    times = timesteps_for(steps)
    history = []
    for i, t in enumerate(times):
        t_batch = torch.full((len(labels),), int(t))
        # x -текущие зашумленные unet картинки, t_batch - текущий шаг шума, lables - классы
        predicted_noise = model(x, t_batch, labels) 
        a = alpha_bar[t]
        if steps == T:
            mean = (x - betas[t] / (1 - a).sqrt() * predicted_noise) / alphas[t].sqrt()
            noise = torch.randn(x.shape, generator=generator)
            x = mean + betas[t].sqrt() * noise if t > 0 else mean
        else:
            x0 = ((x - (1 - a).sqrt() * predicted_noise) / a.sqrt()).clamp(-1, 1)
            if i + 1 < steps:
                a_prev = alpha_bar[times[i + 1]]
                # Заново вычисляем шум после обрезки x0, чтобы шаг оставался согласованным.
                noise_estimate = (x - a.sqrt() * x0) / (1 - a).sqrt()
                x = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * noise_estimate
            else:
                x = x0
        if keep_every and (i % keep_every == 0 or i == steps - 1):
            history.append(x.clone())
    return (x, history) if keep_every else x
