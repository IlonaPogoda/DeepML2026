"""Агент REINFORCE: нейросеть сразу выдаёт вероятности действий.

Идея обучения: действия, после которых прибыль оказалась выше средней, агент
выбирает чаще, а действия с результатом ниже среднего — реже.
"""
import numpy as np
import torch
from torch import nn


class ReinforceAgent:
    def __init__(self, state_size, n_actions, gamma, lr=3e-3, entropy_coef=0.01):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.policy = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.Tanh(),
            nn.Linear(64, n_actions),  # на выходе логиты, softmax применяется ниже
        )
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def probabilities(self, state: np.ndarray) -> torch.Tensor:
        with torch.no_grad():
            return torch.softmax(self.policy(torch.from_numpy(state)), dim=-1)

    def sample_action(self, state: np.ndarray) -> int:
        """При обучении действие случайно: так агент исследует разные варианты."""
        return int(torch.multinomial(self.probabilities(state), 1))

    def best_action(self, state: np.ndarray) -> int:
        """При проверке выбирается самое вероятное действие."""
        return int(torch.argmax(self.probabilities(state)))

    # вычисляет дисконтированную сумму наград для эпизода 60 дн
    def returns_to_go(self, rewards: list) -> np.ndarray:
        """Для каждого дня считает G_t = r_t + gamma * r_(t+1) + gamma^2 * r_(t+2) + ..."""
        returns = np.zeros(len(rewards), dtype=np.float32)
        total = 0.0
        for t in reversed(range(len(rewards))):
            total = rewards[t] + self.gamma * total
            returns[t] = total
        return returns

    def update(self, episodes: list) -> float:
        """Один шаг обучения по пачке эпизодов. Эпизод — это (states, actions, rewards)."""
        states = torch.from_numpy(np.concatenate([e[0] for e in episodes]))
        actions = torch.from_numpy(np.concatenate([e[1] for e in episodes]))
        returns = torch.from_numpy(
            np.concatenate([self.returns_to_go(e[2]) for e in episodes])
        )

        # Вычитаем среднее: теперь важно, лучше или хуже среднего сработало действие.
        advantages = (returns - returns.mean()) / (returns.std() + 1e-8)

        dist = torch.distributions.Categorical(logits=self.policy(states))
        policy_loss = -(dist.log_prob(actions) * advantages).mean()
        # Бонус за энтропию не даёт агенту слишком рано перестать исследовать.
        loss = policy_loss - self.entropy_coef * dist.entropy().mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss)
