"""Среда: сервис проката решает, сколько велосипедов держать на улицах.

Один шаг среды — один день:
1. Утром агент видит состояние и меняет размер парка.
2. Днём приходит реальный спрос из датасета.
3. Вечером агент получает награду — потери относительно идеального дня.
"""
import numpy as np
import pandas as pd

ACTIONS = np.array([-1000, -250, 0, 250, 1000])  # изменение парка за утро
MAX_FLEET = 10000  # больше велосипедов в парке быть не может

PRICE = 1.0  # доход от одной аренды, у. е.
BIKE_COST = 0.3  # обслуживание одного велосипеда на улице за день
MOVE_COST = 0.1  # перевозка одного велосипеда на улицу или на склад

STATE_COLUMNS = [
    "temp", "hum", "windspeed", "workingday", "holiday",
    "weather_1", "weather_2", "weather_3", "year_sin", "year_cos",
]
STATE_SIZE = 5 + len(STATE_COLUMNS)  # 5 признаков парка и спроса + колонки выше


def round_fleet(value: float) -> int:
    """Округляет размер парка до шага 250 велосипедов."""
    return int(round(value / 250) * 250)


class BikeEnv:
    def __init__(self, days: pd.DataFrame):
        self.demand = days["cnt"].to_numpy(dtype=float)
        self.lag1 = days["demand_lag1"].to_numpy(dtype=float)
        self.avg7 = days["demand_avg7"].to_numpy(dtype=float)
        self.context = days[STATE_COLUMNS].to_numpy(dtype=np.float32)
        self.t = 0
        self.end = 0
        self.fleet = 0

    def reset(self, start=0, length=None, fleet=None) -> np.ndarray:
        """Начинает эпизод с дня start. По умолчанию парк равен вчерашнему спросу."""
        self.t = start
        self.end = len(self.demand) if length is None else start + length
        self.fleet = round_fleet(self.lag1[start]) if fleet is None else fleet
        return self.state()

    def state(self) -> np.ndarray:
        """Всё, что агент знает утром. Спрос текущего дня сюда не входит."""
        fleet = self.fleet / MAX_FLEET
        lag1 = self.lag1[self.t] / MAX_FLEET
        avg7 = self.avg7[self.t] / MAX_FLEET
        return np.concatenate(
            ([fleet, fleet - lag1, fleet - avg7, lag1, avg7], self.context[self.t]),
            dtype=np.float32,
        )
    # Один шаг взаимодействия со средой
    def step(self, action: int):
        """Выполняет действие и возвращает состояние, награду, флаг конца и детали дня."""
        new_fleet = int(np.clip(self.fleet + ACTIONS[action], 0, MAX_FLEET))
        moved = abs(new_fleet - self.fleet)
        self.fleet = new_fleet

        demand = self.demand[self.t]
        served = min(demand, self.fleet)
        profit = PRICE * served - BIKE_COST * self.fleet - MOVE_COST * moved
        info = {
            "demand": demand,
            "fleet": self.fleet,
            "served": served,
            "moved": moved,
            "profit": profit,
        }

        self.t += 1
        done = self.t >= self.end
        next_state = None if done else self.state()
        # Награда — потери относительно идеального дня, когда парк равен спросу.
        # Идеальная прибыль не зависит от действий агента, поэтому лучшая стратегия
        # остаётся той же, но сезонные колебания спроса меньше мешают обучению.
        ideal_profit = (PRICE - BIKE_COST) * demand
        # Делим на 1000: нейросети проще учиться на небольших числах.
        reward = (profit - ideal_profit) / 1000
        return next_state, reward, done, info


def action_towards(target: float, fleet: int) -> int:
    """Выбирает действие, которое подводит парк ближе всего к target (для простых правил)."""
    return int(np.argmin(np.abs(fleet + ACTIONS - target)))
