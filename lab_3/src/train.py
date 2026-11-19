"""Обучает REINFORCE, сравнивает его с простыми правилами и сохраняет результаты."""
import copy
import json
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import report
from agent import ReinforceAgent
from env import ACTIONS, BIKE_COST, PRICE, STATE_SIZE, BikeEnv, action_towards

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

EPISODE_DAYS = 60  # длина учебного эпизода
EPISODES_PER_UPDATE = 16  # сколько эпизодов идёт в один шаг обучения
UPDATES = 600  # число шагов обучения
EVAL_EVERY = 10  # как часто проверять агента на валидации
GAMMAS = [0.0, 0.9, 0.99]  # варианты коэффициента дисконтирования
SEEDS = [1, 2, 3, 4, 5]  # каждый вариант обучается пять раз


def run_policy(days: pd.DataFrame, choose_action) -> pd.DataFrame:
    """Проводит стратегию по всем дням и возвращает таблицу с итогами каждого дня."""
    env = BikeEnv(days)
    state = env.reset()
    log, done = [], False
    while not done:
        action = choose_action(state, env)
        state, _, done, info = env.step(action)
        log.append({**info, "action": int(ACTIONS[action])})
    result = pd.DataFrame(log)
    result["date"] = days["dteday"].to_numpy()
    return result


def summarize(log: pd.DataFrame) -> dict:
    """Считает основные метрики стратегии за период."""
    return {
        "profit": log["profit"].sum() / 1000,
        "lost_share": 100 * (1 - log["served"].sum() / log["demand"].sum()),
        "idle_share": 100 * (1 - log["served"].sum() / log["fleet"].sum()),
        "moved": log["moved"].sum() / 1000,
    }


def rule_strategies(val: pd.DataFrame) -> dict:
    """Простые стратегии для сравнения. Постоянный парк подбирается на валидации."""
    best_profit, best_target = -np.inf, 0
    for target in range(0, 10001, 250):
        log = run_policy(val, lambda s, env, t=target: action_towards(t, env.fleet))
        if log["profit"].sum() > best_profit:
            best_profit, best_target = log["profit"].sum(), target

    def random_action(state, env, rng=np.random.default_rng(0)):
        return int(rng.integers(len(ACTIONS)))

    return {
        "Случайные действия": random_action,
        f"Постоянный парк ({best_target})": lambda s, env: action_towards(best_target, env.fleet),
        "Как вчерашний спрос": lambda s, env: action_towards(env.lag1[env.t], env.fleet),
        "Как средний спрос за 7 дней": lambda s, env: action_towards(env.avg7[env.t], env.fleet),
    }


def collect_episode(agent, env, rng):
    """Проводит один учебный эпизод со случайным началом и случайным размером парка."""
    start = int(rng.integers(0, len(env.demand) - EPISODE_DAYS + 1))
    fleet = int(rng.integers(0, 33)) * 250  # от 0 до 8000
    state = env.reset(start, EPISODE_DAYS, fleet)
    states, actions, rewards = [], [], []
    done = False
    while not done:
        action = agent.sample_action(state)
        states.append(state)
        actions.append(action)
        state, reward, done, _ = env.step(action)
        rewards.append(reward)
    return np.stack(states), np.array(actions), rewards


def train_one(gamma, seed, train, val) -> dict:
    """Обучает одного агента и оставляет веса, лучшие на валидации."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = BikeEnv(train)
    agent = ReinforceAgent(STATE_SIZE, len(ACTIONS), gamma)
    greedy = lambda state, _env: agent.best_action(state)

    history, best_val, best_weights = [], -np.inf, None
    for update in range(1, UPDATES + 1):
        episodes = [collect_episode(agent, env, rng) for _ in range(EPISODES_PER_UPDATE)]
        loss = agent.update(episodes)
        if update % EVAL_EVERY == 0:
            val_profit = summarize(run_policy(val, greedy))["profit"]
            # Средняя сумма наград за учебный эпизод: это потери, чем ближе к нулю, тем лучше.
            train_reward = float(np.mean([sum(e[2]) for e in episodes]))
            history.append({"gamma": gamma, "seed": seed, "update": update, "loss": loss,
                            "train_reward": train_reward, "val_profit": val_profit})
            if val_profit > best_val:
                best_val = val_profit
                best_weights = copy.deepcopy(agent.policy.state_dict())

    agent.policy.load_state_dict(best_weights)
    return {"agent": agent, "history": history, "best_val": best_val, "gamma": gamma, "seed": seed}


def main():
    OUTPUTS.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    started = time.time()

    data = pd.read_csv(ROOT / "data" / "bike_days.csv", parse_dates=["dteday"])
    train, val, test = (data[data["split"] == s].reset_index(drop=True)
                        for s in ("train", "val", "test"))

    # 1. Обучаем все варианты gamma и выбираем лучший только по валидации
    jobs = [(gamma, seed, train, val) for gamma in GAMMAS for seed in SEEDS]
    print(f"Запусков обучения: {len(jobs)}", flush=True)
    with multiprocessing.Pool() as pool:
        runs = pool.starmap(train_one, jobs)
    for run in runs:
        print(f"gamma={run['gamma']} seed={run['seed']} валидация={run['best_val']:.1f}")

    gamma_table = (pd.DataFrame([{"gamma": r["gamma"], "val": r["best_val"]} for r in runs])
                   .groupby("gamma")["val"].agg(["mean", "std", "min", "max"]).reset_index())
    best_gamma = float(gamma_table.loc[gamma_table["mean"].idxmax(), "gamma"])
    best_runs = [r for r in runs if r["gamma"] == best_gamma]

    # Прибыль на тесте для каждого gamma показывается только для справки.
    # На выбор она не влияет: best_gamma уже определён по валидации.
    for run in runs:
        run["test"] = summarize(run_policy(test, lambda s, _e, a=run["agent"]: a.best_action(s)))["profit"]
    test_means = pd.DataFrame([{"gamma": r["gamma"], "test": r["test"]} for r in runs])
    gamma_table["test_mean"] = test_means.groupby("gamma")["test"].mean().to_numpy()

    # 2. Проверяем агентов лучшего варианта на тесте, который они ещё не видели.
    agent_results, agent_logs = [], []
    for run in best_runs:
        greedy = lambda state, _env, a=run["agent"]: a.best_action(state)
        log = run_policy(test, greedy)
        agent_results.append({"seed": run["seed"], "val": run["best_val"], **summarize(log)})
        agent_logs.append(log)
    agent_table = pd.DataFrame(agent_results)

    # 3. Считаем простые правила и теоретический предел на тех же днях.
    rows = [{
        "strategy": "REINFORCE",
        **agent_table[["profit", "lost_share", "idle_share", "moved"]].mean().to_dict(),
        "profit_std": agent_table["profit"].std(),
        "val_profit": agent_table["val"].mean(),
    }]
    rule_logs = {}
    for name, rule in rule_strategies(val).items():
        val_profit = summarize(run_policy(val, rule))["profit"]
        rule_logs[name] = run_policy(test, rule)
        rows.append({"strategy": name, **summarize(rule_logs[name]),
                     "profit_std": np.nan, "val_profit": val_profit})
    results = pd.DataFrame(rows)
    upper_bound = (PRICE - BIKE_COST) * test["cnt"].sum() / 1000

    # 4. Сохраняем всё для отчёта. Для графика берём агента с медианной прибылью.
    median_index = agent_table["profit"].rank(method="first").astype(int).eq(3).idxmax()
    agent_logs[median_index].to_csv(OUTPUTS / "test_days_reinforce.csv", index=False)
    best_rule = results.iloc[1:].sort_values("profit", ascending=False).iloc[0]["strategy"]
    rule_logs[best_rule].to_csv(OUTPUTS / "test_days_best_rule.csv", index=False)
    results.to_csv(OUTPUTS / "results.csv", index=False)
    agent_table.to_csv(OUTPUTS / "seeds.csv", index=False)
    gamma_table.to_csv(OUTPUTS / "gamma_search.csv", index=False)
    pd.DataFrame([h for r in runs for h in r["history"]]).to_csv(OUTPUTS / "history.csv", index=False)
    torch.save(best_runs[median_index]["agent"].policy.state_dict(), OUTPUTS / "policy.pt")

    summary = {
        "best_gamma": best_gamma,
        "best_rule": best_rule,
        "upper_bound": upper_bound,
        "train_minutes": (time.time() - started) / 60,
        "settings": {"episode_days": EPISODE_DAYS, "episodes_per_update": EPISODES_PER_UPDATE,
                     "updates": UPDATES, "gammas": GAMMAS, "seeds": SEEDS},
    }
    (OUTPUTS / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(gamma_table.round(2).to_string(index=False))
    print(results.round(1).to_string(index=False))
    print(f"Предел: {upper_bound:.1f}. Время: {summary['train_minutes']:.1f} мин.")

    report.build()
    print(f"Отчёт готов: {OUTPUTS / 'report.html'}")


if __name__ == "__main__":
    main()
