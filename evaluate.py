"""Compare les agents (aléatoire, glouton, DQN) sur les mêmes parties.

Usage :
    python3 evaluate.py --episodes 20
    python3 evaluate.py --episodes 20 --model models/dqn.pt
"""

from __future__ import annotations

import argparse
import os
import statistics

from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from mini_metro.env import MiniMetroEnv


def run_episode(env: MiniMetroEnv, agent, seed: int) -> tuple[int, float]:
    obs, _ = env.reset(seed=seed)
    mask = env.action_mask()
    done = False
    info = {"score": 0, "time": 0.0}
    while not done:
        if isinstance(agent, GreedyAgent):
            action = agent.act(obs, mask, env)
        else:
            action = agent.act(obs, mask)
        obs, _, terminated, truncated, info = env.step(action)
        mask = env.action_mask()
        done = terminated or truncated
    return info["score"], info["time"]


def evaluate(episodes: int = 20, model: str = "models/dqn.pt", base_seed: int = 10_000) -> dict:
    agents: dict[str, object] = {
        "aléatoire": RandomAgent(seed=0),
        "glouton": GreedyAgent(),
    }
    if os.path.exists(model):
        from agents.dqn import DQNAgent

        dqn = DQNAgent()
        dqn.load(model)
        agents["DQN"] = dqn
    else:
        print(f"(pas de modèle DQN trouvé à {model} — lancez train.py d'abord)")

    env = MiniMetroEnv()
    results = {}
    for name, agent in agents.items():
        scores, survivals = [], []
        for ep in range(episodes):
            score, survival = run_episode(env, agent, seed=base_seed + ep)
            scores.append(score)
            survivals.append(survival)
        results[name] = (scores, survivals)
        print(
            f"{name:>10} | score moyen {statistics.mean(scores):6.1f} "
            f"± {statistics.stdev(scores) if len(scores) > 1 else 0:5.1f} "
            f"| survie moyenne {statistics.mean(survivals):6.0f}s "
            f"| meilleur score {max(scores)}"
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--model", default="models/dqn.pt")
    args = parser.parse_args()
    evaluate(args.episodes, args.model)
