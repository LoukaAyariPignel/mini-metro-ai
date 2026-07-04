"""Entraînement DQN sur Mini Metro.

Usage :
    python3 train.py --episodes 300
    python3 train.py --episodes 2000 --out models/dqn_long.pt

Écrit le modèle dans models/dqn.pt et le journal dans logs/train_log.csv.
"""

from __future__ import annotations

import argparse
import csv
import os
import time

from agents.dqn import DQNAgent
from mini_metro.env import MiniMetroEnv


def train(
    episodes: int = 300,
    seed: int = 0,
    out: str = "models/dqn.pt",
    log_path: str = "logs/train_log.csv",
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay_episodes: int | None = None,
    update_every: int = 2,
) -> DQNAgent:
    env = MiniMetroEnv(seed=seed)
    agent = DQNAgent()
    eps_decay_episodes = eps_decay_episodes or max(int(episodes * 0.6), 1)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["episode", "score", "survival_s", "reward", "epsilon", "loss"])

    best_score = -1.0
    t0 = time.time()
    for ep in range(episodes):
        frac = min(ep / eps_decay_episodes, 1.0)
        epsilon = eps_start + (eps_end - eps_start) * frac

        obs, _ = env.reset()
        mask = env.action_mask()
        done = False
        ep_reward, losses, step = 0.0, [], 0
        while not done:
            action = agent.act(obs, mask, epsilon)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_mask = env.action_mask()
            done = terminated or truncated
            agent.buffer.push(obs, action, reward, next_obs, next_mask, terminated)
            step += 1
            if step % update_every == 0:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
            obs, mask = next_obs, next_mask
            ep_reward += reward

        score, survival = info["score"], info["time"]
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        writer.writerow([ep, score, f"{survival:.0f}", f"{ep_reward:.1f}",
                         f"{epsilon:.3f}", f"{mean_loss:.4f}"])
        log.flush()

        if score > best_score and epsilon <= 0.3:
            best_score = score
            agent.save(out)

        if (ep + 1) % 10 == 0:
            print(
                f"ép. {ep + 1:4d}/{episodes} | score {score:4d} | "
                f"survie {survival:5.0f}s | eps {epsilon:.2f} | "
                f"{time.time() - t0:5.0f}s écoulées",
                flush=True,
            )

    agent.save(out.replace(".pt", "_final.pt"))
    log.close()
    print(f"Meilleur score : {best_score:.0f} — modèle : {out}")
    return agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="models/dqn.pt")
    parser.add_argument("--log", default="logs/train_log.csv")
    args = parser.parse_args()
    train(args.episodes, args.seed, args.out, args.log)
