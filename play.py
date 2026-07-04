"""Regarder un agent jouer, en ASCII dans le terminal.

Usage :
    python3 play.py                       # agent glouton
    python3 play.py --agent dqn           # agent DQN entraîné
    python3 play.py --agent random --fps 30
"""

from __future__ import annotations

import argparse
import os
import time

from agents.greedy_agent import GreedyAgent
from agents.random_agent import RandomAgent
from mini_metro.env import MiniMetroEnv
from mini_metro.game import Shape

W, H = 64, 24
SHAPE_CHARS = {
    Shape.CIRCLE: "o",
    Shape.TRIANGLE: "^",
    Shape.SQUARE: "#",
    Shape.STAR: "*",
    Shape.PENTAGON: "5",
}
LINE_CHARS = ["1", "2", "3"]


def render(env: MiniMetroEnv) -> str:
    g = env.game
    grid = [[" "] * W for _ in range(H)]

    def cell(x: float, y: float) -> tuple[int, int]:
        return min(int(x * (W - 1)), W - 1), min(int(y * (H - 1)), H - 1)

    # trace les lignes (interpolation grossière entre stations)
    for li, line in enumerate(g.lines):
        for a_idx, b_idx in zip(line.stations, line.stations[1:]):
            a, b = g.stations[a_idx], g.stations[b_idx]
            steps = 40
            for k in range(steps + 1):
                t = k / steps
                cx, cy = cell(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
                if grid[cy][cx] == " ":
                    grid[cy][cx] = LINE_CHARS[li]

    # trains
    for tr in g.trains:
        line = g.lines[tr.line_idx]
        if len(line) >= 2:
            a = g.stations[line.stations[tr.edge]]
            b = g.stations[line.stations[tr.edge + 1]]
            cx, cy = cell(a.x + (b.x - a.x) * tr.t, a.y + (b.y - a.y) * tr.t)
            grid[cy][cx] = "T"

    # stations (par-dessus tout le reste)
    for st in g.stations:
        cx, cy = cell(st.x, st.y)
        grid[cy][cx] = SHAPE_CHARS[st.shape]
        if len(st.queue) > 0 and cx + 1 < W:
            grid[cy][cx + 1 if cx + 1 < W else cx] = str(min(len(st.queue), 9))

    lines_txt = "\n".join("".join(row) for row in grid)
    status = (
        f"t={g.time:5.0f}s  score={g.score:4d}  attente={g.total_waiting():3d}  "
        f"trains dispo={g.trains_pool}  {'GAME OVER' if g.game_over else ''}"
    )
    return lines_txt + "\n" + "-" * W + "\n" + status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["greedy", "random", "dqn"], default="greedy")
    parser.add_argument("--model", default="models/dqn.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=15.0)
    args = parser.parse_args()

    env = MiniMetroEnv(seed=args.seed)
    if args.agent == "dqn":
        from agents.dqn import DQNAgent

        agent = DQNAgent()
        if not os.path.exists(args.model):
            raise SystemExit(f"Modèle introuvable : {args.model} — lancez train.py d'abord.")
        agent.load(args.model)
    elif args.agent == "random":
        agent = RandomAgent(seed=args.seed)
    else:
        agent = GreedyAgent()

    obs, _ = env.reset()
    mask = env.action_mask()
    done = False
    while not done:
        if isinstance(agent, GreedyAgent):
            action = agent.act(obs, mask, env)
        else:
            action = agent.act(obs, mask)
        obs, _, terminated, truncated, _ = env.step(action)
        mask = env.action_mask()
        done = terminated or truncated
        print("\033[2J\033[H" + render(env), flush=True)
        time.sleep(1.0 / args.fps)
    print("Partie terminée.")


if __name__ == "__main__":
    main()
