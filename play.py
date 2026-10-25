"""Play Othello in the terminal against a trained agent (or watch two agents).

Players can be: human, random, greedy, minimax, alphabeta (optionally with a depth,
e.g. "alphabeta:6"), or a path / algorithm name of trained weights
(e.g. "ppo" means weights/ppo.pt).

Examples:
    python play.py --black human --white ppo
    python play.py --black dqn --white human --device cuda
    python play.py --black ppo --white a2c --games 10        # watch / compare
    python play.py --black human --white alphabeta --depth 6
    python play.py --black minimax:3 --white alphabeta:5 --games 10
"""
from __future__ import annotations

import argparse
import os

from agents import ALGORITHMS, GreedyAgent, NeuralAgent, RandomAgent, get_device, play_match
from env import BLACK, OthelloEnv, parse_square, square_name
from search import SEARCH_AGENTS, make_search_agent


class HumanAgent:
    name = "human"

    def select_action(self, env: OthelloEnv) -> int:
        legal = env.legal_moves()
        hint = " ".join(square_name(m) for m in legal)
        while True:
            text = input(f"Your move ({hint}) or 'q' to quit: ").strip()
            if text.lower() in ("q", "quit", "exit"):
                raise SystemExit
            try:
                m = parse_square(text)
            except ValueError as e:
                print(e)
                continue
            if m in legal:
                return m
            print("Illegal move.")


def make_agent(spec: str, device, weights_dir: str, depth: int | None = None):
    spec = spec.lower()
    if spec.partition(":")[0] in SEARCH_AGENTS:
        return make_search_agent(spec, depth)
    if spec == "human":
        return HumanAgent()
    if spec == "random":
        return RandomAgent()
    if spec == "greedy":
        return GreedyAgent()
    path = os.path.join(weights_dir, f"{spec}.pt") if spec in ALGORITHMS or not spec.endswith(".pt") else spec
    if not os.path.exists(path):
        raise SystemExit(f"weights not found: {path} (train first: python training.py --algo {spec})")
    return NeuralAgent.load(path, device)


def play_interactive(black, white):
    env = OthelloEnv()
    print(env.render())
    while not env.done:
        player = env.current
        agent = black if player == BLACK else white
        move = agent.select_action(env)
        _, _, _, info = env.step(move)
        who = "Black" if player == BLACK else "White"
        stats = getattr(agent, "last_stats", None)
        searched = f" (searched {stats['nodes']:,} positions in {stats['ms']:.0f} ms)" if stats and stats["nodes"] else ""
        print(f"\n{who} ({agent.name}) plays {square_name(move)}, flips {info['flips']}{searched}")
        if info["passed"]:
            print(f"{'White' if player == BLACK else 'Black'} has no legal move and passes.")
        print(env.render())
    b, w = env.score()
    result = "Draw" if env.winner == 0 else ("Black" if env.winner == BLACK else "White") + " wins"
    print(f"\n{result}!  {b} - {w}")


def main():
    p = argparse.ArgumentParser(description="Play Othello against trained RL agents or search opponents.")
    p.add_argument("--black", default="human",
                   help="human | random | greedy | minimax[:depth] | alphabeta[:depth] | "
                        "dqn | ddqn | reinforce | a2c | ppo | path.pt")
    p.add_argument("--white", default="ppo")
    p.add_argument("--device", default="cpu", help="auto | cuda | cpu")
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--games", type=int, default=1, help=">1 runs a silent match between two non-human agents")
    p.add_argument("--depth", type=int, default=None, help="search depth for minimax / alphabeta (default 3 / 5)")
    args = p.parse_args()

    device = get_device(args.device)
    black = make_agent(args.black, device, args.weights_dir, args.depth)
    white = make_agent(args.white, device, args.weights_dir, args.depth)

    if args.games > 1 and not isinstance(black, HumanAgent) and not isinstance(white, HumanAgent):
        r = play_match(black, white, args.games)
        print(f"{black.name} vs {white.name} over {args.games} games (colours alternate): "
              f"W {r['wins']} / L {r['losses']} / D {r['draws']}  ({r['win_rate']:.1%})")
    else:
        play_interactive(black, white)


if __name__ == "__main__":
    main()
