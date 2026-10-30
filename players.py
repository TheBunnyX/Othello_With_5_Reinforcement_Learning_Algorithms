"""Build a player from a short spec string (shared by bench.py and the tools).

    random | greedy                      baselines
    ppo | ddqn | a2c | ...               trained network, greedy move (weights/<name>.pt)
    minimax:3 | alphabeta:5              classic search
    ppo+mcts:100                         network-guided MCTS with 100 simulations (see mcts.py)
"""
from __future__ import annotations

import os

from agents import GreedyAgent, NeuralAgent, RandomAgent
from search import SEARCH_AGENTS, make_search_agent


def make_player(spec: str, weights_dir: str = "weights", device: str = "cpu"):
    spec = spec.strip().lower()
    if spec == "random":
        return RandomAgent()
    if spec == "greedy":
        return GreedyAgent()
    if spec.partition(":")[0] in SEARCH_AGENTS:
        return make_search_agent(spec)
    if "+mcts" in spec:
        from mcts import make_mcts_player  # imported lazily: needs torch only when used
        return make_mcts_player(spec, weights_dir, device)
    path = f"{spec}.pt" if "/" in spec or "\\" in spec else os.path.join(weights_dir, f"{spec}.pt")
    if not os.path.exists(path):
        raise ValueError(f"unknown player {spec!r} (no {path})")
    return NeuralAgent.load(path, device)
