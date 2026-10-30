"""Strength benchmark: play a player against reference opponents in parallel and report win rates.

    python bench.py --players ppo,ddqn,a2c --opponents alphabeta:1,alphabeta:3,alphabeta:5 --games 100
    python bench.py --players ppo,ppo+mcts:50 --opponents alphabeta:3 --games 100 --workers 14

Games come in pairs: the same random opening is played with the player as black and as white,
so luck of the opening cancels.  `--opening` random plies make the games differ even though
network and search players are deterministic.  Win rate counts a draw as 0.5; the +/- is a
95% normal-approximation interval.  Elo is the rating gap implied by the win rate.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import random
import time

from env import BLACK, OthelloEnv
from players import make_player

_cache: dict = {}
_cfg: dict = {}


def _init(weights_dir: str, device: str):
    import torch
    torch.set_num_threads(1)  # one process per core; torch threads on top would oversubscribe
    _cfg.update(weights_dir=weights_dir, device=device)


def _player(spec: str):
    if spec not in _cache:
        _cache[spec] = make_player(spec, _cfg["weights_dir"], _cfg["device"])
    return _cache[spec]


def play_game(task) -> tuple[str, str, float]:
    """One game; returns (player spec, opponent spec, score for the player: 1 / 0.5 / 0)."""
    spec, opp_spec, seed, player_is_black, opening = task
    player, opp = _player(spec), _player(opp_spec)
    rng = random.Random(seed)
    random.seed(seed)  # tie-breaks inside the agents
    env = OthelloEnv()
    mine = BLACK if player_is_black else -BLACK
    while not env.done:
        if env.move_count < opening:
            move = rng.choice(env.legal_moves())
        else:
            move = (player if env.current == mine else opp).select_action(env)
        env.step(move)
    score = 0.5 if env.winner == 0 else 1.0 if env.winner == mine else 0.0
    return spec, opp_spec, score


def elo(p: float) -> float:
    p = min(max(p, 0.005), 0.995)
    return -400 * math.log10(1 / p - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--players", required=True, help="comma list of player specs")
    ap.add_argument("--opponents", default="alphabeta:1,alphabeta:3,alphabeta:5")
    ap.add_argument("--games", type=int, default=100, help="games per pairing (even; played as colour pairs)")
    ap.add_argument("--opening", type=int, default=8, help="random opening plies")
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    ap.add_argument("--weights-dir", default="weights")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    players = [s.strip() for s in args.players.split(",") if s.strip()]
    opponents = [s.strip() for s in args.opponents.split(",") if s.strip()]
    pairs = max(1, args.games // 2)
    tasks = []
    for opp in opponents:  # opponent-major so partial results are meaningful
        for spec in players:
            for i in range(pairs):
                seed = args.seed * 1_000_003 + i
                tasks += [(spec, opp, seed, True, args.opening), (spec, opp, seed, False, args.opening)]
    # slowest games first keeps all workers busy to the end
    tasks.sort(key=lambda t: -(("mcts" in t[0]) * 3 + (t[1].startswith(("alphabeta:5", "alphabeta:6", "alphabeta:7")))))

    start = time.time()
    results: dict[tuple[str, str], list[float]] = {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init, initargs=(args.weights_dir, args.device)) as pool:
        for n, (spec, opp, score) in enumerate(pool.imap_unordered(play_game, tasks, chunksize=1), 1):
            results.setdefault((spec, opp), []).append(score)
            if n % 50 == 0 or n == len(tasks):
                print(f"\r{n}/{len(tasks)} games  {time.time() - start:6.0f}s", end="", flush=True)
    print()

    width = max(len(p) for p in players) + 2
    print(f"\nWin rate of the player (draw = 0.5), {pairs * 2} games per cell, {args.opening} random opening plies\n")
    print(f"{'player':<{width}}" + "".join(f"{o:>24}" for o in opponents))
    for spec in players:
        row = f"{spec:<{width}}"
        for opp in opponents:
            r = results.get((spec, opp), [])
            p = sum(r) / len(r)
            ci = 1.96 * math.sqrt(max(p * (1 - p), 1e-9) / len(r))
            row += f"{p:>13.1%} ±{ci:4.1%} {elo(p):+5.0f}e"
        print(row)
    print("\n(e = Elo gap to that opponent)")


if __name__ == "__main__":
    main()
