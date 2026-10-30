"""AlphaZero-style training: self-play guided by MCTS, then fit the network to the search.

    python alphazero.py --device cuda --init weights/ppo.pt --save-dir weights_az --log-dir logs_az
    python alphazero.py --device cuda --resume                      # continue from <save-dir>/az_last.pt

Each iteration
    1. play --games games of self-play in parallel; every move is chosen by MCTS (mcts.py) and the
       visit distribution is recorded as the policy target (Dirichlet noise at the root,
       temperature 1 for the first --temp-plies plies, then the most visited move)
    2. the last --solve-empties plies are played with the exact endgame solver (perfect labels)
    3. value target = (1 - margin) * result(+1/0/-1) + margin * final disc difference / 64
    4. train on the replay buffer (all 8 board symmetries) with  cross-entropy(policy) + MSE(value)
Every --eval-every iterations the latest net plays the best net so far (both with MCTS, 8 random
opening plies); it replaces the best one (<save-dir>/az.pt) only if it wins at least --gate of the games.
--league of the self-play games are played against an older snapshot instead of against itself.

The net is the same OthelloNet as the other algorithms, so weights/ppo.pt is a good starting point.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from agents import NeuralAgent, build_net, load_checkpoint, masked, save_checkpoint, get_device
from mcts import BatchedMCTS, Evaluator
from solver import best_moves
from training import SYM_PERMS, transform_obs
from vec_env import U, VecOthello, popcount, random_actions, unpack

CONFIG = {"channels": 64, "blocks": 4, "dueling": False}


# ------------------------------------------------------------------ symmetries
def symmetry_tables(device):
    """(obs_idx[8, 64], perms[8, 64]): gather indices that realise transform_obs(x, sym) on a flattened
    board, and perms[sym][a] = square a after the transform (to move policy targets the same way)."""
    squares = torch.arange(64).view(1, 1, 8, 8)
    obs_idx = torch.stack([transform_obs(squares, k).reshape(-1) for k in range(8)])
    perms = torch.as_tensor(np.stack(SYM_PERMS))
    return obs_idx.to(device), perms.to(device)


def apply_symmetry(obs: torch.Tensor, pi: torch.Tensor, sym: int, obs_idx, perms):
    """Transform a batch of observations (n, 3, 8, 8) and policy targets (n, 64) by symmetry `sym`."""
    if sym == 0:
        return obs, pi
    obs = obs.flatten(2).index_select(2, obs_idx[sym]).view_as(obs)
    new_pi = torch.zeros_like(pi)
    new_pi[:, perms[sym]] = pi
    return obs, new_pi


# ------------------------------------------------------------------ replay buffer
class Buffer:
    def __init__(self, capacity: int, device):
        self.cap, self.device, self.size, self.pos = capacity, device, 0, 0
        self.obs = torch.zeros((capacity, 3, 8, 8), dtype=torch.uint8, device=device)
        self.pi = torch.zeros((capacity, 64), dtype=torch.float16, device=device)
        self.z = torch.zeros(capacity, device=device)

    def add(self, obs: np.ndarray, pi: np.ndarray, z: np.ndarray):
        n = len(z)
        idx = (torch.arange(n, device=self.device) + self.pos) % self.cap
        self.obs[idx] = torch.from_numpy(obs).to(self.device)
        self.pi[idx] = torch.from_numpy(pi.astype(np.float16)).to(self.device)
        self.z[idx] = torch.from_numpy(z.astype(np.float32)).to(self.device)
        self.pos, self.size = (self.pos + n) % self.cap, min(self.size + n, self.cap)

    def sample(self, n: int):
        i = torch.randint(0, self.size, (n,), device=self.device)
        return self.obs[i].float(), self.pi[i].float(), self.z[i]


# ------------------------------------------------------------------ self-play
def _sample_rows(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample a column per row of the probability matrix p (Gumbel trick)."""
    g = -np.log(-np.log(rng.random(p.shape).clip(1e-12, 1 - 1e-12)))
    return np.where(p > 0, np.log(np.maximum(p, 1e-12)) + g, -np.inf).argmax(1)


def self_play(learner: Evaluator, games: int, sims: int, rng: np.random.Generator, temp_plies: int = 12,
              margin: float = 0.25, solve_empties: int = 8, opponent: Evaluator | None = None,
              n_league: int = 0, c_puct: float = 1.5):
    """Play `games` games; the last `n_league` are learner vs `opponent` (only the learner's moves are
    recorded).  Returns (obs[n,3,8,8] uint8, pi[n,64], z[n], info)."""
    venv = VecOthello(games, seed=int(rng.integers(1 << 31)))
    league = np.zeros(games, bool)
    league[games - n_league:] = n_league > 0
    learner_colour = np.where(rng.random(games) < 0.5, 1, -1)  # which colour the learner has in league games
    mcts_l = BatchedMCTS(learner, sims, c_puct)
    mcts_o = BatchedMCTS(opponent, sims, c_puct) if opponent is not None and n_league else None

    rec_own, rec_opp, rec_leg, rec_pi, rec_game, rec_mover = [], [], [], [], [], []
    solver_nodes = searches = 0
    while not venv.done.all():
        idx = np.flatnonzero(~venv.done)
        own, opp, leg, mover, ply = venv.own[idx], venv.opp[idx], venv.legal[idx], venv.player[idx], venv.ply[idx]
        empties = 64 - popcount(own | opp).astype(np.int64)
        pi = np.zeros((len(idx), 64), np.float32)
        act = np.zeros(len(idx), np.int64)
        learner_turn = ~league[idx] | (mover == learner_colour[idx])

        solve = (empties <= solve_empties) if solve_empties > 0 else np.zeros(len(idx), bool)
        for j in np.flatnonzero(solve):
            _, moves, nodes = best_moves(int(own[j]), int(opp[j]))
            solver_nodes += nodes
            pi[j, moves] = 1.0 / len(moves)
            act[j] = moves[rng.integers(len(moves))]

        for turn_mask, searcher, noise in ((learner_turn, mcts_l, True), (~learner_turn, mcts_o, False)):
            sel = np.flatnonzero(turn_mask & ~solve)
            if not sel.size:
                continue
            visits, _, _ = searcher.search(own[sel], opp[sel], noise=noise, rng=rng)
            searches += sel.size
            p = visits / np.maximum(visits.sum(1, keepdims=True), 1)
            pi[sel] = p
            explore = ply[sel] < temp_plies
            act[sel] = np.where(explore, _sample_rows(p, rng), visits.argmax(1))

        keep = learner_turn  # the opponent's moves in league games are not trained on
        rec_own.append(own[keep]); rec_opp.append(opp[keep]); rec_leg.append(leg[keep])
        rec_pi.append(pi[keep]); rec_game.append(idx[keep]); rec_mover.append(mover[keep])
        full = np.zeros(games, np.int64)
        full[idx] = act
        venv.step(full)

    black, white = venv.black_white()
    b, w = popcount(black).astype(np.int64), popcount(white).astype(np.int64)
    empt = 64 - b - w
    score_black = np.where(b > w, b - w + empt, np.where(w > b, b - w - empt, 0))  # empties to the winner

    own, opp, leg = np.concatenate(rec_own), np.concatenate(rec_opp), np.concatenate(rec_leg)
    pi, game, mover = np.concatenate(rec_pi), np.concatenate(rec_game), np.concatenate(rec_mover)
    s = (score_black[game] * mover).astype(np.float32)  # final disc difference for the player to move
    z = (1 - margin) * np.sign(s) + margin * s / 64.0
    obs = np.empty((len(own), 3, 64), np.uint8)
    obs[:, 0], obs[:, 1], obs[:, 2] = unpack(own), unpack(opp), unpack(leg)
    info = {"games": games, "positions": len(z), "avg_len": float(venv.ply.mean()), "searches": searches,
            "solver_nodes": solver_nodes, "black_wins": float((score_black > 0).mean())}
    return obs.reshape(-1, 3, 8, 8), pi, z.astype(np.float32), info


# ------------------------------------------------------------------ arena
def arena(a: Evaluator, b: Evaluator, games: int, sims: int, rng: np.random.Generator, opening: int = 8) -> float:
    """Win rate of `a` against `b` (draw = 0.5): MCTS vs MCTS, a plays black in even games."""
    venv = VecOthello(games, seed=int(rng.integers(1 << 31)))
    a_black = np.arange(games) % 2 == 0
    side = np.where(a_black, 1, -1)
    ma, mb = BatchedMCTS(a, sims), BatchedMCTS(b, sims)
    while not venv.done.all():
        idx = np.flatnonzero(~venv.done)
        own, opp, mover = venv.own[idx], venv.opp[idx], venv.player[idx]
        act = np.zeros(len(idx), np.int64)
        a_turn = mover == side[idx]
        for mask, searcher in ((a_turn, ma), (~a_turn, mb)):
            sel = np.flatnonzero(mask)
            if sel.size:
                act[sel] = searcher.search(own[sel], opp[sel])[0].argmax(1)
        rand = venv.ply[idx] < opening
        if rand.any():
            full_mask = venv.legal_mask()[idx]
            act = np.where(rand, random_actions(full_mask, rng), act)
        full = np.zeros(games, np.int64)
        full[idx] = act
        venv.step(full)
    result = venv.winner * side
    return float(((result > 0).sum() + 0.5 * (result == 0).sum()) / games)


# ------------------------------------------------------------------ training
def train_steps(net, opt, buf: Buffer, steps: int, batch: int, tables):
    obs_idx, perms = tables
    net.train()
    pol_sum = val_sum = 0.0
    for _ in range(steps):
        obs, pi, z = buf.sample(batch)
        obs, pi = apply_symmetry(obs, pi, int(torch.randint(0, 8, (1,))), obs_idx, perms)
        logits, v = net(obs)
        logp = torch.log_softmax(masked(logits.float(), obs[:, 2].flatten(1) > 0.5), 1).clamp(min=-1e4)
        pol = -(pi * logp).sum(1).mean()
        val = F.mse_loss(v.float(), z)
        opt.zero_grad(set_to_none=True)
        (pol + val).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        pol_sum += pol.detach()
        val_sum += val.detach()
    net.eval()
    return float(pol_sum) / steps, float(val_sum) / steps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--iterations", type=int, default=10_000)
    ap.add_argument("--hours", type=float, default=None, help="stop after this many hours")
    ap.add_argument("--games", type=int, default=128, help="self-play games per iteration (played in parallel)")
    ap.add_argument("--sims", type=int, default=100, help="MCTS simulations per move in self-play")
    ap.add_argument("--temp-plies", type=int, default=12)
    ap.add_argument("--margin", type=float, default=0.25, help="weight of the disc-difference term in the value target")
    ap.add_argument("--solve-empties", type=int, default=8, help="exact solver for the last N plies (0 = off)")
    ap.add_argument("--league", type=float, default=0.15, help="share of games vs an older snapshot")
    ap.add_argument("--buffer", type=int, default=600_000, help="replay buffer size (positions)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--reuse", type=float, default=4.0, help="times each new position is trained on")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-every", type=int, default=10, help="iterations between gating matches")
    ap.add_argument("--eval-games", type=int, default=100)
    ap.add_argument("--eval-sims", type=int, default=50)
    ap.add_argument("--gate", type=float, default=0.55)
    ap.add_argument("--init", default="weights/ppo.pt", help="checkpoint to start from")
    ap.add_argument("--save-dir", default="weights_az")
    ap.add_argument("--log-dir", default="logs_az")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = get_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    tables = symmetry_tables(device)

    last_path, best_path = os.path.join(args.save_dir, "az_last.pt"), os.path.join(args.save_dir, "az.pt")
    start_iter, games_done = 0, 0
    src = last_path if args.resume and os.path.exists(last_path) else args.init
    net, ckpt = load_checkpoint(src, device)
    if src == last_path:
        start_iter, games_done = ckpt["stats"].get("iteration", 0), ckpt["stats"].get("games", 0)
    if net.dueling:
        raise SystemExit("start from a policy-gradient checkpoint (ppo / a2c): dueling Q-networks have no policy head")
    best_net, _ = load_checkpoint(best_path if os.path.exists(best_path) and args.resume else src, device)
    pool = [copy.deepcopy(best_net)]
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    buf = Buffer(args.buffer, device)
    log = open(os.path.join(args.log_dir, "az.jsonl"), "a", encoding="utf-8")
    print(f"AlphaZero | {device} | start {src} | {args.games} games x {args.sims} sims per iteration", flush=True)

    t_start = time.time()
    for it in range(start_iter + 1, start_iter + args.iterations + 1):
        t0 = time.time()
        n_league = int(round(args.league * args.games)) if len(pool) > 1 or it > 5 else 0
        opponent = Evaluator(pool[rng.integers(len(pool))], "ppo", device) if n_league else None
        obs, pi, z, info = self_play(Evaluator(net, "ppo", device), args.games, args.sims, rng, args.temp_plies,
                                     args.margin, args.solve_empties, opponent, n_league)
        t_play = time.time() - t0
        buf.add(obs, pi, z)
        games_done += args.games
        steps = max(1, int(args.reuse * len(z) / args.batch)) if buf.size >= args.batch else 0
        pol, val = train_steps(net, opt, buf, steps, args.batch, tables) if steps else (float("nan"),) * 2

        rec = {"iter": it, "games": games_done, "positions": info["positions"], "avg_len": round(info["avg_len"], 1),
               "policy_loss": round(pol, 4), "value_loss": round(val, 4), "play_s": round(t_play, 1),
               "total_s": round(time.time() - t0, 1), "league": n_league, "z_mean": round(float(z.mean()), 3)}
        line = (f"[{it:4d}] {games_done:7d} games | play {t_play:5.1f}s ({args.games / t_play:4.1f} g/s) | train {steps:3d} steps "
                f"policy {pol:.3f} value {val:.3f} | len {info['avg_len']:.0f} | league {n_league}")

        save_checkpoint(last_path, net, "az", CONFIG, {"iteration": it, "games": games_done})
        if it % args.eval_every == 0:
            wr = arena(Evaluator(net, "ppo", device), Evaluator(best_net, "ppo", device), args.eval_games,
                       args.eval_sims, rng)
            promoted = wr >= args.gate
            rec.update(arena_vs_best=round(wr, 3), promoted=promoted)
            line += f" | arena vs best {wr:.1%}{' -> NEW BEST' if promoted else ''}"
            if promoted:
                best_net = copy.deepcopy(net).eval()
                save_checkpoint(best_path, net, "az", CONFIG, {"iteration": it, "games": games_done, "arena": wr})
                pool.append(copy.deepcopy(net).eval())
                pool = pool[-6:]
        print(line, flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if args.hours and time.time() - t_start > args.hours * 3600:
            print("time limit reached", flush=True)
            break


if __name__ == "__main__":
    main()
