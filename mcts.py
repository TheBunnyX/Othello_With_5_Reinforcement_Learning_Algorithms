"""Monte-Carlo tree search guided by a trained network (PUCT, as in AlphaZero).

One search runs for many games at once: every game has its own tree stored in numpy arrays, each
simulation descends all trees together, and the new leaves of all games go through the network in a
single batch.  The same code serves self-play training (alphazero.py, thousands of games at a time)
and playing one game (MCTSAgent, used by the web app and bench.py).

Conventions
    * A node stores its position from the point of view of the player who is to move there
      (own, opp, legal).  Values are from that player's point of view too.
    * Backing a value up an edge flips its sign, except across a pass (the same player moves
      again), where the edge sign is +1.
    * Terminal nodes use the exact result, no network call.

Network -> (priors, value)
    ppo / a2c / reinforce   softmax of the policy logits, the value head
    dqn / ddqn              softmax(Q / tau) as the prior, max legal Q as the value
    (reinforce and dqn have no trained value head, so they make poor search guides.)
"""
from __future__ import annotations

import math
import os
import random
import time

import numpy as np
import torch

from agents import VALUE_BASED, NeuralAgent, masked
from env import OthelloEnv
from solver import SolverLimit, solve
from vec_env import U, popcount, unpack, vec_flips, vec_moves

NEG = np.float32(-1e9)


def observations(own: np.ndarray, opp: np.ndarray, legal: np.ndarray) -> np.ndarray:
    out = np.empty((len(own), 3, 64), np.float32)
    out[:, 0], out[:, 1], out[:, 2] = unpack(own), unpack(opp), unpack(legal)
    return out.reshape(-1, 3, 8, 8)


class Evaluator:
    """Network wrapper: (own, opp, legal) bitboards -> (priors[n, 64], values[n])."""

    def __init__(self, net, algo: str, device, tau: float = 0.1):
        self.net, self.algo, self.device, self.tau = net.eval(), algo, torch.device(device), tau
        self.calls = 0

    @classmethod
    def from_agent(cls, agent: NeuralAgent, tau: float = 0.1):
        return cls(agent.net, agent.algo, agent.device, tau)

    @torch.inference_mode()
    def __call__(self, own, opp, legal):
        self.calls += 1
        obs = torch.from_numpy(observations(own, opp, legal)).to(self.device)
        head, value = self.net(obs)
        mask = obs[:, 2].flatten(1) > 0.5
        head = head.float()
        if self.algo in VALUE_BASED:
            logits = masked(head / self.tau, mask)
            v = masked(head, mask).max(1).values.clamp(-1, 1)
        else:
            logits = masked(head, mask)
            v = value.float()
        priors = torch.softmax(logits, 1) * mask
        priors = priors / priors.sum(1, keepdim=True).clamp_min(1e-9)
        return priors.cpu().numpy(), v.cpu().numpy()


class BatchedMCTS:
    def __init__(self, evaluator: Evaluator, sims: int = 100, c_puct: float = 1.5,
                 dirichlet_alpha: float = 0.5, noise_eps: float = 0.25):
        self.ev, self.sims, self.c_puct = evaluator, int(sims), c_puct
        self.alpha, self.eps = dirichlet_alpha, noise_eps

    def search(self, own, opp, noise: bool = False, rng: np.random.Generator | None = None):
        """Run `sims` simulations from each root position (games with no legal move are skipped).

        Returns (visits[G, 64], q[G, 64], value[G]) for the roots: visit counts, mean value of each
        root move for the player to move, and the root's mean value."""
        G, M = len(own), self.sims + 2
        own, opp = np.asarray(own, U), np.asarray(opp, U)
        s_own, s_opp, s_legal = (np.zeros((G, M), U) for _ in range(3))
        nxt = np.full((G, M, 64), -1, np.int32)
        sgn = np.zeros((G, M, 64), np.int8)
        N = np.zeros((G, M, 64), np.float32)
        W = np.zeros((G, M, 64), np.float32)
        P = np.zeros((G, M, 64), np.float32)
        LM = np.zeros((G, M, 64), bool)
        term = np.zeros((G, M), bool)
        tval = np.zeros((G, M), np.float32)
        count = np.ones(G, np.int64)
        gi = np.arange(G)

        s_own[:, 0], s_opp[:, 0] = own, opp
        s_legal[:, 0] = vec_moves(own, opp)
        LM[:, 0] = unpack(s_legal[:, 0]).astype(bool)
        active = s_legal[:, 0] != 0
        if active.any():
            pri, _ = self.ev(s_own[active, 0], s_opp[active, 0], s_legal[active, 0])
            if noise:
                rng = rng or np.random.default_rng()
                lm = LM[active, 0]
                dir_ = rng.gamma(self.alpha, size=pri.shape).astype(np.float32) * lm
                dir_ /= dir_.sum(1, keepdims=True).clip(1e-9)
                pri = (1 - self.eps) * pri + self.eps * dir_
            P[active, 0] = pri

        D = self.sims + 2
        pn = np.zeros((G, D), np.int64)
        pa = np.zeros((G, D), np.int64)
        for _ in range(self.sims):
            cur = np.zeros(G, np.int64)
            plen = np.zeros(G, np.int64)
            alive = active.copy()
            leaf_value = np.zeros(G, np.float32)
            expand_par = np.full(G, -1, np.int64)
            expand_act = np.zeros(G, np.int64)
            while alive.any():
                idx = np.flatnonzero(alive)
                c = cur[idx]
                n, w, p = N[idx, c], W[idx, c], P[idx, c]
                q = np.where(n > 0, w / np.maximum(n, 1), 0.0)
                u = self.c_puct * p * np.sqrt(n.sum(1, keepdims=True) + 1.0) / (1.0 + n)
                a = np.where(LM[idx, c], q + u, NEG).argmax(1)
                pn[idx, plen[idx]], pa[idx, plen[idx]] = c, a
                plen[idx] += 1
                child = nxt[idx, c, a]
                new = child < 0
                expand_par[idx[new]], expand_act[idx[new]] = c[new], a[new]
                alive[idx[new]] = False
                old = ~new
                cur[idx[old]] = child[old]
                hit_term = old & term[idx, np.where(old, child, 0)]
                leaf_value[idx[hit_term]] = tval[idx[hit_term], child[hit_term]]
                alive[idx[hit_term]] = False

            # ---- expansion of the unexpanded edges reached this simulation
            E = np.flatnonzero(expand_par >= 0)
            if E.size:
                par, act = expand_par[E], expand_act[E]
                o, p_ = s_own[E, par], s_opp[E, par]
                bit = U(1) << act.astype(U)
                fl = vec_flips(o, p_, bit)
                a_own, a_opp = o | bit | fl, p_ & ~fl
                m_opp, m_own = vec_moves(a_opp, a_own), vec_moves(a_own, a_opp)
                has = m_opp != 0
                passed = ~has & (m_own != 0)
                ended = ~has & (m_own == 0)
                n_own = np.where(passed, a_own, a_opp)
                n_opp = np.where(passed, a_opp, a_own)
                n_leg = np.where(has, m_opp, np.where(passed, m_own, U(0)))
                new_i = count[E]
                count[E] += 1
                s_own[E, new_i], s_opp[E, new_i], s_legal[E, new_i] = n_own, n_opp, n_leg
                LM[E, new_i] = unpack(n_leg).astype(bool)
                nxt[E, par, act] = new_i
                sgn[E, par, act] = np.where(passed, 1, -1)
                term[E, new_i] = ended
                res = np.sign(popcount(n_own).astype(np.int64) - popcount(n_opp).astype(np.int64)).astype(np.float32)
                tval[E, new_i] = np.where(ended, res, 0.0)
                v = np.where(ended, res, 0.0).astype(np.float32)
                live = ~ended
                if live.any():
                    pri, val = self.ev(n_own[live], n_opp[live], n_leg[live])
                    P[E[live], new_i[live]] = pri
                    v[live] = val
                leaf_value[E] = v

            # ---- back up: walk each path from its last edge to the root
            v = leaf_value.copy()
            for d in range(int(plen.max()) - 1, -1, -1):
                g = np.flatnonzero(plen > d)
                node, a = pn[g, d], pa[g, d]
                v[g] *= sgn[g, node, a]
                N[g, node, a] += 1.0
                W[g, node, a] += v[g]

        visits = N[:, 0]
        q_root = np.where(visits > 0, W[:, 0] / np.maximum(visits, 1), 0.0)
        total = visits.sum(1)
        value = np.where(total > 0, W[:, 0].sum(1) / np.maximum(total, 1), 0.0)
        return visits, q_root, value


class MCTSAgent:
    """One-game player: MCTS around a trained network, plus the exact solver in the endgame."""

    algo = "mcts"

    def __init__(self, base: NeuralAgent, sims: int = 100, c_puct: float = 1.5, tau: float = 0.1,
                 solve_empties: int = 10, solve_seconds: float = 3.0, name: str | None = None):
        self.base = base
        self.search_ = BatchedMCTS(Evaluator.from_agent(base, tau), sims, c_puct)
        self.sims, self.solve_empties, self.solve_seconds = sims, solve_empties, solve_seconds
        self.name = name or f"{base.name}+mcts:{sims}"
        self.label = f"{base.name.upper()} + MCTS"
        self.last_stats = {"sims": 0, "ms": 0.0, "solved": False}

    def evaluate(self, env: OthelloEnv):
        """(scores {move: score}, value in [-1, 1], stats).  Scores are visit fractions, or exact
        final disc differences when the endgame solver ran (stats['solved'])."""
        own, opp = env._own_opp()
        legal = env.legal_moves()
        start = time.perf_counter()
        empties = 64 - (own | opp).bit_count()
        if len(legal) == 1:
            self.last_stats = {"sims": 0, "ms": 0.0, "solved": False}
            return {legal[0]: 1.0}, 0.0, self.last_stats
        if empties <= self.solve_empties:
            try:
                scores, nodes = solve(own, opp, time_limit=self.solve_seconds)
                best = max(scores.values())
                self.last_stats = {"sims": 0, "ms": (time.perf_counter() - start) * 1000, "solved": True,
                                   "nodes": nodes, "exact": int(best)}
                return {m: float(v) for m, v in scores.items()}, math.tanh(best / 8), self.last_stats
            except SolverLimit:
                pass  # too slow for the time budget: fall back to the search
        visits, _, value = self.search_.search(np.array([own], U), np.array([opp], U))
        v = visits[0]
        scores = {m: float(v[m] / max(v.sum(), 1)) for m in legal}
        self.last_stats = {"sims": self.sims, "ms": (time.perf_counter() - start) * 1000, "solved": False}
        return scores, float(np.clip(value[0], -1, 1)), self.last_stats

    def select_action(self, env: OthelloEnv) -> int:
        scores, _, stats = self.evaluate(env)
        best = max(scores.values())
        return random.choice([m for m, v in scores.items() if v == best])


def make_mcts_player(spec: str, weights_dir: str = "weights", device: str = "cpu") -> MCTSAgent:
    """'ppo+mcts:100', 'ddqn+mcts:200:solve=10:c=1.5:tau=0.1'"""
    base_name, _, rest = spec.partition("+mcts")
    opts = [t for t in rest.split(":") if t]
    kw: dict = {}
    sims = 100
    for t in opts:
        if "=" in t:
            k, v = t.split("=", 1)
            kw[{"solve": "solve_empties", "c": "c_puct", "tau": "tau", "secs": "solve_seconds"}[k]] = float(v)
        else:
            sims = int(t)
    if "solve_empties" in kw:
        kw["solve_empties"] = int(kw["solve_empties"])
    # a bare name lives in weights_dir; something with a slash is a path ("weights_az/az+mcts:100")
    path = f"{base_name}.pt" if "/" in base_name or "\\" in base_name else os.path.join(weights_dir, f"{base_name}.pt")
    base = NeuralAgent.load(path, device)
    return MCTSAgent(base, sims, name=spec, **kw)
