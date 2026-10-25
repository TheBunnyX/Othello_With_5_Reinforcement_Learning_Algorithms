"""Classic game-tree search opponents: Minimax and Minimax with alpha-beta pruning.

Both use the negamax formulation (Othello is zero-sum, so a position is worth
v to the player to move and -v to the opponent) directly on bitboards.

    MinimaxAgent(depth)     explores every move down to `depth` plies
    AlphaBetaAgent(depth)   same result, but skips branches that cannot change
                            the decision, so it can search deeper in the same time

Evaluation (from the point of view of the player to move):
    * game over          -> +/- 10000 + disc difference (a win beats any heuristic)
    * otherwise          -> positional square weights + mobility (number of legal moves)
"""
from __future__ import annotations

import random
import time

import numpy as np

from env import BLACK, POSITION_WEIGHTS, OthelloEnv, get_flips, get_moves

WIN_SCORE = 10000
MOBILITY_WEIGHT = 5

# Square weights summed a byte (one board row) at a time: ROW_WEIGHTS[row][byte].
ROW_WEIGHTS = [
    [sum(int(POSITION_WEIGHTS[r * 8 + c]) for c in range(8) if b >> c & 1) for b in range(256)]
    for r in range(8)
]
# Alpha-beta prunes most when good moves come first: try squares by positional weight.
MOVE_ORDER = sorted(range(64), key=lambda sq: -POSITION_WEIGHTS[sq])


def positional(bits: int) -> int:
    return sum(ROW_WEIGHTS[r][(bits >> (8 * r)) & 0xFF] for r in range(8))


def evaluate(own: int, opp: int) -> float:
    """Heuristic value of a non-terminal position for the player to move (`own`)."""
    mobility = get_moves(own, opp).bit_count() - get_moves(opp, own).bit_count()
    return positional(own) - positional(opp) + MOBILITY_WEIGHT * mobility


def terminal_score(own: int, opp: int) -> float:
    diff = own.bit_count() - opp.bit_count()
    return (WIN_SCORE if diff > 0 else -WIN_SCORE if diff < 0 else 0) + diff


def ordered(moves: int) -> list[int]:
    return [sq for sq in MOVE_ORDER if moves >> sq & 1]


class Searcher:
    """Negamax search; `alpha_beta=False` is plain minimax."""

    def __init__(self, alpha_beta: bool):
        self.alpha_beta = alpha_beta
        self.nodes = 0

    def negamax(self, own: int, opp: int, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        moves = get_moves(own, opp)
        if not moves:
            if not get_moves(opp, own):
                return terminal_score(own, opp)
            # Pass: the opponent moves again from the same position (no ply consumed,
            # and it cannot repeat forever because a double pass is terminal).
            return -self.negamax(opp, own, depth, -beta, -alpha)
        if depth == 0:
            return evaluate(own, opp)

        best = -float("inf")
        for sq in ordered(moves):
            bit = 1 << sq
            flips = get_flips(own, opp, bit)
            value = -self.negamax(opp & ~flips, own | bit | flips, depth - 1, -beta, -alpha)
            if value > best:
                best = value
            if self.alpha_beta:
                if value > alpha:
                    alpha = value
                if alpha >= beta:  # the opponent will never allow this line: prune
                    break
        return best

    def root(self, own: int, opp: int, depth: int, exact: bool) -> dict[int, float]:
        """Value of every legal move.  With alpha-beta and exact=False, moves that cannot beat
        the best one so far are only bounded (their value is <= the reported one)."""
        scores = {}
        alpha = -float("inf")
        for sq in ordered(get_moves(own, opp)):
            bit = 1 << sq
            flips = get_flips(own, opp, bit)
            window = -float("inf") if exact or not self.alpha_beta else alpha
            value = -self.negamax(opp & ~flips, own | bit | flips, depth - 1, -float("inf"), -window)
            scores[sq] = value
            alpha = max(alpha, value)
        return scores


class MinimaxAgent:
    """Picks the move with the best minimax value at a fixed search depth."""

    alpha_beta = False
    algo = "minimax"
    label = "Minimax search"

    def __init__(self, depth: int = 3, name: str | None = None):
        self.depth = max(1, int(depth))
        self.name = name or f"{self.algo}:{self.depth}"
        self.last_stats = {"nodes": 0, "ms": 0.0, "depth": self.depth}

    def _search(self, env: OthelloEnv, exact: bool):
        own, opp = env._own_opp()
        searcher = Searcher(self.alpha_beta)
        start = time.perf_counter()
        scores = searcher.root(own, opp, self.depth, exact)
        stats = {"nodes": searcher.nodes, "ms": (time.perf_counter() - start) * 1000, "depth": self.depth}
        self.last_stats = stats
        return scores, stats

    def select_action(self, env: OthelloEnv) -> int:
        legal = env.legal_moves()
        if len(legal) == 1:
            self.last_stats = {"nodes": 0, "ms": 0.0, "depth": self.depth}
            return legal[0]
        scores, _ = self._search(env, exact=False)
        best = max(scores.values())
        return random.choice([m for m, v in scores.items() if v == best])  # random tie-break

    def evaluate(self, env: OthelloEnv):
        """Exact search value of every legal move, plus stats (used for hints / the web UI)."""
        scores, stats = self._search(env, exact=True)
        return scores, stats


class AlphaBetaAgent(MinimaxAgent):
    alpha_beta = True
    algo = "alphabeta"
    label = "Minimax + alpha-beta pruning"


SEARCH_AGENTS = {"minimax": MinimaxAgent, "alphabeta": AlphaBetaAgent}


def make_search_agent(spec: str, default_depth: int | None = None):
    """'minimax', 'alphabeta', or with an explicit depth: 'alphabeta:6'."""
    name, _, depth = spec.lower().partition(":")
    cls = SEARCH_AGENTS[name]
    if depth:
        return cls(int(depth))
    return cls(default_depth) if default_depth else cls()


def env_from_vec(venv, i: int) -> OthelloEnv:
    """Scalar OthelloEnv for game i of a VecOthello (to run sequential agents on it)."""
    env = OthelloEnv.__new__(OthelloEnv)
    own, opp = int(venv.own[i]), int(venv.opp[i])
    env.current = int(venv.player[i])
    env.black, env.white = (own, opp) if env.current == BLACK else (opp, own)
    env.done = bool(venv.done[i])
    env.winner = None
    env.move_count = int(venv.ply[i])
    env.last_move = None
    return env


class VecSequential:
    """Adapts any agent with select_action(env) to the vectorised-match interface (one game at a time)."""

    def __init__(self, agent):
        self.agent = agent
        self.name = getattr(agent, "name", "agent")

    def act(self, venv, mask, rng, active=None):
        actions = np.zeros(venv.n, dtype=np.int64)
        todo = ~venv.done if active is None else active & ~venv.done
        for i in np.flatnonzero(todo):
            actions[i] = self.agent.select_action(env_from_vec(venv, i))
        return actions

