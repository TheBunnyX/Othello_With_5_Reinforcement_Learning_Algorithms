"""Exact endgame solver: perfect play to the end of the game (alpha-beta on the final disc difference).

Scores are the final disc difference for the player to move with the empty squares credited to the
winner (the usual Othello scoring), so +2 is a win by 2 and 0 is a draw.  The solver is plain
Python on bitboards: roughly 20-50k positions/s, so it is meant for the last ~8-10 empty squares.

    solve(own, opp)               -> {move: score} (exact for every optimal move, bounds for the rest)
    best_moves(own, opp)          -> (score, [optimal moves])
"""
from __future__ import annotations

import time

from env import FULL, POSITION_WEIGHTS, get_flips, get_moves

INF = 1_000
MOVE_ORDER = sorted(range(64), key=lambda sq: -POSITION_WEIGHTS[sq])
ORDER_BITS = [1 << sq for sq in MOVE_ORDER]
FASTEST_FIRST_EMPTIES = 7  # above this many empties, order moves by the opponent's mobility


class SolverLimit(Exception):
    """Raised when the node or time budget runs out."""


def final_score(own: int, opp: int) -> int:
    diff = own.bit_count() - opp.bit_count()
    empties = 64 - own.bit_count() - opp.bit_count()
    return diff + empties if diff > 0 else diff - empties if diff < 0 else 0


class Solver:
    def __init__(self, node_limit: int | None = None, time_limit: float | None = None):
        self.nodes = 0
        self.node_limit = node_limit
        self.deadline = None if time_limit is None else time.perf_counter() + time_limit

    def negamax(self, own: int, opp: int, alpha: int, beta: int) -> int:
        self.nodes += 1
        if self.node_limit is not None and self.nodes > self.node_limit:
            raise SolverLimit
        if self.deadline is not None and not self.nodes & 1023 and time.perf_counter() > self.deadline:
            raise SolverLimit
        moves = get_moves(own, opp)
        if not moves:
            if not get_moves(opp, own):
                return final_score(own, opp)
            return -self.negamax(opp, own, -beta, -alpha)  # pass: same position, other player

        empties = 64 - (own | opp).bit_count()
        if empties > FASTEST_FIRST_EMPTIES:
            children = []
            for bit in ORDER_BITS:
                if moves & bit:
                    flips = get_flips(own, opp, bit)
                    n_own, n_opp = opp & ~flips, own | bit | flips
                    children.append((get_moves(n_own, n_opp).bit_count(), n_own, n_opp))
            children.sort(key=lambda c: c[0])
            children = [(a, b) for _, a, b in children]
        else:
            children = []
            for bit in ORDER_BITS:
                if moves & bit:
                    flips = get_flips(own, opp, bit)
                    children.append((opp & ~flips, own | bit | flips))

        best = -INF
        for n_own, n_opp in children:
            value = -self.negamax(n_own, n_opp, -beta, -alpha)
            if value > best:
                best = value
                if value > alpha:
                    alpha = value
                    if alpha >= beta:
                        break
        return best


def solve(own: int, opp: int, node_limit: int | None = None, time_limit: float | None = None,
          exact_all: bool = False):
    """Score every legal move.  Returns (scores, nodes).

    All optimal moves get their exact (equal) score.  Worse moves are only bounded (<= best - 1)
    unless `exact_all`.  Raises SolverLimit when the budget is exceeded."""
    solver = Solver(node_limit, time_limit)
    moves = get_moves(own, opp)
    if not moves:
        raise ValueError("no legal move")
    scores: dict[int, int] = {}
    best = -INF
    for bit in ORDER_BITS:
        if not moves & bit:
            continue
        flips = get_flips(own, opp, bit)
        n_own, n_opp = opp & ~flips, own | bit | flips
        lo = -INF if exact_all else best - 1  # window just below the best so ties stay exact
        value = -solver.negamax(n_own, n_opp, -INF, -lo)
        scores[bit.bit_length() - 1] = value
        best = max(best, value)
    return scores, solver.nodes


def best_moves(own: int, opp: int, **kw):
    scores, nodes = solve(own, opp, **kw)
    best = max(scores.values())
    return best, [m for m, v in scores.items() if v == best], nodes
