"""Vectorised Othello: N games stepped at once with numpy uint64 bitboards.

This is the "multi-env" used for training.  Every operation works on whole
arrays, so stepping 256 games costs about as much as stepping a handful with
the scalar `env.OthelloEnv` (same rules, verified against it in test.py).

State is stored canonically: `own` = stones of the player to move,
`opp` = stones of the other player, `player` = colour to move (1 / -1).
"""
from __future__ import annotations

import numpy as np

from env import BLACK, POSITION_WEIGHTS, square

U = np.uint64
NOT_A = U(0xFEFEFEFEFEFEFEFE)
NOT_H = U(0x7F7F7F7F7F7F7F7F)
_1, _7, _8, _9 = U(1), U(7), U(8), U(9)

SHIFTS = (
    lambda x: (x << _1) & NOT_A,  # E
    lambda x: (x >> _1) & NOT_H,  # W
    lambda x: x << _8,            # S
    lambda x: x >> _8,            # N
    lambda x: (x << _9) & NOT_A,  # SE
    lambda x: (x << _7) & NOT_H,  # SW
    lambda x: (x >> _7) & NOT_A,  # NE
    lambda x: (x >> _9) & NOT_H,  # NW
)

INIT_BLACK = U((1 << square(3, 4)) | (1 << square(4, 3)))
INIT_WHITE = U((1 << square(3, 3)) | (1 << square(4, 4)))
SQUARE_BITS = U(1) << np.arange(64, dtype=U)


def vec_moves(own: np.ndarray, opp: np.ndarray) -> np.ndarray:
    empty = ~(own | opp)
    moves = np.zeros(np.broadcast(own, opp).shape, U)
    for sh in SHIFTS:
        x = sh(own) & opp
        x |= sh(x) & opp
        x |= sh(x) & opp
        x |= sh(x) & opp
        x |= sh(x) & opp
        x |= sh(x) & opp
        moves |= sh(x) & empty
    return moves


def vec_flips(own: np.ndarray, opp: np.ndarray, move: np.ndarray) -> np.ndarray:
    """Stones flipped when `own` plays `move` (one bit per element). Broadcasts."""
    flips = np.zeros(np.broadcast(own, opp, move).shape, U)
    for sh in SHIFTS:
        x = sh(move) & opp
        x |= sh(x) & opp
        x |= sh(x) & opp
        x |= sh(x) & opp
        x |= sh(x) & opp
        x |= sh(x) & opp
        flips |= np.where((sh(x) & own) != 0, x, U(0))
    return flips


def unpack(bits: np.ndarray) -> np.ndarray:
    """(N,) uint64 -> (N, 64) uint8, column i = bit i."""
    bits = np.ascontiguousarray(bits, dtype=U)
    return np.unpackbits(bits.view(np.uint8).reshape(-1, 8), axis=1, bitorder="little")


def popcount(bits: np.ndarray) -> np.ndarray:
    return np.bitwise_count(bits).astype(np.int16)


class VecOthello:
    """N independent Othello games.  Finished games stay finished until `reset(mask)`."""

    def __init__(self, n: int, seed: int | None = None):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.own = np.zeros(n, U)
        self.opp = np.zeros(n, U)
        self.legal = np.zeros(n, U)
        self.player = np.ones(n, np.int8)
        self.done = np.zeros(n, bool)
        self.winner = np.zeros(n, np.int8)
        self.ply = np.zeros(n, np.int16)
        self.reset()

    def reset(self, mask: np.ndarray | None = None):
        idx = slice(None) if mask is None else mask
        self.own[idx] = INIT_BLACK
        self.opp[idx] = INIT_WHITE
        self.player[idx] = BLACK
        self.done[idx] = False
        self.winner[idx] = 0
        self.ply[idx] = 0
        self.legal[idx] = vec_moves(self.own[idx], self.opp[idx])

    # ------------------------------------------------------------- queries
    def observation(self) -> np.ndarray:
        """(N, 3, 8, 8) uint8: own stones, opponent stones, legal moves."""
        out = np.empty((self.n, 3, 64), np.uint8)
        out[:, 0] = unpack(self.own)
        out[:, 1] = unpack(self.opp)
        out[:, 2] = unpack(self.legal)
        return out.reshape(self.n, 3, 8, 8)

    def legal_mask(self) -> np.ndarray:
        return unpack(self.legal).astype(bool)

    def black_white(self) -> tuple[np.ndarray, np.ndarray]:
        is_black = self.player == BLACK
        return np.where(is_black, self.own, self.opp), np.where(is_black, self.opp, self.own)

    def board_string(self, i: int) -> str:
        """64 chars 'b' / 'w' / '.' for game i (used by the training monitor)."""
        b, w = self.black_white()
        b, w = int(b[i]), int(w[i])
        return "".join("b" if b >> k & 1 else "w" if w >> k & 1 else "." for k in range(64))

    # ---------------------------------------------------------------- step
    def step(self, actions: np.ndarray):
        """Play one move in every unfinished game (actions for finished games are ignored).

        Returns (reward, ended, passed, mover), each shape (N,):
            reward  +1 / -1 / 0 for the player who moved, non-zero only when the game ended
            ended   game finished with this move
            passed  opponent had no move, so the same player moves again
            mover   colour that moved (0 for games that were already finished)
        """
        n = self.n
        reward = np.zeros(n, np.float32)
        ended_all = np.zeros(n, bool)
        passed_all = np.zeros(n, bool)
        mover_all = np.zeros(n, np.int8)
        idx = np.flatnonzero(~self.done)
        if idx.size == 0:
            return reward, ended_all, passed_all, mover_all

        bit = U(1) << np.asarray(actions)[idx].astype(U)
        if np.any((self.legal[idx] & bit) == 0):
            bad = idx[(self.legal[idx] & bit) == 0][0]
            raise ValueError(f"illegal move {int(actions[bad])} in game {bad}")

        own, opp = self.own[idx], self.opp[idx]
        flips = vec_flips(own, opp, bit)
        own = own | bit | flips
        opp = opp & ~flips
        mover = self.player[idx]

        m_opp = vec_moves(opp, own)
        m_own = vec_moves(own, opp)
        has_opp = m_opp != 0
        passed = ~has_opp & (m_own != 0)
        ended = ~has_opp & (m_own == 0)

        self.own[idx] = np.where(has_opp, opp, own)
        self.opp[idx] = np.where(has_opp, own, opp)
        self.player[idx] = np.where(has_opp, -mover, mover)
        self.legal[idx] = np.where(has_opp, m_opp, np.where(passed, m_own, U(0)))
        self.done[idx] = ended
        self.ply[idx] += 1

        diff = np.sign(popcount(own) - popcount(opp)).astype(np.int8)  # from the mover's view
        self.winner[idx] = np.where(ended, mover * diff, 0)
        reward[idx] = np.where(ended, diff, 0)
        ended_all[idx] = ended
        passed_all[idx] = passed
        mover_all[idx] = mover
        return reward, ended_all, passed_all, mover_all


# ------------------------------------------------------------ vector baselines
def random_actions(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Uniform random legal move per row of an (N, 64) bool mask."""
    scores = rng.random(mask.shape)
    scores[~mask] = -1.0
    return scores.argmax(1)


def greedy_actions(venv: VecOthello, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Same heuristic as agents.GreedyAgent: square weight + stones flipped."""
    flips = vec_flips(venv.own[:, None], venv.opp[:, None], SQUARE_BITS[None, :])
    scores = POSITION_WEIGHTS[None, :] + popcount(flips) + rng.random(mask.shape) * 0.1
    scores[~mask] = -1e9
    return scores.argmax(1)

