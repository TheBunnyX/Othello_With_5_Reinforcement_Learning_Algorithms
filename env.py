"""Othello (Reversi) environment backed by 64-bit bitboards.

Square index = row * 8 + col  (row 0 is the top row "1", col 0 is column "a").
Players: BLACK = 1 (moves first), WHITE = -1.

The environment automatically handles passes: if the player to move has no
legal move, the turn goes back to the other player.  When neither player can
move the game is over.  Agents therefore only ever choose among 64 squares.

Observation (canonical, from the point of view of the player to move):
    float32 array of shape (3, 8, 8)
        [0] stones of the player to move
        [1] stones of the opponent
        [2] legal moves of the player to move
"""
from __future__ import annotations

import numpy as np

BOARD_SIZE = 8
NUM_ACTIONS = 64
BLACK, WHITE, EMPTY = 1, -1, 0

FULL = 0xFFFFFFFFFFFFFFFF
NOT_A = 0xFEFEFEFEFEFEFEFE  # clears column a (col 0)
NOT_H = 0x7F7F7F7F7F7F7F7F  # clears column h (col 7)

# The 8 directions as bitboard shift functions.
SHIFTS = (
    lambda x: (x << 1) & NOT_A,   # E
    lambda x: (x >> 1) & NOT_H,   # W
    lambda x: (x << 8) & FULL,    # S
    lambda x: x >> 8,             # N
    lambda x: (x << 9) & NOT_A,   # SE
    lambda x: (x << 7) & NOT_H,   # SW
    lambda x: (x >> 7) & NOT_A,   # NE
    lambda x: (x >> 9) & NOT_H,   # NW
)

COLS = "abcdefgh"

# Classic positional weights: corners good, X/C squares bad.
POSITION_WEIGHTS = np.array([
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, 1, 1, 1, 1, -2, 10],
    [5, -2, 1, 0, 0, 1, -2, 5],
    [5, -2, 1, 0, 0, 1, -2, 5],
    [10, -2, 1, 1, 1, 1, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100],
]).reshape(-1)


def square(row: int, col: int) -> int:
    return row * BOARD_SIZE + col


def square_name(action: int) -> str:
    return f"{COLS[action % 8]}{action // 8 + 1}"


def parse_square(text: str) -> int:
    """'d3' -> action index."""
    text = text.strip().lower()
    if len(text) != 2 or text[0] not in COLS or text[1] not in "12345678":
        raise ValueError(f"invalid square: {text!r}")
    return square(int(text[1]) - 1, COLS.index(text[0]))


def get_moves(own: int, opp: int) -> int:
    """Bitboard of legal moves for `own`."""
    empty = ~(own | opp) & FULL
    moves = 0
    for sh in SHIFTS:
        x = sh(own) & opp
        x |= sh(x) & opp
        x |= sh(x) & opp
        x |= sh(x) & opp
        x |= sh(x) & opp
        x |= sh(x) & opp
        moves |= sh(x) & empty
    return moves


def get_flips(own: int, opp: int, move_bit: int) -> int:
    """Bitboard of opponent stones flipped when `own` plays `move_bit`."""
    flips = 0
    for sh in SHIFTS:
        line = 0
        f = sh(move_bit)
        while f & opp:
            line |= f
            f = sh(f)
        if f & own:
            flips |= line
    return flips


def bits_to_array(bits: int) -> np.ndarray:
    """Bitboard -> uint8 array of 64 (index i = bit i)."""
    raw = np.frombuffer(bits.to_bytes(8, "little"), dtype=np.uint8)
    return np.unpackbits(raw, bitorder="little")


def bits_to_list(bits: int) -> list[int]:
    out = []
    while bits:
        low = bits & -bits
        out.append(low.bit_length() - 1)
        bits ^= low
    return out


class OthelloEnv:
    """Two-player Othello with a gym-like API."""

    num_actions = NUM_ACTIONS

    def __init__(self):
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> np.ndarray:
        self.black = (1 << square(3, 4)) | (1 << square(4, 3))
        self.white = (1 << square(3, 3)) | (1 << square(4, 4))
        self.current = BLACK
        self.done = False
        self.winner: int | None = None
        self.move_count = 0
        self.last_move: int | None = None
        return self.get_observation()

    @classmethod
    def from_array(cls, board, current: int = BLACK) -> "OthelloEnv":
        """Build a position from a 64-length (or 8x8) array of 1 / -1 / 0."""
        board = np.asarray(board, dtype=np.int64).reshape(-1)
        if board.size != NUM_ACTIONS:
            raise ValueError("board must have 64 squares")
        env = cls.__new__(cls)
        env.black = sum(1 << i for i in range(64) if board[i] == BLACK)
        env.white = sum(1 << i for i in range(64) if board[i] == WHITE)
        env.current = BLACK if current == BLACK else WHITE
        env.done = False
        env.winner = None
        env.move_count = int(np.count_nonzero(board)) - 4
        env.last_move = None
        env._resolve_turn(env.current)
        return env

    def clone(self) -> "OthelloEnv":
        env = OthelloEnv.__new__(OthelloEnv)
        env.__dict__.update(self.__dict__)
        return env

    # ---------------------------------------------------------------- queries
    def _own_opp(self, player: int | None = None) -> tuple[int, int]:
        player = self.current if player is None else player
        return (self.black, self.white) if player == BLACK else (self.white, self.black)

    def legal_moves_bits(self, player: int | None = None) -> int:
        if self.done:
            return 0
        return get_moves(*self._own_opp(player))

    def legal_moves(self, player: int | None = None) -> list[int]:
        return bits_to_list(self.legal_moves_bits(player))

    def legal_mask(self) -> np.ndarray:
        return bits_to_array(self.legal_moves_bits()).astype(bool)

    def score(self) -> tuple[int, int]:
        return self.black.bit_count(), self.white.bit_count()

    def board_array(self) -> np.ndarray:
        """8x8 int8 board with 1 = black, -1 = white, 0 = empty."""
        b = bits_to_array(self.black).astype(np.int8) - bits_to_array(self.white).astype(np.int8)
        return b.reshape(BOARD_SIZE, BOARD_SIZE)

    def get_observation(self) -> np.ndarray:
        own, opp = self._own_opp()
        obs = np.empty((3, NUM_ACTIONS), dtype=np.float32)
        obs[0] = bits_to_array(own)
        obs[1] = bits_to_array(opp)
        obs[2] = bits_to_array(0 if self.done else get_moves(own, opp))
        return obs.reshape(3, BOARD_SIZE, BOARD_SIZE)

    # ------------------------------------------------------------------- step
    def step(self, action: int):
        """Play `action` for the current player.

        Returns (obs, reward, done, info).  `reward` is from the point of view
        of the player who just moved: +1 win, -1 loss, 0 draw / game ongoing.
        `info["passed"]` is True when the opponent had to pass, i.e. the same
        player moves again.
        """
        if self.done:
            raise RuntimeError("game is over, call reset()")
        action = int(action)
        own, opp = self._own_opp()
        bit = 1 << action
        if not (0 <= action < NUM_ACTIONS) or not (get_moves(own, opp) & bit):
            raise ValueError(f"illegal move {action} ({square_name(action) if 0 <= action < 64 else '?'})")

        flips = get_flips(own, opp, bit)
        own |= bit | flips
        opp &= ~flips
        if self.current == BLACK:
            self.black, self.white = own, opp
        else:
            self.white, self.black = own, opp

        mover = self.current
        self.move_count += 1
        self.last_move = action
        passed = self._resolve_turn(-mover)

        reward = 0.0
        if self.done:
            reward = 0.0 if self.winner == EMPTY else (1.0 if self.winner == mover else -1.0)
        info = {"player": mover, "passed": passed, "flips": flips.bit_count(), "winner": self.winner}
        return self.get_observation(), reward, self.done, info

    def _resolve_turn(self, player: int) -> bool:
        """Give the turn to `player`, passing / ending the game if needed.
        Returns True if `player` had to pass."""
        self.current = player
        if get_moves(*self._own_opp(player)):
            return False
        if get_moves(*self._own_opp(-player)):
            self.current = -player
            return True
        self.done = True
        b, w = self.score()
        self.winner = BLACK if b > w else WHITE if w > b else EMPTY
        return False

    # ----------------------------------------------------------------- render
    def render(self, show_moves: bool = True) -> str:
        board = self.board_array()
        legal = set(self.legal_moves()) if show_moves else set()
        lines = ["   " + " ".join(COLS)]
        for r in range(BOARD_SIZE):
            row = []
            for c in range(BOARD_SIZE):
                v = board[r, c]
                idx = square(r, c)
                if v == BLACK:
                    ch = "X" if idx != self.last_move else "x"
                elif v == WHITE:
                    ch = "O" if idx != self.last_move else "o"
                else:
                    ch = "*" if idx in legal else "."
                row.append(ch)
            lines.append(f"{r + 1:2d} " + " ".join(row))
        b, w = self.score()
        turn = "game over" if self.done else ("Black (X)" if self.current == BLACK else "White (O)") + " to move"
        lines.append(f"Black {b} - White {w}   {turn}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()
