"""Tests for the exact endgame solver, MCTS and the AlphaZero helpers (pytest)."""
import random

import numpy as np
import pytest

from env import OthelloEnv, get_flips, get_moves
from solver import SolverLimit, best_moves, final_score, solve


def brute(own: int, opp: int) -> int:
    """Plain minimax to the end, no pruning: the reference for the solver."""
    moves = get_moves(own, opp)
    if not moves:
        if not get_moves(opp, own):
            return final_score(own, opp)
        return -brute(opp, own)
    best = -1000
    for sq in range(64):
        bit = 1 << sq
        if moves & bit:
            flips = get_flips(own, opp, bit)
            best = max(best, -brute(opp & ~flips, own | bit | flips))
    return best


def endgame_positions(empties: int, n: int, seed: int):
    rng, out = random.Random(seed), []
    while len(out) < n:
        env = OthelloEnv()
        while not env.done and 64 - (env.black | env.white).bit_count() > empties:
            env.step(rng.choice(env.legal_moves()))
        if not env.done:
            out.append(env._own_opp())
    return out


@pytest.mark.parametrize("empties", [3, 5, 7])
def test_solver_matches_brute_force(empties):
    for own, opp in endgame_positions(empties, 12, seed=empties):
        scores, _ = solve(own, opp, exact_all=True)
        for sq, value in scores.items():
            flips = get_flips(own, opp, 1 << sq)
            assert value == -brute(opp & ~flips, own | (1 << sq) | flips)


def test_solver_best_moves_are_exactly_the_optimal_ones():
    for own, opp in endgame_positions(6, 15, seed=99):
        exact, _ = solve(own, opp, exact_all=True)
        best, moves, _ = best_moves(own, opp)
        assert best == max(exact.values())
        assert sorted(moves) == sorted(m for m, v in exact.items() if v == best)


def test_solver_handles_passes():
    """Positions where the side to move must pass (the env hides these, so build them by hand)."""
    from solver import Solver
    rng, found = random.Random(3), 0
    while found < 6:
        env = OthelloEnv()
        while not env.done:
            own, opp = env._own_opp()
            move = rng.choice(env.legal_moves())
            bit = 1 << move
            flips = get_flips(own, opp, bit)
            n_own, n_opp = own | bit | flips, opp & ~flips
            if (not get_moves(n_opp, n_own) and get_moves(n_own, n_opp)
                    and 64 - (n_own | n_opp).bit_count() <= 8):
                assert Solver().negamax(n_opp, n_own, -1000, 1000) == brute(n_opp, n_own)
                found += 1
                break
            env.step(move)


def test_solver_node_limit():
    own, opp = endgame_positions(12, 1, seed=5)[0]
    with pytest.raises(SolverLimit):
        solve(own, opp, node_limit=50)


# ------------------------------------------------------------------------- MCTS
class UniformEvaluator:
    """Stand-in network: uniform priors over legal moves, value 0 (so only terminal results matter)."""

    def __call__(self, own, opp, legal):
        from vec_env import unpack
        mask = unpack(legal).astype(np.float32)
        return mask / mask.sum(1, keepdims=True), np.zeros(len(own), np.float32)


def test_mcts_only_plays_legal_moves():
    from mcts import BatchedMCTS
    from vec_env import U
    rng = random.Random(1)
    envs = []
    for _ in range(20):
        env = OthelloEnv()
        for _ in range(rng.randrange(0, 40)):
            if env.done:
                break
            env.step(rng.choice(env.legal_moves()))
        if not env.done:
            envs.append(env)
    own = np.array([e._own_opp()[0] for e in envs], U)
    opp = np.array([e._own_opp()[1] for e in envs], U)
    visits, _, _ = BatchedMCTS(UniformEvaluator(), sims=30).search(own, opp)
    for e, v in zip(envs, visits):
        assert v.sum() == 30
        assert set(np.flatnonzero(v)) <= set(e.legal_moves())


@pytest.mark.parametrize("empties", [4, 6])
def test_mcts_finds_the_winning_side_in_endgames(empties):
    """With exact terminal values, a deep enough search must agree with the solver on win/draw/loss.
    Many endgames contain passes, so this also checks that signs are not flipped across a pass."""
    from mcts import BatchedMCTS
    from vec_env import U
    positions = endgame_positions(empties, 25, seed=empties + 40)
    own = np.array([p[0] for p in positions], U)
    opp = np.array([p[1] for p in positions], U)
    visits, _, _ = BatchedMCTS(UniformEvaluator(), sims=1500, c_puct=1.0).search(own, opp)
    wrong = 0
    for (o, p_), v in zip(positions, visits):
        scores, _ = solve(o, p_, exact_all=True)
        best = max(scores.values())
        pick = scores[int(v.argmax())]
        wrong += np.sign(pick) != np.sign(best)
    assert wrong <= 1  # rare: a near-tied move can out-visit the best one


def test_mcts_value_sign_is_from_the_player_to_move():
    from mcts import BatchedMCTS
    from vec_env import U
    # find a position that is a forced win for the side to move and one that is a forced loss
    for own, opp in endgame_positions(6, 60, seed=77):
        best, _, _ = best_moves(own, opp)
        if best != 0:
            _, _, val = BatchedMCTS(UniformEvaluator(), sims=1500, c_puct=1.0).search(
                np.array([own], U), np.array([opp], U))
            assert np.sign(val[0]) == np.sign(best)
            return
    pytest.fail("no decisive endgame found")


# ------------------------------------------------------------------ AlphaZero helpers
def test_symmetry_keeps_policy_targets_on_legal_squares():
    """After any of the 8 board symmetries the policy target must still sit exactly on the legal moves
    of the transformed board (checked by recomputing the legal moves from the transformed stones)."""
    import torch

    from alphazero import apply_symmetry, symmetry_tables
    tables = symmetry_tables("cpu")
    rng = random.Random(5)
    obs, pis = [], []
    for _ in range(40):
        env = OthelloEnv()
        for _ in range(rng.randrange(0, 50)):
            if env.done:
                break
            env.step(rng.choice(env.legal_moves()))
        if env.done:
            continue
        obs.append(env.get_observation())
        mask = env.legal_mask().astype(np.float32)
        pis.append(mask / mask.sum())
    obs, pi = torch.tensor(np.stack(obs)), torch.tensor(np.stack(pis))
    for sym in range(8):
        o, p = apply_symmetry(obs, pi, sym, *tables)
        assert torch.allclose(p.sum(1), torch.ones(len(p)))
        for k in range(len(o)):
            bits = lambda ch: sum(1 << i for i, v in enumerate(o[k, ch].flatten().tolist()) if v > 0.5)
            own, opp = bits(0), bits(1)
            assert bits(2) == get_moves(own, opp)  # transformed legal channel is consistent
            assert {i for i, v in enumerate(p[k].tolist()) if v > 0} == {i for i in range(64) if bits(2) >> i & 1}


def test_self_play_targets_are_well_formed():
    import torch

    from alphazero import self_play
    from agents import build_net
    from mcts import Evaluator
    net = build_net({"channels": 8, "blocks": 1, "dueling": False}).eval()
    obs, pi, z, info = self_play(Evaluator(net, "ppo", "cpu"), games=6, sims=8, rng=np.random.default_rng(0),
                                 solve_empties=6)
    assert obs.shape[1:] == (3, 8, 8) and len(obs) == len(pi) == len(z) == info["positions"]
    assert np.allclose(pi.sum(1), 1.0, atol=1e-4)
    legal = obs[:, 2].reshape(len(obs), 64) > 0
    assert (pi[~legal] == 0).all()  # never any probability on an illegal move
    assert np.all(np.abs(z) <= 1.0 + 1e-6)
