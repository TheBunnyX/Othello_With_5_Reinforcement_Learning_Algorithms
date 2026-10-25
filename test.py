"""Tests and evaluation.

    python -m pytest test.py -q              unit tests (rules, network, training pieces)
    python test.py                           tournament of all weights/*.pt vs random, greedy and each other
    python test.py --games 200 --device cuda
    python test.py --search-games 20 --alphabeta-depth 6   (minimax / alpha-beta columns)
    python test.py --no-search                             (skip the slower search opponents)
"""
from __future__ import annotations

import argparse
import glob
import os
import random

import numpy as np
import pytest
import torch

from agents import (
    ALGORITHMS, GreedyAgent, NeuralAgent, RandomAgent, VecGreedy, VecNeural, VecRandom, build_net,
    load_checkpoint, masked, play_game, play_match, save_checkpoint, vec_match,
)
from env import BLACK, EMPTY, WHITE, OthelloEnv, parse_square, square, square_name
from search import AlphaBetaAgent, MinimaxAgent, VecSequential
from vec_env import VecOthello, random_actions


# ------------------------------------------------------------------ env rules
def test_initial_position():
    env = OthelloEnv()
    assert env.score() == (2, 2)
    assert env.current == BLACK
    assert sorted(square_name(m) for m in env.legal_moves()) == ["c4", "d3", "e6", "f5"]


def test_move_flips_stone():
    env = OthelloEnv()
    _, r, done, info = env.step(parse_square("d3"))
    assert info["flips"] == 1 and r == 0 and not done
    assert env.score() == (4, 1)
    assert env.board_array()[3, 3] == BLACK  # d4 flipped
    assert env.current == WHITE


def test_illegal_move_rejected():
    env = OthelloEnv()
    with pytest.raises(ValueError):
        env.step(square(0, 0))


def test_flips_in_multiple_directions():
    # Black plays a1 and flanks white lines along the row, the column and the diagonal.
    board = np.zeros((8, 8), dtype=int)
    board[0, 4] = board[4, 0] = board[4, 4] = BLACK
    for i in (1, 2, 3):
        board[0, i] = board[i, 0] = board[i, i] = WHITE
    env = OthelloEnv.from_array(board, BLACK)
    _, _, _, info = env.step(square(0, 0))
    assert info["flips"] == 9
    assert env.score() == (13, 0)


def test_pass_keeps_same_player():
    board = np.zeros((8, 8), dtype=int)
    board[0, 0], board[0, 1] = BLACK, WHITE
    board[7, 0], board[7, 1] = BLACK, WHITE
    env = OthelloEnv.from_array(board, BLACK)
    _, _, done, info = env.step(square(0, 2))
    assert not done and info["passed"] and env.current == BLACK


def test_game_over_and_reward():
    board = np.zeros((8, 8), dtype=int)
    board[0, 0], board[0, 1] = BLACK, WHITE
    env = OthelloEnv.from_array(board, BLACK)
    _, r, done, info = env.step(square(0, 2))
    assert done and r == 1.0 and env.winner == BLACK and info["winner"] == BLACK
    assert env.legal_moves() == []


def test_observation_is_canonical():
    env = OthelloEnv()
    env.step(parse_square("d3"))  # white to move
    obs = env.get_observation()
    assert obs.shape == (3, 8, 8) and obs.dtype == np.float32
    assert obs[0].sum() == 1 and obs[1].sum() == 4  # own = white
    assert np.array_equal(obs[2].reshape(-1).astype(bool), env.legal_mask())


def test_random_games_are_consistent():
    random.seed(1)
    for _ in range(50):
        env = OthelloEnv()
        while not env.done:
            b, w = env.score()
            moves = env.legal_moves()
            assert moves
            env.step(random.choice(moves))
            nb, nw = env.score()
            assert nb + nw == b + w + 1
        b, w = env.score()
        assert env.winner == (BLACK if b > w else WHITE if w > b else EMPTY)


def test_from_array_roundtrip():
    env = OthelloEnv()
    for m in ("d3", "c3", "c4"):
        env.step(parse_square(m))
    copy = OthelloEnv.from_array(env.board_array(), env.current)
    assert copy.black == env.black and copy.white == env.white and copy.current == env.current


# ------------------------------------------------------------- vectorised env
def test_vec_env_matches_scalar_env():
    rng = np.random.default_rng(0)
    n = 64
    venv, envs = VecOthello(n), [OthelloEnv() for _ in range(n)]
    while not venv.done.all():
        mask = venv.legal_mask()
        obs = venv.observation()
        for i, e in enumerate(envs):
            if not e.done:
                assert np.array_equal(mask[i], e.legal_mask())
                assert np.array_equal(obs[i].astype(np.float32), e.get_observation())
        acts = random_actions(mask, rng)
        reward, ended, passed, mover = venv.step(acts)
        for i, e in enumerate(envs):
            if not e.done:
                _, r, d, info = e.step(int(acts[i]))
                assert (r, d, info["passed"], info["player"]) == (reward[i], ended[i], passed[i], mover[i])
    for i, e in enumerate(envs):
        assert venv.winner[i] == e.winner
        assert venv.board_string(i).count("b") == e.score()[0]


def test_vec_env_rejects_illegal_and_resets():
    venv = VecOthello(3)
    with pytest.raises(ValueError):
        venv.step(np.zeros(3, dtype=int))
    venv.done[1] = True
    venv.reset(np.array([False, True, False]))
    assert not venv.done.any() and venv.legal_mask().sum(1).tolist() == [4, 4, 4]


def test_vec_match_baselines():
    r = vec_match(VecGreedy(), VecRandom(), games=100)
    assert r["wins"] + r["losses"] + r["draws"] == 100 and r["win_rate"] > 0.6
    rec = vec_match(VecRandom(), VecRandom(), games=4, record=True)
    assert rec["boards"][0].count(".") == 60 and len(rec["boards"]) > 50


def test_masked_never_picks_illegal_even_with_huge_logits():
    head = torch.tensor([[-1e12, 5.0, 3.0], [1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, False], [False, False, False]])
    out = masked(head, mask)
    assert out[0].argmax() == 0  # only legal move wins no matter how negative
    assert torch.isfinite(out[1]).all()  # row without legal moves stays finite (no NaN softmax)


def test_signed_gae_negamax():
    from training import signed_gae
    # two plies, players alternate, mover of the last ply wins (+1)
    rew = torch.tensor([[0.0], [1.0]])
    zero = torch.zeros(2, 1)
    player = torch.tensor([[1], [-1]])
    adv = signed_gae(rew, zero, next_val=zero, player=player,
                     next_player=torch.tensor([[-1], [0]]), cont=torch.tensor([[1.0], [0.0]]), gamma=1.0, lam=1.0)
    assert adv[:, 0].tolist() == [-1.0, 1.0]  # the first mover lost


# ------------------------------------------------------------ network & agents
@pytest.mark.parametrize("dueling", [False, True])
def test_network_shapes(dueling):
    net = build_net({"channels": 16, "blocks": 1, "dueling": dueling})
    head, value = net(torch.zeros(5, 3, 8, 8))
    assert head.shape == (5, 64) and value.shape == (5,)


def test_checkpoint_roundtrip_and_agent_plays_legal(tmp_path):
    cfg = {"channels": 16, "blocks": 1, "dueling": False}
    net = build_net(cfg)
    path = str(tmp_path / "ppo.pt")
    save_checkpoint(path, net, "ppo", cfg, {"games": 3})
    net2, ckpt = load_checkpoint(path)
    assert ckpt["algo"] == "ppo" and ckpt["stats"]["games"] == 3
    for k, v in net.state_dict().items():
        assert torch.equal(v, net2.state_dict()[k])
    agent = NeuralAgent.load(path)
    assert play_game(agent, RandomAgent()) in (BLACK, WHITE, EMPTY)


def test_greedy_beats_random():
    random.seed(0)
    assert play_match(GreedyAgent(), RandomAgent(), 40)["win_rate"] > 0.6


def test_symmetry_permutations():
    from training import SYM_PERMS, transform_obs
    env = OthelloEnv()
    env.step(parse_square("d3"))
    obs = torch.from_numpy(env.get_observation()).unsqueeze(0)
    for sym, perm in enumerate(SYM_PERMS):
        t = transform_obs(obs, sym)[0].reshape(3, -1)
        for a in env.legal_moves():
            assert t[2, perm[a]] == 1  # legal move maps onto a legal move


@pytest.mark.parametrize("algo", ALGORITHMS)
def test_each_algorithm_trains(algo, tmp_path):
    from training import main
    main(["--algo", algo, "--device", "cpu", "--games", "16", "--n-envs", "8", "--eval-every", "16",
          "--eval-games", "2", "--channels", "8", "--blocks", "1", "--learn-start", "64", "--batch-size", "32",
          "--save-dir", str(tmp_path), "--log-dir", str(tmp_path)])
    assert os.path.exists(tmp_path / f"{algo}.pt")
    import json
    status = json.load(open(tmp_path / f"{algo}_status.json"))
    assert status["state"] == "done" and status["games"] >= 16 and status["eval"]["sample"]["boards"]


# ------------------------------------------------------------- search agents
def _random_position(plies: int, seed: int) -> OthelloEnv:
    rnd = random.Random(seed)
    env = OthelloEnv()
    while env.move_count < plies and not env.done:
        env.step(rnd.choice(env.legal_moves()))
    return env


@pytest.mark.parametrize("plies", [4, 20, 40])
def test_alphabeta_matches_minimax(plies):
    from search import Searcher
    own, opp = _random_position(plies, seed=plies)._own_opp()
    minimax, alphabeta = Searcher(alpha_beta=False), Searcher(alpha_beta=True)
    assert minimax.root(own, opp, 3, exact=True) == alphabeta.root(own, opp, 3, exact=True)
    assert alphabeta.nodes < minimax.nodes  # same answer, fewer positions


def test_search_scores_game_end_and_handles_pass():
    from search import WIN_SCORE, Searcher
    board = np.zeros((8, 8), dtype=int)
    board[0, 0], board[0, 1] = BLACK, WHITE  # black c1 captures the last white stone
    env = OthelloEnv.from_array(board, BLACK)
    assert Searcher(alpha_beta=True).root(*env._own_opp(), 3, exact=True) == {square(0, 2): WIN_SCORE + 3}

    board[7, 0], board[7, 1] = BLACK, WHITE  # after c1 white must pass and black moves again
    env = OthelloEnv.from_array(board, BLACK)
    for agent in (MinimaxAgent(4), AlphaBetaAgent(4)):
        assert agent.select_action(env) in env.legal_moves()


def test_search_agents_beat_random():
    random.seed(0)
    assert play_match(AlphaBetaAgent(2), RandomAgent(), 10)["win_rate"] >= 0.8
    assert play_match(MinimaxAgent(1), RandomAgent(), 10)["win_rate"] >= 0.6


def test_search_agent_in_vec_match():
    r = vec_match(VecSequential(MinimaxAgent(1)), VecRandom(), games=4)
    assert r["wins"] + r["losses"] + r["draws"] == 4


# ------------------------------------------------------------------- web app
def test_web_search_opponents(monkeypatch):
    import app as web
    monkeypatch.setenv("ALPHABETA_DEPTH", "2")
    c = web.app.test_client()
    models = {m["id"]: m for m in c.get("/api/models").get_json()}
    assert models["alphabeta"]["depth"] == 2 and models["minimax"]["algo"] == "search"
    s = c.post("/api/new", json={}).get_json()
    r = c.post("/api/ai", json={"board": s["board"], "current": 1, "model": "alphabeta"}).get_json()
    assert r["move"] in s["legal"]
    assert r["analysis"]["kind"] == "eval" and r["analysis"]["stats"]["depth"] == 2
    h = c.post("/api/hint", json={"board": s["board"], "current": 1, "model": "minimax"}).get_json()
    assert h["analysis"]["best"] in s["legal"]


@pytest.mark.parametrize("value, enabled", [("true", True), ("false", False), ("0", False), ("on", True)])
def test_training_monitor_toggle(monkeypatch, value, enabled):
    import app as web
    monkeypatch.setenv("ENABLE_TRAINING_MONITOR", value)
    c = web.app.test_client()
    assert c.get("/api/config").get_json() == {"training_monitor": enabled}
    expected = 200 if enabled else 404
    page = c.get("/monitor")
    assert page.status_code == expected
    if not enabled:  # browsers get an explanation page, APIs get JSON
        assert page.content_type.startswith("text/html") and b"ENABLE_TRAINING_MONITOR=true" in page.data
        assert c.get("/api/monitor").get_json()["error"].startswith("Training Monitor is disabled")
    assert c.get("/api/monitor").status_code == expected
    assert c.post("/api/train/stop", json={}).status_code == (409 if enabled else 404)
    assert c.get("/").status_code == 200  # the game itself is always available


def test_dotenv_parsing(tmp_path, monkeypatch):
    import app as web
    env = tmp_path / ".env"
    env.write_text('# comment\nexport ENABLE_TRAINING_MONITOR="false"  # off\nOTHER = x\n', encoding="utf-8")
    monkeypatch.setattr(web, "ENV_PATH", str(env))
    monkeypatch.setattr(web, "_dotenv", {"mtime": None, "values": {}})
    monkeypatch.delenv("ENABLE_TRAINING_MONITOR", raising=False)
    assert web.read_dotenv() == {"ENABLE_TRAINING_MONITOR": "false", "OTHER": "x"}
    assert not web.monitor_enabled()


# ------------------------------------------------------------------ tournament
def tournament(weights_dir: str, games: int, device: str, search_games: int = 10,
               minimax_depth: int = 3, alphabeta_depth: int = 5, with_search: bool = True):
    paths = sorted(p for p in glob.glob(os.path.join(weights_dir, "*.pt")) if not p.endswith("_last.pt"))
    neural = [VecNeural.from_agent(NeuralAgent.load(p, device)) for p in paths]
    search = [VecSequential(MinimaxAgent(minimax_depth)),
              VecSequential(AlphaBetaAgent(alphabeta_depth))] if with_search else []
    if not neural:
        print(f"No weights in {weights_dir}/ - train first: python training.py --algo all")
    players = search + neural
    if not players:
        return
    columns = [VecRandom(), VecGreedy()] + players
    is_search = lambda x: isinstance(x, VecSequential)  # one game at a time -> fewer games

    print(f"\nWin rate (draw = 0.5), colours alternate: {games} games per pairing, "
          f"{search_games} when a search agent plays\n")
    w = max(11, max(len(c.name) for c in columns) + 2)
    header = f"{'agent':<{w}}" + "".join(f"{c.name:>{w}}" for c in columns)
    print(header)
    print("-" * len(header))
    totals = {}
    for a in players:
        row, rates = f"{a.name:<{w}}", []
        for opp in columns:
            if opp is a:
                row += f"{'-':>{w}}"
                continue
            n = search_games if is_search(a) or is_search(opp) else games
            rate = vec_match(a, opp, n)["win_rate"]
            rates.append(rate)
            row += f"{rate:>{w}.1%}"
        totals[a.name] = float(np.mean(rates))
        print(row, flush=True)

    print("\nRanking (mean win rate):")
    for i, (n, s) in enumerate(sorted(totals.items(), key=lambda x: -x[1]), 1):
        print(f"  {i}. {n:<{w}} {s:.1%}")


def main():
    p = argparse.ArgumentParser(description="Evaluate trained Othello agents.")
    p.add_argument("--weights-dir", default="weights")
    p.add_argument("--games", type=int, default=400, help="games per pairing (played in parallel)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--search-games", type=int, default=10, help="games per pairing involving minimax / alphabeta")
    p.add_argument("--minimax-depth", type=int, default=3)
    p.add_argument("--alphabeta-depth", type=int, default=5)
    p.add_argument("--no-search", action="store_true", help="leave the search agents out")
    args = p.parse_args()
    random.seed(0)
    tournament(args.weights_dir, args.games, args.device, args.search_games,
               args.minimax_depth, args.alphabeta_depth, not args.no_search)


if __name__ == "__main__":
    main()
