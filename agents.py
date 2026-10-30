"""Neural network, checkpoint I/O and agents shared by training, play, test and the web app."""
from __future__ import annotations

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from env import BLACK, NUM_ACTIONS, POSITION_WEIGHTS, WHITE, OthelloEnv, get_flips
from vec_env import VecOthello, greedy_actions, random_actions

ALGORITHMS = ("dqn", "ddqn", "reinforce", "a2c", "ppo")
VALUE_BASED = ("dqn", "ddqn")
ALGO_NAMES = {
    "dqn": "Deep Q-Network",
    "ddqn": "Double Dueling DQN",
    "reinforce": "REINFORCE",
    "a2c": "Advantage Actor-Critic",
    "ppo": "Proximal Policy Optimization",
}


# ---------------------------------------------------------------------- device
def get_device(name: str = "auto") -> torch.device:
    """'auto' picks CUDA when available, otherwise CPU."""
    name = (name or "auto").lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is False. "
            "Install a CUDA build of PyTorch or use --device cpu."
        )
    return torch.device(name)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)})"
    return "cpu"


# --------------------------------------------------------------------- network
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return F.relu(x + self.c2(F.relu(self.c1(x))))


class OthelloNet(nn.Module):
    """Residual CNN with a 64-way head (policy logits or Q-values) and a scalar value head.

    With dueling=True the 64-way head is combined with the value head as
    Q = V + A - mean(A)  (Dueling DQN).
    """

    def __init__(self, channels: int = 64, blocks: int = 4, dueling: bool = False):
        super().__init__()
        self.dueling = dueling
        self.channels_last = False  # set by the trainer with --amp: tensor cores like NHWC
        # No BatchNorm: it behaves differently in train / eval mode, which breaks
        # PPO's probability ratios and makes self-play targets drift.
        self.stem = nn.Sequential(nn.Conv2d(3, channels, 3, padding=1), nn.ReLU())
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.head_conv = nn.Sequential(nn.Conv2d(channels, 2, 1), nn.ReLU())
        self.head_fc = nn.Linear(2 * 64, NUM_ACTIONS)
        self.value_conv = nn.Sequential(nn.Conv2d(channels, 1, 1), nn.ReLU())
        self.value_fc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 1), nn.Tanh())

    def forward(self, x):
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)
        h = self.body(self.stem(x))
        head = self.head_fc(self.head_conv(h).flatten(1))
        value = self.value_fc(self.value_conv(h).flatten(1)).squeeze(-1)
        if self.dueling:
            head = value.unsqueeze(-1) + head - head.mean(dim=1, keepdim=True)
        return head, value


def masked(head: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Illegal moves -> -inf.  Rows without any legal move (finished games) stay unmasked to avoid NaN."""
    mask = mask | ~mask.any(1, keepdim=True)
    return head.masked_fill(~mask, float("-inf"))


def obs_mask(obs: torch.Tensor) -> torch.Tensor:
    """Legal-move mask is channel 2 of the observation."""
    return obs[:, 2].flatten(1) > 0.5


# ------------------------------------------------------------------ checkpoint
def build_net(config: dict) -> OthelloNet:
    return OthelloNet(channels=config["channels"], blocks=config["blocks"], dueling=config.get("dueling", False))


def save_checkpoint(path: str, net: OthelloNet, algo: str, config: dict, stats: dict | None = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    torch.save({"algo": algo, "config": config, "state_dict": state, "stats": stats or {}}, path)


def load_checkpoint(path: str, device: torch.device | str = "cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    net = build_net(ckpt["config"])
    net.load_state_dict(ckpt["state_dict"])
    net.to(device).eval()
    return net, ckpt


# ---------------------------------------------------------------------- agents
class RandomAgent:
    name = "random"

    def select_action(self, env: OthelloEnv) -> int:
        return random.choice(env.legal_moves())


class GreedyAgent:
    """One-ply heuristic: positional weight of the square + stones flipped."""
    name = "greedy"

    def select_action(self, env: OthelloEnv) -> int:
        own, opp = env._own_opp()
        best, best_score = None, -1e9
        for m in env.legal_moves():
            s = POSITION_WEIGHTS[m] + get_flips(own, opp, 1 << m).bit_count() + random.random() * 0.1
            if s > best_score:
                best, best_score = m, s
        return best


class NeuralAgent:
    """Wraps a trained network.  Picks argmax over legal moves (or samples with temperature)."""

    def __init__(self, net: OthelloNet, algo: str, device, name: str | None = None, temperature: float = 0.0):
        self.net = net.eval()
        self.algo = algo
        self.device = torch.device(device)
        self.name = name or algo
        self.temperature = temperature

    @classmethod
    def load(cls, path: str, device="cpu", temperature: float = 0.0) -> "NeuralAgent":
        net, ckpt = load_checkpoint(path, device)
        name = os.path.splitext(os.path.basename(path))[0]
        agent = cls(net, ckpt["algo"], device, name=name, temperature=temperature)
        agent.stats = ckpt.get("stats", {})
        return agent

    @torch.no_grad()
    def evaluate(self, env: OthelloEnv):
        """Returns (scores[64] for legal moves, value estimate for the player to move)."""
        obs = torch.from_numpy(env.get_observation()).unsqueeze(0).to(self.device)
        head, value = self.net(obs)
        head = masked(head, obs_mask(obs))[0]
        if self.algo in VALUE_BASED:
            scores = head
            v = head.max()  # Q of best move ~ value of state
        else:
            scores = torch.softmax(head, dim=0)
            v = value[0]
        return scores.cpu().numpy(), float(v.clamp(-1, 1))

    def select_action(self, env: OthelloEnv) -> int:
        legal = env.legal_moves()
        if len(legal) == 1:
            return legal[0]
        scores, _ = self.evaluate(env)
        if self.temperature > 0 and self.algo not in VALUE_BASED:
            p = scores[legal] ** (1.0 / self.temperature)
            return int(np.random.choice(legal, p=p / p.sum()))
        return int(legal[int(np.argmax(scores[legal]))])


def play_game(black_agent, white_agent, env: OthelloEnv | None = None, random_opening: int = 0) -> int:
    """Plays one game, returns winner (BLACK, WHITE or 0 for a draw).
    The first `random_opening` plies are random so deterministic agents see varied games."""
    env = env or OthelloEnv()
    env.reset()
    while not env.done:
        if env.move_count < random_opening:
            env.step(random.choice(env.legal_moves()))
            continue
        agent = black_agent if env.current == BLACK else white_agent
        env.step(agent.select_action(env))
    return env.winner


def play_match(agent, opponent, games: int = 100, random_opening: int = 2) -> dict:
    """Agent plays `games` games, alternating colours.  Results from agent's view."""
    wins = losses = draws = 0
    for g in range(games):
        if g % 2 == 0:
            w = play_game(agent, opponent, random_opening=random_opening)
            me = BLACK
        else:
            w = play_game(opponent, agent, random_opening=random_opening)
            me = WHITE
        if w == me:
            wins += 1
        elif w == 0:
            draws += 1
        else:
            losses += 1
    return {"wins": wins, "losses": losses, "draws": draws, "win_rate": (wins + 0.5 * draws) / max(games, 1)}



# ------------------------------------------------------- vectorised matches
# Policies that pick moves for every game of a VecOthello at once.
# act(venv, mask, rng, active) -> actions; `active` marks the games where it is this policy's turn.  Used for
# fast evaluation during training and for the tournament in test.py.
class VecRandom:
    name = "random"

    def act(self, venv, mask, rng, active=None):
        return random_actions(mask, rng)


class VecGreedy:
    name = "greedy"

    def act(self, venv, mask, rng, active=None):
        return greedy_actions(venv, mask, rng)


class VecNeural:
    def __init__(self, net: OthelloNet, device, name: str = "net"):
        self.net, self.device, self.name = net, torch.device(device), name

    @classmethod
    def from_agent(cls, agent: NeuralAgent) -> "VecNeural":
        return cls(agent.net, agent.device, agent.name)

    @torch.inference_mode()
    def act(self, venv, mask, rng, active=None):
        obs = torch.from_numpy(venv.observation()).to(self.device).float()
        head, _ = self.net(obs)
        return masked(head.float(), obs_mask(obs)).argmax(1).cpu().numpy()


def vec_match(agent, opponent, games: int = 200, random_opening: int = 2, seed: int = 0,
              record: bool = False) -> dict:
    """`agent` plays black in even games and white in odd games, all games in parallel.
    The first `random_opening` plies are random so deterministic agents see varied positions.
    With record=True the boards of game 0 (agent = black) are returned as 64-char strings."""
    rng = np.random.default_rng(seed)
    venv = VecOthello(games)
    side = np.where(np.arange(games) % 2 == 0, BLACK, WHITE).astype(np.int8)
    boards = [venv.board_string(0)] if record else None
    while not venv.done.all():
        mask = venv.legal_mask()
        # `active` tells each side which games are its turn (vectorised agents may ignore it,
        # sequential ones such as search agents use it to skip the other side's games)
        mine = (venv.player == side) & (venv.ply >= random_opening)
        theirs = (venv.player != side) & (venv.ply >= random_opening)
        a = np.where(mine, agent.act(venv, mask, rng, active=mine), opponent.act(venv, mask, rng, active=theirs))
        a = np.where(venv.ply < random_opening, random_actions(mask, rng), a)
        was_done = bool(venv.done[0])
        venv.step(a)
        if record and not was_done:
            boards.append(venv.board_string(0))
    result = venv.winner * side
    wins, losses = int((result > 0).sum()), int((result < 0).sum())
    draws = games - wins - losses
    out = {"wins": wins, "losses": losses, "draws": draws, "win_rate": (wins + 0.5 * draws) / max(games, 1)}
    if record:
        out["boards"] = boards
        out["first_result"] = int(result[0])  # game 0 from the agent's view: 1 win, -1 loss, 0 draw
    return out
