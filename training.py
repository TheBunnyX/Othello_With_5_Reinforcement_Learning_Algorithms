"""Train Othello agents by self-play with 5 RL algorithms.

    dqn        Deep Q-Network (replay buffer + target network)
    ddqn       Double DQN with a Dueling head
    reinforce  Monte-Carlo policy gradient
    a2c        Advantage Actor-Critic (GAE, single pass per rollout)
    ppo        Proximal Policy Optimization (clipped, multi-epoch)

Both colours are played by the same network on canonical observations
(the board as seen by the player to move).  Because Othello is zero-sum,
bootstrapped values change sign when the turn passes to the opponent
("negamax"), and keep their sign when the opponent has to pass.

Speed-ups:
    * VecOthello steps hundreds of games at once with numpy bitboards (--n-envs)
    * evaluation runs all evaluation games in parallel
    * DQN replay buffer lives on the training device (no host->GPU copy per batch)
    * --parallel trains several algorithms at the same time, one process each
    * --amp uses bfloat16 autocast on CUDA; TF32 matmuls are always enabled

Progress is written to logs/<algo>_status.json and logs/<algo>.jsonl, which
the monitor GUI (python monitor.py) reads.

Examples:
    python training.py --algo all --device cuda --parallel
    python training.py --algo ppo,a2c --device cpu --games 20000
    python training.py --algo dqn --device auto --games 200000 --resume
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import time
import traceback
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from agents import (
    ALGO_NAMES, ALGORITHMS, VecGreedy, VecNeural, VecRandom, build_net, describe_device,
    get_device, load_checkpoint, masked, obs_mask, save_checkpoint, vec_match,
)
from vec_env import VecOthello, random_actions

# Per-algorithm default learning rates.
DEFAULT_LR = {"dqn": 1e-4, "ddqn": 1e-4, "reinforce": 3e-4, "a2c": 3e-4, "ppo": 3e-4}
# Per-algorithm default number of parallel games (--n-envs).
DEFAULT_ENVS = {"dqn": 256, "ddqn": 256, "reinforce": 64, "a2c": 64, "ppo": 256}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def setup_torch(args):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if args.threads:
        torch.set_num_threads(args.threads)


# ------------------------------------------------------------- symmetries (DQN)
def _symmetry_perms() -> list[np.ndarray]:
    """For each of the 8 board symmetries: perm[a] = index of square a after the transform."""
    base = np.arange(64).reshape(8, 8)
    perms = []
    for k in range(4):
        for flip in (False, True):
            t = np.rot90(base, k)
            t = np.fliplr(t) if flip else t
            perms.append(np.argsort(t.reshape(-1)))
    return perms


SYM_PERMS = _symmetry_perms()


def transform_obs(x: torch.Tensor, sym: int) -> torch.Tensor:
    k, flip = divmod(sym, 2)
    x = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if flip else x


# ---------------------------------------------------------- monitor output
class RunMonitor:
    """Writes logs/<algo>_status.json (latest state) and logs/<algo>.jsonl (history) for the GUI."""

    def __init__(self, algo: str, args, device, resume: bool):
        os.makedirs(args.log_dir, exist_ok=True)
        self.status_path = os.path.join(args.log_dir, f"{algo}_status.json")
        self.history_path = os.path.join(args.log_dir, f"{algo}.jsonl")
        if not resume:
            open(self.history_path, "w").close()
        self.status = {
            "algo": algo, "label": ALGO_NAMES[algo], "state": "starting", "pid": os.getpid(),
            "device": describe_device(device), "total_games": args.games, "started": time.time(),
            "config": {k: getattr(args, k) for k in ("n_envs", "batch_size", "lr", "channels", "blocks", "amp")},
        }

    def write(self, **fields):
        self.status.update(fields)
        self.status["updated"] = time.time()
        tmp = f"{self.status_path}.{os.getpid()}.tmp"
        with open(tmp, "w") as f:
            json.dump(self.status, f)
        for _ in range(20):  # the web server may be reading the file (Windows locks it)
            try:
                os.replace(tmp, self.status_path)
                return
            except PermissionError:
                time.sleep(0.05)

    def log(self, record: dict):
        with open(self.history_path, "a") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------- base trainer
class Trainer:
    fixed_shapes = False  # cudnn.benchmark only pays off when batch shapes never change
    def __init__(self, algo: str, args, device: torch.device):
        self.algo, self.args, self.device = algo, args, device
        self.config = {"channels": args.channels, "blocks": args.blocks, "dueling": algo == "ddqn"}
        self.net = build_net(self.config).to(device)
        self.lr = args.lr or DEFAULT_LR[algo]
        args.lr = self.lr
        # fused Adam = one CUDA kernel per step instead of several per parameter tensor
        self.graph_on = bool(getattr(args, "graph", False) and device.type == "cuda")
        self.opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, fused=device.type == "cuda",
                                    capturable=self.graph_on)
        torch.backends.cudnn.benchmark = self.fixed_shapes and device.type == "cuda"
        # bf16 only pays off inside a graphed update; REINFORCE's batch size varies, so it stays fp32
        self.amp = bool((args.amp or (self.graph_on and algo != "reinforce")) and device.type == "cuda")
        self.infer_graphs = {}
        if self.amp:
            self.net.channels_last = True
            self.net.to(memory_format=torch.channels_last)
        self.rng = np.random.default_rng(args.seed)
        self.games = 0
        self.best_score = -1.0
        self.path = os.path.join(args.save_dir, f"{algo}.pt")
        self.last_path = os.path.join(args.save_dir, f"{algo}_last.pt")

        resumed = False
        if args.resume and os.path.exists(self.last_path):
            net, ckpt = load_checkpoint(self.last_path, device)
            if ckpt["config"] == self.config:
                self.net.load_state_dict(net.state_dict())
                self.games = ckpt["stats"].get("games", 0)
                self.best_score = ckpt["stats"].get("best_score", -1.0)
                resumed = True
                print(f"[{algo}] resumed from {self.last_path} at {self.games} games")
            else:
                print(f"[{algo}] network config differs from {self.last_path}, starting fresh")
        self.monitor = RunMonitor(algo, args, device, resumed)

    def autocast(self):
        return torch.autocast("cuda", dtype=torch.bfloat16) if self.amp else nullcontext()

    def to_obs(self, obs_np: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(obs_np).to(self.device, non_blocking=True).float()

    def infer(self, obs: torch.Tensor):
        """Network forward without gradients; a CUDA graph per batch size when --graph is on."""
        if not self.graph_on:
            with torch.no_grad(), self.autocast():
                return self.net(obs)
        entry = self.infer_graphs.get(obs.shape[0])
        if entry is None:
            static = obs.clone()
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side), torch.no_grad(), self.autocast():
                for _ in range(3):
                    self.net(static)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.no_grad(), self.autocast(), torch.cuda.graph(graph):
                out = self.net(static)
            entry = self.infer_graphs[obs.shape[0]] = (graph, static, out)
        graph, static, out = entry
        static.copy_(obs)
        graph.replay()
        return tuple(o.clone() for o in out)

    def optimize(self, loss: torch.Tensor, max_norm: float):
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm)
        self.opt.step()

    def train_iteration(self) -> tuple[int, int, dict]:
        """Returns (games finished, samples collected, metrics)."""
        raise NotImplementedError

    def evaluate(self):
        agent = VecNeural(self.net, self.device, self.algo)
        n = self.args.eval_games
        r = vec_match(agent, VecRandom(), n, seed=self.games)
        g = vec_match(agent, VecGreedy(), n, seed=self.games + 1, record=True)
        return r["win_rate"], g["win_rate"], g

    def stats(self, **extra) -> dict:
        return {"games": self.games, "best_score": self.best_score, "lr": self.lr, **extra}

    def run(self):
        args = self.args
        print(f"[{self.algo}] {ALGO_NAMES[self.algo]} | {describe_device(self.device)} | {args.games} games | "
              f"{args.n_envs} envs | lr {self.lr} | amp {self.amp}", flush=True)
        start = last_t = time.time()
        last_games, samples, last_samples = self.games, 0, 0
        next_eval = (self.games // args.eval_every + 1) * args.eval_every
        losses, metrics = [], {}
        last_eval = {}
        self.monitor.write(state="running", games=self.games, elapsed=0.0)
        try:
            while self.games < args.games:
                finished, n_samples, info = self.train_iteration()
                self.games += finished
                samples += n_samples
                if "loss" in info:
                    losses.append(info.pop("loss"))
                metrics.update(info)

                now = time.time()
                if now - last_t >= args.status_interval:
                    gps = (self.games - last_games) / (now - last_t)
                    sps = (samples - last_samples) / (now - last_t)
                    loss = float(np.mean([float(x) for x in losses])) if losses else None
                    rec = {"t": "train", "games": self.games, "elapsed": round(now - start, 2),
                           "gps": round(gps, 2), "sps": round(sps, 1), "loss": loss,
                           **{k: round(float(v), 5) for k, v in metrics.items()}}
                    self.monitor.log(rec)
                    self.monitor.write(state="running", games=self.games, elapsed=now - start, gps=gps, sps=sps,
                                       eta=(args.games - self.games) / gps if gps > 0 else None,
                                       loss=loss, metrics=rec, best_score=self.best_score, eval=last_eval)
                    last_t, last_games, last_samples = now, self.games, samples
                    losses.clear()

                if self.games >= next_eval or self.games >= args.games:
                    next_eval = (self.games // args.eval_every + 1) * args.eval_every
                    last_eval = self.eval_and_save(start)
            self.monitor.write(state="done", games=self.games, elapsed=time.time() - start, eta=0,
                               best_score=self.best_score, eval=last_eval)
            print(f"[{self.algo}] done in {time.time() - start:.1f}s, best score {self.best_score:.3f} -> {self.path}",
                  flush=True)
        except KeyboardInterrupt:
            self.monitor.write(state="stopped")
            raise
        except Exception:
            self.monitor.write(state="error", error=traceback.format_exc())
            raise

    def eval_and_save(self, start: float) -> dict:
        t = time.time()
        vs_r, vs_g, game = self.evaluate()
        score = 0.5 * vs_r + 0.5 * vs_g
        stats = self.stats(vs_random=vs_r, vs_greedy=vs_g)
        best = score > self.best_score
        if best:
            self.best_score = score
            stats["best_score"] = score
            save_checkpoint(self.path, self.net, self.algo, self.config, stats)
        save_checkpoint(self.last_path, self.net, self.algo, self.config, stats)
        elapsed = time.time() - start
        self.monitor.log({"t": "eval", "games": self.games, "elapsed": round(elapsed, 2), "vs_random": vs_r,
                          "vs_greedy": vs_g, "score": score, "best": best})
        result = {"games": self.games, "vs_random": vs_r, "vs_greedy": vs_g, "score": score,
                  "seconds": round(time.time() - t, 2),
                  "sample": {"boards": game["boards"], "result": game["first_result"]}}
        self.monitor.write(eval=result, best_score=self.best_score)
        print(f"[{self.algo}] games {self.games:8d} | {elapsed:7.1f}s | vs random {vs_r:5.1%} | "
              f"vs greedy {vs_g:5.1%}{'  * saved best' if best else ''}", flush=True)
        return result


# --------------------------------------------------------------- DQN / DDQN
class ReplayBuffer:
    """Circular buffer stored on the training device; observations kept as uint8."""

    def __init__(self, capacity: int, device):
        self.capacity, self.device = capacity, device
        self.obs = torch.zeros((capacity, 3, 8, 8), dtype=torch.uint8, device=device)
        self.next_obs = torch.zeros((capacity, 3, 8, 8), dtype=torch.uint8, device=device)
        self.act = torch.zeros(capacity, dtype=torch.long, device=device)
        self.rew = torch.zeros(capacity, device=device)
        self.done = torch.zeros(capacity, device=device)
        self.same = torch.zeros(capacity, device=device)  # 1 if the same player moves next (opponent passed)
        self.size = self.pos = 0
        self.size_t = torch.zeros((), device=device)  # = size, as a tensor so CUDA graphs can read it

    def add(self, obs, act, rew, next_obs, done, same):
        n = len(act)
        idx = (torch.arange(n, device=self.device) + self.pos) % self.capacity
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(self.device, non_blocking=True)
        self.obs[idx] = t(obs)
        self.next_obs[idx] = t(next_obs)
        self.act[idx] = t(act.astype(np.int64))
        self.rew[idx] = t(rew.astype(np.float32))
        self.done[idx] = t(done.astype(np.float32))
        self.same[idx] = t(same.astype(np.float32))
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        self.size_t.fill_(self.size)

    def sample(self, n: int):
        idx = torch.minimum((torch.rand(n, device=self.device) * self.size_t).long(), (self.size_t - 1).long())
        return (self.obs[idx].float(), self.act[idx], self.rew[idx], self.next_obs[idx].float(),
                self.done[idx], self.same[idx])

    def __len__(self):
        return self.size


class DQNTrainer(Trainer):
    fixed_shapes = True
    def __init__(self, algo, args, device):
        super().__init__(algo, args, device)
        self.double = algo == "ddqn"
        self.target = copy.deepcopy(self.net).eval().requires_grad_(False)
        self.buffer = ReplayBuffer(args.buffer_size, device)
        self.venv = VecOthello(args.n_envs, seed=args.seed)
        self.perms = torch.as_tensor(np.stack(SYM_PERMS), device=device)
        # square index maps that reproduce transform_obs(x, sym) as one gather (CUDA-graph friendly)
        squares = torch.arange(64, device=device).view(1, 1, 8, 8)
        self.obs_idx = torch.stack([transform_obs(squares, k).reshape(-1) for k in range(8)])
        self.graph = None
        self.updates = 0
        self.update_credit = 0.0

    def epsilon(self) -> float:
        a = self.args
        frac = min(1.0, self.games / max(1, a.eps_decay * a.games))
        return a.eps_start + frac * (a.eps_end - a.eps_start)

    def train_iteration(self):
        a, v = self.args, self.venv
        eps = self.epsilon()
        obs_np, mask = v.observation(), v.legal_mask()
        obs = self.to_obs(obs_np)
        q, _ = self.infer(obs)
        greedy = masked(q.float(), obs_mask(obs)).argmax(1).cpu().numpy()
        explore = self.rng.random(v.n) < eps
        act = np.where(explore, random_actions(mask, self.rng), greedy)

        rew, ended, passed, _ = v.step(act)
        self.buffer.add(obs_np, act, rew, v.observation(), ended, passed & ~ended)
        v.reset(ended)

        info = {"eps": eps}
        if len(self.buffer) >= a.learn_start:
            self.update_credit += v.n * a.replay_ratio / a.batch_size
            losses = []
            while self.update_credit >= 1:
                losses.append(self.update())
                self.update_credit -= 1
            if losses:
                info["loss"] = torch.stack(losses).mean()  # stays on the GPU: no sync per step
        return int(ended.sum()), v.n, info

    def update(self) -> torch.Tensor:
        if self.graph_on and self.graph is None:
            self.capture_graph()
        if self.graph is None:
            loss = self.update_step()
        else:
            self.graph.replay()
            loss = self.graph_loss.clone()  # the static output is overwritten by the next replay

        self.updates += 1
        if self.updates % self.args.target_update == 0:
            self.target.load_state_dict(self.net.state_dict())
        return loss

    def capture_graph(self):
        """Record one whole gradient step (sample, forward, backward, clip, Adam) as a CUDA graph.
        Replaying it skips ~hundreds of Python-side kernel launches, the bottleneck at batch 512."""
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):  # warm-up: cudnn autotune and Adam state must exist before capture
            for _ in range(3):
                self.update_step()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.graph_loss = self.update_step()
        self.graph = graph

    def update_step(self) -> torch.Tensor:
        a = self.args
        o, act, r, no, d, same = self.buffer.sample(a.batch_size)
        # random board symmetry as data augmentation, drawn on the device so it can live inside a graph
        sym = torch.randint(0, 8, (1,), device=self.device)
        ix = self.obs_idx.index_select(0, sym).squeeze(0)
        o = o.flatten(2).index_select(2, ix).view_as(o)
        no = no.flatten(2).index_select(2, ix).view_as(no)
        act = self.perms.index_select(0, sym).squeeze(0)[act]

        with self.autocast():
            q, _ = self.net(o)
        q_sa = q.float().gather(1, act.unsqueeze(1)).squeeze(1)
        with torch.no_grad(), self.autocast():
            nmask = obs_mask(no)
            q_t, _ = self.target(no)
            q_t = q_t.float()
            if self.double:
                q_online, _ = self.net(no)
                best = masked(q_online.float(), nmask).argmax(1, keepdim=True)
                q_next = q_t.gather(1, best).squeeze(1)
            else:
                q_next = masked(q_t, nmask).max(1).values
            q_next = torch.where(nmask.any(1), q_next, torch.zeros_like(q_next))
            sign = same * 2 - 1  # opponent moves next -> its value counts negatively
            target = r + a.gamma * (1 - d) * sign * q_next
        loss = F.smooth_l1_loss(q_sa, target)
        self.optimize(loss, 10.0)
        return loss.detach()


# ------------------------------------------------- REINFORCE / A2C / PPO
def sample_actions(logits: torch.Tensor) -> torch.Tensor:
    """Gumbel-max sampling from masked logits.  Same distribution as Categorical.sample(), but
    CUDA multinomial can occasionally return a zero-probability (illegal) move; this never does."""
    u = torch.rand_like(logits).clamp_(1e-10, 1 - 1e-10)
    return (logits - torch.log(-torch.log(u))).argmax(1)


def signed_gae(rew, val, next_val, player, next_player, cont, gamma: float, lam: float) -> torch.Tensor:
    """GAE over a (T, N) rollout of a two-player zero-sum game.

    next_val[t] / next_player[t] describe the state after step t and cont[t] = 0 when the game
    ended at step t.  A value is negated when the opponent is the one to move next."""
    adv = torch.zeros_like(rew)
    last = torch.zeros_like(rew[0])
    for t in reversed(range(rew.shape[0])):
        sign = torch.where(next_player[t] == player[t], 1.0, -1.0)
        delta = rew[t] + gamma * sign * next_val[t] * cont[t] - val[t]
        last = delta + gamma * lam * sign * cont[t] * last
        adv[t] = last
    return adv


class PolicyGradientTrainer(Trainer):
    epochs = 1
    graph = None
    minibatches = False  # True: shuffled --batch-size minibatches (PPO); False: one gradient step per rollout

    def act(self, obs: torch.Tensor):
        logits, val = self.infer(obs)
        logits = masked(logits.float(), obs_mask(obs))
        act = sample_actions(logits)
        logp = torch.log_softmax(logits, 1).gather(1, act[:, None]).squeeze(1)
        return act, logp, val.float()

    def collect(self):
        """Play `n_envs` complete self-play games in parallel (Monte-Carlo episodes)."""
        a, dev = self.args, self.device
        v = VecOthello(a.n_envs, seed=int(self.rng.integers(1 << 31)))
        obs_l, act_l, logp_l, val_l, rew_l, alive_l, player_l = ([] for _ in range(7))
        while not v.done.all():
            alive_l.append(torch.from_numpy(~v.done).to(dev))
            player_l.append(torch.from_numpy(v.player.copy()).to(dev))
            obs = self.to_obs(v.observation())
            act, logp, val = self.act(obs)
            obs_l.append(obs.to(torch.uint8))
            act_l.append(act)
            logp_l.append(logp)
            val_l.append(val)
            rew, _, _, _ = v.step(act.cpu().numpy())
            rew_l.append(torch.from_numpy(rew).to(dev))

        alive, player, rew = torch.stack(alive_l), torch.stack(player_l), torch.stack(rew_l)
        use_critic = self.algo != "reinforce"
        val = torch.stack(val_l) if use_critic else torch.zeros_like(rew)
        zero = torch.zeros_like(rew[:1])
        adv = signed_gae(rew, val,
                         next_val=torch.cat([val[1:], zero]),
                         next_player=torch.cat([player[1:], zero.to(player.dtype)]),
                         cont=torch.cat([alive[1:].float(), zero]),
                         player=player, gamma=a.gamma, lam=a.gae_lambda if use_critic else 1.0)
        keep = alive.reshape(-1)
        flat = lambda x: x.reshape(-1, *x.shape[2:])[keep]
        batch = {"obs": flat(torch.stack(obs_l)), "act": flat(torch.stack(act_l)),
                 "logp": flat(torch.stack(logp_l)), "adv": flat(adv), "ret": flat(adv + val)}
        return batch, a.n_envs

    def loss(self, mb, logp, ent, value):
        raise NotImplementedError

    def step_on(self, mb) -> dict[str, torch.Tensor]:
        """One gradient step on a (mini)batch; the returned metrics stay on the GPU."""
        obs = mb["obs"].float()
        with self.autocast():
            logits, value = self.net(obs)
        # same maths as Categorical(logits).log_prob / .entropy, without its validation syncs
        # (a sync cannot be captured in a CUDA graph)
        logp_all = torch.log_softmax(masked(logits.float(), obs_mask(obs)), 1)
        logp = logp_all.gather(1, mb["act"].unsqueeze(1)).squeeze(1)
        ent = -(logp_all.exp() * logp_all.clamp(min=torch.finfo(logp_all.dtype).min)).sum(1)
        loss, parts = self.loss(mb, logp, ent, value.float())
        self.optimize(loss, self.args.max_grad_norm)
        return {"loss": loss.detach(), "entropy": ent.mean().detach(), **parts}

    def graphed_step(self, mb) -> dict[str, torch.Tensor]:
        """step_on as a CUDA graph for batches of the captured size (PPO minibatches, A2C rollouts);
        any other size (the last partial minibatch, REINFORCE) runs eagerly."""
        if not self.graph_on or not (self.minibatches or self.algo == "a2c"):
            return self.step_on(mb)
        n = mb["act"].shape[0]
        if self.graph is None:
            if n != (self.args.batch_size if self.minibatches else self.args.n_envs * self.args.n_steps):
                return self.step_on(mb)
            self.static = {k: v.clone() for k, v in mb.items()}
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):  # warm-up before capture; these are real updates on this batch
                for _ in range(3):
                    self.step_on(self.static)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.graph_out = self.step_on(self.static)
        elif n != self.static["act"].shape[0]:
            return self.step_on(mb)
        for k, v in mb.items():
            self.static[k].copy_(v)
        self.graph.replay()
        return {k: v.clone() for k, v in self.graph_out.items()}

    def train_iteration(self):
        b, games = self.collect()
        n = b["act"].shape[0]
        size = self.args.batch_size if self.minibatches else n
        totals: dict[str, list] = {}  # per-minibatch metrics stay on the GPU: .item() would sync every step
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_kl = []
            for start in range(0, n, size):
                mb = {k: v[perm[start:start + size]] for k, v in b.items()} if self.minibatches else b
                out = self.graphed_step(mb)
                for k, val in out.items():
                    totals.setdefault(k, []).append(val)
                if "kl" in out:
                    epoch_kl.append(out["kl"])
            # PPO: stop after the epoch in which the policy moved too far (one sync per epoch)
            if epoch_kl and torch.stack(epoch_kl).mean().item() > self.args.target_kl:
                break
        return games, n, {k: torch.stack(v).mean().item() for k, v in totals.items()}


class ReinforceTrainer(PolicyGradientTrainer):
    def loss(self, mb, logp, ent, value):
        return -(logp * mb["ret"]).mean() - self.args.ent_coef * ent.mean(), {}


class A2CTrainer(PolicyGradientTrainer):
    """Synchronous n-step actor-critic: `n_envs` games that never stop (finished games restart),
    `n_steps` moves each, then a single gradient step that bootstraps from the critic."""

    def __init__(self, algo, args, device):
        super().__init__(algo, args, device)
        self.venv = VecOthello(args.n_envs, seed=args.seed)

    def collect(self):
        a, v, dev = self.args, self.venv, self.device
        obs_l, act_l, logp_l, val_l, rew_l, ended_l, player_l = ([] for _ in range(7))
        finished = 0
        for _ in range(a.n_steps):
            player_l.append(torch.from_numpy(v.player.copy()).to(dev))
            obs = self.to_obs(v.observation())
            act, logp, val = self.act(obs)
            obs_l.append(obs.to(torch.uint8))
            act_l.append(act)
            logp_l.append(logp)
            val_l.append(val)
            rew, ended, _, _ = v.step(act.cpu().numpy())
            rew_l.append(torch.from_numpy(rew).to(dev))
            ended_l.append(torch.from_numpy(ended).to(dev))
            finished += int(ended.sum())
            v.reset(ended)
        _, _, last_val = self.act(self.to_obs(v.observation()))

        player, val = torch.stack(player_l), torch.stack(val_l)
        adv = signed_gae(torch.stack(rew_l), val,
                         next_val=torch.cat([val[1:], last_val[None]]),
                         next_player=torch.cat([player[1:], torch.from_numpy(v.player.copy()).to(dev)[None]]),
                         cont=1.0 - torch.stack(ended_l).float(),
                         player=player, gamma=a.gamma, lam=a.gae_lambda)
        ret = adv + val
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        batch = {"obs": flat(torch.stack(obs_l)), "act": flat(torch.stack(act_l)),
                 "logp": flat(torch.stack(logp_l)), "adv": flat(adv), "ret": flat(ret)}
        return batch, finished

    def loss(self, mb, logp, ent, value):
        v_loss = F.mse_loss(value, mb["ret"])
        loss = -(logp * mb["adv"]).mean() + self.args.vf_coef * v_loss - self.args.ent_coef * ent.mean()
        return loss, {"v_loss": v_loss.detach()}


class PPOTrainer(PolicyGradientTrainer):
    minibatches = True

    @property
    def epochs(self):
        return self.args.ppo_epochs

    def loss(self, mb, logp, ent, value):
        a = self.args
        adv = (mb["adv"] - mb["adv"].mean()) / (mb["adv"].std() + 1e-8)
        ratio = torch.exp(logp - mb["logp"])
        pg_loss = -torch.min(ratio * adv, ratio.clamp(1 - a.clip, 1 + a.clip) * adv).mean()
        v_loss = F.mse_loss(value, mb["ret"])
        loss = pg_loss + a.vf_coef * v_loss - a.ent_coef * ent.mean()
        return loss, {"v_loss": v_loss.detach(), "kl": (mb["logp"] - logp).mean().detach()}


TRAINERS = {
    "dqn": DQNTrainer,
    "ddqn": DQNTrainer,
    "reinforce": ReinforceTrainer,
    "a2c": A2CTrainer,
    "ppo": PPOTrainer,
}


def parse_algos(text: str) -> list[str]:
    if text == "all":
        return list(ALGORITHMS)
    algos = [s.strip().lower() for s in text.split(",") if s.strip()]
    bad = [x for x in algos if x not in ALGORITHMS]
    if bad or not algos:
        raise argparse.ArgumentTypeError(f"unknown algorithm(s) {bad}; choose from {', '.join(ALGORITHMS)} or all")
    return algos


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train Othello agents with RL (self-play).")
    p.add_argument("--algo", default="all", help="all, one of dqn/ddqn/reinforce/a2c/ppo, or a comma list")
    p.add_argument("--device", default="auto", help="auto | cuda | cpu | cuda:1 ...")
    p.add_argument("--games", type=int, default=50000, help="self-play games per algorithm")
    p.add_argument("--n-envs", type=int, default=None, help="games played in parallel (default per algorithm)")
    p.add_argument("--parallel", action="store_true", help="train the selected algorithms simultaneously")
    p.add_argument("--amp", action="store_true", help="bfloat16 autocast on CUDA")
    p.add_argument("--graph", action="store_true",
                   help="CUDA graphs for the gradient step (implies --amp): fewer, same-size updates run ~2.5x faster")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads per process (0 = default)")
    p.add_argument("--lr", type=float, default=None, help="learning rate (default per algorithm)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=512)
    # DQN
    p.add_argument("--buffer-size", type=int, default=300_000)
    p.add_argument("--learn-start", type=int, default=10_000)
    p.add_argument("--replay-ratio", type=float, default=4.0, help="times each transition is sampled on average")
    p.add_argument("--target-update", type=int, default=500, help="gradient steps between target syncs")
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay", type=float, default=0.6, help="fraction of training to anneal epsilon")
    # policy gradient
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--target-kl", type=float, default=0.05, help="PPO stops a rollout's epochs above this KL")
    p.add_argument("--n-steps", type=int, default=16, help="A2C rollout length per env")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    # bookkeeping
    p.add_argument("--eval-every", type=int, default=2000, help="evaluate / checkpoint every N games")
    p.add_argument("--eval-games", type=int, default=200, help="games vs random and vs greedy (run in parallel)")
    p.add_argument("--status-interval", type=float, default=2.0, help="seconds between monitor updates")
    p.add_argument("--save-dir", default="weights")
    p.add_argument("--log-dir", default="logs")
    p.add_argument("--resume", action="store_true", help="continue from <save-dir>/<algo>_last.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    try:
        args.algos = parse_algos(args.algo)
    except argparse.ArgumentTypeError as e:
        p.error(str(e))
    return args


def train_algo(algo: str, args_dict: dict):
    """Entry point for one algorithm (also the target of --parallel worker processes)."""
    args = argparse.Namespace(**args_dict)
    args.n_envs = args.n_envs or DEFAULT_ENVS[algo]
    setup_torch(args)
    set_seed(args.seed)
    device = get_device(args.device)
    TRAINERS[algo](algo, args, device).run()


def main(argv=None):
    args = parse_args(argv)
    device = get_device(args.device)  # fail fast if CUDA was requested but is missing
    print(f"Device: {describe_device(device)} | algorithms: {', '.join(args.algos)}"
          f"{' (parallel)' if args.parallel and len(args.algos) > 1 else ''}", flush=True)

    if args.parallel and len(args.algos) > 1:
        if device.type == "cpu" and not args.threads:
            args.threads = max(1, (os.cpu_count() or 2) // len(args.algos))
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=train_algo, args=(algo, vars(args)), name=algo) for algo in args.algos]
        for p in procs:
            p.start()
        try:
            for p in procs:
                p.join()
        except KeyboardInterrupt:
            for p in procs:
                p.terminate()
            raise
        failed = [p.name for p in procs if p.exitcode != 0]
        if failed:
            raise SystemExit(f"failed: {', '.join(failed)}")
    else:
        for algo in args.algos:
            train_algo(algo, vars(args))


if __name__ == "__main__":
    main()
