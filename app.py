"""Web app: play Othello against the trained weights, and monitor / control training.

    python app.py                      # http://127.0.0.1:8002  (play)   /monitor (training GUI)
    python app.py --device cuda --port 8002

The game API is stateless: the browser sends the board with every request.
The monitor reads logs/<algo>_status.json and logs/<algo>.jsonl written by training.py.

Settings come from environment variables or a .env file next to this script
(real environment variables win).  .env is re-read whenever it changes:
    ENABLE_TRAINING_MONITOR=true   show the Training Monitor menu, page and training APIs
    MINIMAX_DEPTH=3                search depth of the Minimax opponent (1-8)
    ALPHABETA_DEPTH=5              search depth of the Alpha-Beta opponent (1-8)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time

from html import escape

from flask import Flask, abort, jsonify, request, send_from_directory

from agents import ALGO_NAMES, ALGORITHMS, GreedyAgent, NeuralAgent, RandomAgent, get_device
from env import NUM_ACTIONS, OthelloEnv, square_name
from search import SEARCH_AGENTS, MinimaxAgent, make_search_agent

ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)
app.config["WEIGHTS_DIR"] = os.path.join(ROOT, "weights")
app.config["LOG_DIR"] = os.path.join(ROOT, "logs")
app.config["DEVICE"] = "cpu"
_agents: dict[str, tuple[float, NeuralAgent]] = {}
TRAIN = {"proc": None, "cmd": None, "started": None}
ENV_PATH = os.path.join(ROOT, ".env")
_dotenv = {"mtime": None, "values": {}}
# Routes that belong to the Training Monitor; all return 404 when it is disabled.
MONITOR_ROUTES = ("/monitor", "/api/monitor", "/api/system", "/api/train/")


# ================================================================ settings
def read_dotenv() -> dict[str, str]:
    """KEY=VALUE pairs from .env (comments, blank lines, `export` and quotes allowed)."""
    try:
        mtime = os.path.getmtime(ENV_PATH)
    except OSError:
        return {}
    if mtime != _dotenv["mtime"]:
        values = {}
        with open(ENV_PATH, encoding="utf-8-sig") as f:  # tolerate a BOM (PowerShell writes one)
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.removeprefix("export ").split("=", 1)
                value = value.split(" #", 1)[0].strip().strip("\"'")
                values[key.strip()] = value
        _dotenv.update(mtime=mtime, values=values)
    return _dotenv["values"]


def setting(name: str, default: str) -> str:
    return os.environ.get(name, read_dotenv().get(name, default))


def monitor_enabled() -> bool:
    return setting("ENABLE_TRAINING_MONITOR", "true").strip().lower() in ("1", "true", "yes", "on")


@app.before_request
def guard_monitor():
    if request.path.startswith(MONITOR_ROUTES) and not monitor_enabled():
        abort(404, "Training Monitor is disabled (ENABLE_TRAINING_MONITOR in .env)")


@app.get("/api/config")
def config():
    return jsonify({"training_monitor": monitor_enabled()})


# =================================================================== game
def weight_files() -> dict[str, str]:
    files = glob.glob(os.path.join(app.config["WEIGHTS_DIR"], "*.pt"))
    return {os.path.splitext(os.path.basename(p))[0]: p for p in sorted(files)}


SEARCH_DEFAULT_DEPTH = {"minimax": 3, "alphabeta": 5}


def search_depth(algo: str) -> int:
    default = SEARCH_DEFAULT_DEPTH[algo]
    try:
        return max(1, min(8, int(setting(f"{algo.upper()}_DEPTH", str(default)))))
    except ValueError:
        return default


def get_agent(model: str):
    if model in SEARCH_AGENTS:  # cheap to build; depth follows .env
        return make_search_agent(model, search_depth(model))
    if model == "random":
        return RandomAgent()
    if model == "greedy":
        return GreedyAgent()
    path = weight_files().get(model)
    if path is None:
        abort(404, f"unknown model {model!r}")
    mtime = os.path.getmtime(path)
    cached = _agents.get(model)
    if cached is None or cached[0] != mtime:  # reload when training writes new weights
        _agents[model] = (mtime, NeuralAgent.load(path, app.config["DEVICE"]))
    return _agents[model][1]


def env_from_request(data: dict) -> OthelloEnv:
    board = data.get("board")
    if board is None:
        return OthelloEnv()
    if len(board) != NUM_ACTIONS or any(v not in (-1, 0, 1) for v in board):
        abort(400, "board must be 64 values of -1, 0, 1")
    return OthelloEnv.from_array(board, int(data.get("current", 1)))


def state(env: OthelloEnv, **extra) -> dict:
    b, w = env.score()
    return {
        "board": env.board_array().reshape(-1).tolist(),
        "current": env.current,
        "legal": env.legal_moves(),
        "score": {"black": b, "white": w},
        "done": env.done,
        "winner": env.winner,
        **extra,
    }


def apply_move(env: OthelloEnv, move: int) -> dict:
    before = env.board_array().reshape(-1)
    try:
        _, _, _, info = env.step(move)
    except ValueError as e:
        abort(400, str(e))
    after = env.board_array().reshape(-1)
    flipped = [i for i in range(NUM_ACTIONS) if before[i] != 0 and before[i] != after[i]]
    return state(env, move=move, move_name=square_name(move), player=info["player"],
                 flipped=flipped, passed=info["passed"])


def analysis_for(agent: NeuralAgent, env: OthelloEnv) -> dict:
    scores, value = agent.evaluate(env)
    return {"value": value, "kind": "q" if agent.algo in ("dqn", "ddqn") else "prob",
            "scores": {int(m): float(scores[m]) for m in env.legal_moves()}}


def think(agent, env: OthelloEnv):
    """(move, analysis) for any agent; analysis is None for the random / greedy baselines."""
    if isinstance(agent, MinimaxAgent):
        scores, stats = agent.evaluate(env)  # one search gives both the move and the hints
        best = max(scores.values())
        move = random.choice([m for m, v in scores.items() if v == best])
        return move, {"value": math.tanh(best / 150), "kind": "eval", "stats": stats,
                      "scores": {int(m): float(v) for m, v in scores.items()}}
    if isinstance(agent, NeuralAgent):
        return agent.select_action(env), analysis_for(agent, env)
    return agent.select_action(env), None


@app.get("/")
def index():
    return send_from_directory(os.path.join(ROOT, "web"), "index.html")


@app.get("/monitor")
def monitor_page():
    return send_from_directory(os.path.join(ROOT, "web"), "monitor.html")


@app.get("/static/<path:name>")
def static_files(name):
    return send_from_directory(os.path.join(ROOT, "web"), name)


@app.get("/api/models")
def models():
    out = []
    for name in weight_files():
        if name.endswith("_last"):
            continue
        try:
            agent = get_agent(name)
        except Exception as e:  # corrupt / incompatible file
            out.append({"id": name, "error": str(e)})
            continue
        s = getattr(agent, "stats", {})
        out.append({
            "id": name,
            "algo": agent.algo,
            "label": ALGO_NAMES.get(agent.algo, agent.algo),
            "games": s.get("games"),
            "vs_random": s.get("vs_random"),
            "vs_greedy": s.get("vs_greedy"),
        })
    out += [{"id": algo, "algo": "search", "label": cls.label, "depth": search_depth(algo)}
            for algo, cls in SEARCH_AGENTS.items()]
    out += [{"id": "greedy", "algo": "baseline", "label": "Greedy heuristic"},
            {"id": "random", "algo": "baseline", "label": "Random mover"}]
    return jsonify(out)


@app.post("/api/new")
def new_game():
    return jsonify(state(OthelloEnv()))


@app.post("/api/move")
def human_move():
    data = request.get_json(force=True)
    env = env_from_request(data)
    return jsonify(apply_move(env, int(data["move"])))


@app.post("/api/ai")
def ai_move():
    data = request.get_json(force=True)
    env = env_from_request(data)
    if env.done:
        abort(400, "game is over")
    move, analysis = think(get_agent(data.get("model", "ppo")), env)
    return jsonify({**apply_move(env, move), "analysis": analysis})


@app.post("/api/hint")
def hint():
    data = request.get_json(force=True)
    env = env_from_request(data)
    if env.done:
        return jsonify({"analysis": None})
    move, analysis = think(get_agent(data.get("model", "ppo")), env)
    return jsonify({"analysis": analysis and {**analysis, "best": move}})


# ================================================================ monitor
def read_json(path: str):
    for _ in range(5):  # the trainer may be replacing the file right now
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return None
        except (PermissionError, ValueError):
            time.sleep(0.03)
    return None


def read_history(path: str, offset: int):
    """Complete JSON lines of a history file from line `offset` on.
    Returns (records, new_offset, reset) - reset when the file was restarted."""
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.readlines()
    except (FileNotFoundError, PermissionError):
        return [], offset, False
    if lines and not lines[-1].endswith("\n"):  # line still being written
        lines.pop()
    reset = offset > len(lines)
    if reset:
        offset = 0
    records = []
    for line in lines[offset:]:
        try:
            records.append(json.loads(line))
        except ValueError:
            pass
    return records, len(lines), reset


def trainer_info() -> dict:
    proc = TRAIN["proc"]
    running = proc is not None and proc.poll() is None
    return {"running": running, "cmd": TRAIN["cmd"], "started": TRAIN["started"],
            "exit_code": None if proc is None or running else proc.returncode}


@app.get("/api/monitor")
def monitor_data():
    try:
        offsets = json.loads(request.args.get("offsets", "{}"))
    except ValueError:
        offsets = {}
    log_dir = app.config["LOG_DIR"]
    algos = {}
    for algo in ALGORITHMS:
        records, offset, reset = read_history(os.path.join(log_dir, f"{algo}.jsonl"), int(offsets.get(algo, 0)))
        algos[algo] = {"status": read_json(os.path.join(log_dir, f"{algo}_status.json")),
                       "history": records, "offset": offset, "reset": reset}
    return jsonify({"algos": algos, "names": ALGO_NAMES, "now": time.time(), "trainer": trainer_info()})


@app.get("/api/system")
def system_info():
    import torch
    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "cpu_count": os.cpu_count(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3).stdout
            vals = [v.strip() for v in out.splitlines()[0].split(",")]
            num = lambda v: float(v) if v.replace(".", "", 1).isdigit() else None
            info.update(gpu_util=num(vals[0]), mem_used=num(vals[1]), mem_total=num(vals[2]),
                        temp=num(vals[3]), power=num(vals[4]))
        except Exception:
            pass
    return jsonify(info)


@app.post("/api/train/start")
def train_start():
    if trainer_info()["running"]:
        abort(409, "training is already running")
    d = request.get_json(force=True)
    algos = [a for a in d.get("algos", []) if a in ALGORITHMS]
    if not algos:
        abort(400, "select at least one algorithm")
    device = d.get("device", "auto")
    if device not in ("auto", "cuda", "cpu"):
        abort(400, "device must be auto, cuda or cpu")
    cmd = [sys.executable, "-u", os.path.join(ROOT, "training.py"), "--algo", ",".join(algos),
           "--device", device, "--games", str(max(1, int(d.get("games", 50000)))),
           "--log-dir", app.config["LOG_DIR"], "--save-dir", app.config["WEIGHTS_DIR"]]
    if d.get("n_envs"):
        cmd += ["--n-envs", str(max(1, int(d["n_envs"])))]
    for flag in ("parallel", "resume", "amp"):
        if d.get(flag):
            cmd.append(f"--{flag}")

    os.makedirs(app.config["LOG_DIR"], exist_ok=True)
    console = open(os.path.join(app.config["LOG_DIR"], "console.log"), "w", encoding="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    TRAIN["proc"] = subprocess.Popen(cmd, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                                     creationflags=flags, env=env, start_new_session=os.name != "nt")
    TRAIN["cmd"] = " ".join(["python"] + cmd[2:])
    TRAIN["started"] = time.time()
    return jsonify(trainer_info())


@app.post("/api/train/stop")
def train_stop():
    proc = TRAIN["proc"]
    if proc is None or proc.poll() is not None:
        abort(409, "no training started from this page is running")
    if os.name == "nt":  # kill the whole tree (--parallel spawns one process per algorithm)
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        import signal
        os.killpg(proc.pid, signal.SIGTERM)
    proc.wait(timeout=10)
    for algo in ALGORITHMS:  # killed processes cannot update their own status
        path = os.path.join(app.config["LOG_DIR"], f"{algo}_status.json")
        status = read_json(path)
        if status and status.get("state") in ("running", "starting"):
            status["state"] = "stopped"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(status, f)
    return jsonify(trainer_info())


@app.get("/api/train/console")
def train_console():
    path = os.path.join(app.config["LOG_DIR"], "console.log")
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-120:]
    except FileNotFoundError:
        lines = []
    return jsonify({"lines": [l.rstrip("\n") for l in lines]})


def notice_page(code: int, title: str, message: str, hint: str = ""):
    """Friendly HTML page for browser requests (API requests keep getting JSON)."""
    with open(os.path.join(ROOT, "web", "notice.html"), encoding="utf-8") as f:
        page = f.read()
    for key, value in {"code": str(code), "title": escape(title), "message": escape(message), "hint": hint}.items():
        page = page.replace("{" + key + "}", value)
    return page, code, {"Content-Type": "text/html; charset=utf-8"}


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(409)
def bad_request(e):
    if request.path.startswith("/api/"):
        return jsonify({"error": e.description}), e.code
    if e.code == 404 and request.path.startswith(MONITOR_ROUTES) and not monitor_enabled():
        return notice_page(404, "Training Monitor is disabled",
                           "This page is turned off on this server. To enable it, set this in the .env file "
                           "next to app.py (no restart needed):",
                           "<pre>ENABLE_TRAINING_MONITOR=true</pre>")
    if e.code == 404:
        return notice_page(404, "Page not found", f"There is no page at {request.path}.")
    return notice_page(e.code, "Request error", str(e.description))


def make_parser(default_port: int = 8002) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Othello RL web app")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=default_port)
    p.add_argument("--device", default="cpu", help="auto | cuda | cpu (inference device for the game)")
    p.add_argument("--weights-dir", default=app.config["WEIGHTS_DIR"])
    p.add_argument("--log-dir", default=app.config["LOG_DIR"])
    return p


def serve(args, open_path: str | None = None):
    app.config["DEVICE"] = str(get_device(args.device))
    app.config["WEIGHTS_DIR"] = os.path.abspath(args.weights_dir)
    app.config["LOG_DIR"] = os.path.abspath(args.log_dir)
    url = f"http://{args.host}:{args.port}"
    print(f"Weights: {app.config['WEIGHTS_DIR']}  |  logs: {app.config['LOG_DIR']}  |  device: {app.config['DEVICE']}")
    print(f"Play:    {url}/\nMonitor: {url}/monitor")
    if open_path is not None:
        import threading
        import webbrowser
        threading.Timer(1.0, lambda: webbrowser.open(url + open_path)).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    serve(make_parser().parse_args())
