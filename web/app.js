// Othello RL Arena - front-end.  The server is stateless; we keep the game here.
const COLS = "abcdefgh";
const nameOf = (i) => COLS[i % 8] + (Math.floor(i / 8) + 1);

const el = (id) => document.getElementById(id);
const boardEl = el("board");
const cells = [];

const S = {
  game: null,        // server state
  human: 1,          // 1 = black, -1 = white
  model: null,
  history: [],       // snapshots before each human move (for undo)
  log: [],
  busy: false,
  hint: false,
  token: 0,          // invalidates in-flight AI turns after new game / undo
};

async function api(path, body) {
  const res = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || res.statusText);
  return data;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function toast(msg) {
  el("toast").textContent = msg || "";
}

// ---------------------------------------------------------------- board
function buildBoard() {
  for (let i = 0; i < 64; i++) {
    const c = document.createElement("div");
    c.className = "cell";
    c.setAttribute("role", "gridcell");
    c.setAttribute("aria-label", nameOf(i));
    c.addEventListener("click", () => onCellClick(i));
    boardEl.appendChild(c);
    cells.push(c);
  }
}

function render(prevBoard, placed) {
  const g = S.game;
  const humanTurn = !g.done && g.current === S.human && !S.busy;
  const legal = new Set(humanTurn ? g.legal : []);
  for (let i = 0; i < 64; i++) {
    const c = cells[i];
    const v = g.board[i];
    c.classList.toggle("legal", legal.has(i));
    c.classList.toggle("last", g.move === i);
    c.classList.remove("best");
    c.querySelectorAll(".hint-tag").forEach((n) => n.remove());

    let disc = c.querySelector(".disc");
    if (v === 0) {
      if (disc) disc.remove();
      continue;
    }
    const color = v === 1 ? "black" : "white";
    if (!disc) {
      disc = document.createElement("div");
      c.appendChild(disc);
    }
    disc.className = "disc " + color;
    if (prevBoard) {
      if (i === placed) disc.classList.add("place");
      else if (prevBoard[i] !== 0 && prevBoard[i] !== v) {
        void disc.offsetWidth; // restart animation
        disc.classList.add("flip");
      }
    }
  }

  el("count-black").textContent = g.score.black;
  el("count-white").textContent = g.score.white;
  el("side-black").classList.toggle("turn", !g.done && g.current === 1);
  el("side-white").classList.toggle("turn", !g.done && g.current === -1);
  el("name-black").textContent = S.human === 1 ? "You" : "AI · " + (S.model || "").toUpperCase();
  el("name-white").textContent = S.human === -1 ? "You" : "AI · " + (S.model || "").toUpperCase();

  let status;
  if (g.done) {
    const h = S.human === 1 ? g.score.black : g.score.white;
    const a = S.human === 1 ? g.score.white : g.score.black;
    status = h > a ? "You win! 🎉" : h < a ? "AI wins" : "Draw";
  } else {
    status = g.current === S.human ? "Your turn" : "AI is thinking…";
  }
  el("status").textContent = status;
  el("btn-undo").disabled = S.busy || S.history.length === 0;
  renderLog();
}

function renderLog() {
  const ol = el("log");
  ol.innerHTML = "";
  S.log.forEach((m, k) => {
    const li = document.createElement("li");
    if (m.pass) {
      li.className = "pass";
      li.textContent = (m.player === 1 ? "Black" : "White") + " passes (no legal move)";
    } else {
      li.innerHTML = `<span class="n">${k + 1}.</span><span class="chip sm ${m.player === 1 ? "black" : "white"}"></span>${m.name}`;
    }
    ol.appendChild(li);
  });
  ol.scrollTop = ol.scrollHeight;
}

function setMeter(aiValue) {
  if (aiValue === null || aiValue === undefined) {
    el("meter-fill").style.width = "50%";
    el("meter-text").textContent = "—";
    return;
  }
  const humanShare = (1 - aiValue) / 2;
  // dark fill = black's share, so flip when the human plays white
  const blackShare = S.human === 1 ? humanShare : 1 - humanShare;
  el("meter-fill").style.width = (blackShare * 100).toFixed(1) + "%";
  el("meter-text").textContent = `AI ${Math.round((aiValue + 1) * 50)}%`;
}

// Search scores: heuristic points, or a forced result once the search sees the end of the game.
function searchLabel(v) {
  if (v >= 10000) return "WIN";
  if (v <= -10000) return "LOSS";
  return (v >= 0 ? "+" : "") + Math.round(v);
}

function showSearchStats(stats) {
  let text = "";
  if (stats && stats.solved) {
    const e = stats.exact;
    text = `Endgame solved exactly in ${Math.round(stats.ms)} ms: ${e > 0 ? "AI wins by " + e : e < 0 ? "AI loses by " + -e : "draw"} with perfect play`;
  } else if (stats && stats.sims !== undefined) {
    text = stats.sims ? `Tree search: ${stats.sims} simulations in ${Math.round(stats.ms)} ms` : "";
  } else if (stats) {
    text = `Searched ${stats.nodes.toLocaleString()} positions in ${Math.round(stats.ms)} ms (depth ${stats.depth})`;
  }
  el("meter-info").textContent = text;
}

function showHints(analysis) {
  if (!analysis || !S.hint) return;
  const vals = Object.values(analysis.scores);
  const max = Math.max(...vals), min = Math.min(...vals);
  for (const [k, v] of Object.entries(analysis.scores)) {
    const i = Number(k);
    const tag = document.createElement("span");
    tag.className = "hint-tag";
    const t = max === min ? 1 : (v - min) / (max - min);
    tag.style.background = `rgba(255,255,255,${0.45 + 0.5 * t})`;
    tag.textContent = analysis.kind === "prob" ? Math.round(v * 100) + "%"
      : analysis.kind === "eval" ? searchLabel(v)
      : analysis.kind === "disc" ? (v > 0 ? "+" : "") + Math.round(v)
      : (v >= 0 ? "+" : "") + v.toFixed(2);
    cells[i].appendChild(tag);
  }
  if (analysis.best !== undefined) cells[analysis.best].classList.add("best");
}

// ---------------------------------------------------------------- flow
function record(res) {
  S.log.push({ player: res.player, name: res.move_name });
  if (res.passed) S.log.push({ pass: true, player: -res.player });
}

async function refreshHint() {
  const g = S.game;
  if (!S.hint || g.done || g.current !== S.human) return;
  const token = S.token;
  try {
    const { analysis } = await api("/api/hint", { board: g.board, current: g.current, model: S.model });
    if (token !== S.token || !analysis) return;
    showHints(analysis);
    setMeter(-analysis.value);
  } catch (e) {
    toast(e.message);
  }
}

async function onCellClick(i) {
  const g = S.game;
  if (S.busy || g.done || g.current !== S.human || !g.legal.includes(i)) return;
  S.history.push({ game: g, log: S.log.slice() });
  try {
    const res = await api("/api/move", { board: g.board, current: g.current, move: i });
    const prev = g.board;
    S.game = res;
    record(res);
    toast(res.passed ? "AI has no legal move — your turn again" : "");
    render(prev, i);
    if (!res.done && res.current !== S.human) await aiTurn();
    else refreshHint();
  } catch (e) {
    S.history.pop();
    toast(e.message);
  }
}

async function aiTurn() {
  const token = S.token;
  S.busy = true;
  render();
  try {
    while (!S.game.done && S.game.current !== S.human) {
      await sleep(420);
      if (token !== S.token) return;
      const g = S.game;
      const res = await api("/api/ai", { board: g.board, current: g.current, model: S.model });
      if (token !== S.token) return;
      S.game = res;
      record(res);
      if (res.analysis) setMeter(res.analysis.value);
      showSearchStats(res.analysis?.stats);
      toast(res.passed ? "You have no legal move — AI moves again" : "");
      render(g.board, res.move);
    }
  } catch (e) {
    toast(e.message);
  } finally {
    if (token === S.token) {
      S.busy = false;
      render();
      refreshHint();
    }
  }
}

async function newGame() {
  S.token++;
  S.busy = false;
  S.history = [];
  S.log = [];
  toast("");
  setMeter(null);
  showSearchStats(null);
  S.game = await api("/api/new", {});
  cells.forEach((c) => c.querySelector(".disc")?.remove());
  render(new Array(64).fill(0), -1);
  if (S.game.current !== S.human) await aiTurn();
  else refreshHint();
}

function undo() {
  if (S.busy || !S.history.length) return;
  S.token++;
  const snap = S.history.pop();
  S.game = snap.game;
  S.log = snap.log;
  toast("");
  render();
  refreshHint();
}

// ---------------------------------------------------------------- models
function pct(x) {
  return x === null || x === undefined ? "–" : Math.round(x * 100) + "%";
}

async function loadModels() {
  const list = await api("/api/models");
  const box = el("models");
  box.innerHTML = "";
  const trained = list.filter((m) => !["baseline", "search", "mcts"].includes(m.algo) && !m.error);
  if (!trained.length) {
    const p = document.createElement("p");
    p.className = "empty";
    p.innerHTML = "No trained weights yet — train first with<br><code>python training.py --algo all</code><br>You can play the baselines meanwhile.";
    box.appendChild(p);
  }
  const groups = [
    ["Trained networks", (m) => !["baseline", "search", "mcts"].includes(m.algo)],
    ["Networks with tree search", (m) => m.algo === "mcts"],
    ["Classic opponents", (m) => ["baseline", "search"].includes(m.algo)],
  ];
  for (const [title, belongs] of groups) {
    const members = list.filter((m) => !m.error && belongs(m));
    if (!members.length) continue;
    const group = document.createElement("section");
    group.className = "model-group";
    group.innerHTML = `<h3>${title}</h3><div class="model-grid"></div>`;
    for (const m of members) {
      const b = document.createElement("button");
      b.className = "model";
      b.dataset.id = m.id;
      b.title = m.label + (m.games ? " · " + m.games.toLocaleString() + " games" : "");
      const stat = m.algo === "mcts" ? `<b>${m.sims}</b> simulations`
        : m.algo === "search" ? `depth <b>${m.depth}</b>`
        : m.algo === "baseline" ? (m.id === "greedy" ? "one-move lookahead" : "random moves")
        : `vs greedy <b>${pct(m.vs_greedy)}</b>`;
      const level = m.level ? `<span class="level ${m.level}">${m.level}</span>` : "";
      b.innerHTML = `<span class="top"><span class="id">${m.id}</span>${level}</span><span class="stat">${stat}</span>`;
      b.addEventListener("click", () => selectModel(m.id));
      group.querySelector(".model-grid").appendChild(b);
    }
    box.appendChild(group);
  }
  // every visit starts with PPO+MCTS (or PPO when tree search is off); the last choice is not remembered
  const ids = list.filter((m) => !m.error).map((m) => m.id);
  selectModel(["ppo+mcts", "ppo"].find((id) => ids.includes(id)) || ids[0], false);
}

function selectModel(id, restart = true) {
  S.model = id;
  document.querySelectorAll(".model").forEach((b) => b.classList.toggle("active", b.dataset.id === id));
  if (restart) newGame();
}

function localStorageGet(k) { try { return localStorage.getItem(k); } catch { return null; } }
function localStorageSet(k, v) { try { localStorage.setItem(k, v); } catch { /* ignore */ } }

// ---------------------------------------------------------------- init
document.querySelectorAll("#color-seg button").forEach((b) =>
  b.addEventListener("click", () => {
    S.human = Number(b.dataset.color);
    document.querySelectorAll("#color-seg button").forEach((x) => x.classList.toggle("active", x === b));
    newGame();
  })
);
el("btn-new").addEventListener("click", newGame);
el("btn-undo").addEventListener("click", undo);
el("chk-hint").addEventListener("change", (e) => {
  S.hint = e.target.checked;
  render();
  refreshHint();
});

// The Training Monitor link is only shown when ENABLE_TRAINING_MONITOR is on (.env).
api("/api/config")
  .then((c) => { el("nav-monitor").hidden = !c.training_monitor; })
  .catch(() => {});

buildBoard();
loadModels().then(newGame).catch((e) => toast(e.message));
