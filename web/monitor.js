// Training monitor: polls /api/monitor and draws progress, charts and the latest evaluation game.
const ALGOS = ["dqn", "ddqn", "reinforce", "a2c", "ppo"];
const css = getComputedStyle(document.documentElement);
const COLOR = Object.fromEntries(ALGOS.map((a) => [a, css.getPropertyValue(`--s-${a}`).trim()]));
const INK2 = css.getPropertyValue("--ink-2").trim();
const AXIS = css.getPropertyValue("--axis").trim();
const GRID = css.getPropertyValue("--grid").trim();
const STATE_TEXT = {
  idle: "Idle", starting: "Starting", running: "Training", done: "Done",
  stopped: "Stopped", error: "Error", stalled: "Stalled",
};
const METRICS = {
  loss: "Loss", entropy: "Entropy", eps: "Epsilon (exploration)", v_loss: "Value loss", kl: "KL divergence", sps: "Samples / second",
};

const el = (id) => document.getElementById(id);
const S = {
  offsets: {},
  hist: Object.fromEntries(ALGOS.map((a) => [a, { train: [], eval: [] }])),
  status: {},
  names: {},
  now: Date.now() / 1000,
  selected: "ppo",
  trainer: { running: false },
  replay: { algo: null, boards: [], i: 0, timer: null, key: null },
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

// ------------------------------------------------------------------ format
const fmtInt = (n) => (n === null || n === undefined ? "—" : Math.round(n).toLocaleString());
const fmtPct = (x) => (x === null || x === undefined ? "—" : (x * 100).toFixed(1) + "%");
function fmtDur(s) {
  if (s === null || s === undefined || !isFinite(s)) return "—";
  s = Math.max(0, Math.round(s));
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
  return h ? `${h}:${String(m).padStart(2, "0")}:${String(sec).padStart(2, "0")}` : `${m}:${String(sec).padStart(2, "0")}`;
}
function fmtNum(x) {
  if (x === null || x === undefined) return "—";
  const a = Math.abs(x);
  return a >= 100 ? x.toFixed(0) : a >= 1 ? x.toFixed(2) : x.toFixed(4);
}

// ------------------------------------------------------------------ charts
Chart.defaults.font.family = '"Noto Sans", "Noto Sans Thai", system-ui, sans-serif';
Chart.defaults.color = AXIS;

function makeChart(canvas, { xTitle, yTitle, pct = false }) {
  return new Chart(canvas, {
    type: "line",
    data: { datasets: [] },
    options: {
      animation: false,
      responsive: true,
      maintainAspectRatio: false,
      parsing: false,
      interaction: { mode: "nearest", axis: "x", intersect: false },
      plugins: {
        legend: {
          position: "top", align: "end",
          labels: { color: INK2, boxWidth: 12, boxHeight: 2, padding: 12, font: { size: 11 } },
        },
        tooltip: {
          backgroundColor: "#ffffff", borderColor: "#c9d8eb", borderWidth: 1,
          titleColor: "#0f2747", bodyColor: INK2, padding: 10, boxWidth: 8, boxHeight: 8,
          callbacks: {
            title: (items) => `${xTitle}: ${fmtNum(items[0].parsed.x)}`,
            label: (it) => ` ${it.dataset.label}: ${pct ? it.parsed.y.toFixed(1) + "%" : fmtNum(it.parsed.y)}`,
          },
        },
      },
      scales: {
        x: {
          type: "linear", title: { display: true, text: xTitle, color: AXIS, font: { size: 11 } },
          grid: { color: GRID }, border: { color: "#c3d2e6" }, ticks: { color: AXIS, maxTicksLimit: 7 },
        },
        y: {
          title: { display: !!yTitle, text: yTitle, color: AXIS, font: { size: 11 } },
          grid: { color: GRID }, border: { display: false }, ticks: { color: AXIS, maxTicksLimit: 6 },
          ...(pct ? { min: 0, max: 100, ticks: { color: AXIS, callback: (v) => v + "%" } } : {}),
        },
      },
      elements: {
        line: { borderWidth: 2, cubicInterpolationMode: "monotone" },
        point: { radius: 0, hoverRadius: 5, hitRadius: 8, hoverBorderWidth: 2, hoverBorderColor: "#ffffff" },
      },
    },
  });
}

// Trailing moving average: games finish in bursts (a whole rollout at once), so raw speed is spiky.
function smooth(points, k) {
  return points.map((p, i) => {
    const win = points.slice(Math.max(0, i - k + 1), i + 1);
    return { x: p.x, y: win.reduce((s, q) => s + q.y, 0) / win.length };
  });
}

function seriesFor(algo, points) {
  return {
    label: algo.toUpperCase(), data: points, borderColor: COLOR[algo], backgroundColor: COLOR[algo],
    pointBackgroundColor: COLOR[algo],
  };
}

const charts = {
  greedy: makeChart(el("c-greedy"), { xTitle: "Games trained", pct: true }),
  random: makeChart(el("c-random"), { xTitle: "Games trained", pct: true }),
  speed: makeChart(el("c-speed"), { xTitle: "Time (minutes)", yTitle: "Games / s (10 s average)" }),
  metric: makeChart(el("c-metric"), { xTitle: "Games trained" }),
};

function updateCharts() {
  const withData = (key) => ALGOS.filter((a) => S.hist[a][key].length);
  charts.greedy.data.datasets = withData("eval").map((a) =>
    seriesFor(a, S.hist[a].eval.map((r) => ({ x: r.games, y: r.vs_greedy * 100 }))));
  charts.random.data.datasets = withData("eval").map((a) =>
    seriesFor(a, S.hist[a].eval.map((r) => ({ x: r.games, y: r.vs_random * 100 }))));
  charts.speed.data.datasets = withData("train").map((a) =>
    seriesFor(a, smooth(S.hist[a].train.map((r) => ({ x: r.elapsed / 60, y: r.gps })), 5)));
  for (const c of [charts.greedy, charts.random, charts.speed]) {
    c.data.datasets.forEach((d) => (d.pointRadius = c === charts.speed ? 0 : 2.5));
    c.update("none");
  }
  updateMetricChart();
}

function updateMetricChart() {
  const algo = el("metric-algo").value;
  const recs = S.hist[algo]?.train || [];
  const available = Object.keys(METRICS).filter((k) => recs.some((r) => r[k] !== null && r[k] !== undefined));
  const sel = el("metric-name");
  const want = sel.value;
  const opts = available.map((k) => `<option value="${k}">${METRICS[k]}</option>`).join("");
  if (sel.dataset.opts !== opts) {
    sel.innerHTML = opts || `<option value="">—</option>`;
    sel.dataset.opts = opts;
    sel.value = available.includes(want) ? want : available[0] || "";
  }
  const key = sel.value;
  const pts = recs.filter((r) => r[key] !== null && r[key] !== undefined).map((r) => ({ x: r.games, y: r[key] }));
  charts.metric.data.datasets = key ? [{ ...seriesFor(algo, pts), label: `${algo.toUpperCase()} · ${METRICS[key]}` }] : [];
  charts.metric.options.plugins.legend.display = false;
  charts.metric.update("none");
}

// ------------------------------------------------------------------ cards
function effectiveState(st) {
  if (!st) return "idle";
  if ((st.state === "running" || st.state === "starting") && S.now - st.updated > 30) return "stalled";
  return st.state;
}

function renderCards() {
  const box = el("algo-cards");
  if (!box.children.length) {
    for (const a of ALGOS) {
      const card = document.createElement("article");
      card.className = "algo-card";
      card.style.setProperty("--c", COLOR[a]);
      card.dataset.algo = a;
      card.addEventListener("click", () => selectAlgo(a));
      box.appendChild(card);
    }
  }
  for (const card of box.children) {
    const a = card.dataset.algo;
    const st = S.status[a];
    const state = effectiveState(st);
    const games = st?.games ?? 0;
    const total = st?.total_games ?? 0;
    const frac = total ? Math.min(1, games / total) : 0;
    const ev = st?.eval || {};
    const live = state === "running";
    const gps = live ? recentSpeed(a) ?? st.gps : null;
    const eta = gps ? (total - games) / gps : null;
    card.classList.toggle("selected", a === S.selected);
    card.innerHTML = `
      <div class="ac-head"><b>${a}</b><span class="badge ${state}">${STATE_TEXT[state] || state}</span></div>
      <div class="ac-label">${S.names[a] || ""}${st?.device ? " · " + st.device.replace(/ \(.*\)/, "") : ""}</div>
      <div class="progress"><div style="width:${(frac * 100).toFixed(1)}%"></div></div>
      <div class="ac-row"><span>${fmtInt(games)} / ${fmtInt(total || null)} games</span><span>${(frac * 100).toFixed(0)}%</span></div>
      <dl class="ac-stats">
        <div><dt>Speed</dt><dd>${live ? fmtNum(gps) + " <small>games/s</small>" : "—"}</dd></div>
        <div><dt>${live ? "Time left" : "Time taken"}</dt><dd>${fmtDur(live ? eta : st?.elapsed)}</dd></div>
        <div><dt>vs Random</dt><dd>${fmtPct(ev.vs_random)}</dd></div>
        <div><dt>vs Greedy</dt><dd>${fmtPct(ev.vs_greedy)}</dd></div>
        <div><dt>Best score</dt><dd>${st && st.best_score >= 0 ? st.best_score.toFixed(3) : "—"}</dd></div>
        <div><dt>Loss</dt><dd>${fmtNum(st?.loss)}</dd></div>
      </dl>
      ${state === "error" && st.error ? `<div class="ac-error">${escapeHtml(st.error.trim().split("\n").slice(-3).join("\n"))}</div>` : ""}`;
  }
  const total = ALGOS.reduce((s, a) => s + (effectiveState(S.status[a]) === "running" ? recentSpeed(a) ?? S.status[a].gps ?? 0 : 0), 0);
  el("t-speed").textContent = total ? fmtNum(total) : "—";
}

// Average speed over the last ~10 s of history (single 2 s samples are spiky).
function recentSpeed(algo, k = 5) {
  const recs = S.hist[algo].train.slice(-k);
  return recs.length ? recs.reduce((s, r) => s + r.gps, 0) / recs.length : null;
}

function escapeHtml(s) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function selectAlgo(a) {
  S.selected = a;
  el("metric-algo").value = a;
  el("replay-algo").value = a;
  renderCards();
  updateMetricChart();
  loadReplay(true);
}

// ------------------------------------------------------------------ replay
const mini = el("mini-board");
for (let i = 0; i < 64; i++) mini.appendChild(document.createElement("div"));

function drawBoard(str, prev) {
  [...mini.children].forEach((cell, i) => {
    const c = str ? str[i] : ".";
    cell.innerHTML = c === "." ? "" : `<i class="${c}"></i>`;
    cell.classList.toggle("new", !!prev && prev[i] === "." && c !== ".");
  });
  if (str) {
    const b = [...str].filter((c) => c === "b").length, w = [...str].filter((c) => c === "w").length;
    el("replay-score").textContent = `${b} – ${w}`;
  }
}

function showPly(i) {
  const R = S.replay;
  R.i = Math.max(0, Math.min(i, R.boards.length - 1));
  el("rp-slider").value = R.i;
  drawBoard(R.boards[R.i], R.boards[R.i - 1]);
  el("replay-ply").textContent = R.boards.length ? `Ply ${R.i} / ${R.boards.length - 1}` : "";
}

function stopReplay() {
  clearInterval(S.replay.timer);
  S.replay.timer = null;
  el("rp-play").textContent = "▶";
}

function playReplay() {
  const R = S.replay;
  if (!R.boards.length) return;
  if (R.i >= R.boards.length - 1) showPly(0);
  el("rp-play").textContent = "⏸";
  R.timer = setInterval(() => {
    if (R.i >= R.boards.length - 1) return stopReplay();
    showPly(R.i + 1);
  }, 260);
}

function loadReplay(force = false) {
  const algo = el("replay-algo").value;
  const ev = S.status[algo]?.eval;
  const sample = ev?.sample;
  const key = sample ? `${algo}:${ev.games}` : `${algo}:none`;
  if (!force && key === S.replay.key) return;
  S.replay.key = key;
  stopReplay();
  if (!sample) {
    S.replay.boards = [];
    el("rp-slider").max = 0;
    drawBoard(null);
    el("replay-score").textContent = "";
    el("replay-ply").textContent = "";
    el("replay-caption").textContent = `${algo.toUpperCase()}: no evaluation game yet`;
    return;
  }
  S.replay.boards = sample.boards;
  el("rp-slider").max = sample.boards.length - 1;
  const res = sample.result > 0 ? "won" : sample.result < 0 ? "lost" : "draw";
  el("replay-caption").innerHTML =
    `<b>${algo.toUpperCase()}</b> (black) vs Greedy (white) — <b>${res}</b><br>` +
    `<span style="color:var(--muted)">after ${fmtInt(ev.games)} training games · vs greedy ${fmtPct(ev.vs_greedy)}</span>`;
  showPly(0);
  playReplay();
}

el("rp-play").addEventListener("click", () => (S.replay.timer ? stopReplay() : playReplay()));
el("rp-slider").addEventListener("input", (e) => { stopReplay(); showPly(Number(e.target.value)); });

// ------------------------------------------------------------------ polling
async function pollMonitor() {
  try {
    const data = await api(`/api/monitor?offsets=${encodeURIComponent(JSON.stringify(S.offsets))}`);
    S.now = data.now;
    S.names = data.names;
    let changed = false;
    for (const a of ALGOS) {
      const d = data.algos[a];
      if (d.reset) { S.hist[a] = { train: [], eval: [] }; changed = true; }
      for (const r of d.history) {
        (r.t === "eval" ? S.hist[a].eval : S.hist[a].train).push(r);
        changed = true;
      }
      S.offsets[a] = d.offset;
      S.status[a] = d.status;
    }
    setTrainer(data.trainer);
    renderCards();
    if (changed) updateCharts();
    loadReplay();
    el("live").className = "live ok";
    el("live-text").textContent = "Connected · " + new Date().toLocaleTimeString();
  } catch (e) {
    el("live").className = "live err";
    el("live-text").textContent = "Disconnected";
  }
}

async function pollSystem() {
  try {
    const s = await api("/api/system");
    el("gpu-name").textContent = s.gpu ? `${s.gpu} · PyTorch ${s.torch}` : `No CUDA GPU found · PyTorch ${s.torch} · ${s.cpu_count} CPU threads`;
    const util = s.gpu_util, memPct = s.mem_total ? s.mem_used / s.mem_total : null;
    el("t-util").textContent = util === undefined || util === null ? "—" : util.toFixed(0) + "%";
    el("b-util").style.width = (util || 0) + "%";
    el("t-mem").textContent = memPct === null ? "—" : `${(s.mem_used / 1024).toFixed(1)} / ${(s.mem_total / 1024).toFixed(1)} GB`;
    el("b-mem").style.width = ((memPct || 0) * 100).toFixed(1) + "%";
    el("t-temp").textContent = s.temp ? `${s.temp.toFixed(0)}°C${s.power ? " · " + s.power.toFixed(0) + " W" : ""}` : "—";
    if (!s.cuda) {
      const dev = el("f-device");
      if (dev.value === "cuda") dev.value = "cpu";
      dev.querySelector('option[value="cuda"]').disabled = true;
    }
  } catch { /* the monitor poll reports connectivity */ }
}

async function pollConsole() {
  if (!S.trainer.running && S.consoleLoaded) return;
  try {
    const { lines } = await api("/api/train/console");
    if (!lines.length) return;
    const pre = el("console");
    const atBottom = pre.scrollTop + pre.clientHeight >= pre.scrollHeight - 20;
    pre.textContent = lines.join("\n");
    if (atBottom || !S.consoleLoaded) pre.scrollTop = pre.scrollHeight;
    S.consoleLoaded = true;
  } catch { /* ignore */ }
}

// ------------------------------------------------------------------ controls
function setTrainer(t) {
  const wasRunning = S.trainer.running;
  S.trainer = t;
  el("btn-start").disabled = t.running;
  el("btn-stop").disabled = !t.running;
  el("cmd").textContent = t.cmd
    ? (t.running ? "Running: " : `Finished (exit ${t.exit_code}): `) + t.cmd
    : "";
  if (wasRunning && !t.running) { S.consoleLoaded = false; pollConsole(); }
}

function buildChecks() {
  const box = el("algo-checks");
  for (const a of ALGOS) {
    const lab = document.createElement("label");
    lab.className = "algo-chip on";
    lab.style.setProperty("--c", COLOR[a]);
    lab.innerHTML = `<input type="checkbox" value="${a}" checked><span class="swatch"></span><span class="mono">${a}</span>`;
    const input = lab.querySelector("input");
    input.addEventListener("change", () => lab.classList.toggle("on", input.checked));
    box.appendChild(lab);
  }
  for (const id of ["metric-algo", "replay-algo"]) {
    el(id).innerHTML = ALGOS.map((a) => `<option value="${a}">${a.toUpperCase()}</option>`).join("");
    el(id).value = S.selected;
  }
  el("metric-algo").addEventListener("change", (e) => selectAlgo(e.target.value));
  el("replay-algo").addEventListener("change", (e) => selectAlgo(e.target.value));
  el("metric-name").addEventListener("change", updateMetricChart);
}

el("btn-start").addEventListener("click", async () => {
  const algos = [...document.querySelectorAll("#algo-checks input:checked")].map((i) => i.value);
  const body = {
    algos,
    device: el("f-device").value,
    games: Number(el("f-games").value) || 50000,
    n_envs: Number(el("f-envs").value) || null,
    parallel: el("f-parallel").checked,
    resume: el("f-resume").checked,
    amp: el("f-amp").checked,
  };
  try {
    el("btn-start").disabled = true;
    setTrainer(await api("/api/train/start", body));
    S.consoleLoaded = false;
    el("console").textContent = "Starting…";
  } catch (e) {
    el("btn-start").disabled = false;
    el("cmd").textContent = "Could not start: " + e.message;
  }
});

el("btn-stop").addEventListener("click", async () => {
  if (!confirm("Stop all training started from this page? (Weights already saved are kept.)")) return;
  try {
    setTrainer(await api("/api/train/stop", {}));
    pollMonitor();
  } catch (e) {
    el("cmd").textContent = e.message;
  }
});

buildChecks();
pollMonitor().then(() => loadReplay(true));
pollSystem();
pollConsole();
setInterval(pollMonitor, 2000);
setInterval(pollSystem, 4000);
setInterval(pollConsole, 2500);
