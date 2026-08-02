/* OrionQuant — single-page UI logic */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  let currentScorecard = null;
  let charts = {};
  const PALETTE = ["#4f8cff","#32d583","#ff5470","#ffb547","#a881ff","#29d6d6","#ff86c2"];

  // ------- Navigation -------
  $$("nav button").forEach(b => {
    b.addEventListener("click", () => {
      $$("nav button").forEach(x => x.classList.remove("active"));
      b.classList.add("active");
      $$(".view").forEach(v => v.classList.remove("active"));
      $("#" + (b.id === "nav-evaluate" ? "view-evaluate" : "view-compare")).classList.add("active");
      if (b.id === "nav-compare") refreshSavedList();
    });
  });

  // ------- Upload form -------
  $("#upload-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const file = $("#file-input").files[0];
    if (!file) return;
    const fd = new FormData();
    fd.append("file", file);
    const cfg = { start_balance: Number($("#balance-input").value) || 10000 };
    fd.append("config", JSON.stringify(cfg));
    const seed = $("#seed-input").value;
    if (seed) fd.append("seed", seed);
    const name = $("#strategy-name").value || file.name;
    fd.append("strategy_name", name);
    const status = $("#upload-status");
    status.textContent = "Evaluating…";
    status.className = "status";
    $("#results-area").style.display = "none";
    try {
      const t0 = performance.now();
      const res = await fetch("/evaluate", { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      const card = await res.json();
      const ms = Math.round(performance.now() - t0);
      status.textContent = `Done in ${ms} ms`;
      status.className = "status ok";
      currentScorecard = card;
      renderScorecard(card);
      $("#save-btn").style.display = "inline-block";
      $("#results-area").style.display = "block";
    } catch (err) {
      status.textContent = "Error: " + err.message;
      status.className = "status error";
    }
  });

  $("#save-btn").addEventListener("click", async () => {
    if (!currentScorecard) return;
    try {
      const r = await fetch("/evaluations", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scorecard: currentScorecard })
      });
      if (!r.ok) throw new Error("Save failed");
      $("#save-btn").textContent = "✓ Saved";
      setTimeout(() => $("#save-btn").textContent = "💾 Save this run", 1500);
    } catch (e) {
      alert(e.message);
    }
  });

  // ------- Scorecard rendering -------
  function renderScorecard(card) {
    const s = card.scoring;
    $("#final-score").textContent = s.final_score;
    $("#score-label").textContent = s.label;
    $("#n-trades").textContent = card.metrics.basic.total_trades;
    $("#seed-used").textContent = card.seed ?? "—";
    $("#fmt-used").textContent = card.input_file.format;

    const koPill = $("#knockout-pill");
    const koReason = $("#knockout-reason");
    if (s.status === "KNOCKED_OUT") {
      koPill.style.display = "inline-block";
      koReason.style.display = "block";
      koReason.textContent = s.knockout_reason || "";
    } else {
      koPill.style.display = "none";
      koReason.style.display = "none";
    }

    const corr = $("#correlation-pill");
    corr.style.display = card.correlation_flag ? "inline-block" : "none";

    renderCategoryScores(s.category_scores);
    renderEquity(card.equity_curve);
    renderDrawdown(card.equity_curve);
    renderPnlDistribution(card.metrics, card.equity_curve);
    renderMCDistributions(card.monte_carlo);
    renderScoreBreakdown(s.category_scores);
    renderMonthlyHeat(card.monthly_returns);
    renderMetricsTable(card);
  }

  function renderCategoryScores(cats) {
    const host = $("#cat-scores");
    host.innerHTML = "";
    if (!cats) return;
    const defs = [
      ["Profitability", cats.profitability, 0.25],
      ["Risk-Adj Return", cats.risk_adj, 0.25],
      ["Drawdown & Cap Pres", cats.drawdown, 0.20],
      ["Consistency & Robust", cats.robustness, 0.15],
      ["Behavioural Sanity", cats.sanity, 0.10],
      ["Statistical Suffic", cats.sufficiency, 0.05],
    ];
    for (const [name, v, w] of defs) {
      const row = document.createElement("div");
      row.className = "cat-bar";
      row.innerHTML = `
        <div>${name} <span style="color:var(--muted);">(${Math.round(w * 100)}%)</span></div>
        <div class="bar"><span style="width:${Math.max(0, Math.min(100, v * 10))}%"></span></div>
        <div style="text-align:right;font-variant-numeric:tabular-nums;">${v.toFixed(2)}</div>`;
      host.appendChild(row);
    }
  }

  function _mkChart(id, cfg) {
    const el = document.getElementById(id);
    if (!el) return;
    if (charts[id]) charts[id].destroy();
    charts[id] = new Chart(el.getContext("2d"), cfg);
  }

  function renderEquity(eq) {
    const labels = eq.map(p => new Date(p.t).toISOString().slice(0, 10));
    const bal = eq.map(p => p.balance);
    const peak = eq.map(p => p.peak);
    _mkChart("equity-chart", {
      type: "line",
      data: {
        labels,
        datasets: [
          { label: "Equity", data: bal, borderColor: PALETTE[0], backgroundColor: "rgba(79,140,255,.18)", fill: true, pointRadius: 0, tension: .2 },
          { label: "Peak", data: peak, borderColor: PALETTE[3], borderDash: [4, 4], pointRadius: 0, fill: false },
        ],
      },
      options: { responsive: true, plugins: { legend: { position: "bottom", labels: { color: "#e6ecff" } } },
        scales: { x: { ticks: { color: "#8ea0c8" } }, y: { ticks: { color: "#8ea0c8" } } } },
    });
  }

  function renderDrawdown(eq) {
    const labels = eq.map(p => new Date(p.t).toISOString().slice(0, 10));
    const dd = eq.map(p => p.dd * 100);
    const mdd = Math.min(...dd);
    _mkChart("drawdown-chart", {
      type: "line",
      data: { labels, datasets: [
        { label: "Drawdown %", data: dd, borderColor: PALETTE[2], backgroundColor: "rgba(255,84,112,.25)", fill: true, pointRadius: 0 },
        { label: `MDD ${mdd.toFixed(2)}%`, data: new Array(dd.length).fill(mdd), borderColor: PALETTE[2], borderDash: [6, 4], pointRadius: 0, fill: false },
      ]},
      options: { responsive: true, plugins: { legend: { labels: { color: "#e6ecff" } } },
        scales: { x: { ticks: { color: "#8ea0c8" } }, y: { ticks: { color: "#8ea0c8" } } } },
    });
  }

  function renderPnlDistribution(metrics, eq) {
    // Build 30-bin histogram from equity_curve trade-returns.
    const perTrade = [];
    for (let i = 1; i < eq.length; i++) {
      const prev = eq[i - 1].peak > 0 ? eq[i - 1].balance : 1;
      perTrade.push((eq[i].balance - eq[i - 1].balance) / prev * 100);
    }
    const bins = 30;
    const lo = Math.min(...perTrade, -1);
    const hi = Math.max(...perTrade, 1);
    const step = (hi - lo) / bins;
    const counts = new Array(bins).fill(0);
    const labels = new Array(bins).fill(0).map((_, i) => (lo + step * (i + 0.5)).toFixed(2) + "%");
    for (const r of perTrade) {
      let idx = Math.floor((r - lo) / step);
      if (idx < 0) idx = 0;
      if (idx >= bins) idx = bins - 1;
      counts[idx]++;
    }
    const var95 = -metrics.risk_adj.var_95 * 100;
    const cvar95 = -metrics.risk_adj.cvar_95 * 100;
    _mkChart("pnl-dist-chart", {
      type: "bar",
      data: { labels, datasets: [
        { label: "Count", data: counts, backgroundColor: PALETTE[0] },
      ]},
      options: { responsive: true, plugins: {
          legend: { labels: { color: "#e6ecff" } },
          annotation: {},
        }, scales: { x: { ticks: { color: "#8ea0c8", maxRotation: 90 } }, y: { ticks: { color: "#8ea0c8" } } } },
    });
  }

  function renderMCDistributions(mc) {
    // Violin-ish: show p0/p5/p25/p50/p75/p95/p100 as box plot series for TR, MDD, SR, PF.
    const ta = mc.test_a.percentiles;
    const series = [
      { label: "TR %", vals: ta.total_return.map(x => x * 100) },
      { label: "MDD %", vals: ta.max_drawdown.map(x => x * 100) },
      { label: "Sharpe", vals: ta.sharpe_ratio },
      { label: "PF", vals: ta.profit_factor },
    ];
    const labels = ["p0", "p5", "p25", "p50", "p75", "p95", "p100"];
    _mkChart("mc-dist-chart", {
      type: "line",
      data: { labels, datasets: series.map((s, i) => ({
        label: s.label, data: s.vals, borderColor: PALETTE[i],
        backgroundColor: PALETTE[i] + "22", fill: true, tension: .3, pointRadius: 3,
      }))},
      options: { responsive: true, plugins: { legend: { position: "bottom", labels: { color: "#e6ecff" } } },
        scales: { x: { ticks: { color: "#8ea0c8" } }, y: { ticks: { color: "#8ea0c8" } } } },
    });
  }

  function renderScoreBreakdown(cats) {
    if (!cats) { _mkChart("score-chart", {}); return; }
    const labels = ["Profit", "RiskAdj", "DD", "Robust", "Sanity", "Suff"];
    const vals = [cats.profitability, cats.risk_adj, cats.drawdown, cats.robustness, cats.sanity, cats.sufficiency];
    _mkChart("score-chart", {
      type: "bar",
      data: { labels, datasets: [{ label: "Sub-score (/10)", data: vals, backgroundColor: PALETTE.slice(0, 6) }] },
      options: { responsive: true, plugins: { legend: { labels: { color: "#e6ecff" } } },
        scales: { x: { ticks: { color: "#8ea0c8" } }, y: { max: 10, ticks: { color: "#8ea0c8" } } } },
    });
  }

  function renderMonthlyHeat(mr) {
    const host = $("#monthly-heat");
    host.innerHTML = "";
    const entries = Object.entries(mr).sort();
    if (!entries.length) { host.innerHTML = '<p class="muted">Insufficient data for monthly breakdown.</p>'; return; }
    const byYear = {};
    for (const [k, v] of entries) {
      const [y, m] = k.split("-");
      (byYear[y] = byYear[y] || {})[m] = v;
    }
    const maxAbs = Math.max(1e-9, ...entries.map(([, v]) => Math.abs(v)));
    const mn = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
    const row = document.createElement("div");
    row.className = "heatmap-row";
    row.innerHTML = `<div class="heatmap-year"></div>` + mn.map(m => `<div style="width:40px;text-align:center;color:var(--muted);">${m}</div>`).join("");
    host.appendChild(row);
    for (const y of Object.keys(byYear).sort()) {
      const r = document.createElement("div");
      r.className = "heatmap-row";
      r.innerHTML = `<div class="heatmap-year">${y}</div>`;
      for (let i = 1; i <= 12; i++) {
        const k = String(i).padStart(2, "0");
        const v = byYear[y][k];
        const cell = document.createElement("div");
        cell.className = "heatmap-cell";
        if (v == null) { cell.style.background = "#182238"; cell.textContent = ""; }
        else {
          const t = Math.max(-1, Math.min(1, v / maxAbs));
          const col = t >= 0 ? `rgba(50,213,131,${0.2 + t * 0.7})` : `rgba(255,84,112,${0.2 + -t * 0.7})`;
          cell.style.background = col;
          cell.textContent = (v * 100).toFixed(1);
        }
        r.appendChild(cell);
      }
      host.appendChild(r);
    }
  }

  function renderMetricsTable(card) {
    const tbl = $("#metrics-table");
    const rows = [];
    const push = (name, val, fmt = v => v) => {
      const v = val == null || (typeof val === "number" && !isFinite(val)) ? "—" : fmt(val);
      rows.push({ name, val: v });
    };
    const { basic, growth, drawdown, risk_adj, behav } = card.metrics;
    push("Total Trades", basic.total_trades, v => v);
    push("Win Rate", basic.win_rate, v => (v * 100).toFixed(2) + "%");
    push("Profit Factor", basic.profit_factor, v => v.toFixed(3));
    push("Payoff Ratio", basic.payoff_ratio, v => v.toFixed(3));
    push("Expectancy / $", basic.expectancy_per_unit_risk, v => v.toFixed(3));
    push("Net P&L", growth.net_pnl, v => v.toFixed(2));
    push("Total Return", growth.total_return, v => (v * 100).toFixed(2) + "%");
    push("CAGR", growth.cagr, v => (v * 100).toFixed(2) + "%");
    push("Max Drawdown", drawdown.max_drawdown, v => (v * 100).toFixed(2) + "%");
    push("Ulcer Index", drawdown.ulcer_index, v => (v * 100).toFixed(2) + "%");
    push("Recovery Factor", drawdown.recovery_factor, v => v.toFixed(2));
    push("Sharpe Ratio", risk_adj.sharpe_ratio, v => v.toFixed(3));
    push("Sortino Ratio", risk_adj.sortino_ratio, v => v.toFixed(3));
    push("Calmar Ratio", risk_adj.calmar_ratio, v => v.toFixed(3));
    push("Omega Ratio", risk_adj.omega_ratio, v => v.toFixed(3));
    push("Tail Ratio", risk_adj.tail_ratio, v => v.toFixed(3));
    push("VaR 95%", risk_adj.var_95, v => (v * 100).toFixed(2) + "%");
    push("CVaR 95%", risk_adj.cvar_95, v => (v * 100).toFixed(2) + "%");
    push("Max Win Streak", behav.max_win_streak);
    push("Max Loss Streak", behav.max_loss_streak);
    push("Runs Z-score", behav.runs_z_score, v => v.toFixed(3));
    push("Skew (returns)", growth.skew_returns, v => v.toFixed(3));
    push("Excess Kurtosis", growth.excess_kurtosis_returns, v => v.toFixed(3));
    push("% Profitable Months", behav.profitable_month_fraction, v => (v * 100).toFixed(0) + "%");

    tbl.innerHTML = `<thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>` +
      rows.map(r => `<tr><td>${r.name}</td><td style="text-align:right;font-variant-numeric:tabular-nums;">${r.val}</td></tr>`).join("") +
      `</tbody>`;
  }

  // ------- Saved / Compare view -------
  async function refreshSavedList() {
    const host = $("#saved-list");
    host.innerHTML = "";
    try {
      const r = await fetch("/evaluations");
      const list = await r.json();
      if (!list.length) { host.innerHTML = '<p class="muted">No saved evaluations yet.</p>'; return; }
      for (const item of list) {
        const div = document.createElement("label");
        div.className = "saved-item";
        const isKO = item.status === "KNOCKED_OUT";
        div.innerHTML = `
          <input type="checkbox" data-id="${item.id}" class="saved-check"/>
          <div class="mini-score" style="color:${isKO ? "var(--bad)" : "var(--text)"}">${item.final_score}</div>
          <h4>${item.name}${isKO ? ' <span class="pill ko" style="font-size:.65rem;padding:.1rem .4rem;">KO</span>' : ''}</h4>
          <div class="meta">${item.label || ""} · N=${item.total_trades} · ${item.created_at ? new Date(item.created_at).toLocaleString() : ""}</div>`;
        host.appendChild(div);
      }
      $$(".saved-check").forEach(c => c.addEventListener("change", updateCompareBtn));
    } catch (e) {
      host.textContent = "Failed to load saved evaluations: " + e.message;
    }
  }

  function updateCompareBtn() {
    const ids = $$(".saved-check:checked").map(c => c.dataset.id);
    $("#compare-btn").disabled = ids.length < 1;
    $$(".saved-item").forEach(el => el.classList.remove("selected"));
    $$(".saved-check:checked").forEach(c => c.closest(".saved-item").classList.add("selected"));
  }

  $("#compare-btn").addEventListener("click", async () => {
    const ids = $$(".saved-check:checked").map(c => c.dataset.id);
    if (!ids.length) return;
    try {
      const r = await fetch(`/evaluations/compare?ids=${ids.join(",")}`);
      if (!r.ok) throw new Error("Compare failed");
      const data = await r.json();
      renderCompare(data);
    } catch (e) { alert(e.message); }
  });

  function renderCompare(data) {
    $("#compare-area").style.display = "block";
    // Equity overlay
    const ids = Object.keys(data.scorecards);
    const datasets = [];
    const card0 = data.scorecards[ids[0]];
    const labels0 = card0.equity_curve.map(p => new Date(p.t).toISOString().slice(0, 10));
    for (let i = 0; i < ids.length; i++) {
      const c = data.scorecards[ids[i]];
      datasets.push({
        label: c.name + " (" + c.scoring.final_score + ")",
        data: c.equity_curve.map(p => p.balance),
        borderColor: PALETTE[i % PALETTE.length],
        pointRadius: 0, tension: .2, fill: false,
      });
    }
    _mkChart("compare-equity-chart", {
      type: "line", data: { labels: labels0, datasets },
      options: { responsive: true, plugins: { legend: { position: "bottom", labels: { color: "#e6ecff" } } },
        scales: { x: { ticks: { color: "#8ea0c8" } }, y: { ticks: { color: "#8ea0c8" } } } },
    });
    // Radar (category scores)
    const catLabels = ["Profit", "RiskAdj", "DD", "Robust", "Sanity", "Suff"];
    const radarDS = ids.map((id, i) => {
      const cs = data.scorecards[id].scoring.category_scores;
      const vals = cs ? [cs.profitability, cs.risk_adj, cs.drawdown, cs.robustness, cs.sanity, cs.sufficiency] : [0, 0, 0, 0, 0, 0];
      return { label: data.scorecards[id].name, data: vals, borderColor: PALETTE[i % PALETTE.length],
        backgroundColor: PALETTE[i % PALETTE.length] + "22", pointBackgroundColor: PALETTE[i % PALETTE.length] };
    });
    _mkChart("compare-radar-chart", {
      type: "radar",
      data: { labels: catLabels, datasets: radarDS },
      options: { responsive: true, scales: { r: { min: 0, max: 10, ticks: { color: "#8ea0c8" },
        grid: { color: "#233055" }, pointLabels: { color: "#e6ecff" } } },
        plugins: { legend: { position: "bottom", labels: { color: "#e6ecff" } } } },
    });
    // Compare table
    const rows = data.rows || [];
    const host = $("#compare-table");
    const thead = `<thead><tr><th>Metric</th><th>Baseline (${data.baseline_id})</th>` +
      ids.filter(i => i !== data.baseline_id).map(i => `<th>${i}<br/><span style="color:var(--muted);">Δ %</span></th>`).join("") + `</tr></thead>`;
    const tbody = rows.map(row => {
      const baselinVal = formatVal(row.metric, row.baseline);
      const cols = ids.filter(i => i !== data.baseline_id).map(i => {
        const v = formatVal(row.metric, row.by_id[i]);
        const d = row.delta_pct && row.delta_pct[i];
        const delta = d == null ? "" : `<div class="${d >= 0 ? "delta-pos" : "delta-neg"}">${(d >= 0 ? "+" : "") + d.toFixed(2)}%</div>`;
        return `<td style="text-align:right;">${v}${delta}</td>`;
      }).join("");
      return `<tr><td>${row.metric}</td><td style="text-align:right;">${baselinVal}</td>${cols}</tr>`;
    }).join("");
    host.innerHTML = thead + "<tbody>" + tbody + "</tbody>";
  }

  function formatVal(metric, v) {
    if (v == null || (typeof v === "number" && !isFinite(v))) return "—";
    if (typeof v === "number") {
      if (/(rate|ratio|return|cagr|drawdown|volatility|factor|sharpe|sortino|calmar|omega|var|cvar|skew|kurt|fraction|ulcer|tail|probability|stability|rate)$/i.test(metric)) {
        return (v * 100).toFixed(2) + "%";
      }
      return v.toFixed(3);
    }
    return String(v);
  }

  // Load saved list once on first switch to compare (already handled by nav click).
})();
