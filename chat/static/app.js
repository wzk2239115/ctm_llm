if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
    const radii = Array.isArray(r) ? r : [r, r, r, r];
    this.moveTo(x + radii[0], y);
    this.lineTo(x + w - radii[1], y);
    this.quadraticCurveTo(x + w, y, x + w, y + radii[1]);
    this.lineTo(x + w, y + h - radii[2]);
    this.quadraticCurveTo(x + w, y + h, x + w - radii[2], y + h);
    this.lineTo(x + radii[3], y + h);
    this.quadraticCurveTo(x, y + h, x, y + h - radii[3]);
    this.lineTo(x, y + radii[0]);
    this.quadraticCurveTo(x, y, x + radii[0], y);
    this.closePath();
  };
}

const LAYER_COLORS = [
  '#10a37f', '#3b82f6', '#8b5cf6', '#f59e0b',
  '#ef4444', '#ec4899', '#06b6d4', '#84cc16',
  '#f97316', '#6366f1', '#14b8a6', '#a855f7',
  '#e11d48', '#0ea5e9', '#22c55e', '#d946ef',
];

let messages = [];
let isGenerating = false;
let currentTracking = null;
let vizVisible = false;
let confidenceThreshold = 0.8;
let tokenTickLog = [];
let numTicks = 4;
let syncAnimState = null;
let syncAnimFrame = null;

const inputEl = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const messagesEl = document.getElementById('messages');
const welcomeEl = document.getElementById('welcome');
const confSlider = document.getElementById('confidence-slider');

inputEl.addEventListener('input', () => {
  inputEl.style.height = 'auto';
  inputEl.style.height = Math.min(inputEl.scrollHeight, 120) + 'px';
  sendBtn.disabled = !inputEl.value.trim() || isGenerating;
});

function handleKey(e) {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
}

function updateConfLabel(val) {
  confidenceThreshold = parseFloat(val);
  document.getElementById('confidence-label').textContent = confidenceThreshold.toFixed(2);
}

function toggleSidebar() {
  document.getElementById('sidebar').classList.toggle('collapsed');
}

function toggleVizPanel() {
  vizVisible = !vizVisible;
  document.getElementById('viz-panel').classList.toggle('hidden', !vizVisible);
  document.getElementById('viz-toggle').classList.toggle('active', vizVisible);
  if (vizVisible && currentTracking) renderAllViz(currentTracking);
}

function switchVizTab(tab) {
  document.querySelectorAll('.viz-tab').forEach(t => t.classList.toggle('active', t.dataset.tab === tab));
  document.querySelectorAll('.viz-section').forEach(s => s.classList.toggle('active', s.id === 'viz-' + tab));
  if (currentTracking) renderAllViz(currentTracking);
}

function setPrompt(text) {
  inputEl.value = text;
  inputEl.dispatchEvent(new Event('input'));
  inputEl.focus();
}

function newChat() {
  messages = [];
  currentTracking = null;
  tokenTickLog = [];
  messagesEl.innerHTML = '';
  messagesEl.appendChild(welcomeEl);
  welcomeEl.style.display = 'flex';
  clearCanvases();
  document.getElementById('tick-stats').classList.add('hidden');
}

function addMessage(role, text) {
  welcomeEl.style.display = 'none';
  const div = document.createElement('div');
  div.className = `message ${role}`;
  div.innerHTML = `
    <div>
      <div class="message-label">${role === 'user' ? 'You' : 'CTM-LLM'}</div>
      <div class="message-bubble">${escapeHtml(text)}</div>
    </div>`;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
  return div;
}

function escapeHtml(t) {
  return t.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function updateTickStatsBar() {
  if (tokenTickLog.length === 0) return;

  const statsEl = document.getElementById('tick-stats');
  const barEl = document.getElementById('tick-stats-bar');
  const summaryEl = document.getElementById('tick-stats-summary');
  statsEl.classList.remove('hidden');

  const counts = new Array(numTicks).fill(0);
  tokenTickLog.forEach(t => { if (t >= 0 && t < numTicks) counts[t]++; });
  const maxCount = Math.max(...counts, 1);

  barEl.innerHTML = '';
  counts.forEach((c, i) => {
    const seg = document.createElement('div');
    seg.className = 'tick-bar-segment';
    const pct = (c / maxCount) * 100;
    const minH = 4;
    const h = minH + (pct / 100) * 20;
    seg.style.height = h + 'px';
    seg.style.background = c > 0
      ? `linear-gradient(to top, ${LAYER_COLORS[i % LAYER_COLORS.length]}, ${LAYER_COLORS[i % LAYER_COLORS.length]}88)`
      : 'var(--border)';
    seg.innerHTML = `<span class="tick-bar-label">t${i}:${c}</span>`;
    barEl.appendChild(seg);
  });

  const earlyStops = tokenTickLog.filter(t => t < numTicks - 1).length;
  const avgTick = (tokenTickLog.reduce((a, b) => a + b, 0) / tokenTickLog.length).toFixed(1);
  summaryEl.innerHTML = `<strong>${tokenTickLog.length}</strong> tokens, avg tick <strong>${avgTick}</strong>, <strong>${earlyStops}</strong> early stops`;
}

async function sendMessage() {
  const text = inputEl.value.trim();
  if (!text || isGenerating) return;

  messages.push({ role: 'user', content: text });
  addMessage('user', text);

  inputEl.value = '';
  inputEl.style.height = 'auto';
  sendBtn.disabled = true;
  isGenerating = true;
  tokenTickLog = [];

  const assistantDiv = addMessage('assistant', '');
  const bubble = assistantDiv.querySelector('.message-bubble');
  bubble.textContent = '';

  const indicator = document.getElementById('thinking-indicator');
  indicator.classList.remove('hidden');
  document.getElementById('tick-count').textContent = numTicks;

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        messages,
        max_new_tokens: 128,
        temperature: 0.3,
        top_p: 0.8,
        top_k: 40,
        repetition_penalty: 1.08,
        confidence_threshold: confidenceThreshold,
      }),
    });

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let fullText = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split('\n');
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        try {
          const data = JSON.parse(line.slice(6));
          if (data.type === 'metadata') {
            indicator.classList.add('hidden');
            if (data.num_ticks) numTicks = data.num_ticks;
            if (data.tick_metrics) {
              document.getElementById('tick-count').textContent = data.tick_metrics.losses.length;
            }
            if (data.tracking) {
              currentTracking = data.tracking;
              if (data.tracking.neuron_activations) {
                stopSyncLoop();
                initSyncAnimation(data.tracking.neuron_activations);
              }
              if (vizVisible) renderAllViz(data.tracking);
            }
          } else if (data.type === 'token') {
            if (data.eos) { /* skip */ }
            else {
              fullText += data.text;
              bubble.textContent = fullText;
              messagesEl.scrollTop = messagesEl.scrollHeight;
              if (data.tick !== undefined) {
                tokenTickLog.push(data.tick);
                updateTickStatsBar();
              }
            }
          } else if (data.type === 'done') {
            indicator.classList.add('hidden');
          }
        } catch (e) { /* skip */ }
      }
    }

    messages.push({ role: 'assistant', content: fullText });
  } catch (err) {
    bubble.textContent = 'Error: ' + err.message;
    indicator.classList.add('hidden');
  }

  isGenerating = false;
  sendBtn.disabled = !inputEl.value.trim();
}

function clearCanvases() {
  ['clock-canvas', 'elf-canvas', 'neuron-canvas', 'attention-canvas'].forEach(id => {
    const c = document.getElementById(id);
    const ctx = c.getContext('2d');
    ctx.clearRect(0, 0, c.width, c.height);
  });
  document.getElementById('clock-legend').innerHTML = '';
  document.getElementById('elf-top-tokens').innerHTML = '';
}

function renderAllViz(tracking) {
  if (tracking.neuron_activations) {
    stopSyncLoop();
    initSyncAnimation(tracking.neuron_activations);
  }
  if (tracking.tick_metrics) renderELF(tracking.tick_metrics);
  if (tracking.attention) renderClock(tracking.attention);
  if (tracking.neuron_firing) renderNeurons(tracking.neuron_firing);
  if (tracking.attention) renderAttention(tracking.attention);
}

function setupCanvas(canvasId, w, h) {
  const c = document.getElementById(canvasId);
  const dpr = window.devicePixelRatio || 1;
  c.width = w * dpr;
  c.height = h * dpr;
  c.style.width = w + 'px';
  c.style.height = h + 'px';
  const ctx = c.getContext('2d');
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, w, h);
  return { c, ctx, w, h };
}

function drawAxes(ctx, w, h, pad, xLabel, yLabel) {
  ctx.strokeStyle = '#333';
  ctx.lineWidth = 0.5;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, h - pad.bottom);
  ctx.lineTo(w - pad.right, h - pad.bottom);
  ctx.stroke();

  ctx.fillStyle = '#555';
  ctx.font = '10px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  if (xLabel) ctx.fillText(xLabel, (pad.left + w - pad.right) / 2, h - 4);
  ctx.textAlign = 'right';
  if (yLabel) {
    ctx.save();
    ctx.translate(10, (pad.top + h - pad.bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();
  }
}

function renderClock(attentionData) {
  const { ctx, w, h } = setupCanvas('clock-canvas', 440, 260);
  const pad = { top: 20, right: 20, bottom: 35, left: 50 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  drawAxes(ctx, w, h, pad, 'Tick', 'Activation Variance');

  const layers = Object.keys(attentionData).sort((a, b) => {
    return attentionData[a].layer_id - attentionData[b].layer_id;
  });

  let maxVal = 0;
  layers.forEach(l => { maxVal = Math.max(maxVal, ...attentionData[l].activation_variance); });
  if (maxVal === 0) maxVal = 1;

  const legend = document.getElementById('clock-legend');
  legend.innerHTML = '';

  const showLayers = layers.length > 6 ? layers.filter((_, i) => i % Math.ceil(layers.length / 6) === 0 || i === layers.length - 1) : layers;

  showLayers.forEach((layerName, idx) => {
    const data = attentionData[layerName];
    const color = LAYER_COLORS[idx % LAYER_COLORS.length];
    const ticks = data.activation_variance;
    const n = ticks.length;
    if (n < 2) return;

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const x = pad.left + (i / (n - 1)) * plotW;
      const y = pad.top + plotH - (ticks[i] / maxVal) * plotH;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    const dot = document.createElement('span');
    dot.className = 'legend-item';
    dot.innerHTML = `<span class="legend-dot" style="background:${color}"></span>L${data.layer_id}`;
    legend.appendChild(dot);
  });
}

function renderELF(tickMetrics) {
  const { ctx, w, h } = setupCanvas('elf-canvas', 440, 280);
  const pad = { top: 20, right: 50, bottom: 35, left: 50 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  drawAxes(ctx, w, h, pad, 'Tick', 'Value');

  const confidences = tickMetrics.confidences || tickMetrics.entropies.map(e => 1 - e);
  const losses = tickMetrics.losses;
  const entropies = tickMetrics.entropies;
  const n = losses.length;
  if (n < 2) return;

  const allVals = [...losses, ...entropies, ...confidences];
  let gMin = Math.min(...allVals), gMax = Math.max(...allVals);
  const range = gMax - gMin || 1;
  gMin -= range * 0.1;
  gMax += range * 0.1;
  const totalRange = gMax - gMin;

  function toY(v) { return pad.top + plotH - ((v - gMin) / totalRange) * plotH; }
  function toX(i) { return pad.left + (i / (n - 1)) * plotW; }

  // Confidence threshold line
  const threshY = toY(confidenceThreshold);
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = '#10a37f88';
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, threshY);
  ctx.lineTo(w - pad.right, threshY);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#10a37f';
  ctx.font = '9px -apple-system, sans-serif';
  ctx.textAlign = 'left';
  ctx.fillText(`thresh=${confidenceThreshold.toFixed(2)}`, w - pad.right + 4, threshY + 3);

  function drawLine(data, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((v, i) => {
      const x = toX(i), y = toY(v);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = color;
    data.forEach((v, i) => {
      ctx.beginPath();
      ctx.arc(toX(i), toY(v), 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  drawLine(losses, '#ef4444');
  drawLine(entropies, '#f59e0b');
  drawLine(confidences, '#10a37f');

  // Highlight selected tick (first tick >= threshold)
  let selectedTick = n - 1;
  for (let t = 0; t < n; t++) {
    if (confidences[t] >= confidenceThreshold) {
      selectedTick = t;
      break;
    }
  }
  const selX = toX(selectedTick);
  ctx.strokeStyle = '#ffffff44';
  ctx.lineWidth = 1;
  ctx.setLineDash([2, 2]);
  ctx.beginPath();
  ctx.moveTo(selX, pad.top);
  ctx.lineTo(selX, pad.top + plotH);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = '#fff';
  ctx.font = '9px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(`selected: t=${selectedTick}`, selX, pad.top - 6);

  const legend = document.getElementById('clock-legend');
  if (legend.childElementCount === 0) {
    [
      { color: '#ef4444', label: 'Loss' },
      { color: '#f59e0b', label: 'Entropy' },
      { color: '#10a37f', label: 'Confidence' },
      { color: '#10a37f88', label: 'Threshold', dash: true },
    ].forEach(({ color, label, dash }) => {
      const s = document.createElement('span');
      s.className = 'legend-item';
      s.innerHTML = dash
        ? `<span class="legend-dot" style="background:transparent;border:1px dashed ${color}"></span>${label}`
        : `<span class="legend-dot" style="background:${color}"></span>${label}`;
      legend.appendChild(s);
    });
  }

  const topTokens = tickMetrics.top_tokens;
  const container = document.getElementById('elf-top-tokens');
  if (topTokens && topTokens.length > 0) {
    container.innerHTML = '<div style="font-size:11px;color:#666;margin-bottom:8px;font-weight:600;">Per-tick top token predictions</div>';
    topTokens.forEach((tokens, t) => {
      const isSelected = t === selectedTick;
      const row = document.createElement('div');
      row.className = 'top-tokens-row';
      if (isSelected) row.style.background = 'var(--accent-dim)';
      row.innerHTML = `<span class="tick-label" style="${isSelected ? 'color:var(--accent);font-weight:700' : ''}">t=${t}${isSelected ? ' ✓' : ''}</span>` +
        tokens.map(tok => `<span class="token-chip">${escapeHtml(tok.token)}</span>`).join('');
      container.appendChild(row);
    });
  }
}

function renderNeurons(neuronData) {
  const { ctx, w, h } = setupCanvas('neuron-canvas', 440, 260);
  const pad = { top: 20, right: 20, bottom: 35, left: 50 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  drawAxes(ctx, w, h, pad, 'Neuron Index', 'Activation');

  const grid = neuronData.firing_grid;
  const numTicksGrid = grid.length;
  const numNeurons = grid[0].length;
  if (numTicksGrid === 0 || numNeurons === 0) return;

  let allVals = grid.flat();
  let vmin = Math.min(...allVals), vmax = Math.max(...allVals);
  const vrange = vmax - vmin || 1;

  const cellW = plotW / numNeurons;
  const cellH = plotH / numTicksGrid;

  for (let t = 0; t < numTicksGrid; t++) {
    for (let n = 0; n < numNeurons; n++) {
      const val = (grid[t][n] - vmin) / vrange;
      const x = pad.left + n * cellW;
      const y = pad.top + t * cellH;

      const r = Math.round(16 + val * 0);
      const g = Math.round(163 * val);
      const b = Math.round(127 * val);
      ctx.fillStyle = `rgba(${r},${g},${b},${0.3 + val * 0.7})`;
      ctx.fillRect(x, y, cellW - 0.5, cellH - 0.5);
    }
  }

  ctx.fillStyle = '#555';
  ctx.font = '10px -apple-system, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('Tick 0', pad.left, pad.top - 4);
  ctx.fillText(`Tick ${numTicksGrid - 1}`, pad.left, pad.top + numTicksGrid * cellH + 14);
}

function renderAttention(attentionData) {
  const { ctx, w, h } = setupCanvas('attention-canvas', 440, 260);
  const pad = { top: 20, right: 20, bottom: 35, left: 50 };
  const plotW = w - pad.left - pad.right;
  const plotH = h - pad.top - pad.bottom;

  drawAxes(ctx, w, h, pad, 'Layer', 'Sync Norm');

  const layers = Object.keys(attentionData).sort((a, b) => {
    return attentionData[a].layer_id - attentionData[b].layer_id;
  });

  const numLayers = layers.length;
  if (numLayers === 0) return;

  const lastTickSyncNorms = layers.map(l => {
    const sn = attentionData[l].sync_norm;
    return sn.length > 0 ? sn[sn.length - 1] : 0;
  });

  let maxV = Math.max(...lastTickSyncNorms, 0.001);

  const barW = Math.min(20, plotW / numLayers - 4);
  const gap = (plotW - barW * numLayers) / (numLayers + 1);

  lastTickSyncNorms.forEach((v, i) => {
    const x = pad.left + gap + i * (barW + gap);
    const barH = (v / maxV) * plotH;
    const y = pad.top + plotH - barH;

    const gradient = ctx.createLinearGradient(x, y, x, pad.top + plotH);
    gradient.addColorStop(0, LAYER_COLORS[i % LAYER_COLORS.length]);
    gradient.addColorStop(1, 'rgba(10,10,10,0.8)');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.roundRect(x, y, barW, barH, [3, 3, 0, 0]);
    ctx.fill();

    ctx.fillStyle = '#555';
    ctx.font = '8px -apple-system, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(`L${i}`, x + barW / 2, pad.top + plotH + 12);
  });
}

const SYNC_SIZE = 320;

function fibSpiralPositions(n, cx, cy, maxR) {
  const positions = [];
  const golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const r = maxR * Math.sqrt((i + 0.5) / n);
    const theta = i * golden;
    positions.push({ x: cx + r * Math.cos(theta), y: cy + r * Math.sin(theta) });
  }
  return positions;
}

function initSyncAnimation(neuronData) {
  if (!neuronData) return;
  const d_model = neuronData.d_model || 512;
  const acts = neuronData.activations;
  const numT = neuronData.num_ticks;
  const pairsL = neuronData.sync_pairs_left || [];
  const pairsR = neuronData.sync_pairs_right || [];

  const cx = SYNC_SIZE / 2, cy = SYNC_SIZE / 2;
  const maxR = SYNC_SIZE / 2 - 12;
  const positions = fibSpiralPositions(d_model, cx, cy, maxR);

  syncAnimState = { positions, activations: acts, numTicks: numT, d_model, pairsL, pairsR };
  document.getElementById('sync-total-count').textContent = d_model;
  startSyncLoop();
}

function drawSyncFrame(ctx, state, tickFrac, canvasSize) {
  const { positions, activations, numTicks, d_model, pairsL, pairsR } = state;
  const w = canvasSize, h = canvasSize;
  const cx = w / 2, cy = h / 2;

  ctx.clearRect(0, 0, w, h);

  for (let r = 30; r < cx; r += 50) {
    ctx.strokeStyle = 'rgba(255,255,255,0.025)';
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();
  }

  const tickIdx = Math.floor(tickFrac);
  const frac = tickFrac - tickIdx;
  const acts0 = tickIdx < 0 ? new Array(d_model).fill(0) : activations[Math.min(tickIdx, numTicks - 1)];
  const acts1 = tickIdx + 1 < numTicks ? activations[tickIdx + 1] : acts0;

  const interp = new Float64Array(d_model);
  for (let i = 0; i < d_model; i++) {
    interp[i] = acts0[i] * (1 - frac) + acts1[i] * frac;
  }

  const showPairs = Math.min(pairsL.length, 24);
  ctx.lineWidth = 0.3;
  for (let p = 0; p < showPairs; p++) {
    const li = pairsL[p], ri = pairsR[p];
    if (li >= d_model || ri >= d_model) continue;
    const aL = Math.abs(interp[li]), aR = Math.abs(interp[ri]);
    const strength = (aL + aR) / 2;
    if (strength < 0.15) continue;
    ctx.strokeStyle = `rgba(239,68,68,${Math.min(strength * 0.4, 0.35)})`;
    ctx.beginPath();
    ctx.moveTo(positions[li].x, positions[li].y);
    ctx.lineTo(positions[ri].x, positions[ri].y);
    ctx.stroke();
  }

  let activeCount = 0;
  for (let i = 0; i < d_model; i++) {
    const v = interp[i];
    const absV = Math.abs(v);
    if (absV > 0.2) activeCount++;
    const pos = positions[i];
    const baseR = 2;
    const r = baseR + absV * 2.5;

    let fillColor;
    if (v > 0) {
      const t = Math.min(v, 1);
      fillColor = `rgba(${60 + 179 * t | 0},${60 * (1 - t) | 0},${60 * (1 - t) | 0},${0.3 + t * 0.7})`;
    } else if (v < 0) {
      const t = Math.min(-v, 1);
      fillColor = `rgba(${60 * (1 - t) | 0},${60 + 80 * t | 0},${60 + 196 * t | 0},${0.3 + t * 0.7})`;
    } else {
      fillColor = 'rgba(50,50,50,0.25)';
    }

    if (absV > 0.5) {
      ctx.shadowColor = v > 0 ? 'rgba(239,68,68,0.5)' : 'rgba(59,130,246,0.5)';
      ctx.shadowBlur = absV * 6;
    } else {
      ctx.shadowColor = 'transparent';
      ctx.shadowBlur = 0;
    }

    ctx.fillStyle = fillColor;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
    ctx.fill();
  }

  ctx.shadowColor = 'transparent';
  ctx.shadowBlur = 0;
  document.getElementById('sync-active-count').textContent = activeCount;
}

let syncLoopRunning = false;
let syncStartTime = 0;
const TICK_DUR = 700;
const TICK_GAP = 300;

function setupSyncCanvas() {
  const canvas = document.getElementById('sync-canvas');
  const dpr = window.devicePixelRatio || 1;
  canvas.width = SYNC_SIZE * dpr;
  canvas.height = SYNC_SIZE * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  return ctx;
}

function drawIdleSync() {
  if (syncAnimState) return;
  const ctx = setupSyncCanvas();
  const cx = SYNC_SIZE / 2, cy = SYNC_SIZE / 2;
  ctx.clearRect(0, 0, SYNC_SIZE, SYNC_SIZE);

  const d_model = 512;
  const maxR = SYNC_SIZE / 2 - 12;
  const positions = fibSpiralPositions(d_model, cx, cy, maxR);

  for (let i = 0; i < d_model; i++) {
    ctx.fillStyle = 'rgba(50,50,50,0.25)';
    ctx.beginPath();
    ctx.arc(positions[i].x, positions[i].y, 2, 0, Math.PI * 2);
    ctx.fill();
  }
  document.getElementById('sync-active-count').textContent = '0';
}

function startSyncLoop() {
  if (!syncAnimState) return;
  if (syncLoopRunning) return;
  syncLoopRunning = true;
  syncStartTime = performance.now();

  const ctx = setupSyncCanvas();

  function loop(now) {
    if (!syncLoopRunning) return;
    if (!syncAnimState) { syncLoopRunning = false; return; }

    const elapsed = now - syncStartTime;
    const cycleLen = syncAnimState.numTicks * (TICK_DUR + TICK_GAP) + 800;
    const t = elapsed % cycleLen;
    const tickFrac = Math.min(t / (TICK_DUR + TICK_GAP), syncAnimState.numTicks - 0.001);
    const currentTick = Math.floor(tickFrac);

    drawSyncFrame(ctx, syncAnimState, tickFrac, SYNC_SIZE);
    document.getElementById('sync-tick-label').textContent = `tick ${currentTick}/${syncAnimState.numTicks - 1}`;

    syncAnimFrame = requestAnimationFrame(loop);
  }
  syncAnimFrame = requestAnimationFrame(loop);
}

function stopSyncLoop() {
  syncLoopRunning = false;
  if (syncAnimFrame) cancelAnimationFrame(syncAnimFrame);
}

function renderSyncStatic() {
  if (!syncAnimState) return;
  stopSyncLoop();
  const ctx = setupSyncCanvas();
  const lastTick = syncAnimState.numTicks - 1;
  drawSyncFrame(ctx, syncAnimState, lastTick, SYNC_SIZE);
  document.getElementById('sync-tick-label').textContent = `tick ${lastTick}/${lastTick} (final)`;
}

async function loadModelInfo() {
  try {
    const resp = await fetch('/api/model_info');
    const info = await resp.json();
    const statusEl = document.getElementById('model-status');
    if (info.status === 'loaded') {
      statusEl.textContent = 'Model ready';
      statusEl.className = 'model-status ready';
      numTicks = info.iterations;
      document.getElementById('param-ticks').textContent = info.iterations;
      document.getElementById('param-layers').textContent = info.num_hidden_layers;
      document.getElementById('param-neurons').textContent = `d=${info.d_model}`;
      document.getElementById('sync-total-count').textContent = info.d_model;
    } else {
      statusEl.textContent = 'Model not loaded';
      statusEl.className = 'model-status error';
    }
  } catch (e) {
    document.getElementById('model-status').textContent = 'Connection error';
    document.getElementById('model-status').className = 'model-status error';
  }
  drawIdleSync();
}

loadModelInfo();
