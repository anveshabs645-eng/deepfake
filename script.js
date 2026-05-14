// ── Animated background network ──
const canvas = document.getElementById('bgCanvas');
const ctx = canvas.getContext('2d');
let nodes = [];

function resizeCanvas() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
}

function initNodes() {
  nodes = [];
  for (let i = 0; i < 55; i++) {
    nodes.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      r: Math.random() * 2 + 1
    });
  }
}

function drawNetwork() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  nodes.forEach(n => {
    n.x += n.vx; n.y += n.vy;
    if (n.x < 0 || n.x > canvas.width) n.vx *= -1;
    if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = '#e8609a';
    ctx.fill();
  });
  for (let i = 0; i < nodes.length; i++) {
    for (let j = i + 1; j < nodes.length; j++) {
      const dx = nodes[i].x - nodes[j].x;
      const dy = nodes[i].y - nodes[j].y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 140) {
        ctx.beginPath();
        ctx.moveTo(nodes[i].x, nodes[i].y);
        ctx.lineTo(nodes[j].x, nodes[j].y);
        ctx.strokeStyle = `rgba(232,96,154,${0.15 * (1 - dist / 140)})`;
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    }
  }
  requestAnimationFrame(drawNetwork);
}

resizeCanvas(); initNodes(); drawNetwork();
window.addEventListener('resize', () => { resizeCanvas(); initNodes(); });

// ── Auto-fetch stats from /health ──
async function fetchStats() {
  try {
    const res = await fetch('http://127.0.0.1:5000/health');
    const data = await res.json();
    const videosEl   = document.getElementById('statVideos');
    const accuracyEl = document.getElementById('statAccuracy');
    if (videosEl)   videosEl.textContent   = data.videos_trained ?? '—';
    if (accuracyEl) accuracyEl.textContent = data.val_accuracy   ? data.val_accuracy + '%' : '—';
  } catch (e) {
    console.warn('Could not fetch stats:', e);
  }
}
fetchStats();

// ── Tab switching ──
function switchTab(tab, el) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById(tab + 'Tab').classList.add('active');
  el.classList.add('active');
}

// ── File input & drag drop ──
const videoInput  = document.getElementById('videoInput');
const dropZone    = document.getElementById('dropZone');
const fileInfo    = document.getElementById('fileInfo');
const videoPreview = document.getElementById('videoPreview');
const videoPlayer  = document.getElementById('videoPlayer');

videoInput.addEventListener('change', () => handleFile(videoInput.files[0]));

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file) return;
  fileInfo.textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)`;
  const url = URL.createObjectURL(file);
  videoPlayer.src = url;
  videoPreview.style.display = 'block';
}

// ── Loading animation ──
const steps = [
  'Extracting video frames...',
  'Running EfficientNet-B4...',
  'Extracting audio features...',
  'Running Gradient Boosting classifier...',
  'Fusion MLP scoring...',
  'Computing SHA-256 hash...',
  'Generating verdict...'
];

function animateLoading() {
  const bar  = document.getElementById('progressBar');
  const text = document.getElementById('loadingText');
  const step = document.getElementById('loadingStep');
  let i = 0;
  bar.style.width = '0%';
  const interval = setInterval(() => {
    if (i < steps.length) {
      text.textContent = steps[i];
      if (step) step.textContent = `step ${i + 1} of ${steps.length}`;
      bar.style.width = ((i + 1) / steps.length * 90) + '%';
      i++;
    }
  }, 900);
  return interval;
}

// ── Show results ──
function showResults(data) {
  document.getElementById('loading').style.display = 'none';
  document.getElementById('results').style.display = 'block';

  const verdict = data.verdict;
  const vScore  = parseFloat(data.V_score);
  const aScore  = parseFloat(data.A_score);
  const fScore  = parseFloat(data.final_score);

  const card   = document.getElementById('verdictCard');
  const vText  = document.getElementById('verdictText');
  const vSub   = document.getElementById('verdictSub');
  const vCaveat = document.getElementById('verdictCaveat');
  const vGlow  = card.querySelector('.v-glow');

  vText.textContent = verdict;

  if (verdict === 'DEEPFAKE' || verdict === 'DEEPFAKE — POSSIBLE FACESWAP') {
    card.style.borderColor = '#ff3a3a';
    vText.style.color = '#ff3a3a';
    if (vGlow) vGlow.style.background = 'radial-gradient(circle,rgba(255,58,58,0.18),transparent 70%)';
    vSub.textContent = verdict === 'DEEPFAKE — POSSIBLE FACESWAP'
      ? 'Face-swap likely — audio appears genuine'
      : 'High confidence — manipulated media detected';
    if (vCaveat) vCaveat.style.display = 'none';
  } else if (verdict === 'SUSPICIOUS' || verdict === 'SUSPICIOUS — POSSIBLE FACESWAP') {
    card.style.borderColor = '#ffaa00';
    vText.style.color = '#ffaa00';
    if (vGlow) vGlow.style.background = 'radial-gradient(circle,rgba(255,170,0,0.18),transparent 70%)';
    vSub.textContent = verdict === 'SUSPICIOUS — POSSIBLE FACESWAP'
      ? 'Possible face-swap — manual review recommended'
      : 'Inconclusive — manual review recommended';
    if (vCaveat) { vCaveat.style.display = 'block'; vCaveat.textContent = 'Score falls between thresholds. Consider additional verification.'; }
  } else {
    card.style.borderColor = '#00ff88';
    vText.style.color = '#00ff88';
    if (vGlow) vGlow.style.background = 'radial-gradient(circle,rgba(0,255,136,0.18),transparent 70%)';
    vSub.textContent = 'Low confidence of manipulation — video appears authentic';
    if (vCaveat) vCaveat.style.display = 'none';
  }
  
  // Gauges
  setTimeout(() => {
    setGauge('vGauge', vScore);
    setGauge('aGauge', aScore);
    setGauge('fGauge', fScore);
    document.getElementById('vScoreVal').textContent = (vScore * 100).toFixed(1) + '%';
    document.getElementById('aScoreVal').textContent = (aScore * 100).toFixed(1) + '%';
    document.getElementById('fScoreVal').textContent = (fScore * 100).toFixed(1) + '%';
    setBar('vBar', 'vBarNum', vScore);
    setBar('aBar', 'aBarNum', aScore);
    setBar('fBar', 'fBarNum', fScore);
  }, 200);

  // Meta
  document.getElementById('resFilename').textContent = data.filename || data.url || '—';
  document.getElementById('resSha256').textContent   = data.sha256 || '—';
  document.getElementById('resTime').textContent     = data.elapsed_sec ? data.elapsed_sec + 's' : '—';

  // PDF button
  document.getElementById('pdfBtn').onclick = () => {
    window.open(`http://127.0.0.1:5000/report?hash=${data.sha256}`, '_blank');
  };
}

function setGauge(id, score) {
  const total = 141.37;
  document.getElementById(id).style.strokeDashoffset = total - (score * total);
}

function setBar(barId, numId, score) {
  document.getElementById(barId).style.width = (score * 100) + '%';
  document.getElementById(numId).textContent = (score * 100).toFixed(1) + '%';
}

// ── Reset ──
function resetAll() {
  document.getElementById('results').style.display  = 'none';
  document.getElementById('loading').style.display  = 'none';
  document.getElementById('fileInfo').textContent   = 'no file selected';
  document.getElementById('videoPreview').style.display = 'none';
  document.getElementById('videoPlayer').src        = '';
  document.getElementById('urlInput').value         = '';
  videoInput.value = '';
}

// ── Analyze file ──
document.getElementById('analyzeBtn').addEventListener('click', async () => {
  const file = videoInput.files[0];
  if (!file) { alert('Please select a video file first.'); return; }

  document.getElementById('loading').style.display  = 'block';
  document.getElementById('results').style.display  = 'none';
  document.getElementById('progressBar').style.width = '0%';

  const interval  = animateLoading();
  const formData  = new FormData();
  formData.append('video', file);

  try {
    const res  = await fetch('http://127.0.0.1:5000/analyze', { method: 'POST', body: formData });
    const data = await res.json();
    clearInterval(interval);
    document.getElementById('progressBar').style.width = '100%';
    setTimeout(() => showResults(data), 400);
  } catch (err) {
    clearInterval(interval);
    document.getElementById('loading').style.display = 'none';
    alert('Could not connect to Flask backend. Make sure app.py is running on port 5000.');
    console.error(err);
  }
});

// ── Analyze URL ──
document.getElementById('analyzeUrlBtn').addEventListener('click', async () => {
  const url = document.getElementById('urlInput').value.trim();
  if (!url) { alert('Please enter a video URL.'); return; }

  document.getElementById('loading').style.display  = 'block';
  document.getElementById('results').style.display  = 'none';
  document.getElementById('progressBar').style.width = '0%';

  const interval = animateLoading();

  try {
    const res  = await fetch('http://127.0.0.1:5000/analyze_url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const data = await res.json();
    clearInterval(interval);
    document.getElementById('progressBar').style.width = '100%';
    setTimeout(() => showResults(data), 400);
  } catch (err) {
    clearInterval(interval);
    document.getElementById('loading').style.display = 'none';
    alert('Could not connect to Flask backend. Make sure app.py is running on port 5000.');
    console.error(err);
  }
});