const BACKEND = 'http://127.0.0.1:5000';
const DEEPGUARD_URL = 'http://127.0.0.1:5000/app';

const STEPS = [
  'extracting video frames...',
  'running efficientnet-b4...',
  'extracting audio features...',
  'running gradient boosting...',
  'fusion mlp scoring...',
  'computing sha-256...',
  'generating verdict...'
];

let currentTabUrl = '';

document.addEventListener('DOMContentLoaded', async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  currentTabUrl = tab.url || '';
  document.getElementById('currentUrl').textContent = currentTabUrl || 'no url detected';

  try {
    const res = await fetch(`${BACKEND}/health`);
    if (res.ok) {
      document.getElementById('statusDot').classList.remove('offline');
    } else {
      document.getElementById('statusDot').classList.add('offline');
    }
  } catch {
    document.getElementById('statusDot').classList.add('offline');
  }
});

document.getElementById('analyzeBtn').addEventListener('click', () => {
  if (!currentTabUrl) { showError('no url detected on this page.'); return; }
  startAnalysis(currentTabUrl);
});

document.getElementById('retryBtn').addEventListener('click', () => {
  showIdle();
  if (currentTabUrl) startAnalysis(currentTabUrl);
});

document.getElementById('resetBtn').addEventListener('click', showIdle);

document.getElementById('visitBtn').addEventListener('click', () => {
  chrome.tabs.create({ url: DEEPGUARD_URL });
});

document.getElementById('pdfBtn').addEventListener('click', () => {
  const hash = document.getElementById('pdfBtn').dataset.hash;
  if (hash) {
    chrome.tabs.create({ url: `${BACKEND}/report?hash=${hash}` });
  }
});

async function startAnalysis(url) {
  showLoading();

  let i = 0;
  const interval = setInterval(() => {
    if (i < STEPS.length) {
      document.getElementById('loadStep').textContent = STEPS[i];
      document.getElementById('progFill').style.width = ((i + 1) / STEPS.length * 88) + '%';
      i++;
    }
  }, 850);

  try {
    const res = await fetch(`${BACKEND}/analyze_url`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });

    clearInterval(interval);
    document.getElementById('progFill').style.width = '100%';

    if (!res.ok) {
      const err = await res.json();
      showError(err.error || 'analysis failed. try a direct video url.');
      return;
    }

    const data = await res.json();
    setTimeout(() => showResult(data), 400);

  } catch (err) {
    clearInterval(interval);
    showError('could not reach backend.\nmake sure deepguard is running on port 5000.');
  }
}

function showResult(data) {
  const verdict = data.verdict;
  const vScore  = parseFloat(data.V_score);
  const aScore  = parseFloat(data.A_score);
  const fScore  = parseFloat(data.final_score);

  const card    = document.getElementById('verdictCard');
  const vText   = document.getElementById('vText');
  const vSub    = document.getElementById('vSub');
  const vCaveat = document.getElementById('vCaveat');

  vText.textContent = verdict;
  vCaveat.style.display = 'none';

  if (verdict.includes('SYNTHETIC')) {
    card.style.borderColor = '#ff3a7a';
    vText.style.color      = '#ff3a7a';
    vText.style.textShadow = '0 0 24px rgba(255,58,122,0.8)';
    vSub.textContent       = '⚠ entirely AI-generated media';
  } else if (verdict.includes('DEEPFAKE')) {
    card.style.borderColor = '#ff3a7a';
    vText.style.color      = '#ff3a7a';
    vText.style.textShadow = '0 0 24px rgba(255,58,122,0.8)';
    vSub.textContent       = verdict.includes('FACESWAP')
      ? '⚠ face-swap likely — audio genuine'
      : '⚠ manipulated media detected';
  } else if (verdict.includes('MANIPULATED')) {
    card.style.borderColor = '#4d9fff';
    vText.style.color      = '#4d9fff';
    vText.style.textShadow = '0 0 24px rgba(77,159,255,0.8)';
    vSub.textContent       = verdict.includes('VOICE CLONE')
      ? '◈ possible voice clone detected'
      : '◈ significant manipulation detected';
  } else if (verdict.includes('SUSPICIOUS')) {
    card.style.borderColor = '#ff1493';
    vText.style.color      = '#ff1493';
    vText.style.textShadow = '0 0 24px rgba(255,20,147,0.8)';
    vSub.textContent       = verdict.includes('FACESWAP')
      ? '◈ possible face-swap detected'
      : verdict.includes('VOICE CLONE')
      ? '◈ possible voice clone detected'
      : '◈ manual review recommended';
  } else {
    card.style.borderColor = '#44ffb0';
    vText.style.color      = '#44ffb0';
    vText.style.textShadow = '0 0 24px rgba(68,255,176,0.8)';
    vSub.textContent       = '✦ video appears authentic';
  }

  document.getElementById('sVideo').textContent  = pct(vScore);
  document.getElementById('sAudio').textContent  = pct(aScore);
  document.getElementById('sFusion').textContent = pct(fScore);

  // Show PDF button with hash stored
  const pdfBtn = document.getElementById('pdfBtn');
  if (data.sha256) {
    pdfBtn.dataset.hash = data.sha256;
    pdfBtn.style.display = 'block';
  }

  setState('result');
}

function pct(v) { return (v * 100).toFixed(1) + '%'; }

function showIdle() {
  document.getElementById('pdfBtn').style.display = 'none';
  document.getElementById('pdfBtn').dataset.hash = '';
  setState('idle');
}

function showLoading() { setState('loading'); }

function showError(msg) {
  document.getElementById('errorText').textContent = msg;
  setState('error');
}

function setState(state) {
  document.getElementById('idleState').style.display    = state === 'idle'    ? 'block' : 'none';
  document.getElementById('loadingState').style.display = state === 'loading' ? 'block' : 'none';
  document.getElementById('resultState').style.display  = state === 'result'  ? 'block' : 'none';
  document.getElementById('errorState').style.display   = state === 'error'   ? 'block' : 'none';
}