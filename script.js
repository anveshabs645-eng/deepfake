// ── Network background ──
const canvas = document.getElementById('bgCanvas');
const ctx    = canvas.getContext('2d');
let nodes    = [];

function resize(){
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight;
}

function initNodes(){
  nodes = [];
  for(let i = 0; i < 65; i++){
    nodes.push({
      x:  Math.random() * canvas.width,
      y:  Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.40,
      vy: (Math.random() - 0.5) * 0.40,
      r:  Math.random() * 1.8 + 0.8
    });
  }
}

function draw(){
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  nodes.forEach(n => {
    n.x += n.vx; n.y += n.vy;
    if(n.x < 0 || n.x > canvas.width)  n.vx *= -1;
    if(n.y < 0 || n.y > canvas.height) n.vy *= -1;
    ctx.beginPath();
    ctx.arc(n.x, n.y, n.r, 0, Math.PI * 2);
    ctx.fillStyle = '#e8609a';
    ctx.fill();
  });
  for(let i = 0; i < nodes.length; i++){
    for(let j = i + 1; j < nodes.length; j++){
      const dx   = nodes[i].x - nodes[j].x;
      const dy   = nodes[i].y - nodes[j].y;
      const dist = Math.sqrt(dx*dx + dy*dy);
      if(dist < 130){
        ctx.beginPath();
        ctx.moveTo(nodes[i].x, nodes[i].y);
        ctx.lineTo(nodes[j].x, nodes[j].y);
        ctx.strokeStyle = `rgba(232,96,154,${0.11*(1-dist/130)})`;
        ctx.lineWidth   = 0.7;
        ctx.stroke();
      }
    }
  }
  requestAnimationFrame(draw);
}

resize(); initNodes(); draw();
window.addEventListener('resize', ()=>{ resize(); initNodes(); });

// ── Tab switching ──
function switchTab(tab, btn){
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
  document.getElementById(tab + 'Tab').classList.add('active');
  btn.classList.add('active');
}

// ── File drop ──
const videoInput   = document.getElementById('videoInput');
const dropZone     = document.getElementById('dropZone');
const fileInfo     = document.getElementById('fileInfo');
const videoPreview = document.getElementById('videoPreview');
const videoPlayer  = document.getElementById('videoPlayer');

videoInput.addEventListener('change', () => handleFile(videoInput.files[0]));

dropZone.addEventListener('dragover',  e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop',      e  => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  handleFile(e.dataTransfer.files[0]);
});

function handleFile(file){
  if(!file) return;
  fileInfo.textContent       = `${file.name}  ·  ${(file.size/1024/1024).toFixed(2)} MB`;
  videoPlayer.src            = URL.createObjectURL(file);
  videoPreview.style.display = 'block';
}

// ── Loading steps ──
const STEPS = [
  'extracting video frames...',
  'running efficientnet-b4...',
  'extracting audio features...',
  'running gradient boosting classifier...',
  'fusion mlp scoring...',
  'computing sha-256 hash...',
  'generating verdict...'
];

function startLoading(){
  const bar  = document.getElementById('progressBar');
  const txt  = document.getElementById('loadingText');
  const step = document.getElementById('loadingStep');
  bar.style.width = '0%';
  txt.textContent = 'initializing pipeline...';
  let i = 0;
  const id = setInterval(()=>{
    if(i < STEPS.length){
      step.textContent = STEPS[i];
      bar.style.width  = ((i+1)/STEPS.length * 88) + '%';
      i++;
    }
  }, 850);
  return id;
}

// ── Show results ──
function showResults(data){
  document.getElementById('loading').style.display = 'none';
  document.getElementById('results').style.display = 'block';

  const verdict = data.verdict;
  const vScore  = parseFloat(data.V_score);
  const aScore  = parseFloat(data.A_score);
  const fScore  = parseFloat(data.final_score);

  const card  = document.getElementById('verdictCard');
  const vText = document.getElementById('verdictText');
  const vSub  = document.getElementById('verdictSub');
  const glow  = card.querySelector('.v-glow');

  vText.textContent = verdict;

  if(verdict === 'DEEPFAKE'){
    card.style.borderColor = '#ff3a7a';
    vText.style.color      = '#ff3a7a';
    vText.style.textShadow = '0 0 40px rgba(255,58,122,0.8)';
    glow.style.background  = 'radial-gradient(circle,rgba(255,58,122,0.16),transparent 70%)';
    vSub.textContent       = '⚠ high confidence — manipulated media detected';
  } else if(verdict === 'SUSPICIOUS'){
    card.style.borderColor = '#ffaa44';
    vText.style.color      = '#ffaa44';
    vText.style.textShadow = '0 0 40px rgba(255,170,68,0.8)';
    glow.style.background  = 'radial-gradient(circle,rgba(255,170,68,0.14),transparent 70%)';
    vSub.textContent       = '◈ inconclusive — manual review recommended';
  } else {
    card.style.borderColor = '#44ffb0';
    vText.style.color      = '#44ffb0';
    vText.style.textShadow = '0 0 40px rgba(68,255,176,0.8)';
    glow.style.background  = 'radial-gradient(circle,rgba(68,255,176,0.14),transparent 70%)';
    vSub.textContent       = '✦ low risk — video appears authentic';
  }

  setTimeout(()=>{
    setGauge('vGauge', vScore);
    setGauge('aGauge', aScore);
    setGauge('fGauge', fScore);
    document.getElementById('vScoreVal').textContent = pct(vScore);
    document.getElementById('aScoreVal').textContent = pct(aScore);
    document.getElementById('fScoreVal').textContent = pct(fScore);
    setBar('vBar','vBarNum', vScore);
    setBar('aBar','aBarNum', aScore);
    setBar('fBar','fBarNum', fScore);
  }, 200);

  document.getElementById('resFilename').textContent = data.filename || data.url || '—';
  document.getElementById('resSha256').textContent   = data.sha256 || '—';
  document.getElementById('resTime').textContent     = data.elapsed_sec ? data.elapsed_sec + ' seconds' : '—';

  document.getElementById('pdfBtn').onclick = ()=>{
    window.open(`http://127.0.0.1:5000/report?hash=${data.sha256}`, '_blank');
  };
}

function pct(v){ return (v*100).toFixed(1) + '%'; }

function setGauge(id, score){
  document.getElementById(id).style.strokeDashoffset = 141.37 - score * 141.37;
}

function setBar(barId, numId, score){
  document.getElementById(barId).style.width  = (score*100) + '%';
  document.getElementById(numId).textContent  = pct(score);
}

// ── Reset ──
function resetAll(){
  document.getElementById('results').style.display      = 'none';
  document.getElementById('videoPreview').style.display = 'none';
  document.getElementById('fileInfo').textContent       = 'no file selected';
  document.getElementById('progressBar').style.width    = '0%';
  document.getElementById('loadingStep').textContent    = '';
  videoInput.value = '';
  document.getElementById('urlInput').value = '';
}

// ── Analyze file ──
document.getElementById('analyzeBtn').addEventListener('click', async ()=>{
  const file = videoInput.files[0];
  if(!file){ alert('please select a video file first'); return; }

  document.getElementById('loading').style.display = 'block';
  document.getElementById('results').style.display = 'none';

  const interval = startLoading();
  const form     = new FormData();
  form.append('video', file);

  try{
    const res  = await fetch('http://127.0.0.1:5000/analyze', { method:'POST', body:form });
    const data = await res.json();
    clearInterval(interval);
    document.getElementById('progressBar').style.width = '100%';
    setTimeout(()=> showResults(data), 450);
  } catch(err){
    clearInterval(interval);
    document.getElementById('loading').style.display = 'none';
    alert('could not connect to backend — make sure flask is running on port 5000');
    console.error(err);
  }
});

// ── Analyze URL ──
document.getElementById('analyzeUrlBtn').addEventListener('click', async ()=>{
  const url = document.getElementById('urlInput').value.trim();
  if(!url){ alert('please enter a video url'); return; }

  document.getElementById('loading').style.display = 'block';
  document.getElementById('results').style.display = 'none';

  const interval = startLoading();

  try{
    const res  = await fetch('http://127.0.0.1:5000/analyze_url', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ url })
    });
    const data = await res.json();
    clearInterval(interval);
    document.getElementById('progressBar').style.width = '100%';
    setTimeout(()=> showResults(data), 450);
  } catch(err){
    clearInterval(interval);
    document.getElementById('loading').style.display = 'none';
    alert('could not connect to backend');
    console.error(err);
  }
});