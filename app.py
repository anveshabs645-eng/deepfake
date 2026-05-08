from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import hashlib
import os
import subprocess
import logging
import time
import datetime
import io
import torch
from predict import predict
from audio_features import load_audio_classifier
from mlp_model import FusionMLP

# reportlab
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.units import inch

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# LOAD MODELS AT STARTUP
# ─────────────────────────────────────────────
try:
    load_audio_classifier("audio_model.pkl")
    log.info("Audio model loaded")
except FileNotFoundError:
    log.warning("audio_model.pkl not found — audio scores will return 0.5 until trained")

FUSION_THRESHOLD = 0.55
DEEPFAKE_THRESHOLD = 0.65
fusion_model = None

if os.path.exists("fusion_mlp.pth"):
    try:
        checkpoint = torch.load("fusion_mlp.pth", weights_only=False, map_location="cpu")
        fusion_model = FusionMLP()
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            fusion_model.load_state_dict(checkpoint["model_state"])
            FUSION_THRESHOLD = checkpoint.get("threshold", FUSION_THRESHOLD)
        else:
            fusion_model.load_state_dict(checkpoint)
        fusion_model.eval()
        log.info(f"Fusion MLP loaded — threshold={FUSION_THRESHOLD:.2f}")
    except Exception as e:
        log.warning(f"Could not load fusion MLP: {e} — falling back to weighted average")
        fusion_model = None
else:
    log.warning("fusion_mlp.pth not found — falling back to weighted average fusion")

# ─────────────────────────────────────────────
# REPORT CACHE
# ─────────────────────────────────────────────
_report_cache = {}

def cache_result(result):
    if "sha256" in result:
        _report_cache[result["sha256"]] = result

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_SIZE_MB   = 500

def allowed_file(filename):
    return os.path.splitext(filename.lower())[1] in ALLOWED_EXTENSIONS

def compute_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def run_predict(temp_path):
    result = predict(temp_path)
    v = result.get("V_score", 0.5)
    a = result.get("A_score", 0.5)

    if fusion_model is not None:
        with torch.no_grad():
            x = torch.tensor([[v, a]], dtype=torch.float32)
            logit = fusion_model(x)
            final_score = torch.sigmoid(logit).item()
    else:
        final_score = 0.7 * v + 0.3 * a

    if final_score < FUSION_THRESHOLD:
        verdict = "REAL"
    elif final_score < DEEPFAKE_THRESHOLD:
        verdict = "SUSPICIOUS"
    else:
        verdict = "DEEPFAKE"

    result["final_score"] = round(final_score, 4)
    result["verdict"]     = verdict
    result["threshold"]   = round(FUSION_THRESHOLD, 2)
    return result

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "models": {
            "audio":  os.path.exists("audio_model.pkl"),
            "fusion": os.path.exists("fusion_mlp.pth"),
        },
        "threshold": FUSION_THRESHOLD
    })

@app.route("/analyze", methods=["POST"])
def analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(video.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    temp_path = f"temp_upload_{int(time.time())}.mp4"
    video.save(temp_path)

    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        os.remove(temp_path)
        return jsonify({"error": f"File too large ({size_mb:.1f}MB). Max: {MAX_FILE_SIZE_MB}MB"}), 400

    log.info(f"Analyzing upload: {video.filename} ({size_mb:.1f}MB)")
    file_hash = compute_sha256(temp_path)
    t0 = time.time()

    try:
        result = run_predict(temp_path)
        result["filename"]    = video.filename
        result["sha256"]      = file_hash
        result["elapsed_sec"] = round(time.time() - t0, 2)
        cache_result(result)
        log.info(f"Result: {result['verdict']} (final={result['final_score']:.3f}) in {result['elapsed_sec']}s")
    except Exception as e:
        log.error(f"Prediction error: {e}")
        result = {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(result)

@app.route("/analyze_url", methods=["POST"])
def analyze_url():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "No URL provided"}), 400

    url = data["url"]
    temp_path = f"temp_download_{int(time.time())}.mp4"
    log.info(f"Downloading: {url}")

    download = subprocess.run(
        ["yt-dlp", "-o", temp_path, "--merge-output-format", "mp4",
         "--max-filesize", f"{MAX_FILE_SIZE_MB}m", url],
        capture_output=True, text=True
    )

    if not os.path.exists(temp_path):
        log.error(f"yt-dlp failed: {download.stderr}")
        return jsonify({
            "error":   "Failed to download video. URL may not be supported.",
            "details": download.stderr[-300:] if download.stderr else None
        }), 400

    size_mb   = os.path.getsize(temp_path) / (1024 * 1024)
    file_hash = compute_sha256(temp_path)
    t0        = time.time()

    log.info(f"Downloaded: {size_mb:.1f}MB — analyzing...")

    try:
        result = run_predict(temp_path)
        result["url"]         = url
        result["sha256"]      = file_hash
        result["elapsed_sec"] = round(time.time() - t0, 2)
        cache_result(result)
        log.info(f"Result: {result['verdict']} (final={result['final_score']:.3f}) in {result['elapsed_sec']}s")
    except Exception as e:
        log.error(f"Prediction error: {e}")
        result = {"error": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":           "ok",
        "audio_model":      os.path.exists("audio_model.pkl"),
        "fusion_model":     os.path.exists("fusion_mlp.pth"),
        "fusion_threshold": FUSION_THRESHOLD
    })

@app.route("/report", methods=["GET"])
def generate_report():
    file_hash = request.args.get("hash")
    if not file_hash or file_hash not in _report_cache:
        return jsonify({"error": "No result found for this hash. Analyze a video first."}), 404

    data      = _report_cache[file_hash]
    verdict     = data.get("verdict", "UNKNOWN")
    final_score = data.get("final_score", 0)
    v_score     = data.get("V_score", 0)
    a_score     = data.get("A_score", 0)
    filename    = data.get("filename") or data.get("url", "Unknown")
    sha256      = data.get("sha256", "N/A")
    elapsed     = data.get("elapsed_sec", "N/A")
    threshold   = data.get("threshold", 0.55)
    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if verdict == "DEEPFAKE":
        verdict_color = colors.HexColor("#b90303")
    elif verdict == "SUSPICIOUS":
        verdict_color = colors.HexColor("#9B005D")
    else:
        verdict_color = colors.HexColor("#017C3F")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75*inch,
        leftMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )

    styles = getSampleStyleSheet()
    story  = []

    # ── Title ──
    story.append(Paragraph("DEEPGUARD", ParagraphStyle(
        'T', parent=styles['Title'],
        fontSize=22, textColor=colors.HexColor("#053891"), spaceAfter=4
    )))
    story.append(Paragraph("Multimodal Deepfake Detection — Forensic Report", ParagraphStyle(
        'S', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor("#7a9abb"), spaceAfter=16,
        alignment=1
    )))
    story.append(Spacer(1, 0.1*inch))

    # ── Verdict ──
    story.append(Paragraph(verdict, ParagraphStyle(
        'V', parent=styles['Normal'],
        fontSize=28, textColor=verdict_color,
        spaceAfter=18, fontName='Helvetica-Bold'
    )))

    if verdict == "DEEPFAKE":
        sub_msg = "High confidence — manipulated media detected"
    elif verdict == "SUSPICIOUS":
        sub_msg = "Inconclusive — manual review recommended"
    else:
        sub_msg = "Low confidence of manipulation — video appears authentic"

    story.append(Paragraph(sub_msg, ParagraphStyle(
        'SM', parent=styles['Normal'],
        fontSize=10, textColor=colors.HexColor("#7a9abb"), spaceAfter=16
    )))
    story.append(Spacer(1, 0.2*inch))

    section_style = ParagraphStyle(
        'SEC', parent=styles['Normal'],
        fontSize=11, textColor=colors.HexColor("#003ea8"),
        fontName='Helvetica-Bold', spaceAfter=8
    )

    # ── Score Table ──
    story.append(Paragraph("DETECTION SCORES", section_style))
    score_table = Table([
        ["Metric",              "Score",              "Interpretation"],
        ["Video Score (V)",     f"{v_score:.3f}",     "Spatial artifact analysis via EfficientNet-B4"],
        ["Audio Score (A)",     f"{a_score:.3f}",     "Spectral analysis via Gradient Boosting"],
        ["Final Fusion Score",  f"{final_score:.4f}", f"MLP fusion output (threshold: {threshold})"],
    ], colWidths=[1.8*inch, 1.0*inch, 3.8*inch])
    score_table.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0),  colors.HexColor("#053891")),
        ('TEXTCOLOR',      (0,0), (-1,0),  colors.HexColor("#C4C9D1")),
        ('FONTNAME',       (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,0),  10),
        ('FONTSIZE',       (0,1), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f0f4ff"), colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.5, colors.HexColor("#ccddff")),
        ('PADDING',        (0,0), (-1,-1), 8),
        ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.2*inch))

    # ── Thresholds ──
    story.append(Paragraph("CLASSIFICATION THRESHOLDS", section_style))
    thresh_table = Table([
        ["Category",    "Score Range",                                      "Status"],
        ["REAL",        f"< {FUSION_THRESHOLD}",                            "Authentic"],
        ["SUSPICIOUS",  f"{FUSION_THRESHOLD} - {DEEPFAKE_THRESHOLD}",       "Inconclusive"],
        ["DEEPFAKE",    f"> {DEEPFAKE_THRESHOLD}",                          "Manipulated"],
    ], colWidths=[1.8*inch, 1.8*inch, 3.0*inch])
    thresh_table.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0),  colors.HexColor("#053891")),
        ('TEXTCOLOR',      (0,0), (-1,0),  colors.HexColor("#C4C9D1")),
        ('FONTNAME',       (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,0),  10),
        ('FONTSIZE',       (0,1), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f0f4ff"), colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.5, colors.HexColor("#ccddff")),
        ('PADDING',        (0,0), (-1,-1), 8),
    ]))
    story.append(thresh_table)
    story.append(Spacer(1, 0.2*inch))

    # ── File Metadata ──
    story.append(Paragraph("FILE METADATA", section_style))
    meta_table = Table([
        ["Field",          "Value"],
        ["Filename",       filename],
        ["SHA-256 Hash",   sha256],
        ["Analysis Time",  f"{elapsed} seconds"],
        ["Timestamp",      timestamp],
        ["Model",          "EfficientNet-B4 + Gradient Boosting + Fusion MLP"],
    ], colWidths=[1.8*inch, 4.8*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0),  colors.HexColor("#053891")),
        ('TEXTCOLOR',      (0,0), (-1,0),  colors.HexColor("#C4C9D1")),
        ('FONTNAME',       (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,0),  10),
        ('FONTSIZE',       (0,1), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.HexColor("#f0f4ff"), colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.5, colors.HexColor("#ccddff")),
        ('PADDING',        (0,0), (-1,-1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.3*inch))

    # ── Footer ──
    story.append(Paragraph(
        "Generated by DeepGuard — RNSIT B.Tech IDT Project 2026 | For forensic and educational use only.",
        ParagraphStyle('F', parent=styles['Normal'],
            fontSize=8, textColor=colors.HexColor("#7a9abb"), alignment=1)
    ))

    doc.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"deepguard_report_{file_hash[:8]}.pdf"
    )

if __name__ == "__main__":
    app.run(debug=True)