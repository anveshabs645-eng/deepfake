from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import hashlib
import os
import subprocess
import logging
import time
import datetime
import io
import torch
import pandas as pd
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

FUSION_THRESHOLD   = 0.21
DEEPFAKE_THRESHOLD = 0.40
VAL_ACCURACY       = None
VIDEOS_TRAINED     = None
fusion_model       = None

if os.path.exists("fusion_mlp.pth"):
    try:
        checkpoint = torch.load("fusion_mlp.pth", weights_only=False, map_location="cpu")
        fusion_model = FusionMLP()
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            fusion_model.load_state_dict(checkpoint["model_state"])
            # FUSION_THRESHOLD = checkpoint.get("threshold", FUSION_THRESHOLD)
            VAL_ACCURACY     = checkpoint.get("val_accuracy", None)
        else:
            fusion_model.load_state_dict(checkpoint)
        fusion_model.eval()
        log.info(f"Fusion MLP loaded — threshold={FUSION_THRESHOLD:.2f}")
    except Exception as e:
        log.warning(f"Could not load fusion MLP: {e} — falling back to weighted average")
        fusion_model = None
else:
    log.warning("fusion_mlp.pth not found — falling back to weighted average fusion")

if os.path.exists("scores.csv"):
    try:
        _df = pd.read_csv("scores.csv")
        VIDEOS_TRAINED = len(_df)
        log.info(f"scores.csv loaded — {VIDEOS_TRAINED} training samples")
    except Exception as e:
        log.warning(f"Could not read scores.csv: {e}")

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
    final_score = result["final_score"]
    v = result.get("V_score", 0.5)
    a = result.get("A_score", 0.5)

    if final_score < FUSION_THRESHOLD:
        if v > 0.45 and a < 0.35:
            verdict = "SUSPICIOUS — POSSIBLE FACESWAP"
        else:
            verdict = "REAL"

    elif final_score < DEEPFAKE_THRESHOLD:
        if v > 0.35 and a < 0.35:
            verdict = "SUSPICIOUS — POSSIBLE FACESWAP"
        elif v < 0.30 and a > 0.55:
            verdict = "SUSPICIOUS — POSSIBLE VOICE CLONE"
        else:
            verdict = "SUSPICIOUS"

    else:
        if v > 0.60 and a > 0.60:
            verdict = "SYNTHETIC — AI GENERATED"
        elif v > 0.40 and a < 0.35:
            verdict = "DEEPFAKE — POSSIBLE FACESWAP"
        elif v < 0.30 and a > 0.65:
            verdict = "MANIPULATED — POSSIBLE VOICE CLONE"
        else:
            verdict = "MANIPULATED"

    result["final_score"] = round(final_score, 4)
    result["verdict"]     = verdict
    result["threshold"]   = round(FUSION_THRESHOLD, 2)
    cache_result(result)
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

@app.route("/health")
def health():
    return jsonify({
        "status":         "running",
        "videos_trained": VIDEOS_TRAINED,
        "val_accuracy":   round(VAL_ACCURACY * 100, 1) if VAL_ACCURACY else None
    })

@app.route('/app')
def serve_app():
    return send_from_directory('/Users/omaryabs/Desktop/Anvesha/deepfake', 'index.html')

@app.route('/script.js')
def serve_script():
    return send_from_directory('/Users/omaryabs/Desktop/Anvesha/deepfake', 'script.js')

@app.route('/style.css')
def serve_style():
    return send_from_directory('/Users/omaryabs/Desktop/Anvesha/deepfake', 'style.css')

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

@app.route("/report", methods=["GET"])
def generate_report():
    file_hash = request.args.get("hash")
    if not file_hash or file_hash not in _report_cache:
        return jsonify({"error": "No result found for this hash. Analyze a video first."}), 404

    data        = _report_cache[file_hash]
    verdict     = data.get("verdict", "UNKNOWN")
    final_score = data.get("final_score", 0)
    v_score     = data.get("V_score", 0)
    a_score     = data.get("A_score", 0)
    filename    = data.get("filename") or data.get("url", "Unknown")
    sha256      = data.get("sha256", "N/A")
    elapsed     = data.get("elapsed_sec", "N/A")
    threshold   = data.get("threshold", 0.21)
    timestamp   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Theme based on verdict ──
    if "SYNTHETIC" in verdict:
        verdict_color  = colors.HexColor("#cc0000")
        theme_header   = colors.HexColor("#4a0000")
        theme_text     = colors.HexColor("#ffaaaa")
        theme_row      = colors.HexColor("#fff0f0")
        theme_grid     = colors.HexColor("#ffcccc")
        theme_section  = colors.HexColor("#8b0000")
        theme_subtitle = colors.HexColor("#cc6666")
        sub_msg        = "Entirely AI-generated media detected — no authentic source"

    elif "DEEPFAKE" in verdict:
        verdict_color  = colors.HexColor("#cc0000")
        theme_header   = colors.HexColor("#4a0000")
        theme_text     = colors.HexColor("#ffaaaa")
        theme_row      = colors.HexColor("#fff0f0")
        theme_grid     = colors.HexColor("#ffcccc")
        theme_section  = colors.HexColor("#8b0000")
        theme_subtitle = colors.HexColor("#cc6666")
        sub_msg        = "Face-swap likely detected — audio appears genuine" if "FACESWAP" in verdict else "High confidence — manipulated media detected"

    elif "MANIPULATED" in verdict:
        verdict_color  = colors.HexColor("#1565c0")
        theme_header   = colors.HexColor("#0d3b6e")
        theme_text     = colors.HexColor("#b3d1ff")
        theme_row      = colors.HexColor("#f0f5ff")
        theme_grid     = colors.HexColor("#99bbff")
        theme_section  = colors.HexColor("#1a4a8a")
        theme_subtitle = colors.HexColor("#4d88cc")
        if "VOICE CLONE" in verdict:
            sub_msg = "Possible voice clone — video appears authentic"
        else:
            sub_msg = "Significant manipulation detected — possible AI generation or deepfake"

    elif "SUSPICIOUS" in verdict:
        verdict_color  = colors.HexColor("#c2185b")
        theme_header   = colors.HexColor("#4a0a2a")
        theme_text     = colors.HexColor("#ffb3d1")
        theme_row      = colors.HexColor("#fff0f5")
        theme_grid     = colors.HexColor("#ffcce0")
        theme_section  = colors.HexColor("#880044")
        theme_subtitle = colors.HexColor("#cc5588")
        if "FACESWAP" in verdict:
            sub_msg = "Possible face-swap — manual review recommended"
        elif "VOICE CLONE" in verdict:
            sub_msg = "Possible voice clone — manual review recommended"
        else:
            sub_msg = "Inconclusive — manual review recommended"

    else:  # REAL
        verdict_color  = colors.HexColor("#006633")
        theme_header   = colors.HexColor("#002a14")
        theme_text     = colors.HexColor("#a8e8c8")
        theme_row      = colors.HexColor("#f0fff5")
        theme_grid     = colors.HexColor("#a0ddb8")
        theme_section  = colors.HexColor("#004d22")
        theme_subtitle = colors.HexColor("#4a9a70")
        sub_msg        = "Low confidence of manipulation — video appears authentic"

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
        fontSize=35, textColor=colors.HexColor("#0a1f44"), spaceAfter=14
    )))
    story.append(Paragraph("Multimodal Deepfake Detection — Forensic Report", ParagraphStyle(
        'S', parent=styles['Normal'],
        fontSize=10, textColor=theme_subtitle, spaceAfter=18,
        alignment=1
    )))
    story.append(Spacer(1, 0.1*inch))

    # ── Verdict ──
    verdict_fontsize = 18 if len(verdict) > 20 else 22 if len(verdict) > 12 else 28
    story.append(Paragraph(verdict, ParagraphStyle(
        'V', parent=styles['Normal'],
        fontSize=verdict_fontsize, textColor=verdict_color,
        spaceAfter=10, fontName='Helvetica-Bold',
        leading=verdict_fontsize + 6
    )))
    story.append(Paragraph(sub_msg, ParagraphStyle(
        'SM', parent=styles['Normal'],
        fontSize=10, textColor=theme_subtitle, spaceAfter=16
    )))
    story.append(Spacer(1, 0.2*inch))

    section_style = ParagraphStyle(
        'SEC', parent=styles['Normal'],
        fontSize=11, textColor=theme_section,
        fontName='Helvetica-Bold', spaceAfter=8
    )

    # ── Score Table ──
    story.append(Paragraph("DETECTION SCORES", section_style))
    score_table = Table([
        ["Metric",             "Score",              "Interpretation"],
        ["Video Score (V)",    f"{v_score:.3f}",     "Spatial artifact analysis via EfficientNet-B4"],
        ["Audio Score (A)",    f"{a_score:.3f}",     "Spectral analysis via Gradient Boosting"],
        ["Final Fusion Score", f"{final_score:.4f}", f"MLP fusion output (threshold: {threshold})"],
    ], colWidths=[1.8*inch, 1.0*inch, 3.8*inch])
    score_table.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0),  theme_header),
        ('TEXTCOLOR',      (0,0), (-1,0),  theme_text),
        ('FONTNAME',       (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,0),  10),
        ('FONTSIZE',       (0,1), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [theme_row, colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.5, theme_grid),
        ('PADDING',        (0,0), (-1,-1), 8),
        ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
        ('WORDWRAP',       (0,0), (-1,-1), True),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 0.2*inch))

    # ── Thresholds ──
    story.append(Paragraph("CLASSIFICATION THRESHOLDS", section_style))
    f_thresh = round(FUSION_THRESHOLD, 2)
    d_thresh = round(DEEPFAKE_THRESHOLD, 2)
    thresh_table = Table([
        ["Verdict",                          "Condition",                          "Meaning"],
        ["REAL",                             f"Score < {f_thresh}",                "Authentic"],
        ["SUSPICIOUS",                       f"{f_thresh} to {d_thresh}",          "Inconclusive"],
        ["SUSPICIOUS — POSSIBLE FACESWAP",   f"Score < {d_thresh}, V>0.35, A<0.35","Visual anomaly, clean audio"],
        ["SUSPICIOUS — POSSIBLE VOICE CLONE",f"Score < {d_thresh}, V<0.30, A>0.55","Clean video, synthetic audio"],
        ["MANIPULATED",                      f"Score > {d_thresh}",                "AI/deepfake likely"],
        ["MANIPULATED — POSSIBLE VOICE CLONE",f"Score > {d_thresh}, V<0.30, A>0.65","Voice clone detected"],
        ["DEEPFAKE — POSSIBLE FACESWAP",     f"Score > {d_thresh}, V>0.40, A<0.35","Face-swap, real audio"],
        ["SYNTHETIC — AI GENERATED",         f"Score > {d_thresh}, V>0.60, A>0.60","Fully AI generated"],
    ], colWidths=[2.6*inch, 2.0*inch, 2.0*inch])
    thresh_table.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0),  theme_header),
        ('TEXTCOLOR',      (0,0), (-1,0),  theme_text),
        ('FONTNAME',       (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,0),  8),
        ('FONTSIZE',       (0,1), (-1,-1), 7),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [theme_row, colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.5, theme_grid),
        ('PADDING',        (0,0), (-1,-1), 5),
        ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
        ('WORDWRAP',       (0,0), (-1,-1), True),
    ]))
    story.append(thresh_table)
    story.append(Spacer(1, 0.2*inch))

    # ── File Metadata ──
    story.append(Paragraph("FILE METADATA", section_style))
    meta_table = Table([
        ["Field",         "Value"],
        ["Filename",      filename],
        ["SHA-256 Hash",  sha256],
        ["Analysis Time", f"{elapsed} seconds"],
        ["Timestamp",     timestamp],
        ["Model",         "EfficientNet-B4 + Gradient Boosting + Fusion MLP"],
    ], colWidths=[1.8*inch, 4.8*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND',     (0,0), (-1,0),  theme_header),
        ('TEXTCOLOR',      (0,0), (-1,0),  theme_text),
        ('FONTNAME',       (0,0), (-1,0),  'Helvetica-Bold'),
        ('FONTSIZE',       (0,0), (-1,0),  10),
        ('FONTSIZE',       (0,1), (-1,-1), 9),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [theme_row, colors.white]),
        ('GRID',           (0,0), (-1,-1), 0.5, theme_grid),
        ('PADDING',        (0,0), (-1,-1), 8),
        ('VALIGN',         (0,0), (-1,-1), 'MIDDLE'),
        ('WORDWRAP',       (0,0), (-1,-1), True),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.3*inch))

    # ── Footer ──
    story.append(Paragraph(
        "Generated by DeepGuard — RNSIT B.Tech IDT Project 2026 | For forensic and educational use only.",
        ParagraphStyle('F', parent=styles['Normal'],
            fontSize=8, textColor=theme_subtitle, alignment=1)
    ))

    doc.build(story)
    buffer.seek(0)

    return send_file(
        buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=f"deepguard_report_{file_hash[:8]}.pdf"
    )

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status":           "ok",
        "audio_model":      os.path.exists("audio_model.pkl"),
        "fusion_model":     os.path.exists("fusion_mlp.pth"),
        "fusion_threshold": FUSION_THRESHOLD
    })

if __name__ == "__main__":
    app.run(debug=True)