import librosa
import numpy as np
import subprocess
import os
import pickle

# ─────────────────────────────────────────────
# 1. AUDIO EXTRACTION
# ─────────────────────────────────────────────
def extract_audio(video_path, output_wav="temp_audio.wav"):
    cmd = f'ffmpeg -i "{video_path}" -q:a 0 -map a "{output_wav}" -y'
    result = subprocess.run(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0 or not os.path.exists(output_wav):
        return None
    return output_wav

# ─────────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ─────────────────────────────────────────────
def extract_features(y, sr):
    features = []
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features += [np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]
    delta_mfcc = librosa.feature.delta(mfcc)
    features += [np.mean(delta_mfcc, axis=1), np.std(delta_mfcc, axis=1)]
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features += [np.array([np.mean(cent), np.std(cent)])]
    bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features += [np.array([np.mean(bw), np.std(bw)])]
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features += [np.mean(contrast, axis=1)]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features += [np.array([np.mean(rolloff), np.std(rolloff)])]
    zcr = librosa.feature.zero_crossing_rate(y)
    features += [np.array([np.mean(zcr), np.std(zcr)])]
    rms = librosa.feature.rms(y=y)
    features += [np.array([np.mean(rms), np.std(rms)])]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features += [np.mean(chroma, axis=1)]
    try:
        harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
        features += [np.mean(tonnetz, axis=1)]
    except Exception:
        features += [np.zeros(6)]
    try:
        f0, _, _ = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
        f0_clean = f0[~np.isnan(f0)]
        if len(f0_clean) > 0:
            features += [np.array([np.mean(f0_clean), np.std(f0_clean),
                                   np.min(f0_clean), np.max(f0_clean), np.median(f0_clean)])]
        else:
            features += [np.zeros(5)]
    except Exception:
        features += [np.zeros(5)]
    return np.hstack(features).astype(np.float32)

# ─────────────────────────────────────────────
# 3. MODEL
# ─────────────────────────────────────────────
_scorer = None
_threshold = 0.5          # ← Fix 3: replaced at training time with optimal cutoff
_model_path_used = None

def load_audio_classifier(model_path="audio_model.pkl"):
    global _scorer, _threshold, _model_path_used
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Audio model not found: {model_path}")
    with open(model_path, 'rb') as f:
        bundle = pickle.load(f)
    # Support both old (bare pipeline) and new (dict with threshold) formats
    if isinstance(bundle, dict):
        _scorer    = bundle["model"]
        _threshold = bundle.get("threshold", 0.5)
    else:
        _scorer    = bundle          # backward-compat with old pkl
        _threshold = 0.5
    _model_path_used = model_path
    print(f"[audio] Loaded model from {model_path} | threshold={_threshold:.3f}")

def train_audio_classifier(features_list, labels_list, model_path="audio_model.pkl"):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.calibration import CalibratedClassifierCV   # Fix 2
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_curve
    from imblearn.over_sampling import SMOTE                 # Fix 1
    from imblearn.pipeline import Pipeline as ImbPipeline

    X = np.array(features_list)
    y = np.array(labels_list)

    # ── Fix 1: SMOTE resampling inside a pipeline so it only runs on train folds ──
    # ── Fix 2: Platt scaling via CalibratedClassifierCV (isotonic also works)    ──
    base_clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=42
        # NOTE: GBC doesn't accept class_weight directly; SMOTE + calibration
        #       handle the imbalance instead. If you switch to a HistGBC,
        #       add class_weight='balanced' there.
    )

    pipe = ImbPipeline([
        ('smote',   SMOTE(random_state=42, k_neighbors=5)),   # Fix 1
        ('scaler',  StandardScaler()),
        ('clf',     CalibratedClassifierCV(base_clf, cv=3, method='sigmoid')),  # Fix 2
    ])

    pipe.fit(X, y)

    # ── Fix 3: find optimal decision threshold via Youden's J on training data ──
    # Use cross-validated probabilities to avoid over-optimism.
    from sklearn.model_selection import cross_val_predict
    # Re-run a quick CV just for threshold tuning (fast — no SMOTE needed here
    # since we already balanced; use raw X/y so counts are real).
    inner_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf',    GradientBoostingClassifier(
                       n_estimators=100, max_depth=4,
                       learning_rate=0.05, random_state=42)),
    ])
    cv_probs = cross_val_predict(
        inner_pipe, X, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        method='predict_proba'
    )[:, 1]

    fpr, tpr, thresholds = roc_curve(y, cv_probs)
    j_scores = tpr - fpr                                    # Youden's J statistic
    optimal_threshold = float(thresholds[np.argmax(j_scores)])
    print(f"[audio] Optimal threshold (Youden J): {optimal_threshold:.3f}")

    bundle = {"model": pipe, "threshold": optimal_threshold}
    with open(model_path, 'wb') as f:
        pickle.dump(bundle, f)

    global _scorer, _threshold
    _scorer    = pipe
    _threshold = optimal_threshold
    print(f"[audio] Model trained and saved to {model_path}")
    return pipe

# ─────────────────────────────────────────────
# 4. SCORING
# ─────────────────────────────────────────────
def get_audio_score(video_path):
    """
    Returns (raw_probability, binary_verdict).
    The raw probability is Platt-calibrated (Fix 2).
    The binary verdict uses the data-driven threshold (Fix 3/4)
    rather than a blind 0.5 cutoff.
    """
    if _scorer is None:
        print("[audio] WARNING: No model loaded — returning neutral 0.5")
        return 0.5, False

    wav_file = extract_audio(video_path)
    if wav_file is None:
        print("[audio] No audio track found — returning neutral 0.5")
        return 0.5, False

    try:
        y, sr = librosa.load(wav_file, duration=15, sr=22050)
        if len(y) < sr * 0.5:
            return 0.5, False
        features  = extract_features(y, sr).reshape(1, -1)
        proba     = _scorer.predict_proba(features)[0]
        score     = float(proba[1])
        # Fix 4: apply the stored threshold instead of hardcoded 0.5
        is_fake   = score >= _threshold
    except Exception as e:
        print(f"[audio] Processing error: {e}")
        score, is_fake = 0.5, False
    finally:
        if wav_file and os.path.exists(wav_file):
            os.remove(wav_file)

    return score, is_fake

# ─────────────────────────────────────────────
# 5. DATASET TRAINING HELPER
# ─────────────────────────────────────────────
def build_dataset_and_train(real_dir, fake_dir, model_path="audio_model.pkl"):
    X, y = [], []
    for label, folder in [(0, real_dir), (1, fake_dir)]:
        for fname in os.listdir(folder):
            if not fname.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                continue
            vpath = os.path.join(folder, fname)
            wav = extract_audio(vpath, output_wav=f"_tmp_{fname}.wav")
            if wav is None:
                continue
            try:
                audio, sr = librosa.load(wav, duration=15, sr=22050)
                feats = extract_features(audio, sr)
                X.append(feats)
                y.append(label)
                print(f"  [{'REAL' if label==0 else 'FAKE'}] {fname} — {len(feats)} features")
            except Exception as e:
                print(f"  [SKIP] {fname}: {e}")
            finally:
                if os.path.exists(wav):
                    os.remove(wav)

    print(f"\nDataset: {y.count(0)} real, {y.count(1)} fake")
    print(f"Imbalance ratio: 1:{y.count(1)/max(y.count(0),1):.2f}")
    return train_audio_classifier(X, y, model_path)

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        print("Training mode...")
        build_dataset_and_train(sys.argv[1], sys.argv[2])
    else:
        load_audio_classifier()
        path = input("Enter video path: ")
        score, verdict = get_audio_score(path)
        print(f"A_score = {score:.3f} | FAKE={verdict} (threshold={_threshold:.3f})") 