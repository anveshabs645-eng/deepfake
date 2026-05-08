import torch
import os
from mlp_model import FusionMLP
from video_model import get_video_score
from audio_features import get_audio_score, load_audio_classifier

model = FusionMLP()
THRESHOLD = 0.55

if os.path.exists('fusion_mlp.pth'):
    checkpoint = torch.load('fusion_mlp.pth', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state' in checkpoint:
        model.load_state_dict(checkpoint['model_state'])
        THRESHOLD = checkpoint.get('threshold', THRESHOLD)
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded fusion_mlp.pth — threshold={THRESHOLD:.2f}")
else:
    print("Warning: fusion_mlp.pth not found. Train the model first.")
model.eval()

if os.path.exists('audio_model.pkl'):
    load_audio_classifier()

def predict(video_path):
    V_score = get_video_score(video_path)
    A_score, _ = get_audio_score(video_path)

    inputs = torch.tensor([[V_score, A_score]], dtype=torch.float32)
    with torch.no_grad():
        logit = model(inputs)
        final_score = torch.sigmoid(logit).item()  # sigmoid applied here

    if final_score < 0.3:
        verdict = "REAL"
    elif final_score < THRESHOLD:
        verdict = "SUSPICIOUS"
    else:
        verdict = "DEEPFAKE"

    return {
        "final_score": round(final_score, 3),
        "verdict": verdict,
        "V_score": round(V_score, 3),
        "A_score": round(A_score, 3)
    }