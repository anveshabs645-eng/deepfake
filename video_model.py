import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import os

model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(1792, 1)
    # No sigmoid here — applied manually at inference
)

if os.path.exists("video_model.pth"):
    checkpoint = torch.load("video_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    print(f"[video] Loaded video_model.pth (val_acc={checkpoint.get('val_accuracy', '?'):.3f})")
else:
    print("[video] WARNING: video_model.pth not found — scores will be random until trained")

model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_video_score(video_path):
    cap = cv2.VideoCapture(video_path)
    scores = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 5 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = transform(rgb).unsqueeze(0)
            with torch.no_grad():
                logit = model(img).item()
                score = torch.sigmoid(torch.tensor(logit)).item()
                scores.append(score)
        frame_count += 1

    cap.release()
    return float(sum(scores) / len(scores)) if scores else 0.5