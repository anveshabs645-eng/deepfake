import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2

# Load EfficientNet-B4 with pretrained ImageNet weights
model = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)

# Replace classifier for binary output (real/fake)
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(1792, 1),
    nn.Sigmoid()
)
model.eval()

# B4 requires 380x380 input
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
            img = transform(rgb)
            img = img.unsqueeze(0)

            with torch.no_grad():
                score = model(img).item()
                scores.append(score)

        frame_count += 1

    cap.release()

    if len(scores) == 0:
        return 0.5

    return float(sum(scores) / len(scores))