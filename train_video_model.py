import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
import os
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATASET_PATH   = "dataset"
MODEL_OUT      = "video_model.pth"
FRAMES_PER_VID = 8       # slightly more frames per video
BATCH_SIZE     = 4
EPOCHS         = 30
LR_BACKBONE    = 1e-5     # lower LR for pretrained backbone blocks
LR_HEAD        = 3e-5     # higher LR for classifier head
DEVICE         = torch.device("mps" if torch.backends.mps.is_available()
                               else "cuda" if torch.cuda.is_available()
                               else "cpu")
print(f"Using device: {DEVICE}")

# ─────────────────────────────────────────────
# TRANSFORMS  (separate train / val)
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((380, 380)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
def sample_frames(video_path, n=FRAMES_PER_VID):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, max(total - 1, 0), n, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

class DeepfakeFrameDataset(Dataset):
    def __init__(self, video_list, transform, frames_per_vid=8):
        self.samples = []
        self.transform = transform
        for path, label in video_list:
            for _ in range(frames_per_vid):
                self.samples.append((path, label))
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = np.random.randint(0, max(total, 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            frame = np.zeros((380, 380, 3), dtype=np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.transform(frame), torch.tensor(label, dtype=torch.float32)

def build_video_list():
    """Returns list of (video_path, label_int) — one entry per VIDEO."""
    videos = []
    for label_name, label_val in [("real", 0), ("fake", 1)]:
        folder = os.path.join(DATASET_PATH, label_name)
        if not os.path.exists(folder):
            print(f"Warning: {folder} not found, skipping.")
            continue
        vids = [v for v in os.listdir(folder)
                if v.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        print(f"[{label_name}] {len(vids)} videos")
        for v in vids:
            videos.append((os.path.join(folder, v), label_val))
    return videos

def videos_to_frame_samples(video_list):
    """Expand a list of (path, label) videos into (frame_rgb, label) samples."""
    samples = []
    for i, (path, label) in enumerate(video_list, 1):
        frames = sample_frames(path)
        if not frames:
            print(f"  [SKIP] no frames: {path}")
            continue
        for f in frames:
            samples.append((f, label))
        print(f"  [{i}/{len(video_list)}] {os.path.basename(path)} — {len(frames)} frames")
    return samples

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def build_model():
    m = models.efficientnet_b4(weights=models.EfficientNet_B4_Weights.DEFAULT)
    # Freeze everything first
    for param in m.parameters():
        param.requires_grad = False
    # Unfreeze last 2 backbone blocks
    for param in m.features[-2:].parameters():
        param.requires_grad = True
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(1792, 1)
        # No sigmoid — BCEWithLogitsLoss handles it
    )
    return m.to(DEVICE)

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train():
    print("\n=== Building video list ===")
    all_videos = build_video_list()
    video_labels = [v[1] for v in all_videos]

    # ── FIX: Split at VIDEO level, not frame level ──
    train_vids, val_vids = train_test_split(
        all_videos, test_size=0.2,
        stratify=video_labels, random_state=42
    )
    print(f"\nVideos — Train: {len(train_vids)} | Val: {len(val_vids)}")

    print(f"\nVideos — Train: {len(train_vids)} | Val: {len(val_vids)}")

    train_loader = DataLoader(
        DeepfakeFrameDataset(train_vids, train_transform, frames_per_vid=8),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        DeepfakeFrameDataset(val_vids, val_transform, frames_per_vid=8),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = build_model()

    # ── FIX: Optimizer covers BOTH backbone blocks AND classifier head ──
    optimizer = torch.optim.Adam([
        {"params": model.features[-2:].parameters(), "lr": LR_BACKBONE},
        {"params": model.classifier.parameters(),    "lr": LR_HEAD}
    ])

    # Class counts: 147 real, 102 fake — penalise missed fakes proportionally
    pos_weight = torch.tensor([463 / 419]).to(DEVICE)   # ≈ 1.38
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── FIX: CosineAnnealingLR instead of aggressive StepLR ──
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss, correct, total = 0, 0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds   = (torch.sigmoid(logits) >= 0.5).float()
            correct += (preds == lbls).sum().item()
            total   += lbls.size(0)

        train_acc = correct / total

        # ── Val ──
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = model(imgs).squeeze(1)
                val_loss    += criterion(logits, lbls).item()
                preds        = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == lbls).sum().item()
                val_total   += lbls.size(0)

        val_acc = val_correct / val_total
        scheduler.step()

        print(f"Epoch {epoch}/{EPOCHS} | "
              f"Train Loss: {train_loss/len(train_loader):.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f} Acc: {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state":  model.state_dict(),
                "val_accuracy": val_acc,
                "epochs":       epoch
            }, MODEL_OUT)
            print(f"  ✓ Saved best model (val_acc={val_acc:.3f})")

    print(f"\nDone. Best val accuracy: {best_val_acc:.3f} → {MODEL_OUT}")

if __name__ == "__main__":
    train()