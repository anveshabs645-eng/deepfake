import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from mlp_model import FusionMLP

# ─────────────────────────────────────────────
# 1. LOAD & VALIDATE DATA
# ─────────────────────────────────────────────
df = pd.read_csv('scores.csv')
print(f"Dataset: {len(df)} samples — {(df['label']==0).sum()} real, {(df['label']==1).sum()} fake")

assert 'V_score' in df.columns and 'A_score' in df.columns and 'label' in df.columns
if df.isnull().any().any():
    print("WARNING: NaNs found — filling with 0.5")
    df.fillna(0.5, inplace=True)

X = torch.tensor(df[['V_score', 'A_score']].values, dtype=torch.float32)
y = torch.tensor(df['label'].values, dtype=torch.float32).unsqueeze(1)

# ─────────────────────────────────────────────
# 2. CLASS IMBALANCE HANDLING
# ─────────────────────────────────────────────
n_real = (df['label'] == 0).sum()
n_fake = (df['label'] == 1).sum()
pos_weight = torch.tensor([n_real / n_fake], dtype=torch.float32)
print(f"Class weight (pos_weight): {pos_weight.item():.3f}")

# ─────────────────────────────────────────────
# 3. STRATIFIED SPLIT
# ─────────────────────────────────────────────
X_np = X.numpy()
y_np = y.squeeze().numpy()

X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X_np, y_np, test_size=0.2, stratify=y_np, random_state=42
)

X_train = torch.tensor(X_train_np, dtype=torch.float32)
X_val   = torch.tensor(X_val_np,   dtype=torch.float32)
y_train = torch.tensor(y_train_np, dtype=torch.float32).unsqueeze(1)
y_val   = torch.tensor(y_val_np,   dtype=torch.float32).unsqueeze(1)

print(f"Train: {len(X_train)} samples | Val: {len(X_val)} samples")

# ─────────────────────────────────────────────
# 4. MODEL + TRAINING
# ─────────────────────────────────────────────
model     = FusionMLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)
loss_fn   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # applies sigmoid internally

best_val_loss    = float('inf')
best_state       = None
patience_counter = 0
EARLY_STOP_PATIENCE = 80

print("\nTraining...")
for epoch in range(10000):
    model.train()
    pred = model(X_train)        # raw logits
    loss = loss_fn(pred, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)  # raw logits
        val_loss = loss_fn(val_pred, y_val).item()
        scheduler.step(val_loss)

    if val_loss < best_val_loss:
        best_val_loss    = val_loss
        best_state       = {k: v.clone() for k, v in model.state_dict().items()}
        patience_counter = 0
    else:
        patience_counter += 1

    if patience_counter >= EARLY_STOP_PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

    if epoch % 50 == 0:
        print(f"Epoch {epoch:4d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")

model.load_state_dict(best_state)

# ─────────────────────────────────────────────
# 5. EVALUATION + THRESHOLD SEARCH
# ─────────────────────────────────────────────
model.eval()
with torch.no_grad():
    val_logits = model(X_val)
    val_probs  = torch.sigmoid(val_logits).numpy().squeeze()  # apply sigmoid here
    y_true     = y_val.numpy().squeeze()

print("\nThreshold search:")
best_thresh, best_acc = 0.5, 0.0
for t in np.arange(0.3, 0.75, 0.05):
    preds = (val_probs > t).astype(float)
    acc   = (preds == y_true).mean()
    print(f"  t={t:.2f} → acc={acc*100:.1f}%")
    if acc > best_acc:
        best_acc, best_thresh = acc, t

print(f"\nBest threshold: {best_thresh:.2f} @ {best_acc*100:.1f}% val accuracy")

final_preds = (val_probs > best_thresh).astype(int)
print("\nClassification Report:")
print(classification_report(y_true, final_preds, target_names=["REAL", "FAKE"]))

try:
    auc = roc_auc_score(y_true, val_probs)
    print(f"ROC-AUC: {auc:.4f}")
except Exception:
    auc = None

# ─────────────────────────────────────────────
# 6. SAVE
# ─────────────────────────────────────────────
torch.save({
    'model_state':  model.state_dict(),
    'threshold':    best_thresh,
    'val_accuracy': best_acc,
    'val_auc':      auc
}, 'fusion_mlp.pth')

print("\nSaved fusion_mlp.pth")
print("Done — restart app.py to use the new model.")