import os
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# =========================
# Config
# =========================
DATA_DIR = "features_v1"
BATCH_SIZE = 32

EPOCHS = 200
LR = 5e-4
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 512
DROPOUT = 0.0
N_CLASSES = 4

USE_L2_NORM = True
SEED = 42

BEST_CKPT_PATH = "best_mlp_by_val.pt"


# =========================
# Utils
# =========================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


def l2_normalize(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / denom


class MLP(nn.Module):
    def __init__(self, in_dim=1536, hidden=512, n_classes=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


@torch.no_grad()
def eval_acc_and_cm(model, loader, device, n_classes=4):
    model.eval()
    correct, total = 0, 0
    cm = torch.zeros(n_classes, n_classes, dtype=torch.int64)

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        pred = logits.argmax(dim=1)

        correct += (pred == yb).sum().item()
        total += yb.numel()

        for t, p in zip(yb.view(-1), pred.view(-1)):
            cm[t.long(), p.long()] += 1

    acc = correct / total if total > 0 else 0.0
    return acc, cm.cpu().numpy()


@torch.no_grad()
def get_wrong_predictions(model, loader, device):
    model.eval()
    wrong = []
    global_idx = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb).argmax(dim=1)

        for j in range(len(yb)):
            if pred[j] != yb[j]:
                wrong.append((global_idx + j, yb[j].item(), pred[j].item()))

        global_idx += len(yb)

    return wrong


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_count = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        bs = yb.size(0)
        total_loss += loss.item() * bs
        total_count += bs

    return total_loss / total_count if total_count > 0 else 0.0


def main():
    set_seed(SEED)

    # -------------------------
    # Load features
    # -------------------------
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy")).astype(np.float32)
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy")).astype(np.int64)

    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy")).astype(np.float32)
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy")).astype(np.int64)

    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy")).astype(np.float32)
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy")).astype(np.int64)

    if USE_L2_NORM:
        X_train = l2_normalize(X_train)
        X_val = l2_normalize(X_val)
        X_test = l2_normalize(X_test)

    print("Train label counts:", np.unique(y_train, return_counts=True))
    print("Val   label counts:", np.unique(y_val, return_counts=True))
    print("Test  label counts:", np.unique(y_test, return_counts=True))
    print("Shapes:")
    print("  X_train:", X_train.shape, "y_train:", y_train.shape)
    print("  X_val  :", X_val.shape, "y_val  :", y_val.shape)
    print("  X_test :", X_test.shape, "y_test :", y_test.shape)

    # -------------------------
    # Device
    # -------------------------
    device = get_device()
    print("Using device:", device)

    # -------------------------
    # DataLoaders
    # -------------------------
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # -------------------------
    # Model / optimizer
    # -------------------------
    model = MLP(
        in_dim=X_train.shape[1],
        hidden=HIDDEN_DIM,
        n_classes=N_CLASSES,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Train with best model selected by VAL
    # -------------------------
    best_val_acc = -1.0
    best_epoch = -1
    best_state = None

    print("\n>>> START training (pick best checkpoint by VAL)")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_acc, val_cm = eval_acc_and_cm(model, val_loader, device, N_CLASSES)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, BEST_CKPT_PATH)

        print(f"Epoch {epoch:03d} | train_loss = {train_loss:.4f} | val_acc = {val_acc:.3f}")

    print("\n=== TRAINING DONE ===")
    print(f"Best val_acc = {best_val_acc:.3f} at epoch {best_epoch}")
    print(f"[OK] Saved best checkpoint to {BEST_CKPT_PATH}")

    # -------------------------
    # Reload best model
    # -------------------------
    model.load_state_dict(torch.load(BEST_CKPT_PATH, map_location=device))

    # -------------------------
    # Final evaluation on VAL
    # -------------------------
    val_acc, val_cm = eval_acc_and_cm(model, val_loader, device, N_CLASSES)
    print("\n=== Final VAL result (best checkpoint) ===")
    print(f"VAL accuracy = {val_acc:.3f}")
    print("VAL confusion matrix (rows=true, cols=pred):")
    print(val_cm)

    # -------------------------
    # Final evaluation on TEST
    # -------------------------
    test_acc, test_cm = eval_acc_and_cm(model, test_loader, device, N_CLASSES)
    print("\n=== Final TEST result (best checkpoint selected by VAL) ===")
    print(f"TEST accuracy = {test_acc:.3f}")
    print("TEST confusion matrix (rows=true, cols=pred):")
    print(test_cm)

    # -------------------------
    # Print wrong TEST samples
    # -------------------------
    wrong = get_wrong_predictions(model, test_loader, device)
    print("\nWrong TEST predictions (index, gt, pred):")
    print(wrong)

    try:
        with open("test_data.pkl", "rb") as f:
            test_paths, test_labels = pickle.load(f)

        print("\nWrong TEST video files:")
        for idx, gt, pred in wrong:
            print(f"{test_paths[idx]} | GT={gt} | Pred={pred}")

    except Exception as e:
        print("[WARN] Could not map wrong predictions back to test video paths:", repr(e))


if __name__ == "__main__":
    main()