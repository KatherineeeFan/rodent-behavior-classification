import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm

import torch
from transformers import AutoImageProcessor, AutoModel

# =========================
# Config
# =========================
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

N_FRAMES = 32
DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu"
)

TRAIN_PKL = "train_data.pkl"
VAL_PKL = "val_data.pkl"
TEST_PKL = "test_data.pkl"

OUT_DIR = "features_v1"
CACHE_DIR = os.path.join(OUT_DIR, "per_video_meanstd")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# =========================
# Helpers
# =========================
def sample_frame_indices(total_frames: int, n_frames: int) -> list[int]:
    if total_frames <= 0:
        return []

    if total_frames <= n_frames:
        return list(range(total_frames))

    idx = np.linspace(0, total_frames - 1, n_frames)
    idx = [int(round(x)) for x in idx]

    # remove duplicates while preserving order
    seen = set()
    uniq = []
    for i in idx:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def read_frames_opencv(video_path: str, n_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = sample_frame_indices(total_frames, n_frames)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()
    return frames


def infer_hidden_dim(model) -> int:
    if hasattr(model, "config"):
        for attr in ["hidden_size", "embed_dim", "dim"]:
            if hasattr(model.config, attr):
                v = getattr(model.config, attr)
                if isinstance(v, int) and v > 0:
                    return v
    return 768


@torch.no_grad()
def frames_to_embedding_mean_std(frames_rgb: list[np.ndarray], processor, model) -> np.ndarray | None:
    """
    Multi-frame -> per-frame embedding -> mean + std -> concat
    Output dim = 2 * hidden_dim
    """
    if len(frames_rgb) == 0:
        return None

    inputs = processor(images=frames_rgb, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = model(**inputs)

    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        per_frame = outputs.last_hidden_state.mean(dim=1)   # [B, dim]
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        per_frame = outputs.pooler_output
    else:
        return None

    per_frame = per_frame.detach().cpu().numpy().astype(np.float32)

    mu = per_frame.mean(axis=0)
    sigma = per_frame.std(axis=0)
    feat = np.concatenate([mu, sigma], axis=0)
    return feat.astype(np.float32)


def load_split_paths_and_labels(pkl_path: str):
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Cannot find {pkl_path}. Please run prepare_data.py first.")

    with open(pkl_path, "rb") as f:
        paths, labels = pickle.load(f)

    return paths, labels


def process_split(pkl_path: str, split_name: str, processor, model):
    paths, labels = load_split_paths_and_labels(pkl_path)

    dim = infer_hidden_dim(model)
    out_dim = dim * 2

    X, y = [], []

    for video_path, label in tqdm(list(zip(paths, labels)), desc=f"Extract {split_name}"):
        base = os.path.basename(video_path).replace(" ", "_")
        cache_path = os.path.join(CACHE_DIR, base + ".npy")

        if os.path.exists(cache_path):
            video_feat = np.load(cache_path)
        else:
            frames = read_frames_opencv(video_path, N_FRAMES)
            video_feat = frames_to_embedding_mean_std(frames, processor, model)

            if video_feat is None:
                print(f"[WARN] No frames / embedding failed: {video_path} -> use zeros")
                video_feat = np.zeros((out_dim,), dtype=np.float32)

            np.save(cache_path, video_feat)

        if video_feat.shape != (out_dim,):
            print(f"[WARN] Bad feature shape {video_feat.shape} for {video_path}, expected {(out_dim,)}. Skip.")
            continue

        X.append(video_feat)
        y.append(int(label))

    if len(X) == 0:
        raise RuntimeError(f"No features extracted for split = {split_name}")

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y, dtype=np.int64)

    np.save(os.path.join(OUT_DIR, f"X_{split_name}.npy"), X)
    np.save(os.path.join(OUT_DIR, f"y_{split_name}.npy"), y)

    print(f"[OK] {split_name}: X shape = {X.shape}, y shape = {y.shape}")
    print(f"[OK] {split_name} labels:", np.unique(y, return_counts=True))


def main():
    print(f"Loading model: {MODEL_NAME}")
    print(f"N_FRAMES per video: {N_FRAMES}")
    print(f"Using device: {DEVICE}")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    model.eval()

    process_split(TRAIN_PKL, "train", processor, model)
    process_split(VAL_PKL, "val", processor, model)
    process_split(TEST_PKL, "test", processor, model)

    print("\n[DONE] Features saved to:", OUT_DIR)
    print("Expected feature dim = 1536 for ViT-B mean+std")


if __name__ == "__main__":
    main()