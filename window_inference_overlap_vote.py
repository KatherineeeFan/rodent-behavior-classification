import os
import json
import math
import csv
from collections import Counter, defaultdict

import cv2
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

# =========================
# Config
# =========================
MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

LONG_VIDEO_DIR = "long_videos"
CKPT_PATH = "best_mlp_by_val.pt"

# Optimization suggested by PhD:
# run 3-second windows with 1-second stride:
# 0-3, 1-4, 2-5, 3-6, ...
WINDOW_SEC = 3.0
STRIDE_SEC = 1.0

# Final voted output interval. With 1 sec, interval 2-3s is voted by
# windows 0-3, 1-4, and 2-5.
VOTE_INTERVAL_SEC = 1.0

N_FRAMES = 32
HIDDEN_DIM = 512
DROPOUT = 0.0
N_CLASSES = 4
USE_L2_NORM = True

OUT_RAW_JSON = "overlap_window_predictions_raw.json"
OUT_VOTED_JSON = "overlap_voted_predictions_1s.json"
OUT_VOTED_CSV = "overlap_voted_predictions_1s.csv"
OUT_3S_CSV = "overlap_voted_predictions_3s_for_sanity_check.csv"

VALID_EXTS = (".mp4", ".mov", ".avi", ".mkv")

LABEL_NAME_MAP = {
    0: "body grooming",
    1: "face grooming",
    2: "freezing",
    3: "exploratory",
}

SHORT_LABEL_MAP = {
    0: "BG",
    1: "FG",
    2: "F",
    3: "E",
}

# =========================
# Device
# =========================
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()


# =========================
# Model
# =========================
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


def l2_normalize(X: np.ndarray) -> np.ndarray:
    denom = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / denom


def infer_hidden_dim(model) -> int:
    if hasattr(model, "config"):
        for attr in ["hidden_size", "embed_dim", "dim"]:
            if hasattr(model.config, attr):
                v = getattr(model.config, attr)
                if isinstance(v, int) and v > 0:
                    return v
    return 768


# =========================
# Video helpers
# =========================
def get_video_meta(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps is None or fps <= 0:
        fps = 30.0

    duration_sec = frame_count / fps if fps > 0 else 0.0
    return fps, frame_count, duration_sec


def sample_indices_in_window(start_frame: int, end_frame: int, n_frames: int):
    """Sample up to n_frames evenly from [start_frame, end_frame)."""
    total = max(0, end_frame - start_frame)
    if total <= 0:
        return []

    if total <= n_frames:
        return list(range(start_frame, end_frame))

    idx = np.linspace(start_frame, end_frame - 1, n_frames)
    idx = [int(round(x)) for x in idx]

    seen = set()
    uniq = []
    for i in idx:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def read_window_frames(video_path: str, start_sec: float, end_sec: float, n_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    start_frame = max(0, min(start_frame, total_frames))
    end_frame = max(0, min(end_frame, total_frames))

    indices = sample_indices_in_window(start_frame, end_frame, n_frames)

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


@torch.no_grad()
def frames_to_embedding_mean_std(frames_rgb, processor, backbone, out_dim):
    if len(frames_rgb) == 0:
        return np.zeros((out_dim,), dtype=np.float32)

    inputs = processor(images=frames_rgb, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    outputs = backbone(**inputs)

    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        per_frame = outputs.last_hidden_state.mean(dim=1)
    elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        per_frame = outputs.pooler_output
    else:
        return np.zeros((out_dim,), dtype=np.float32)

    per_frame = per_frame.detach().cpu().numpy().astype(np.float32)
    mu = per_frame.mean(axis=0)
    sigma = per_frame.std(axis=0)
    feat = np.concatenate([mu, sigma], axis=0).astype(np.float32)

    if feat.shape != (out_dim,):
        return np.zeros((out_dim,), dtype=np.float32)

    return feat


def format_window_key(start_sec: float, end_sec: float):
    s = int(round(start_sec))
    e = int(round(end_sec))
    return f"{s}-{e}s"


# =========================
# Prediction and voting
# =========================
@torch.no_grad()
def predict_feature(model, feat_1d: np.ndarray):
    x = feat_1d[None, :]

    if USE_L2_NORM:
        x = l2_normalize(x)

    x = torch.from_numpy(x.astype(np.float32)).to(DEVICE)
    logits = model(x)
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return pred, probs.astype(float).tolist()


def majority_vote_with_prob_tiebreak(votes):
    """
    votes: list of dicts with pred_label and probs.

    First use majority vote. If there is a tie, choose the label with the
    largest summed probability across the covering windows.
    """
    if len(votes) == 0:
        return None, {}, []

    labels = [int(v["pred_label"]) for v in votes]
    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = [lab for lab, cnt in counts.items() if cnt == max_count]

    if len(candidates) == 1:
        final_label = candidates[0]
    else:
        prob_sums = defaultdict(float)
        for v in votes:
            for lab, prob in enumerate(v["probs"]):
                prob_sums[lab] += float(prob)
        final_label = max(candidates, key=lambda lab: prob_sums[lab])

    vote_count_dict = {str(k): int(v) for k, v in sorted(counts.items())}
    source_windows = [v["window_key"] for v in votes]
    return int(final_label), vote_count_dict, source_windows


def build_voted_1s_results(raw_windows, duration_sec):
    """
    For each 1-second interval [t, t+1), collect all 3-second windows that
    fully cover that interval, then vote.

    Example: interval 2-3s is covered by 0-3, 1-4, and 2-5.
    """
    voted = {}
    n_intervals = math.ceil(duration_sec / VOTE_INTERVAL_SEC)

    for i in range(n_intervals):
        start_sec = i * VOTE_INTERVAL_SEC
        end_sec = min((i + 1) * VOTE_INTERVAL_SEC, duration_sec)
        if end_sec <= start_sec:
            continue

        covering_votes = []
        for win in raw_windows:
            # window covers the full voted interval
            if win["start_sec"] <= start_sec and win["end_sec"] >= end_sec:
                covering_votes.append(win)

        final_label, vote_counts, source_windows = majority_vote_with_prob_tiebreak(covering_votes)
        key = format_window_key(start_sec, end_sec)

        if final_label is None:
            voted[key] = {
                "pred_label": None,
                "pred_name": None,
                "short_label": None,
                "vote_counts": {},
                "source_windows": [],
            }
        else:
            voted[key] = {
                "pred_label": final_label,
                "pred_name": LABEL_NAME_MAP[final_label],
                "short_label": SHORT_LABEL_MAP[final_label],
                "vote_counts": vote_counts,
                "source_windows": source_windows,
            }

    return voted


def aggregate_1s_to_3s(voted_1s, duration_sec):
    """
    Optional sanity-check view: aggregate the voted 1-second labels back into
    non-overlapping 3-second bins, matching the old Excel format.
    """
    result_3s = {}
    n_bins = math.ceil(duration_sec / WINDOW_SEC)

    for b in range(n_bins):
        start_sec = b * WINDOW_SEC
        end_sec = min((b + 1) * WINDOW_SEC, duration_sec)
        second_keys = []
        labels = []

        t = start_sec
        while t < end_sec:
            key = format_window_key(t, min(t + VOTE_INTERVAL_SEC, duration_sec))
            second_keys.append(key)
            item = voted_1s.get(key)
            if item is not None and item["pred_label"] is not None:
                labels.append(int(item["pred_label"]))
            t += VOTE_INTERVAL_SEC

        if labels:
            counts = Counter(labels)
            max_count = max(counts.values())
            candidates = [lab for lab, cnt in counts.items() if cnt == max_count]
            # deterministic tie-break: choose the earliest label among the tied labels
            final_label = min(candidates)
            key_3s = format_window_key(start_sec, end_sec)
            result_3s[key_3s] = {
                "pred_label": int(final_label),
                "pred_name": LABEL_NAME_MAP[final_label],
                "short_label": SHORT_LABEL_MAP[final_label],
                "vote_counts_from_1s": {str(k): int(v) for k, v in sorted(counts.items())},
                "source_1s_intervals": second_keys,
            }

    return result_3s


def write_voted_1s_csv(all_voted_results, out_path):
    # Use csv.writer so JSON strings like {"0": 1, "1": 2} are escaped correctly.
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video", "time", "pred_label", "pred_name", "short_label",
            "vote_counts", "source_windows"
        ])
        for video_name, pred_dict in all_voted_results.items():
            for time_key, item in pred_dict.items():
                vote_counts = json.dumps(item.get("vote_counts", {}), ensure_ascii=False)
                source_windows = "|".join(item.get("source_windows", []))
                writer.writerow([
                    video_name, time_key, item.get("pred_label"), item.get("pred_name"),
                    item.get("short_label"), vote_counts, source_windows
                ])


def write_voted_3s_csv(all_3s_results, out_path):
    # Use csv.writer so JSON strings like {"0": 1, "1": 2} are escaped correctly.
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video", "time", "pred_label", "pred_name", "short_label",
            "vote_counts_from_1s", "source_1s_intervals"
        ])
        for video_name, pred_dict in all_3s_results.items():
            for time_key, item in pred_dict.items():
                counts = json.dumps(item.get("vote_counts_from_1s", {}), ensure_ascii=False)
                intervals = "|".join(item.get("source_1s_intervals", []))
                writer.writerow([
                    video_name, time_key, item.get("pred_label"), item.get("pred_name"),
                    item.get("short_label"), counts, intervals
                ])


def run_single_video(video_path: str, processor, backbone, clf_model, out_dim: int):
    fps, frame_count, duration_sec = get_video_meta(video_path)

    # 0-3, 1-4, 2-5, ... until video end
    n_windows = math.ceil(duration_sec / STRIDE_SEC)
    raw_windows = []

    print(f"\n=== Processing {os.path.basename(video_path)} ===")
    print(
        f"fps={fps:.3f}, frame_count={frame_count}, duration={duration_sec:.3f}s, "
        f"window={WINDOW_SEC:.1f}s, stride={STRIDE_SEC:.1f}s, windows={n_windows}"
    )

    for w in tqdm(range(n_windows), desc=f"Overlap windows for {os.path.basename(video_path)}"):
        start_sec = w * STRIDE_SEC
        end_sec = min(start_sec + WINDOW_SEC, duration_sec)

        if end_sec <= start_sec:
            continue

        frames = read_window_frames(video_path, start_sec, end_sec, N_FRAMES)
        feat = frames_to_embedding_mean_std(frames, processor, backbone, out_dim)
        pred, probs = predict_feature(clf_model, feat)

        key = format_window_key(start_sec, end_sec)
        raw_windows.append({
            "window_key": key,
            "start_sec": float(start_sec),
            "end_sec": float(end_sec),
            "pred_label": pred,
            "pred_name": LABEL_NAME_MAP[pred],
            "short_label": SHORT_LABEL_MAP[pred],
            "probs": probs,
        })

    voted_1s = build_voted_1s_results(raw_windows, duration_sec)
    voted_3s = aggregate_1s_to_3s(voted_1s, duration_sec)

    return raw_windows, voted_1s, voted_3s


def main():
    if not os.path.isdir(LONG_VIDEO_DIR):
        raise FileNotFoundError(f"Folder not found: {LONG_VIDEO_DIR}")

    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    print("Using device:", DEVICE)
    print("Loading backbone:", MODEL_NAME)

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME)
    backbone.to(DEVICE)
    backbone.eval()

    hidden_dim = infer_hidden_dim(backbone)
    out_dim = hidden_dim * 2

    print("Backbone hidden dim:", hidden_dim)
    print("Feature dim (mean+std):", out_dim)

    clf_model = MLP(
        in_dim=out_dim,
        hidden=HIDDEN_DIM,
        n_classes=N_CLASSES,
        dropout=DROPOUT,
    ).to(DEVICE)

    state = torch.load(CKPT_PATH, map_location=DEVICE)
    clf_model.load_state_dict(state)
    clf_model.eval()

    video_files = []
    for fn in sorted(os.listdir(LONG_VIDEO_DIR)):
        if fn.lower().endswith(VALID_EXTS):
            video_files.append(os.path.join(LONG_VIDEO_DIR, fn))

    if len(video_files) == 0:
        raise RuntimeError(f"No video files found in {LONG_VIDEO_DIR}")

    all_raw = {}
    all_voted_1s = {}
    all_voted_3s = {}

    for video_path in video_files:
        raw_windows, voted_1s, voted_3s = run_single_video(
            video_path=video_path,
            processor=processor,
            backbone=backbone,
            clf_model=clf_model,
            out_dim=out_dim,
        )
        video_name = os.path.basename(video_path)
        all_raw[video_name] = raw_windows
        all_voted_1s[video_name] = voted_1s
        all_voted_3s[video_name] = voted_3s

    with open(OUT_RAW_JSON, "w", encoding="utf-8") as f:
        json.dump(all_raw, f, indent=2, ensure_ascii=False)

    with open(OUT_VOTED_JSON, "w", encoding="utf-8") as f:
        json.dump(all_voted_1s, f, indent=2, ensure_ascii=False)

    write_voted_1s_csv(all_voted_1s, OUT_VOTED_CSV)
    write_voted_3s_csv(all_voted_3s, OUT_3S_CSV)

    print("\n[DONE] Saved raw overlap-window predictions to:", OUT_RAW_JSON)
    print("[DONE] Saved voted 1-second predictions to:", OUT_VOTED_JSON)
    print("[DONE] Saved voted 1-second CSV to:", OUT_VOTED_CSV)
    print("[DONE] Saved 3-second sanity-check CSV to:", OUT_3S_CSV)

    # compact preview
    for video_name, pred_dict in all_voted_1s.items():
        print(f"\n--- {video_name}: voted 1-second preview ---")
        for i, (k, v) in enumerate(pred_dict.items()):
            if i >= 20:
                print("...")
                break
            print(
                f"{k}: {v['pred_label']} ({v['pred_name']}), "
                f"votes={v['vote_counts']}, from={v['source_windows']}"
            )


if __name__ == "__main__":
    main()
