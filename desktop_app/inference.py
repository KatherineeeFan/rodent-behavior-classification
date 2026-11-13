"""
Rat Behavior Recognition - Inference module (real-model version)

This wires together three pieces from the original training project:
    1. Feature extraction (from extract_features.py / window_inference_overlap_vote.py):
       DINOv3 backbone (facebook/dinov3-vitb16-pretrain-lvd1689m) extracts a per-frame
       embedding -> mean+std pooling across frames -> 1536-dim feature vector
    2. The MLP classifier architecture trained in train_mlp.py (1536 -> 512 -> 4, dropout=0)
    3. The sliding-window strategy from window_inference_overlap_vote.py:
       3-second window / 1-second stride -> majority vote (ties broken by summed
       probability) -> per-second output

On first run this needs internet access to download the DINOv3 weights from
HuggingFace (a few hundred MB), and you must have already run
`hf auth login` locally once (DINOv3 is a gated model and requires an
authorized HuggingFace account), otherwise you'll get a 401 permission error.
After the first download the weights are cached locally (~/.cache/huggingface),
so no internet is required after that.
"""

import os
import math
from collections import Counter, defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn

MODEL_NAME = "facebook/dinov3-vitb16-pretrain-lvd1689m"

# ---- must match window_inference_overlap_vote.py ----
WINDOW_SEC = 3.0
STRIDE_SEC = 1.0
VOTE_INTERVAL_SEC = 1.0
N_FRAMES = 32

# ---- must match train_mlp.py ----
HIDDEN_DIM = 512
DROPOUT = 0.0
N_CLASSES = 4
USE_L2_NORM = True

BEHAVIOR_CLASSES = ["body_grooming", "face_grooming", "freezing", "exploratory"]
BEHAVIOR_LABELS = {
    "body_grooming": "Body Grooming",
    "face_grooming": "Face Grooming",
    "freezing": "Freezing",
    "exploratory": "Exploratory",
}
LABEL_IDX_TO_KEY = {0: "body_grooming", 1: "face_grooming", 2: "freezing", 3: "exploratory"}


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class MLP(nn.Module):
    """Structure must exactly match the MLP in train_mlp.py, or load_state_dict will fail."""
    def __init__(self, in_dim, hidden=512, n_classes=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def _l2_normalize(x):
    denom = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / denom


def _infer_hidden_dim(backbone):
    if hasattr(backbone, "config"):
        for attr in ["hidden_size", "embed_dim", "dim"]:
            if hasattr(backbone.config, attr):
                v = getattr(backbone.config, attr)
                if isinstance(v, int) and v > 0:
                    return v
    return 768


class RatBehaviorModel:
    """
    Wraps the real model: DINOv3 backbone (feature extraction) + trained MLP (classification).
    If the weights file can't be found, self.classifier stays None and
    sliding_window_predict() automatically falls back to placeholder logic,
    so the UI can still be tested end to end.
    """

    def __init__(self, weights_path=None):
        self.weights_path = weights_path
        self.device = _get_device()
        self.processor = None
        self.backbone = None
        self.classifier = None
        self.out_dim = None
        self._load()

    def _load(self):
        if self.weights_path is None or not os.path.exists(self.weights_path):
            print("[inference] No weights file found, using placeholder logic (UI testing only)")
            return

        from transformers import AutoImageProcessor, AutoModel

        print("[inference] Loading backbone:", MODEL_NAME)
        self.processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        self.backbone = AutoModel.from_pretrained(MODEL_NAME)
        self.backbone.to(self.device)
        self.backbone.eval()

        hidden_dim = _infer_hidden_dim(self.backbone)
        self.out_dim = hidden_dim * 2

        self.classifier = MLP(
            in_dim=self.out_dim, hidden=HIDDEN_DIM, n_classes=N_CLASSES, dropout=DROPOUT
        ).to(self.device)
        state = torch.load(self.weights_path, map_location=self.device)
        self.classifier.load_state_dict(state)
        self.classifier.eval()
        print(f"[inference] Real model loaded. Feature dim = {self.out_dim}, device = {self.device}")

    @torch.no_grad()
    def _frames_to_embedding(self, frames_rgb):
        """Matches frames_to_embedding_mean_std in extract_features.py exactly."""
        if len(frames_rgb) == 0:
            return np.zeros((self.out_dim,), dtype=np.float32)

        inputs = self.processor(images=frames_rgb, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.backbone(**inputs)

        if getattr(outputs, "last_hidden_state", None) is not None:
            per_frame = outputs.last_hidden_state.mean(dim=1)
        elif getattr(outputs, "pooler_output", None) is not None:
            per_frame = outputs.pooler_output
        else:
            return np.zeros((self.out_dim,), dtype=np.float32)

        per_frame = per_frame.detach().cpu().numpy().astype(np.float32)
        mu = per_frame.mean(axis=0)
        sigma = per_frame.std(axis=0)
        feat = np.concatenate([mu, sigma], axis=0).astype(np.float32)
        if feat.shape != (self.out_dim,):
            return np.zeros((self.out_dim,), dtype=np.float32)
        return feat

    @torch.no_grad()
    def predict_feature(self, feat_1d):
        x = feat_1d[None, :]
        if USE_L2_NORM:
            x = _l2_normalize(x)
        x = torch.from_numpy(x.astype(np.float32)).to(self.device)
        logits = self.classifier(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        return pred, probs.tolist()


def _sample_indices(start_frame, end_frame, n_frames):
    total = max(0, end_frame - start_frame)
    if total <= 0:
        return []
    if total <= n_frames:
        return list(range(start_frame, end_frame))
    idx = np.linspace(start_frame, end_frame - 1, n_frames)
    idx = [int(round(x)) for x in idx]
    seen, uniq = set(), []
    for i in idx:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    return uniq


def _read_window_frames(cap, fps, total_frames, start_sec, end_sec, n_frames):
    start_frame = max(0, min(int(start_sec * fps), total_frames))
    end_frame = max(0, min(int(end_sec * fps), total_frames))
    indices = _sample_indices(start_frame, end_frame, n_frames)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    return frames


def _majority_vote(votes):
    """Matches majority_vote_with_prob_tiebreak in window_inference_overlap_vote.py."""
    if not votes:
        return None
    labels = [v["pred_label"] for v in votes]
    counts = Counter(labels)
    max_count = max(counts.values())
    candidates = [lab for lab, cnt in counts.items() if cnt == max_count]
    if len(candidates) == 1:
        return candidates[0]
    prob_sums = defaultdict(float)
    for v in votes:
        for lab, p in enumerate(v["probs"]):
            prob_sums[lab] += p
    return max(candidates, key=lambda lab: prob_sums[lab])


def _merge_consecutive(per_second):
    if not per_second:
        return []
    merged = [dict(per_second[0])]
    for seg in per_second[1:]:
        last = merged[-1]
        if seg["label"] == last["label"] and seg["start"] == last["end"]:
            last["end"] = seg["end"]
            last["confidence"] = round((last["confidence"] + seg["confidence"]) / 2, 3)
        else:
            merged.append(dict(seg))
    return merged


def sliding_window_predict(video_path, model, window_sec=WINDOW_SEC, step_sec=STRIDE_SEC, progress_cb=None):
    """
    Real-model sliding window + majority vote. Returns per-second-merged
    behavior segments:
    [{"start":0,"end":4,"label":"freezing","confidence":0.82}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0.0

    if model.classifier is None:
        cap.release()
        return _placeholder_predict(video_path)

    n_windows = math.ceil(duration / step_sec) if duration else 0
    raw_windows = []
    for w in range(n_windows):
        start_sec = w * step_sec
        end_sec = min(start_sec + window_sec, duration)
        if end_sec <= start_sec:
            continue
        frames = _read_window_frames(cap, fps, total_frames, start_sec, end_sec, N_FRAMES)
        feat = model._frames_to_embedding(frames)
        pred, probs = model.predict_feature(feat)
        raw_windows.append({"start_sec": start_sec, "end_sec": end_sec, "pred_label": pred, "probs": probs})
        if progress_cb and duration:
            progress_cb(min(end_sec / duration, 1.0))

    cap.release()

    per_second = []
    n_intervals = math.ceil(duration / VOTE_INTERVAL_SEC) if duration else 0
    for i in range(n_intervals):
        start_sec = i * VOTE_INTERVAL_SEC
        end_sec = min(start_sec + VOTE_INTERVAL_SEC, duration)
        if end_sec <= start_sec:
            continue
        covering = [w for w in raw_windows if w["start_sec"] <= start_sec and w["end_sec"] >= end_sec]
        label_idx = _majority_vote(covering)
        if label_idx is None:
            continue
        conf = float(np.mean([w["probs"][label_idx] for w in covering]))
        per_second.append({
            "start": int(round(start_sec)),
            "end": int(round(end_sec)),
            "label": LABEL_IDX_TO_KEY[label_idx],
            "confidence": round(conf, 3),
        })

    return _merge_consecutive(per_second)


def _placeholder_predict(video_path):
    """Placeholder logic used when no real weights are configured. UI testing only,
    does not represent real model performance."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    n = len(frames)
    if n == 0:
        return []
    duration = n / fps

    per_second = []
    total_seconds = int(math.ceil(duration))
    for sec in range(total_seconds):
        start_f = int(sec * fps)
        end_f = min(int((sec + 1) * fps), n)
        clip = frames[start_f:end_f]
        motion = 0.0
        if len(clip) >= 2:
            g1 = cv2.cvtColor(clip[0], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(clip[-1], cv2.COLOR_BGR2GRAY)
            motion = float(np.mean(np.abs(g1.astype(np.int16) - g2.astype(np.int16)))) / 255.0

        if motion < 0.015:
            label, conf = "freezing", round(min(0.5 + (0.015 - motion) * 10, 0.95), 3)
        else:
            idx = int(motion * 137) % len(BEHAVIOR_CLASSES)
            label, conf = BEHAVIOR_CLASSES[idx], round(min(0.45 + motion, 0.9), 3)

        per_second.append({"start": sec, "end": sec + 1, "label": label, "confidence": conf})

    return _merge_consecutive(per_second)
