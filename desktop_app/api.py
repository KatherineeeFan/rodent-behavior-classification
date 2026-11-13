"""
JS <-> Python bridge layer.
The frontend (app.js) calls the methods defined here via
window.pywebview.api.<method_name>(...)
"""
import os
import json
import glob
import shutil
import hashlib
from datetime import datetime
import cv2

import webview

from inference import RatBehaviorModel, sliding_window_predict, BEHAVIOR_CLASSES, BEHAVIOR_LABELS
from video_server import VideoServer

APP_DATA_DIR = os.path.join(os.path.expanduser("~"), "RatBehaviorApp")
ANNOTATIONS_DIR = os.path.join(APP_DATA_DIR, "annotations")
CONFIG_PATH = os.path.join(APP_DATA_DIR, "config.json")

os.makedirs(ANNOTATIONS_DIR, exist_ok=True)

DEFAULT_CONFIG = {"shared_export_folder": ""}


class Api:
    def __init__(self):
        self.model = RatBehaviorModel(weights_path=self._find_weights())
        self._load_config()

    # ---------- config ----------
    def _load_config(self):
        if os.path.exists(CONFIG_PATH):
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        else:
            self.config = dict(DEFAULT_CONFIG)
            self._save_config()

    def _save_config(self):
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    def get_config(self):
        return self.config

    def set_shared_folder(self):
        result = webview.windows[0].create_file_dialog(webview.FOLDER_DIALOG)
        if result:
            self.config["shared_export_folder"] = result[0]
            self._save_config()
        return self.config

    # ---------- locating model weights ----------
    def _find_weights(self):
        """
        At packaging time, place the trained weights file at
        resources/model_weights.pt. This looks it up automatically and
        hands it to RatBehaviorModel. If it can't be found, returns None
        and the model automatically falls back to placeholder logic
        (useful for testing the UI before real weights are wired up).
        """
        candidate = os.path.join(os.path.dirname(__file__), "resources", "model_weights.pt")
        return candidate if os.path.exists(candidate) else None

    # ---------- video playback (local HTTP server, works around WKWebView's file:// restriction) ----------
    def get_video_url(self, video_path):
        if not os.path.exists(video_path):
            return {"error": "Video file not found"}
        url = VideoServer.instance().register(video_path)
        return {"url": url}

    # ---------- video selection ----------
    def select_video(self):
        result = webview.windows[0].create_file_dialog(
            webview.OPEN_DIALOG,
            file_types=("Video files (*.mp4;*.mov;*.avi)", "All files (*.*)")
        )
        if not result:
            return None
        return result[0]

    # ---------- inference ----------
    def run_inference(self, video_path):
        if not os.path.exists(video_path):
            return {"error": "Video file not found. Please check the path."}

        try:
            segments = sliding_window_predict(video_path, self.model)
        except Exception as e:
            return {"error": str(e)}

        return {
            "video_path": video_path,
            "segments": segments,
            "classes": BEHAVIOR_CLASSES,
            "labels": BEHAVIOR_LABELS,
        }

    # ---------- save / load annotations ----------
    def _annotation_path(self, video_path):
        base = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(ANNOTATIONS_DIR, base + ".json")

    def save_annotation(self, video_path, segments, annotator=""):
        payload = {
            "video_path": video_path,
            "video_name": os.path.basename(video_path),
            "annotator": annotator,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "segments": segments,
        }
        path = self._annotation_path(video_path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return {"saved_path": path}

    def load_annotation(self, video_path):
        path = self._annotation_path(video_path)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def list_annotations(self):
        items = []
        for fname in sorted(os.listdir(ANNOTATIONS_DIR)):
            if fname.endswith(".json"):
                with open(os.path.join(ANNOTATIONS_DIR, fname), "r", encoding="utf-8") as f:
                    data = json.load(f)
                items.append({
                    "video_name": data.get("video_name"),
                    "saved_at": data.get("saved_at"),
                    "path": os.path.join(ANNOTATIONS_DIR, fname),
                })
        return items

    # ---------- exporting clips to the shared folder (feeding data back to the research team) ----------
    LABEL_PREFIX = {
        "body_grooming": "body",
        "face_grooming": "face",
        "freezing": "freezing",
        "exploratory": "exploratory",
    }
    SHORT_CODE = {
        "body_grooming": "BG",
        "face_grooming": "FG",
        "freezing": "F",
        "exploratory": "E",
    }

    def _cut_clip(self, video_path, start_sec, end_sec, out_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = max(0, min(int(start_sec * fps), total))
        end_frame = max(0, min(int(end_sec * fps), total))
        if end_frame <= start_frame:
            cap.release()
            return False

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for _ in range(end_frame - start_frame):
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
        writer.release()
        cap.release()
        return True

    def _segments_signature(self, segments):
        """Fingerprints a set of reviewed segments, used to detect whether a
        video's annotations have changed since the last export."""
        key = "|".join(
            f"{s['start']}-{s['end']}:{s['label']}"
            for s in sorted(segments, key=lambda s: s["start"])
        )
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def export_clips(self, video_path):
        """
        Exports only the currently loaded video's reviewed segments — never
        scans or re-exports any other video's local annotation record. This
        keeps export an explicit, per-video action: nothing you tested with
        earlier (or annotated by mistake) gets swept in just because it's
        sitting in local storage.

        Cuts every "confirmed correct" or "corrected" segment of this video
        into a short clip, named with a body_/face_/freezing_/exploratory_
        prefix (matching the convention prepare_data.py expects), and writes
        them to a clips_export/ subfolder inside the configured shared
        folder, along with a manifest CSV for this video.

        A "signature" of the video's annotations is used to skip re-cutting
        if nothing changed since the last export of this same video.
        """
        target = self.config.get("shared_export_folder")
        if not target:
            return {"error": "No shared folder set yet. Please click 'Set Shared Folder' in the top right first."}
        if not os.path.exists(target):
            return {"error": "The shared folder path no longer exists. Please check that your cloud sync client is running."}
        if not video_path:
            return {"error": "No video is currently loaded."}

        ann_path = self._annotation_path(video_path)
        if not os.path.exists(ann_path):
            return {"error": "No reviewed segments found for this video yet. Confirm or correct at least one segment before exporting."}

        with open(ann_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        video_name = data.get("video_name", os.path.basename(video_path))
        segments = data.get("segments", [])
        reviewed_segments = [s for s in segments if s.get("reviewed")]
        if not reviewed_segments:
            return {"error": "No reviewed segments found for this video yet. Confirm or correct at least one segment before exporting."}

        if not os.path.exists(video_path):
            return {"error": "The original video file can no longer be found at its original location."}

        clips_dir = os.path.join(target, "clips_export")
        clips_subdir = os.path.join(clips_dir, "clips")
        manifests_subdir = os.path.join(clips_dir, "manifests")
        os.makedirs(clips_subdir, exist_ok=True)
        os.makedirs(manifests_subdir, exist_ok=True)

        video_stem = os.path.splitext(video_name)[0]
        manifest_path = os.path.join(manifests_subdir, f"{video_stem}_manifest.csv")

        signature = self._segments_signature(reviewed_segments)
        prev_signature = data.get("_export_signature")

        rows = []
        for seg in reviewed_segments:
            label = seg["label"]
            prefix = self.LABEL_PREFIX.get(label, label)
            start_i, end_i = int(seg["start"]), int(seg["end"])
            clip_filename = f"{prefix}_{video_stem}_{start_i:04d}-{end_i:04d}s.mp4"
            rows.append({
                "video": video_name,
                "start": start_i,
                "end": end_i,
                "duration_sec": end_i - start_i,
                "orig_pred_label": seg.get("orig_label", label),
                "final_label": label,
                "short_label": self.SHORT_CODE.get(label, label),
                "corrected": seg.get("corrected", False),
                "confidence": seg.get("confidence"),
                "clip_filename": clip_filename,
            })

        all_files_exist = all(
            os.path.exists(os.path.join(clips_subdir, r["clip_filename"])) for r in rows
        )

        if signature == prev_signature and all_files_exist:
            updated = False
        else:
            for old_file in glob.glob(os.path.join(clips_subdir, f"*_{video_stem}_*.mp4")):
                try:
                    os.remove(old_file)
                except OSError:
                    pass

            for r in rows:
                out_path = os.path.join(clips_subdir, r["clip_filename"])
                self._cut_clip(video_path, r["start"], r["end"], out_path)

            data["_export_signature"] = signature
            with open(ann_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            updated = True

        import csv
        fieldnames = ["video", "start", "end", "duration_sec", "orig_pred_label",
                      "final_label", "short_label", "corrected", "confidence", "clip_filename"]
        with open(manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows({k: r[k] for k in fieldnames} for r in rows)

        return {
            "video_name": video_name,
            "clip_count": len(rows),
            "updated": updated,
            "clips_dir": clips_dir,
            "manifest_name": os.path.basename(manifest_path),
        }

    def export_all(self):
        """Legacy: copies the raw annotation JSON files as-is. Kept only as
        a debugging backup; the frontend now uses export_clips() instead."""
        target = self.config.get("shared_export_folder")
        if not target:
            return {"error": "No shared folder set yet. Please click 'Set Shared Folder' in the top right first."}
        if not os.path.exists(target):
            return {"error": "The shared folder path no longer exists. Please check that your cloud sync client is running."}

        copied = 0
        for fname in os.listdir(ANNOTATIONS_DIR):
            if fname.endswith(".json"):
                src = os.path.join(ANNOTATIONS_DIR, fname)
                dst = os.path.join(target, fname)
                shutil.copy2(src, dst)
                copied += 1
        return {"copied": copied, "target": target}
