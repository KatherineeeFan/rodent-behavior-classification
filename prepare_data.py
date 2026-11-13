# prepare_data.py
import os
import pickle
import random
from collections import Counter

# =========================
# Config
# =========================
VIDEO_DIRS = ["videos_1", "videos_2"]
SEED = 42

# Per-class split sizes
TEST_PER_CLASS = 6
VAL_PER_CLASS = 4

LABEL_MAP = {
    "body": 0,          # body grooming
    "face": 1,          # face grooming
    "freezing": 2,
    "exploratory": 3,
}

VALID_EXTS = (".mp4", ".mov", ".avi", ".mkv")


# =========================
# Helpers
# =========================
def get_label(filename: str):
    lower = filename.lower()
    for k, v in LABEL_MAP.items():
        if lower.startswith(k):
            return v
    return None


def collect_videos_from_dirs(video_dirs):
    """
    Read all labeled videos from multiple folders.
    Returns:
        paths_by_class: dict[label] = list of full video paths
    """
    paths_by_class = {c: [] for c in LABEL_MAP.values()}

    for video_dir in video_dirs:
        if not os.path.isdir(video_dir):
            raise FileNotFoundError(f"Video folder not found: {video_dir}")

        print(f"Reading videos from: {video_dir}")

        for fn in os.listdir(video_dir):
            if not fn.lower().endswith(VALID_EXTS):
                continue

            lab = get_label(fn)
            if lab is None:
                print(f"[SKIP] Cannot infer label from filename: {fn}")
                continue

            full_path = os.path.join(video_dir, fn)
            paths_by_class[lab].append(full_path)

    return paths_by_class


def print_class_counts(paths_by_class, title="Counts"):
    print(f"\n=== {title} ===")
    inv_label_map = {v: k for k, v in LABEL_MAP.items()}
    for c in sorted(paths_by_class.keys()):
        print(f"class {c} ({inv_label_map[c]}): {len(paths_by_class[c])}")


def save_split(paths, labels, out_pkl):
    with open(out_pkl, "wb") as f:
        pickle.dump((paths, labels), f)


def describe_split(name, labels):
    cnt = Counter(labels)
    print(f"{name}: {len(labels)} samples")
    print(f"{name} label counts:", dict(sorted(cnt.items())))


# =========================
# Main
# =========================
def main():
    rng = random.Random(SEED)

    # 1) collect from both videos_1 and videos_2
    paths_by_class = collect_videos_from_dirs(VIDEO_DIRS)
    print_class_counts(paths_by_class, title="Combined raw counts from all folders")

    # 2) sanity check
    min_required = TEST_PER_CLASS + VAL_PER_CLASS + 1
    for c, arr in paths_by_class.items():
        if len(arr) < min_required:
            raise ValueError(
                f"Class {c} has only {len(arr)} samples, "
                f"but need at least {min_required} "
                f"(test={TEST_PER_CLASS}, val={VAL_PER_CLASS}, and at least 1 for train)."
            )

    # 3) per-class split
    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    test_paths, test_labels = [], []

    for c, arr in paths_by_class.items():
        arr = arr.copy()
        rng.shuffle(arr)

        test_c = arr[:TEST_PER_CLASS]
        val_c = arr[TEST_PER_CLASS: TEST_PER_CLASS + VAL_PER_CLASS]
        train_c = arr[TEST_PER_CLASS + VAL_PER_CLASS:]

        test_paths.extend(test_c)
        test_labels.extend([c] * len(test_c))

        val_paths.extend(val_c)
        val_labels.extend([c] * len(val_c))

        train_paths.extend(train_c)
        train_labels.extend([c] * len(train_c))

    # 4) shuffle overall splits
    train = list(zip(train_paths, train_labels))
    val = list(zip(val_paths, val_labels))
    test = list(zip(test_paths, test_labels))

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    train_paths, train_labels = zip(*train)
    val_paths, val_labels = zip(*val)
    test_paths, test_labels = zip(*test)

    train_paths, train_labels = list(train_paths), list(train_labels)
    val_paths, val_labels = list(val_paths), list(val_labels)
    test_paths, test_labels = list(test_paths), list(test_labels)

    # 5) save
    save_split(train_paths, train_labels, "train_data.pkl")
    save_split(val_paths, val_labels, "val_data.pkl")
    save_split(test_paths, test_labels, "test_data.pkl")

    print("\n[OK] Saved stratified split with TRAIN / VAL / TEST.")
    describe_split("Train", train_labels)
    describe_split("Val", val_labels)
    describe_split("Test", test_labels)

    # 6) optional preview
    print("\nExample files:")
    if len(train_paths) > 0:
        print("Train example:", train_paths[0], "| label =", train_labels[0])
    if len(val_paths) > 0:
        print("Val example  :", val_paths[0], "| label =", val_labels[0])
    if len(test_paths) > 0:
        print("Test example :", test_paths[0], "| label =", test_labels[0])


if __name__ == "__main__":
    main()