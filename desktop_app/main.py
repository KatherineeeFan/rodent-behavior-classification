import os
import sys
import multiprocessing


def resource_path(relative_path):
    """Handles the resource path change that happens after PyInstaller bundling."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)


# When packaged (frozen) by PyInstaller, point HuggingFace at the model cache
# bundled inside the app (resources/hf_cache) and force offline mode, so the
# app never needs network access or a HuggingFace login on the end user's
# machine. In dev mode (plain `python main.py`), this is skipped and the
# normal global ~/.cache/huggingface cache + online download is used instead.
if hasattr(sys, "_MEIPASS"):
    os.environ["HF_HOME"] = resource_path(os.path.join("resources", "hf_cache"))
    os.environ["HF_HUB_OFFLINE"] = "1"

import webview

from api import Api


def main():
    api = Api()
    frontend_index = resource_path(os.path.join("frontend", "index.html"))

    webview.create_window(
        "Rat Behavior Recognition - Annotation Tool",
        frontend_index,
        js_api=api,
        width=1200,
        height=820,
        min_size=(960, 640),
    )
    webview.start(debug=False)


if __name__ == "__main__":
    # Required when frozen by PyInstaller: without this, any internal library
    # (torch/transformers, etc.) that spawns a multiprocessing child process
    # will cause the whole app to be re-executed from scratch in that child,
    # which re-opens a new window, which can spawn again — an infinite loop
    # of new app windows. This call must come before anything else runs.
    multiprocessing.freeze_support()
    main()
