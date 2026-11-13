"""
Local video streaming server.

On Mac, the WKWebView engine that pywebview uses does not allow a <video>
tag to play an arbitrary local file via the file:// protocol for security
reasons. This module runs a small HTTP server bound only to 127.0.0.1 to
serve the selected video to the frontend instead, and supports Range
requests (required for seeking/scrubbing the progress bar).
"""

import os
import hashlib
import mimetypes
import http.server
import socketserver
import threading

# Add MIME types for common video formats that may be missing from the
# system's default library.
mimetypes.add_type("video/mp4", ".mp4")
mimetypes.add_type("video/quicktime", ".mov")
mimetypes.add_type("video/x-msvideo", ".avi")


class _VideoRequestHandler(http.server.BaseHTTPRequestHandler):
    registry = {}  # video_id -> absolute file path, populated by VideoServer

    def log_message(self, format, *args):
        pass  # silence logging to avoid flooding the terminal

    def do_GET(self):
        video_id = self.path.lstrip("/").split("?")[0]
        path = self.registry.get(video_id)
        if not path or not os.path.exists(path):
            self.send_error(404, "Video not found")
            return
        self._serve_file(path)

    def _serve_file(self, path):
        file_size = os.path.getsize(path)
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        range_header = self.headers.get("Range")

        start, end = 0, file_size - 1
        if range_header:
            try:
                spec = range_header.replace("bytes=", "").split("-")
                if spec[0]:
                    start = int(spec[0])
                if len(spec) > 1 and spec[1]:
                    end = int(spec[1])
            except ValueError:
                pass

        length = end - start + 1

        self.send_response(206 if range_header else 200)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if range_header:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        try:
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                chunk_size = 1024 * 1024
                while remaining > 0:
                    data = f.read(min(chunk_size, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            pass  # user scrubbed the progress bar and cancelled the request mid-way; ignore


class VideoServer:
    """
    Singleton: only one local HTTP server is started for the whole app
    lifetime. Use register(path) to turn a video file path into an
    accessible URL.
    """
    _instance = None

    def __init__(self):
        handler_cls = type("Handler", (_VideoRequestHandler,), {"registry": {}})
        self.registry = handler_cls.registry
        self._httpd = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler_cls)
        self.port = self._httpd.server_address[1]
        thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        thread.start()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = VideoServer()
        return cls._instance

    def register(self, path):
        video_id = hashlib.md5(path.encode("utf-8")).hexdigest()
        self.registry[video_id] = path
        return f"http://127.0.0.1:{self.port}/{video_id}"
