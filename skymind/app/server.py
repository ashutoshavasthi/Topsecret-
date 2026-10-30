"""SkyMind control app: a web UI served from the laptop (stdlib only, no extra installs).

    python -m skymind app --config config/nano_room.yaml            # simulated drone
    python -m skymind app --config config/nano_room.yaml --esp-drone udp://192.168.43.42

Open http://localhost:8080 in Chrome. Voice needs a "secure context": it works on
``localhost``, and over the LAN only with HTTPS, so use the laptop for voice and a
phone for text and video.

When serving beyond localhost (``--host 0.0.0.0``) every request needs the access
token printed at start-up, so nobody else on the network can fly your drone.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from skymind.assistant.agent import DroneAssistant
from skymind.live import LiveSession, SimLiveSession

STATIC = Path(__file__).parent / "static"


class SkyMindApp:
    def __init__(self, session: LiveSession, assistant: DroneAssistant, host: str = "127.0.0.1", port: int = 8080):
        self.session = session
        self.assistant = assistant
        self.host, self.port = host, port
        local = host in ("127.0.0.1", "localhost", "::1")
        self.token = None if local else secrets.token_urlsafe(9)
        self._map_cache: tuple[float, bytes] = (0.0, b"")
        self._map_lock = threading.Lock()
        self.httpd: ThreadingHTTPServer | None = None

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("127.0.0.1", "::1") else self.host
        port = self.httpd.server_address[1] if self.httpd else self.port
        return f"http://{host}:{port}/" + (f"?t={self.token}" if self.token else "")

    def map_png(self) -> bytes:
        with self._map_lock:
            t, png = self._map_cache
            if time.time() - t > 0.4 or not png:
                png = self.session.render_map()
                self._map_cache = (time.time(), png)
            return png

    def status(self) -> dict:
        st = self.session.status()
        a = self.assistant
        st.update(
            {
                "assistant": {
                    "name": a.a.name,
                    "wake_word": a.a.wake_word,
                    "busy": a.busy,
                    "llm": a.a.model if a.llm_ok else None,
                    "vision": a.a.vision_model if a.vision_ok else None,
                },
                "places": list(self.session.cfg.mission.places),
                "can_reset": isinstance(self.session, SimLiveSession),
            }
        )
        return st

    def serve(self) -> ThreadingHTTPServer:
        app = self

        class Handler(BaseHTTPRequestHandler):
            server_version = "SkyMind/1.0"

            def log_message(self, fmt, *args):  # quiet console
                pass

            # ---------------------------------------------------- helpers
            def _authorised(self, query: dict) -> bool:
                if app.token is None:
                    return True
                given = self.headers.get("X-SkyMind-Token") or (query.get("t") or [""])[0]
                return secrets.compare_digest(given, app.token)

            def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(body)

            def _json(self, data, code: int = 200) -> None:
                self._send(code, json.dumps(data).encode("utf-8"), "application/json")

            def _body(self) -> dict:
                n = int(self.headers.get("Content-Length") or 0)
                if n <= 0 or n > 65536:
                    return {}
                try:
                    data = json.loads(self.rfile.read(n).decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    return {}
                return data if isinstance(data, dict) else {}

            # ---------------------------------------------------- routes
            def do_GET(self):  # noqa: N802
                u = urlparse(self.path)
                q = parse_qs(u.query)
                if not self._authorised(q):
                    return self._send(HTTPStatus.FORBIDDEN, b"missing or wrong access token", "text/plain")
                if u.path in ("/", "/index.html"):
                    return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
                if u.path == "/api/status":
                    return self._json(app.status())
                if u.path == "/api/messages":
                    after = int((q.get("after") or ["0"])[0] or 0)
                    return self._json(app.assistant.messages_after(after))
                if u.path == "/map.png":
                    return self._send(200, app.map_png(), "image/png")
                if u.path == "/frame.jpg":
                    jpg = app.session.latest_jpeg
                    return self._send(200, jpg, "image/jpeg") if jpg else self._send(204, b"", "text/plain")
                if u.path == "/video.mjpg":
                    return self._mjpeg()
                return self._send(404, b"not found", "text/plain")

            def do_POST(self):  # noqa: N802
                u = urlparse(self.path)
                if not self._authorised(parse_qs(u.query)):
                    return self._send(HTTPStatus.FORBIDDEN, b"missing or wrong access token", "text/plain")
                body = self._body()
                if u.path == "/api/chat":
                    text = str(body.get("text", ""))[:500]
                    app.assistant.handle(text)
                    return self._json({"ok": True})
                if u.path == "/api/tool":
                    tool = str(body.get("tool", ""))
                    if tool not in app.assistant.skills.skills:
                        return self._json({"ok": False, "error": f"unknown tool {tool}"}, 400)
                    args = body.get("args") if isinstance(body.get("args"), dict) else {}
                    app.assistant.run_tool(tool, args)
                    return self._json({"ok": True})
                if u.path == "/api/stop":
                    app.assistant.stop()
                    return self._json({"ok": True})
                if u.path == "/api/reset" and isinstance(app.session, SimLiveSession):
                    app.assistant.stop()
                    app.session.reset()
                    return self._json({"ok": True})
                return self._send(404, b"not found", "text/plain")

            def _mjpeg(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=skymindframe")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                period = 1.0 / max(app.session.cfg.camera.fps, 1.0)
                last = None
                try:
                    while True:
                        jpg = app.session.latest_jpeg
                        if jpg is not None and jpg is not last:
                            self.wfile.write(b"--skymindframe\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpg))
                            self.wfile.write(jpg + b"\r\n")
                            last = jpg
                        time.sleep(period / 2)
                except (BrokenPipeError, ConnectionResetError, OSError):
                    return

        self.httpd = ThreadingHTTPServer((self.host, self.port), Handler)
        self.httpd.daemon_threads = True
        return self.httpd
