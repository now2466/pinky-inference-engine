"""Bounded HTTP image ingress; loopback by default, optional bearer token."""
import argparse
import hmac
import json
import logging
import os
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import cv2
import numpy as np
from .backend import TorchScriptBackend
from .config import Config
from .engine import Engine


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 8

    def __init__(self, address, engine, token=None):
        self.engine, self.token = engine, token
        self.inference_slot = threading.BoundedSemaphore(1)
        self.connections = threading.BoundedSemaphore(8)
        super().__init__(address, Handler)

    def process_request(self, request, address):
        if not self.connections.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request, address)
        except Exception:
            self.connections.release()
            raise

    def process_request_thread(self, request, address):
        try:
            super().process_request_thread(request, address)
        finally:
            self.connections.release()


class Handler(BaseHTTPRequestHandler):
    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.engine.config.request_timeout_seconds)

    def respond(self, code, payload):
        body = json.dumps(payload, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def authorized(self):
        token = self.server.token
        if token and not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
            self.respond(401, {"error": "unauthorized"})
            return False
        return True

    def do_GET(self):
        if not self.authorized():
            return
        if self.path != "/health":
            self.respond(404, {"error": "not_found"})
            return
        self.respond(200, {"ready": True, "model_sha256": self.server.engine.backend.model_id})

    def do_POST(self):
        if not self.authorized():
            return
        if self.path != "/infer":
            self.respond(404, {"error": "not_found"})
            return
        if not self.server.inference_slot.acquire(blocking=False):
            self.respond(429, {"error": "busy", "retry": "send_latest_frame_only"})
            return
        try:
            if self.headers.get("Transfer-Encoding"):
                self.respond(400, {"error": "content_length_required"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = 0
            if not 0 < length <= self.server.engine.config.max_body_bytes:
                self.respond(413, {"error": "invalid_body_size"})
                return
            if self.headers.get("Content-Type", "").split(";")[0] not in ("image/jpeg", "image/png"):
                self.respond(415, {"error": "expected_jpeg_or_png"})
                return
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ValueError("Truncated image")
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            result = self.server.engine.infer(
                image, frame_id=self.headers.get("X-Frame-Id", "")[:128],
                captured_at=self.headers.get("X-Captured-At"),
                include_overlay=self.headers.get("X-Include-Overlay") == "1")
            self.respond(200, result)
        except (ValueError, cv2.error):
            self.respond(400, {"error": "invalid_image_or_input"})
        except (socket.timeout, TimeoutError):
            self.respond(408, {"error": "read_timeout"})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            logging.exception("Inference failed")
            self.respond(500, {"error": "inference_failed"})
        finally:
            self.server.inference_slot.release()

    def log_message(self, format, *args):
        logging.info(format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    token = os.environ.get("PINKY_ENGINE_TOKEN")
    if args.host not in ("127.0.0.1", "localhost", "::1") and not token:
        parser.error("Non-loopback binding requires PINKY_ENGINE_TOKEN; use a trusted LAN/TLS proxy")
    config = Config.load(args.config)
    engine = Engine(TorchScriptBackend(args.model, config), config)
    logging.basicConfig(level=logging.INFO)
    server = Server((args.host, args.port), engine, token)
    logging.info("Ready at %s:%s, model=%s", args.host, args.port, engine.backend.model_id)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
