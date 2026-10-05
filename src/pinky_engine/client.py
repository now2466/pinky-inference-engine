"""Camera / video / OpenCV stream -> latest-frame HTTP inference -> JSONL sink."""
import argparse
import base64
import json
import math
import os
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request
import cv2
from .stream import LatestFrame


def run(args):
    if (not math.isfinite(args.timeout) or not math.isfinite(args.max_age_ms)
            or args.timeout <= 0 or args.max_age_ms <= 0 or args.jpeg_quality not in range(1, 101)):
        raise ValueError("Invalid timeout, max age, or JPEG quality")
    if args.max_frames is not None and args.max_frames < 1:
        raise ValueError("max_frames must be positive")
    source = int(args.source) if args.source.isdecimal() else args.source
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError("Cannot open camera/video/stream")
    if args.output.exists():
        capture.release()
        raise ValueError("Output already exists; choose a new JSONL filename")
    frames = LatestFrame()
    stop = threading.Event()
    reader_errors = []
    # Recorded files are replayed at source rate, rather than racing through EOF.
    recorded = isinstance(source, str) and Path(source).is_file()
    start_seconds = getattr(args, "start_seconds", 0.0)
    if not math.isfinite(start_seconds) or start_seconds < 0 or (start_seconds and not recorded):
        capture.release()
        raise ValueError("start-seconds requires a recorded video and non-negative time")
    if start_seconds and not capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000):
        capture.release()
        raise ValueError("Cannot seek video")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if recorded and not fps > 0:
        capture.release()
        raise ValueError("Video has no valid FPS")

    def read():
        index = 0
        try:
            while not stop.is_set():
                started = time.monotonic()
                ok, bgr = capture.read()
                if not ok:
                    break
                frames.put((index, time.time(), time.monotonic(), bgr))
                index += 1
                if recorded:
                    stop.wait(max(0, 1 / fps - (time.monotonic() - started)))
        except Exception as error:
            reader_errors.append(error)
        finally:
            capture.release()
            frames.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.artifacts:
        # A dedicated new directory prevents accidental overwrite of old results.
        args.artifacts.mkdir(parents=True, exist_ok=False)
    worker = threading.Thread(target=read, daemon=True)
    worker.start()
    processed = stale = busy = 0
    try:
        with args.output.open("x") as stream:
            while args.max_frames is None or processed < args.max_frames:
                item = frames.get()
                if item is None:
                    break
                index, wall_time, received, bgr = item
                if (time.monotonic() - received) * 1000 > args.max_age_ms:
                    stale += 1
                    continue
                ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                if not ok:
                    raise RuntimeError("JPEG encode failed")
                headers = {"Content-Type": "image/jpeg", "X-Frame-Id": str(index),
                           "X-Captured-At": str(wall_time), "X-Include-Overlay": "1" if args.artifacts else "0"}
                token = os.environ.get("PINKY_ENGINE_TOKEN")
                if token:
                    headers["Authorization"] = "Bearer " + token
                request = urllib.request.Request(args.url.rstrip("/") + "/infer", encoded.tobytes(), headers)
                try:
                    with urllib.request.urlopen(request, timeout=args.timeout) as response:
                        result = json.load(response)
                except urllib.error.HTTPError as error:
                    if error.code == 429:
                        busy += 1
                        continue
                    raise
                age_ms = (time.monotonic() - received) * 1000
                if age_ms > args.max_age_ms:
                    stale += 1
                    continue  # never hand a stale prediction downstream
                result["client_age_ms"] = age_ms
                result["client_dropped_pending_frames"] = frames.dropped
                result["source_position_seconds"] = start_seconds + index / fps if recorded else None
                for key, suffix in (("mask_png_base64", "mask"), ("confidence_png_base64", "confidence"),
                                    ("overlay_png_base64", "overlay")):
                    encoded_image = result.pop(key, None)
                    if args.artifacts and encoded_image:
                        path = args.artifacts / f"{index:08d}_{suffix}.png"
                        with path.open("xb") as file:
                            file.write(base64.b64decode(encoded_image, validate=True))
                        result[suffix + "_path"] = str(path.resolve())
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                processed += 1
    finally:
        stop.set()
        frames.close()
        worker.join(timeout=2)
    if reader_errors:
        raise RuntimeError("Capture failed") from reader_errors[0]
    if processed == 0:
        raise RuntimeError("No usable inference result; check source, server and max-age-ms")
    return {"processed": processed, "dropped_pending": frames.dropped, "stale": stale, "busy": busy}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Camera index, video filename or stream URL")
    parser.add_argument("--url", default="http://127.0.0.1:8770")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--artifacts", type=Path, help="New directory for mask/confidence/overlay PNGs")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--start-seconds", type=float, default=0, help="Seek recorded video before replay")
    parser.add_argument("--max-age-ms", type=float, default=500)
    parser.add_argument("--timeout", type=float, default=5)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    print(json.dumps(run(parser.parse_args())))


if __name__ == "__main__":
    main()
