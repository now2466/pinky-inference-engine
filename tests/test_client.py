from argparse import Namespace
from pathlib import Path
import json
import threading
import time
import cv2
import numpy as np
import pytest
from pinky_engine.client import run
from pinky_engine.config import Config
from pinky_engine.engine import Engine
from pinky_engine.server import Server


class Backend:
    model_id = "client-test-fixture"

    def __init__(self, delay=0):
        self.delay = delay

    def predict(self, tensor):
        time.sleep(self.delay)
        result = np.zeros((5, 240, 320), np.float32)
        result[1] = 1
        return result


def video(path):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((48, 64, 3), np.uint8))
    writer.release()


@pytest.mark.parametrize("stale", [False, True])
def test_client_end_to_end_and_stale_failure(tmp_path, monkeypatch, stale):
    monkeypatch.delenv("PINKY_ENGINE_TOKEN", raising=False)
    source = tmp_path / "test.avi"
    video(source)
    server = Server(("127.0.0.1", 0), Engine(Backend(0.03 if stale else 0), Config()))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    args = Namespace(source=str(source), url=f"http://127.0.0.1:{server.server_port}",
                     output=tmp_path / "results.jsonl", artifacts=tmp_path / "frames",
                     timeout=5, max_age_ms=1 if stale else 2000, jpeg_quality=90,
                     max_frames=3, start_seconds=0)
    try:
        if stale:
            with pytest.raises(RuntimeError, match="No usable inference"):
                run(args)
            assert args.output.read_text() == ""
        else:
            report = run(args)
            assert report["processed"] == 3
            rows = [json.loads(line) for line in args.output.read_text().splitlines()]
            assert len(rows) == 3 and all(row["client_age_ms"] <= 2000 for row in rows)
            assert len(list(args.artifacts.glob("*.png"))) == 9
            assert Path(rows[0]["mask_path"]).is_file()
            assert "mask_png_base64" not in rows[0]
            with pytest.raises(ValueError, match="already exists"):
                run(args)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
