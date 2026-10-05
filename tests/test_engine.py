import base64
import json
import threading
import urllib.request
import urllib.error
import cv2
import numpy as np
import pytest
import torch
from pinky_engine.config import Config
from pinky_engine.engine import Engine
from pinky_engine.backend import TorchScriptBackend
from pinky_engine.preprocess import preprocess
from pinky_engine.postprocess import postprocess, overlay
from pinky_engine.stream import LatestFrame
from pinky_engine.server import Server


class Fake:
    model_id = "test-only-not-trained"

    def predict(self, tensor):
        probabilities = np.zeros((5, 240, 320), np.float32)
        probabilities[2] = 0.9
        probabilities[0] = 0.1
        return probabilities


def decode(value):
    return cv2.imdecode(np.frombuffer(base64.b64decode(value), np.uint8), cv2.IMREAD_UNCHANGED)


def test_rgb_contract():
    bgr = np.zeros((480, 640, 3), np.uint8)
    bgr[:, :, 2] = 255
    tensor = preprocess(bgr, Config())
    assert tuple(tensor.shape) == (1, 3, 240, 320)
    assert tensor.dtype == torch.float32
    assert tensor[0, 0].min() == 1 and tensor[0, 2].max() == 0


@pytest.mark.parametrize("image", [None, np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32)])
def test_invalid_image(image):
    with pytest.raises(ValueError):
        preprocess(image, Config())


def test_config_validation():
    for kwargs in ({"ignore_top": 240}, {"classes": ("background", "right_lane")},
                   {"confidence_thresholds": {"speed_bump": 1.1}}, {"cpu_threads": 0}):
        with pytest.raises(ValueError):
            Config(**kwargs)


def test_original_coordinates_and_ignore():
    bgr = np.zeros((480, 640, 3), np.uint8)
    result = Engine(Fake(), Config()).infer(bgr, "frame7", "timestamp", True)
    mask = decode(result["mask_png_base64"])
    assert mask.shape == (480, 640) and mask.dtype == np.uint8
    assert (mask[:220] == 255).all() and (mask[220:] == 2).all()
    confidence = decode(result["confidence_png_base64"])
    assert confidence.dtype == np.uint16 and (confidence[:220] == 0).all()
    assert result["frame_id"] == "frame7"
    assert result["class_stats"]["lane_right"]["pixels"] == 260 * 640
    assert result["timing_ms"]["total"] > 0


def test_default_preserves_tiny_bump_and_roles():
    probabilities = np.zeros((5, 240, 320), np.float32)
    probabilities[0] = 1
    probabilities[:, 150, 10] = 0
    probabilities[4, 150, 10] = 1
    probabilities[:, 150, 300] = 0
    probabilities[1, 150, 300] = 1  # left role on screen right must remain left
    mask, _, changed = postprocess(probabilities, Config())
    assert mask[150, 10] == 4 and mask[150, 300] == 1 and changed == 0


def test_thresholds_reject_to_unknown():
    mask, confidence, changed = postprocess(Fake().predict(None), Config(confidence_thresholds={"lane_right": 0.95}))
    assert (mask == 255).all() and (confidence == 0).all() and changed > 0


def test_overlay_keeps_background():
    bgr = np.full((5, 5, 3), 100, np.uint8)
    mask = np.zeros((5, 5), np.uint8)
    mask[0] = 255
    assert np.array_equal(overlay(bgr, mask), bgr)


def test_latest_frame_and_shutdown():
    buffer = LatestFrame()
    for index in range(100):
        buffer.put(index)
    assert buffer.dropped == 99 and buffer.get() == 99
    buffer.close()
    assert buffer.get() is None and not buffer.put(101)


class Toy(torch.nn.Module):
    def forward(self, x):
        return torch.cat((x, x[:, :2]), dim=1)


def test_torchscript_backend(tmp_path):
    path = tmp_path / "trusted_fixture.pt"
    torch.jit.trace(Toy(), torch.zeros(1, 3, 240, 320)).save(str(path))
    backend = TorchScriptBackend(path, Config(device="cpu"))
    probabilities = backend.predict(torch.zeros(1, 3, 240, 320))
    assert probabilities.shape == (5, 240, 320)
    assert np.allclose(probabilities.sum(0), 1)
    assert len(backend.model_id) == 64
    with pytest.raises(ValueError):
        TorchScriptBackend(path, Config(classes=Config().classes[:4]))


def test_http_smoke_auth_busy_invalid_and_infer():
    server = Server(("127.0.0.1", 0), Engine(Fake(), Config()), "test-only-token")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    token = {"Authorization": "Bearer test-only-token", "Content-Type": "image/png"}
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + "/health")
        assert error.value.code == 401
        with urllib.request.urlopen(urllib.request.Request(url + "/health", headers=token)) as response:
            assert json.load(response)["ready"]
        ok, image = cv2.imencode(".png", np.zeros((240, 320, 3), np.uint8))
        assert ok
        request = urllib.request.Request(url + "/infer", image.tobytes(), token)
        with urllib.request.urlopen(request) as response:
            assert decode(json.load(response)["mask_png_base64"]).shape == (240, 320)
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(urllib.request.Request(url + "/infer", b"invalid", token))
        assert error.value.code == 400
        server.inference_slot.acquire()
        try:
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request)
            assert error.value.code == 429
        finally:
            server.inference_slot.release()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
