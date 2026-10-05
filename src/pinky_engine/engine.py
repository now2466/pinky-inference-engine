import base64
import time
import threading
import cv2
import numpy as np
from .preprocess import preprocess
from .postprocess import postprocess, overlay


def png64(image):
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return base64.b64encode(data).decode("ascii")


class Engine:
    def __init__(self, backend, config):
        self.backend, self.config = backend, config
        self.lock = threading.Lock()

    def infer(self, bgr, frame_id="", captured_at=None, include_overlay=False):
        # Direct callers are serialized; HTTP adapter rejects busy instead of queuing.
        with self.lock:
            started = time.perf_counter()
            tensor = preprocess(bgr, self.config)
            prepared = time.perf_counter()
            probabilities = self.backend.predict(tensor)
            predicted = time.perf_counter()
            mask, confidence, changed = postprocess(probabilities, self.config)
            h, w = bgr.shape[:2]
            large = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            large_conf = cv2.resize(confidence, (w, h), interpolation=cv2.INTER_NEAREST)
            processed = time.perf_counter()
            result = {
                "schema": "pinky.segmentation.v1", "frame_id": frame_id,
                "captured_at": captured_at, "completed_at": time.time(),
                "model_sha256": self.backend.model_id,
                "width": w, "height": h, "model_width": self.config.width,
                "model_height": self.config.height, "ignore_top_model_rows": self.config.ignore_top,
                "classes": {str(i): name for i, name in enumerate(self.config.classes)},
                "ignore_label": 255, "coordinate_system": "input_image_pixels_top_left",
                "mask_png_base64": png64(large),
                "confidence_png_base64": png64(np.rint(large_conf * 65535).astype(np.uint16)),
                "confidence_scale": 65535,
                "class_stats": {name: {
                    "pixels": int((large == i).sum()),
                    "mean_confidence": float(large_conf[large == i].mean()) if (large == i).any() else None
                } for i, name in enumerate(self.config.classes)},
                "postprocess_changed_model_pixels": changed,
                "valid_pixels": int((large != 255).sum()),
                "status": "ok" if (large != 255).any() else "no_valid_pixels",
            }
            if include_overlay:
                result["overlay_png_base64"] = png64(overlay(bgr, large))
            result["timing_ms"] = {
                "preprocess": (prepared-started)*1000,
                "inference": (predicted-prepared)*1000,
                "postprocess": (processed-predicted)*1000,
                "encode": (time.perf_counter()-processed)*1000,
                "total": (time.perf_counter()-started)*1000,
            }
            return result
