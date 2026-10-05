import cv2
import numpy as np
import torch


def preprocess(bgr, config):
    if not isinstance(bgr, np.ndarray) or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("Expected uint8 BGR image [H,W,3]")
    h, w = bgr.shape[:2]
    if h < 1 or w < 1 or h * w > config.max_image_pixels:
        raise ValueError("Input image dimensions exceed configured limits")
    # Identical to the training repository: resize, BGR -> RGB, /255; no CLAHE.
    small = cv2.resize(bgr, (config.width, config.height), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0)
