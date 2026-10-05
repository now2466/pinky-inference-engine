"""No screen-side relabeling, dilation, or automatic lane-role swapping."""
import cv2
import numpy as np

COLORS_BGR = {1: (255, 255, 0), 2: (0, 140, 255), 3: (255, 0, 255), 4: (0, 255, 0)}


def postprocess(probabilities, config):
    expected = (len(config.classes), config.height, config.width)
    if probabilities.shape != expected or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability output")
    mask = probabilities.argmax(0).astype(np.uint8)
    raw = mask.copy()
    confidence = probabilities.max(0).astype(np.float32)
    # Rejected foreground becomes IGNORE, not a falsely certain background.
    for index, name in enumerate(config.classes[1:], 1):
        threshold = config.confidence_thresholds.get(name, 0.0)
        mask[(mask == index) & (confidence < threshold)] = 255
    mask[:config.ignore_top] = 255
    raw[:config.ignore_top] = 255
    for index, name in enumerate(config.classes[1:], 1):
        minimum = config.min_component_pixels.get(name, 0)
        if minimum > 0:
            count, regions, stats, _ = cv2.connectedComponentsWithStats((mask == index).astype(np.uint8), 8)
            for component in range(1, count):
                if stats[component, cv2.CC_STAT_AREA] < minimum:
                    mask[regions == component] = 255
    confidence[mask == 255] = 0
    return mask, confidence, int((mask != raw).sum())


def overlay(bgr, mask, alpha=0.55):
    result = bgr.copy()
    for index, color in COLORS_BGR.items():
        selected = mask == index
        result[selected] = (bgr[selected].astype(np.float32) * (1-alpha) + np.array(color) * alpha).astype(np.uint8)
    return result
