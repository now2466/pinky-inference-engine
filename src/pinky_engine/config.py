"""Explicit deployment contract; not inferred from a checkpoint filename."""
from dataclasses import dataclass, field
import json
from pathlib import Path
import math

CLASSES = ("background", "lane_left", "lane_right", "crosswalk", "speed_bump")


@dataclass(frozen=True)
class Config:
    width: int = 320
    height: int = 240
    ignore_top: int = 110
    classes: tuple[str, ...] = CLASSES
    device: str = "auto"
    cpu_threads: int = 2
    # Conservative default: do not delete small genuine markings.
    confidence_thresholds: dict[str, float] = field(default_factory=dict)
    min_component_pixels: dict[str, int] = field(default_factory=dict)
    max_body_bytes: int = 8 * 1024 * 1024
    max_image_pixels: int = 1920 * 1080
    request_timeout_seconds: float = 5.0

    def __post_init__(self):
        for name in ("width", "height", "cpu_threads", "max_body_bytes", "max_image_pixels"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.ignore_top) is not int or not 0 <= self.ignore_top < self.height:
            raise ValueError("ignore_top out of range")
        if tuple(self.classes) not in (CLASSES[:4], CLASSES):
            raise ValueError("Expected existing LaneUNet class order (4 or 5 classes)")
        if not math.isfinite(self.request_timeout_seconds) or self.request_timeout_seconds <= 0:
            raise ValueError("Invalid timeout")
        for name, threshold in self.confidence_thresholds.items():
            if name not in self.classes[1:] or not math.isfinite(threshold) or not 0 <= threshold <= 1:
                raise ValueError("Invalid class confidence threshold")
        for name, minimum in self.min_component_pixels.items():
            if name not in self.classes[1:] or type(minimum) is not int or minimum < 0:
                raise ValueError("Invalid component threshold")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text())
        if "classes" in data:
            data["classes"] = tuple(data["classes"])
        return cls(**data)
