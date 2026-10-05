"""Replaceable backend, currently compatible with existing TorchScript exports."""
import hashlib
from pathlib import Path
import torch


class TorchScriptBackend:
    def __init__(self, path, config):
        torch.set_num_threads(config.cpu_threads)
        name = config.device
        if name == "auto":
            name = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(name)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices: auto, cpu, cuda, cuda:N")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable")
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        self.model_id = digest.hexdigest()
        # Only load trusted weights; TorchScript is executable code.
        self.model = torch.jit.load(str(path), map_location=self.device).eval()
        self.config = config
        self.predict(torch.zeros(1, 3, config.height, config.width))  # warm-up / contract check

    @torch.inference_mode()
    def predict(self, tensor):
        logits = self.model(tensor.to(self.device))
        expected = (1, len(self.config.classes), self.config.height, self.config.width)
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != expected:
            raise ValueError(f"Model must output logits {expected}")
        if not torch.isfinite(logits).all().item():
            raise ValueError("Model returned non-finite logits")
        return torch.softmax(logits, dim=1)[0].cpu().numpy()
