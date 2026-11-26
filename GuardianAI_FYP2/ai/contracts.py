"""Framework-independent contract: RGB Pillow frame -> per-person predictions."""

from dataclasses import dataclass, field
from typing import Protocol
from PIL.Image import Image


@dataclass(frozen=True)
class Prediction:
    track_id: str
    label: str  # normal, warning, fall, unknown
    confidence: float  # finite [0,1]; a model score, not measured accuracy
    bbox: tuple[float, float, float, float] | None = None  # normalized x1,y1,x2,y2
    evidence: dict = field(default_factory=dict)  # JSON values; include units


@dataclass(frozen=True)
class Result:
    predictions: list[Prediction]
    model_version: str
    ready: bool = True
    message: str = ""


class Detector(Protocol):
    def predict(self, frame: Image, *, camera_id: int, timestamp: float) -> Result: ...
    def reset(self, camera_id: int) -> None: ...
