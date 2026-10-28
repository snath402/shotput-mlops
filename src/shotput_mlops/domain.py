from dataclasses import dataclass
from typing import Literal

AnnotationSource = Literal["human","model"]
ThrowProtocol = Literal[
    "standing_power_position",
    "glide",
    "rotational",
    "unknown",
]
@dataclass(frozen=True)

class EventAnnotation:
    event_name: str
    frame_index: int
    source: AnnotationSource
    confidence: float | None = None
    note: str = ""

    def __post_init__(self) -> None:
        if not self.event_name.strip():
            raise ValueError("event_name cannot be empty")

        if self.frame_index < 0:
            raise ValueError("frame_index cannot be negative")

        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        if self.source not in ("human", "model"):
            raise ValueError("source must be 'human' or 'model'")

@dataclass(frozen=True)
class ThrowAttempt:
    attempt_id: str
    video_path: str
    protocol: ThrowProtocol
    events: tuple[EventAnnotation,...]
    distance_m: float | None = None

    def __post_init__(self) -> None:
        if not self.attempt_id.strip():
            raise ValueError("attempt_id cannot be empty")

        if not self.video_path.strip():
            raise ValueError("video_path cannot be empty")

        if self.protocol not in ("standing_power_position","glide","rotational","unknown"):
            raise ValueError("protocol must be standing_power_position, glide, rotational, or unknown")

        if self.distance_m is not None and self.distance_m < 0:
            raise ValueError("distance_m cannot be negative")
