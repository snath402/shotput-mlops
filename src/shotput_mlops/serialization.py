import json
from pathlib import Path
from .domain import EventAnnotation, ThrowAttempt
from dataclasses import asdict

def load_attempt(path: Path) -> ThrowAttempt:
    data = json.loads(path.read_text(encoding="utf-8"))

    events = tuple(
        EventAnnotation(**event_data)
        for event_data in data["events"]
    )

    return ThrowAttempt(
        attempt_id=data["attempt_id"],
        video_path=data["video_path"],
        protocol=data["protocol"],
        events=events,
        distance_m=data.get("distance_m"),
    )

def save_attempt(attempt: ThrowAttempt, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(asdict(attempt), indent=2)
    path.write_text(json_text, encoding="utf-8")
