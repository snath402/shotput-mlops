"""Inspect video files and write a reproducible dataset inventory."""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import cv2

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
CSV_FIELDS = [
    "relative_path",
    "sha256",
    "size_bytes",
    "readable",
    "width",
    "height",
    "fps",
    "frame_count",
    "duration_seconds",
    "issue",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_video(path: Path, root: Path) -> dict[str, object]:
    """Return media properties; retain unreadable files in the inventory."""
    row: dict[str, object] = {
        "relative_path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "readable": False,
        "width": "",
        "height": "",
        "fps": "",
        "frame_count": "",
        "duration_seconds": "",
        "issue": "",
    }

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            row["issue"] = "Could not open video; codec or file may be unsupported."
            return row

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        row.update(
            readable=True,
            width=width,
            height=height,
            fps=round(fps, 3) if fps > 0 else "",
            frame_count=frame_count if frame_count > 0 else "",
            duration_seconds=round(frame_count / fps, 3)
            if fps > 0 and frame_count > 0
            else "",
        )
        if not width or not height:
            row["issue"] = "Video opened but dimensions are unavailable."
        elif fps <= 0 or frame_count <= 0:
            row["issue"] = "Video opened but timing metadata is unavailable."
        return row
    finally:
        capture.release()


def build_inventory(root: Path) -> list[dict[str, object]]:
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Video directory does not exist: {root}")
    paths = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES
    )
    return [inspect_video(path, root) for path in paths]


def write_inventory(rows: list[dict[str, object]], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory shot-put training video files.")
    parser.add_argument("video_dir", type=Path, help="Directory containing source video clips")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/video_inventory.csv"),
        help="CSV destination (default: data/processed/video_inventory.csv)",
    )
    args = parser.parse_args()

    rows = build_inventory(args.video_dir)
    write_inventory(rows, args.output)
    readable = sum(bool(row["readable"]) for row in rows)
    print(f"Inventoried {len(rows)} clip(s); {readable} readable. CSV: {args.output}")


if __name__ == "__main__":
    main()
