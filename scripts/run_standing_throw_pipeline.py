"""Orchestrate the current standing-throw workflow for one attempt."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_step(label: str, command: list[str], root: Path) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(command, cwd=root, check=True)


def run_stage(label: str, command: list[str], root: Path, expected: list[Path]) -> None:
    """Skip a complete previous stage; stop safely if its outputs are partial."""
    present = [path.exists() for path in expected]
    if all(present):
        print(f"\n=== Reusing completed stage: {label} ===")
        return
    if any(present):
        partial = [str(path) for path, exists in zip(expected, present) if exists]
        raise FileExistsError(
            f"Stage has partial outputs; inspect before continuing: {', '.join(partial)}"
        )
    run_step(label, command, root)

def prepare_new_attempt(config_path: Path, root: Path) -> None:
    """Bootstrap inventory and sample frames, then write a human-review config."""
    attempt_dir = config_path.parent.parent
    raw_dir = attempt_dir / "raw"
    if not raw_dir.is_dir():
        raise NotADirectoryError(f"Attempt raw folder not found: {raw_dir}")

    video_suffixes = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
    videos = sorted(
        path for path in raw_dir.iterdir()
        if path.is_file() and path.suffix.lower() in video_suffixes
    )
    if len(videos) != 1:
        raise ValueError(
            f"Expected exactly one video in {raw_dir}; found {len(videos)}. "
            "Keep one attempt video in each raw folder."
        )

    video = videos[0]
    processed = attempt_dir / "processed"
    inventory = processed / f"{video.stem}_video_inventory.csv"
    frame_dir = processed / f"{video.stem}_frames"
    existing = [str(path) for path in (inventory, frame_dir) if path.exists()]
    if existing:
        raise FileExistsError(
            "Bootstrap will not overwrite existing outputs:\n" + "\n".join(existing)
        )

    processed.mkdir(parents=True, exist_ok=True)
    run_step("1/2 Video inventory", [
        sys.executable, "-m", "shotput_mlops.intake", str(raw_dir),
        "--output", str(inventory),
    ], root)
    run_step("2/2 Sample frames for release review", [
        sys.executable, str(root / "scripts" / "sample_frames.py"), str(video),
        "--output-dir", str(frame_dir), "--interval-seconds", "0.5",
    ], root)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    starter = {
        "schema_version": 1,
        "athlete_id": attempt_dir.parent.parent.name,
        "session_id": attempt_dir.parent.name,
        "attempt_id": "attempt_001",
        "protocol": "standing_power_position",
        "video_path": video.relative_to(root).as_posix(),
        "throwing_hand": None,
        "release_frame": None,
        "window_before_seconds": 1.0,
        "window_after_seconds": 0.4,
        "exclude_landmarks": [],
        "limb_notes": "",
    }
    config_path.write_text(json.dumps(starter, indent=2), encoding="utf-8")
    print(f"\nPreparation complete. Starter config: {config_path}")
    print(f"Review sampled images in: {frame_dir}")
    print("Fill in throwing_hand and the human-confirmed release_frame in the JSON.")
    print("Then run this same pipeline command again to continue the analysis.")

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the standing-throw analysis workflow for one attempt."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    root = project_root()
    config_path = resolve_path(str(args.config), root)
    if not config_path.is_file():
        prepare_new_attempt(config_path, root)
        return
    config = json.loads(config_path.read_text(encoding="utf-8"))

    required = (
        "athlete_id", "session_id", "attempt_id", "protocol", "video_path",
        "throwing_hand", "release_frame",
    )
    missing = [field for field in required if field not in config]
    if missing:
        raise ValueError(f"Attempt config is missing fields: {', '.join(missing)}")
    if config["protocol"] != "standing_power_position":
        raise ValueError("This pipeline currently supports standing_power_position only.")
    if config["throwing_hand"] not in ("left", "right"):
        raise ValueError("throwing_hand must be 'left' or 'right'.")
    if type(config["release_frame"]) is not int or config["release_frame"] < 0:
        raise ValueError("release_frame must be a non-negative, human-confirmed integer.")

    video = resolve_path(config["video_path"], root)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if video.parent.name.lower() != "raw":
        raise ValueError("The source video must be inside this attempt's raw folder.")

    attempt_dir = video.parent.parent
    processed = attempt_dir / "processed"
    stem = video.stem
    athlete_id = str(config["athlete_id"])
    hand = str(config["throwing_hand"])
    release = config["release_frame"]
    before = float(config.get("window_before_seconds", 1.0))
    after = float(config.get("window_after_seconds", 0.4))
    if before <= 0 or after < 0:
        raise ValueError("window_before_seconds must be positive; after must be non-negative.")

    model = root / "models" / "pose_landmarker_lite.task"
    if not model.is_file():
        raise FileNotFoundError(f"Pose model not found: {model}")

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    if min(fps, frame_count, width, height) <= 0:
        raise ValueError("Video metadata is incomplete.")
    if release >= frame_count:
        raise ValueError(f"Release frame {release} exceeds last frame {frame_count - 1}.")

    start = max(0, release - round(before * fps))
    end = min(frame_count - 1, release + round(after * fps))
    window = f"{start:05d}_{end:05d}"

    inventory = processed / f"{stem}_video_inventory.csv"
    frame_dir = processed / f"{stem}_frames"
    pose_csv = processed / f"{stem}_pose_window_{window}_with_feet.csv"
    knee_csv = processed / f"{stem}_knee_angles_{window}.csv"
    knee_plot = processed / f"{stem}_knee_angles_{window}.png"
    hip_csv = processed / f"{stem}_hip_proxy_{window}.csv"
    signals_csv = processed / f"{stem}_kinetic_chain_signals_v3.csv"
    signals_plot = processed / f"{stem}_kinetic_chain_signals_v3.png"
    individual_plot = processed / f"{stem}_kinetic_chain_individual_review_v2.png"
    frames_plot = processed / f"{stem}_kinetic_chain_frames_individual.png"
    report_path = processed / f"{stem}_athlete_review.html"
    runtime_config = processed / f"{stem}_pipeline_runtime_config.json"
    manifest = processed / f"{stem}_pipeline_manifest.json"

    outputs = [
        pose_csv, knee_csv, knee_plot, hip_csv, signals_csv, signals_plot,
        individual_plot, frames_plot, report_path, runtime_config, manifest,
    ]
    all_outputs = [inventory, frame_dir, *outputs]

    processed.mkdir(parents=True, exist_ok=True)
    print(
        f"{athlete_id} | {config['attempt_id']} | {hand} hand | "
        f"{width}x{height} | {fps:.3f} FPS | {frame_count} frames"
    )
    print(f"Human-confirmed release: {release}; analysis window: {start}-{end}")

    if inventory.is_file():
        print(f"1/8 Reusing prepared inventory: {inventory}")
    else:
        run_step("1/10 Video inventory", [
            sys.executable, "-m", "shotput_mlops.intake", str(video.parent),
            "--output", str(inventory),
        ], root)

    if frame_dir.is_dir() and any(frame_dir.iterdir()):
        print(f"2/8 Reusing prepared sample frames: {frame_dir}")
    else:
        run_step("2/10 Sample frames", [
            sys.executable, str(root / "scripts" / "sample_frames.py"), str(video),
            "--output-dir", str(frame_dir), "--interval-seconds", "0.5",
        ], root)
    run_stage("3/10 Track pose landmarks", [
        sys.executable, str(root / "scripts" / "track_pose_window.py"),
        "--video", str(video), "--output", str(pose_csv), "--model", str(model),
        "--start-frame", str(start), "--end-frame", str(end),
    ], root, [pose_csv])
    run_stage("4/10 Calculate knee-angle proxies", [
        sys.executable, str(root / "scripts" / "compute_knee_angles.py"),
        "--input", str(pose_csv), "--output", str(knee_csv),
        "--frame-width", str(width), "--frame-height", str(height),
    ], root, [knee_csv])
    run_stage("5/10 Calculate pelvis/shoulder proxies", [
        sys.executable, str(root / "scripts" / "compute_hip_pivot_proxy.py"),
        "--input", str(pose_csv), "--output", str(hip_csv),
        "--frame-width", str(width), "--frame-height", str(height),
    ], root, [hip_csv])

    runtime = {
        "window_before_seconds": before,
        "window_after_seconds": after,
        "output_csv": str(signals_csv.relative_to(root)).replace("\\", "/"),
        "athletes": [{
            "athlete_id": athlete_id,
            "throwing_hand": hand,
            "release_frame": release,
            "hip_proxy_csv": str(hip_csv.relative_to(attempt_dir)).replace("\\", "/"),
            "exclude_landmarks": config.get("exclude_landmarks", []),
            "limb_note": str(config.get("limb_notes", "")),
        }],
    }
    runtime_text = json.dumps(runtime, indent=2)
    if runtime_config.is_file():
        existing_runtime = json.loads(runtime_config.read_text(encoding="utf-8"))
        if existing_runtime != runtime:
            raise ValueError(
                f"Saved runtime config differs from this attempt: {runtime_config}. "
                "Use a fresh attempt folder to keep runs reproducible."
            )
    else:
        runtime_config.write_text(runtime_text, encoding="utf-8")

    if manifest.exists():
        raise FileExistsError(f"Pipeline is already complete: {manifest}")

    run_stage("6/10 Calculate release-aligned movement signals", [
        sys.executable, str(root / "scripts" / "compute_kinetic_chain_signals.py"),
        "--config", str(runtime_config), "--plot-output", str(signals_plot),
    ], root, [signals_csv, signals_plot])
    run_stage("7/10 Plot knee angles", [
        sys.executable, str(root / "scripts" / "plot_knee_angles.py"),
        "--input", str(knee_csv), "--output", str(knee_plot),
        "--throwing-hand", hand, "--release-frame", str(release),
    ], root, [knee_plot])
    run_stage("8/10 Plot individual movement signals", [
        sys.executable, str(root / "scripts" / "plot_individual_kinetic_chain_signals.py"),
        "--config", str(runtime_config),
    ], root, [individual_plot])
    run_stage("9/10 Create the athlete's annotated frame panel", [
        sys.executable, str(root / "scripts" / "plot_kinetic_chain_frames.py"),
        "--config", str(runtime_config), "--athlete-id", athlete_id,
        "--output", str(frames_plot),
    ], root, [frames_plot])

    run_stage("10/10 Create plain-language attempt report", [
        sys.executable, str(root / "scripts" / "generate_attempt_report.py"),
        "--config", str(config_path),
    ], root, [report_path])

    manifest_data = {
        "pipeline_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "athlete_id": athlete_id,
        "session_id": config["session_id"],
        "attempt_id": config["attempt_id"],
        "protocol": config["protocol"],
        "throwing_hand": hand,
        "release_frame": release,
        "video": str(video.relative_to(root)).replace("\\", "/"),
        "video_sha256": sha256_file(video),
        "video_metadata": {
            "width": width, "height": height, "fps": fps, "frame_count": frame_count,
        },
        "tracked_frame_window": {"start_frame": start, "end_frame": end},
        "outputs": [
            str(path.relative_to(root)).replace("\\", "/") for path in all_outputs
        ],
        "interpretation_note": (
            "Pose-derived 2D proxies support individual visual review. They do not "
            "measure ground force or certify technique success."
        ),
    }
    manifest.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
    print(f"\nPipeline complete. Manifest: {manifest}")


if __name__ == "__main__":
    main()
