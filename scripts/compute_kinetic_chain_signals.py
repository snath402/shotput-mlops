"""Extract release-aligned pose signals for human kinetic-chain review.

This script creates per-frame 2D movement proxies. It does not decide whether
a technique phase was successful or measure ground-reaction force.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


MIN_VISIBILITY = 0.5


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def point(row: dict[str, str], name: str, width: int, height: int,
          excluded: set[str]):
    if name in excluded:
        return None
    try:
        if float(row[f"{name}_visibility"]) < MIN_VISIBILITY:
            return None
        return (
            float(row[f"{name}_x"]) * width,
            float(row[f"{name}_y"]) * height,
        )
    except (KeyError, ValueError, TypeError):
        return None


def midpoint(a, b):
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def distance(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def joint_angle(first, vertex, last):
    if first is None or vertex is None or last is None:
        return None
    a = (first[0] - vertex[0], first[1] - vertex[1])
    b = (last[0] - vertex[0], last[1] - vertex[1])
    la, lb = math.hypot(*a), math.hypot(*b)
    if la == 0 or lb == 0:
        return None
    cosine = max(-1.0, min(1.0, (a[0] * b[0] + a[1] * b[1]) / (la * lb)))
    return math.degrees(math.acos(cosine))


def axis_angle(left, right):
    if left is None or right is None:
        return None
    raw = math.degrees(math.atan2(right[1] - left[1], right[0] - left[0]))
    return (raw + 90.0) % 180.0 - 90.0


def angle_difference(first: float | None, second: float | None):
    if first is None or second is None:
        return None
    return (first - second + 90.0) % 180.0 - 90.0


def speed(current, previous, seconds: float, body_length: float):
    if current is None or previous is None or seconds <= 0 or body_length <= 0:
        return None
    return distance(current, previous) / seconds / body_length


def optional_number(value, decimals=3):
    return "" if value is None else f"{value:.{decimals}f}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create per-frame movement proxies for kinetic-chain review."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--plot-output", type=Path, required=True)
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    before = float(config["window_before_seconds"])
    after = float(config["window_after_seconds"])

    plot_path = args.plot_output
    if not plot_path.is_absolute():
        plot_path = root / plot_path
    plot_path = plot_path.resolve()
    if plot_path.exists():
        raise FileExistsError(f"Plot already exists; choose a new path: {plot_path}")

    output_fields = [
        "athlete_id", "throwing_hand", "release_frame", "frame_index",
        "timestamp_seconds", "seconds_from_release", "pose_detected",
        "throw_foot_speed_torso_lengths_s", "support_foot_speed_torso_lengths_s",
        "throw_knee_angle_deg", "support_knee_angle_deg",
        "pelvis_axis_angle_deg", "shoulder_axis_angle_deg",
        "shoulder_pelvis_separation_signed_deg",
        "shoulder_pelvis_separation_abs_deg", "throw_elbow_angle_deg",
        "support_elbow_angle_deg", "throw_wrist_speed_torso_lengths_s",
        "support_wrist_speed_torso_lengths_s",
        "support_shoulder_speed_torso_lengths_s",
    ]
    signal_specs = [
        ("throw_knee_angle_deg", "Throw knee (°)", (0, 180)),
        ("throw_foot_speed_torso_lengths_s", "Throw foot speed (torso/s)", None),
        ("support_foot_speed_torso_lengths_s", "Support foot speed (torso/s)", None),
        ("support_knee_angle_deg", "Support knee (°)", (0, 180)),
        ("pelvis_axis_angle_deg", "Pelvis axis (°)", (-90, 90)),
        ("shoulder_pelvis_separation_abs_deg", "Torso separation (°)", (0, 90)),
        ("throw_elbow_angle_deg", "Throw elbow (°)", (0, 180)),
        ("throw_wrist_speed_torso_lengths_s", "Throw wrist speed (torso/s)", None),
        ("support_elbow_angle_deg", "Support elbow (°)", (0, 180)),
        ("support_shoulder_speed_torso_lengths_s", "Support shoulder speed (torso/s)", None),
    ]

    athlete_results = []
    for athlete in config["athletes"]:
        athlete_id = str(athlete["athlete_id"])
        hand = str(athlete["throwing_hand"]).lower()
        if hand not in ("left", "right"):
            raise ValueError(f"{athlete_id}: throwing_hand must be left or right")
        support = "right" if hand == "left" else "left"
        release_frame = int(athlete["release_frame"])
        excluded = set(athlete.get("exclude_landmarks", []))

        proxy_path = Path(athlete["hip_proxy_csv"])
        if not proxy_path.is_absolute():
            proxy_path = root / proxy_path
        attempt_dir = proxy_path.resolve().parent.parent
        attempt_name = proxy_path.name.split("_hip_proxy_")[0]
        video_path = attempt_dir / "raw" / f"{attempt_name}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")

        pose_candidates = sorted(
            attempt_dir.joinpath("processed").glob(
                f"{attempt_name}_pose_window_*_with_feet.csv"
            )
        )
        if not pose_candidates:
            raise FileNotFoundError(
                f"Foot-aware pose CSV missing for {athlete_id} in "
                f"{attempt_dir / 'processed'}"
            )
        pose_path = max(pose_candidates, key=lambda path: len(read_csv(path)))
        pose_rows = read_csv(pose_path)
        release_rows = [r for r in pose_rows if int(r["frame_index"]) == release_frame]
        if not release_rows:
            raise ValueError(f"{athlete_id}: release frame missing from {pose_path.name}")

        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            capture.release()
        if fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"Video metadata unavailable: {video_path}")

        release_time = float(release_rows[0]["timestamp_seconds"])
        previous = None
        output_rows = []
        for row in pose_rows:
            frame_index = int(row["frame_index"])
            seconds = float(row["timestamp_seconds"]) - release_time
            if seconds < -before or seconds > after:
                continue

            pts = {
                f"{side}_{name}": point(
                    row, f"{side}_{name}", width, height, excluded
                )
                for side in ("left", "right")
                for name in (
                    "foot_index", "heel", "ankle", "knee", "hip",
                    "shoulder", "elbow", "wrist",
                )
            }
            throw_foot = pts[f"{hand}_foot_index"]
            support_foot = pts[f"{support}_foot_index"]
            throw_knee = joint_angle(
                pts[f"{hand}_hip"], pts[f"{hand}_knee"], pts[f"{hand}_ankle"]
            )
            support_knee = joint_angle(
                pts[f"{support}_hip"], pts[f"{support}_knee"], pts[f"{support}_ankle"]
            )
            throw_elbow = joint_angle(
                pts[f"{hand}_shoulder"], pts[f"{hand}_elbow"], pts[f"{hand}_wrist"]
            )
            support_elbow = joint_angle(
                pts[f"{support}_shoulder"], pts[f"{support}_elbow"], pts[f"{support}_wrist"]
            )
            pelvis_angle = axis_angle(pts["left_hip"], pts["right_hip"])
            shoulder_angle = axis_angle(pts["left_shoulder"], pts["right_shoulder"])
            separation = angle_difference(shoulder_angle, pelvis_angle)

            shoulder_center = (
                midpoint(pts["left_shoulder"], pts["right_shoulder"])
                if pts["left_shoulder"] is not None and pts["right_shoulder"] is not None
                else None
            )
            hip_center = (
                midpoint(pts["left_hip"], pts["right_hip"])
                if pts["left_hip"] is not None and pts["right_hip"] is not None
                else None
            )
            body_length = (
                distance(shoulder_center, hip_center)
                if shoulder_center is not None and hip_center is not None
                else None
            )
            previous_time = (
                float(previous["timestamp_seconds"]) if previous else None
            )
            dt = (
                float(row["timestamp_seconds"]) - previous_time
                if previous_time is not None
                else 0.0
            )
            previous_points = previous["points"] if previous else {}
            foot_speed = speed(
                throw_foot, previous_points.get("throw_foot"), dt,
                body_length or 0.0,
            )
            support_foot_speed = speed(
                support_foot, previous_points.get("support_foot"), dt,
                body_length or 0.0,
            )
            wrist_speed = speed(
                pts[f"{hand}_wrist"], previous_points.get("throw_wrist"), dt,
                body_length or 0.0,
            )
            support_wrist_speed = speed(
                pts[f"{support}_wrist"], previous_points.get("support_wrist"), dt,
                body_length or 0.0,
            )
            support_shoulder_speed = speed(
                pts[f"{support}_shoulder"], previous_points.get("support_shoulder"),
                dt, body_length or 0.0,
            )

            result = {
                "athlete_id": athlete_id,
                "throwing_hand": hand,
                "release_frame": release_frame,
                "frame_index": frame_index,
                "timestamp_seconds": row["timestamp_seconds"],
                "seconds_from_release": f"{seconds:.3f}",
                "pose_detected": row.get("pose_detected", ""),
                "throw_foot_speed_torso_lengths_s": optional_number(foot_speed),
                "support_foot_speed_torso_lengths_s": optional_number(support_foot_speed),
                "throw_knee_angle_deg": optional_number(throw_knee, 2),
                "support_knee_angle_deg": optional_number(support_knee, 2),
                "pelvis_axis_angle_deg": optional_number(pelvis_angle, 2),
                "shoulder_axis_angle_deg": optional_number(shoulder_angle, 2),
                "shoulder_pelvis_separation_signed_deg": optional_number(separation, 2),
                "shoulder_pelvis_separation_abs_deg": optional_number(
                    abs(separation) if separation is not None else None, 2
                ),
                "throw_elbow_angle_deg": optional_number(throw_elbow, 2),
                "support_elbow_angle_deg": optional_number(support_elbow, 2),
                "throw_wrist_speed_torso_lengths_s": optional_number(wrist_speed),
                "support_wrist_speed_torso_lengths_s": optional_number(support_wrist_speed),
                "support_shoulder_speed_torso_lengths_s": optional_number(
                    support_shoulder_speed
                ),
            }
            output_rows.append(result)
            previous = {
                "timestamp_seconds": row["timestamp_seconds"],
                "points": {
                    "throw_foot": throw_foot,
                    "support_foot": support_foot,
                    "throw_wrist": pts[f"{hand}_wrist"],
                    "support_wrist": pts[f"{support}_wrist"],
                    "support_shoulder": pts[f"{support}_shoulder"],
                },
            }

        output_path = attempt_dir / "processed" / f"{attempt_name}_kinetic_chain_signals_v3.csv"
        if output_path.exists():
            raise FileExistsError(f"Output exists; choose a new filename: {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=output_fields)
            writer.writeheader()
            writer.writerows(output_rows)
        athlete_results.append((athlete, output_rows, output_path))
        print(f"{athlete_id}: {len(output_rows)} release-window rows -> {output_path}")

    fig, axes = plt.subplots(
        len(signal_specs), len(athlete_results),
        figsize=(16, 2.05 * len(signal_specs)),
        sharex=True,
        squeeze=False,
    )
    fig.suptitle(
        "Kinetic-chain review signals (2D pose proxies; release aligned)",
        fontsize=15,
        y=0.995,
    )
    fig.subplots_adjust(left=0.12, right=0.985, top=0.955, bottom=0.055,
                        hspace=0.28, wspace=0.12)
    for column, (athlete, rows, _) in enumerate(athlete_results):
        for row_index, (field, label, limits) in enumerate(signal_specs):
            ax = axes[row_index, column]
            xs, ys = [], []
            for row in rows:
                value = row[field]
                if value:
                    xs.append(float(row["seconds_from_release"]))
                    ys.append(float(value))
            if xs:
                ax.plot(xs, ys, marker=".", markersize=3, linewidth=1)
            ax.axvline(0, color="red", linestyle="--", linewidth=1)
            if limits:
                ax.set_ylim(*limits)
            ax.grid(True, alpha=0.25)
            if column == 0:
                ax.set_ylabel(label, fontsize=9)
            if row_index == 0:
                note = athlete.get("limb_note", "")
                ax.set_title(
                    f"{athlete['athlete_id']} — {athlete['throwing_hand']} hand"
                    + (f"\n{note}" if note else "")
                )
            if row_index == len(signal_specs) - 1:
                ax.set_xlabel("Seconds from annotated release")
            if not xs:
                ax.text(
                    0.5, 0.5, "not available / not applicable",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8,
                )

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Review plot saved to {plot_path}")


if __name__ == "__main__":
    main()
