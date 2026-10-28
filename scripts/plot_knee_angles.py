"""Plot knee-angle estimates in a consistent, release-aligned layout."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot throwing-side and block-side knee angles relative to release."
    )
    parser.add_argument("--input", type=Path, required=True, help="Knee-angle CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument(
        "--throwing-hand", choices=("right", "left"), required=True
    )
    parser.add_argument("--release-frame", type=int, required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--before-seconds", type=float, default=1.0)
    parser.add_argument("--after-seconds", type=float, default=0.4)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--y-min", type=float, default=0)
    parser.add_argument("--y-max", type=float, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Angle CSV not found: {args.input}")
    if args.output.exists():
        raise FileExistsError(
            f"Output already exists; choose a new filename to protect it: {args.output}"
        )
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero")
    if args.before_seconds <= 0 or args.after_seconds < 0:
        raise ValueError("--before-seconds must be positive and --after-seconds non-negative")
    if args.y_min >= args.y_max:
        raise ValueError("--y-min must be less than --y-max")

    frames: list[int] = []
    left_angles: list[float | None] = []
    right_angles: list[float | None] = []

    with args.input.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            frame = int(row["frame_index"])
            if args.start_frame is not None and frame < args.start_frame:
                continue
            if args.end_frame is not None and frame > args.end_frame:
                continue

            seconds_from_release = (frame - args.release_frame) / args.fps
            if not -args.before_seconds <= seconds_from_release <= args.after_seconds:
                continue

            left_value = row["left_knee_angle_deg"]
            right_value = row["right_knee_angle_deg"]
            frames.append(frame)
            left_angles.append(float(left_value) if left_value else None)
            right_angles.append(float(right_value) if right_value else None)

    if not frames:
        raise ValueError("No CSV rows fall within the selected release-relative window.")

    block_side = "left" if args.throwing_hand == "right" else "right"
    role_series = [
        ("throwing", args.throwing_hand, "#d76bff", left_angles if args.throwing_hand == "left" else right_angles),
        ("block", block_side, "#22d3c5", left_angles if block_side == "left" else right_angles),
    ]
    seconds = [(frame - args.release_frame) / args.fps for frame in frames]

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 8))
    for axis, (role, side, color, angles) in zip(axes, role_series):
        axis.plot(
            seconds,
            angles,
            marker="o",
            linewidth=2,
            markersize=4,
            color=color,
            label=f"{role.title()} side — {side.title()} knee",
        )
        axis.axvline(
            0,
            color="red",
            linestyle="--",
            linewidth=1.8,
            label=f"Annotated release (frame {args.release_frame})",
        )
        axis.set_title(f"{role.title()} side — {side.title()} knee")
        axis.set_ylabel("Angle (degrees)")
        axis.set_ylim(args.y_min, args.y_max)
        axis.set_xlim(-args.before_seconds, args.after_seconds)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    axes[-1].set_xlabel("Seconds from release")
    fig.suptitle("Shot-put knee angles · release-aligned view")
    fig.text(
        0.5,
        0.012,
        "2D pose estimate; gaps indicate unavailable or low-visibility landmarks.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(f"Plot saved to {args.output}")
    print(f"Throwing side: {args.throwing_hand}; block side: {block_side}")


if __name__ == "__main__":
    main()
