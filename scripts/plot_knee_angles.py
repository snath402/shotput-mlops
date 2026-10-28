import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot left and right knee angles across video frames."
    )
    parser.add_argument("--input", type=Path, required=True, help="Knee-angle CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output image path")
    parser.add_argument(
        "--throwing-hand",
        choices=("right", "left"),
        required=True,
    )
    parser.add_argument("--release-frame", type=int)
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

    if args.y_min >= args.y_max:
        raise ValueError("--y-min must be less than --y-max")

    frames = []
    left_angles = []
    right_angles = []

    with args.input.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            frame = int(row["frame_index"])

            if args.start_frame is not None and frame < args.start_frame:
                continue
            if args.end_frame is not None and frame > args.end_frame:
                continue

            left_value = row["left_knee_angle_deg"]
            right_value = row["right_knee_angle_deg"]

            frames.append(frame)
            left_angles.append(float(left_value) if left_value else None)
            right_angles.append(float(right_value) if right_value else None)

    if not frames:
        raise ValueError("No rows fall within the selected frame range.")

    block_hand = "left" if args.throwing_hand == "right" else "right"

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(11, 8))

    plots = [
        ("left", left_angles),
        ("right", right_angles),
    ]

    for axis, (side, angles) in zip(axes, plots):
        role = "THROWING SIDE" if side == args.throwing_hand else "BLOCK SIDE"
        axis.plot(frames, angles, marker="o", label=f"{side.title()} knee")
        axis.set_title(f"{side.title()} knee — {role}")
        axis.set_ylabel("Angle (degrees)")
        axis.set_ylim(args.y_min, args.y_max)
        axis.grid(True, alpha=0.3)

        if args.release_frame is not None:
            axis.axvline(
                args.release_frame,
                color="red",
                linestyle="--",
                label=f"Release (frame {args.release_frame})",
            )

        axis.legend()

    axes[-1].set_xlabel("Video frame")
    axes[-1].set_xlim(min(frames), max(frames))
    fig.suptitle("Shot-put knee angles")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    print(f"Plot saved to {args.output}")
    print(f"Throwing side: {args.throwing_hand}; block side: {block_hand}")


if __name__ == "__main__":
    main()