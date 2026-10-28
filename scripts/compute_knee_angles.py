import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 2D knee angles from a pose-landmark CSV."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input pose CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output knee-angle CSV")
    parser.add_argument("--frame-width", type=int, required=True)
    parser.add_argument("--frame-height", type=int, required=True)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    return parser.parse_args()


def get_knee_angle(
    row: dict[str, str],
    side: str,
    frame_width: int,
    frame_height: int,
    min_visibility_threshold: float,
) -> tuple[float | None, float | None]:
    hip_name = f"{side}_hip"
    knee_name = f"{side}_knee"
    ankle_name = f"{side}_ankle"

    try:
        visibility_values = [
            float(row[f"{hip_name}_visibility"]),
            float(row[f"{knee_name}_visibility"]),
            float(row[f"{ankle_name}_visibility"]),
        ]
    except (KeyError, TypeError, ValueError):
        return None, None

    min_visibility = min(visibility_values)

    if min_visibility < min_visibility_threshold:
        return None, min_visibility

    try:
        hip = (
            float(row[f"{hip_name}_x"]) * frame_width,
            float(row[f"{hip_name}_y"]) * frame_height,
        )
        knee = (
            float(row[f"{knee_name}_x"]) * frame_width,
            float(row[f"{knee_name}_y"]) * frame_height,
        )
        ankle = (
            float(row[f"{ankle_name}_x"]) * frame_width,
            float(row[f"{ankle_name}_y"]) * frame_height,
        )
    except (KeyError, TypeError, ValueError):
        return None, min_visibility

    vector_to_hip = (hip[0] - knee[0], hip[1] - knee[1])
    vector_to_ankle = (ankle[0] - knee[0], ankle[1] - knee[1])

    length_hip = math.hypot(*vector_to_hip)
    length_ankle = math.hypot(*vector_to_ankle)

    if length_hip == 0 or length_ankle == 0:
        return None, min_visibility

    dot_product = (
        vector_to_hip[0] * vector_to_ankle[0]
        + vector_to_hip[1] * vector_to_ankle[1]
    )

    cosine = dot_product / (length_hip * length_ankle)
    cosine = max(-1.0, min(1.0, cosine))
    angle_degrees = math.degrees(math.acos(cosine))

    return angle_degrees, min_visibility


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    if args.frame_width <= 0 or args.frame_height <= 0:
        raise ValueError("Frame width and height must be positive.")

    if not 0.0 <= args.min_visibility <= 1.0:
        raise ValueError("Minimum visibility must be between 0 and 1.")

    if args.output.exists():
        raise FileExistsError(
            f"Output already exists; choose a new filename to protect it: {args.output}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_fields = [
        "frame_index",
        "timestamp_seconds",
        "left_knee_angle_deg",
        "left_min_visibility",
        "right_knee_angle_deg",
        "right_min_visibility",
    ]

    with args.input.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)

        with args.output.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_fields)
            writer.writeheader()

            for row in reader:
                left_angle, left_visibility = get_knee_angle(
                    row,
                    "left",
                    args.frame_width,
                    args.frame_height,
                    args.min_visibility,
                )
                right_angle, right_visibility = get_knee_angle(
                    row,
                    "right",
                    args.frame_width,
                    args.frame_height,
                    args.min_visibility,
                )

                writer.writerow(
                    {
                        "frame_index": row["frame_index"],
                        "timestamp_seconds": row["timestamp_seconds"],
                        "left_knee_angle_deg": (
                            f"{left_angle:.2f}" if left_angle is not None else ""
                        ),
                        "left_min_visibility": (
                            f"{left_visibility:.2f}"
                            if left_visibility is not None
                            else ""
                        ),
                        "right_knee_angle_deg": (
                            f"{right_angle:.2f}" if right_angle is not None else ""
                        ),
                        "right_min_visibility": (
                            f"{right_visibility:.2f}"
                            if right_visibility is not None
                            else ""
                        ),
                    }
                )

    print(f"Knee-angle CSV saved to {args.output}")


if __name__ == "__main__":
    main()