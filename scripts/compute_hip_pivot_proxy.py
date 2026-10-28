import argparse
import csv
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute 2D pelvis and shoulder axis proxies from pose landmarks."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input pose CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output proxy CSV")
    parser.add_argument("--frame-width", type=int, required=True)
    parser.add_argument("--frame-height", type=int, required=True)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument(
        "--min-span-height-fraction",
        type=float,
        default=0.01,
        help="Minimum axis span as a fraction of image height; provisional default 0.01",

    )
    parser.add_argument(
        "--min-hip-shoulder-ratio",
        type=float,
        default=0.20,
        help="Minimum hip span as a ratio of image height; provisional default 0.20",
    )
    return parser.parse_args()


def point_from_row(
    row: dict[str, str],
    name: str,
    frame_width: int,
    frame_height: int,
) -> tuple[float, float]:
    return (
    float(row[f"{name}_x"]) * frame_width,
    float(row[f"{name}_y"]) * frame_height,
    )

def visibility_from_row(row: dict[str, str], name: str) -> float:
    return float(row[f"{name}_visibility"])

def normalize_axis_angle(angle: float) -> float:
    return (angle + 90) % 180 - 90

def axis_angle_degrees(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    """Return a line angle in the image plane, normalized to [-90, 90)."""
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    angle = math.degrees(math.atan2(dy, dx))
    return normalize_axis_angle(angle)


def distance(
    point_a: tuple[float, float],
    point_b: tuple[float, float],
) -> float:
    return math.hypot(
        point_b[0] - point_a[0],
        point_b[1] - point_a[1],
    )


def main() -> None:
    args = parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    if args.frame_width <= 0 or args.frame_height <= 0:
        raise ValueError("Frame width and height must be positive.")

    if not 0.0 <= args.min_visibility <= 1.0:
        raise ValueError("Minimum visibility must be between 0 and 1.")

    if args.min_span_height_fraction <= 0:
        raise ValueError("Minimum span height must be positive.")

    if args.min_hip_shoulder_ratio <= 0:
        raise ValueError("Minimum hip span ratio must be positive.")

    if args.output.exists():
        raise FileExistsError(
            f"Output already exists; choose a new filename to protect it: {args.output}"
        )
    min_axis_span_px = args.frame_height * args.min_span_height_fraction
    args.output.parent.mkdir(parents=True, exist_ok=True)

    output_fields = [
        "frame_index",
        "timestamp_seconds",
        "pelvis_axis_angle_deg",
        "shoulder_axis_angle_deg",
        "shoulder_vs_pelvis_axis_angle_deg",
        "hip_span_px",
        "shoulder_span_px",
        "hip_to_shoulder_ratio",
        "hip_pair_min_visibility",
        "shoulder_pair_min_visibility",
        "minimum_axis_span_px",
        "minimum_hip_to_shoulder_ratio",
        "angle_usable",
        "angle_quality",
    ]

    with args.input.open("r", newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)

        with args.output.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=output_fields)
            writer.writeheader()

            for row in reader:
                output = {
                    "frame_index": row.get("frame_index", ""),
                    "timestamp_seconds": row.get("timestamp_seconds", ""),
                    "pelvis_axis_angle_deg": "",
                    "shoulder_axis_angle_deg": "",
                    "shoulder_vs_pelvis_axis_angle_deg": "",
                    "hip_span_px": "",
                    "shoulder_span_px": "",
                    "hip_to_shoulder_ratio": "",
                    "hip_pair_min_visibility": "",
                    "shoulder_pair_min_visibility": "",
                    "minimum_axis_span_px": f"{min_axis_span_px: .2f}",
                    "minimum_hip_to_shoulder_ratio": (
                        f"{args.min_hip_shoulder_ratio:.3f}"
                    ),
                    "angle_usable": "False",
                    "angle_quality": "",
                }

                try:
                    hip_min_visibility = min(
                        visibility_from_row(row, "left_hip"),
                        visibility_from_row(row, "right_hip"),
                    )
                    shoulder_min_visibility = min(
                        visibility_from_row(row, "left_shoulder"),
                        visibility_from_row(row, "right_shoulder"),
                    )
                except (KeyError, TypeError, ValueError):
                    writer.writerow(output)
                    continue

                output["hip_pair_min_visibility"] = f"{hip_min_visibility:.3f}"
                output["shoulder_pair_min_visibility"] = (
                    f"{shoulder_min_visibility:.3f}"
                )

                if (
                    hip_min_visibility < args.min_visibility
                    or shoulder_min_visibility < args.min_visibility
                ):

                    output["angle_quality"] = "low_visibility"
                    writer.writerow(output)
                    continue

                try:
                    left_hip = point_from_row(
                        row, "left_hip", args.frame_width, args.frame_height
                    )
                    right_hip = point_from_row(
                        row, "right_hip", args.frame_width, args.frame_height
                    )
                    left_shoulder = point_from_row(
                        row, "left_shoulder", args.frame_width, args.frame_height
                    )
                    right_shoulder = point_from_row(
                        row, "right_shoulder", args.frame_width, args.frame_height
                    )
                except (KeyError, TypeError, ValueError):
                    output["angle_quality"] ="missing_landmark_data"
                    writer.writerow(output)
                    continue

                pelvis_angle = axis_angle_degrees(left_hip, right_hip)
                shoulder_angle = axis_angle_degrees(
                    left_shoulder, right_shoulder
                )
                relative_angle = normalize_axis_angle(
                    shoulder_angle - pelvis_angle
                )

                hip_span = distance(left_hip, right_hip)
                shoulder_span = distance(left_shoulder, right_shoulder)
                hip_to_shoulder_ratio = (
                    hip_span / shoulder_span if shoulder_span > 0 else None
                )

                output.update(
                    {
                        "pelvis_axis_angle_deg": f"{pelvis_angle:.2f}",
                        "shoulder_axis_angle_deg": f"{shoulder_angle:.2f}",
                        "shoulder_vs_pelvis_axis_angle_deg": (
                            f"{relative_angle:.2f}"
                        ),
                        "hip_span_px": f"{hip_span:.2f}",
                        "shoulder_span_px": f"{shoulder_span:.2f}",
                        "hip_to_shoulder_ratio": (
                            f"{hip_to_shoulder_ratio:.3f}"
                            if hip_to_shoulder_ratio is not None
                            else ""
                        ),
                    }
                )

                quality_issues = []

                if hip_span < min_axis_span_px:
                    quality_issues.append("hip_span_small")
                if shoulder_span < min_axis_span_px:
                    quality_issues.append("shoulder_span_small")
                if (
                    hip_to_shoulder_ratio is None
                    or hip_to_shoulder_ratio < args.min_hip_shoulder_ratio

                ):
                    quality_issues.append("hip_span_small_vs_shoulders")

                if quality_issues:
                    output["angle_quality"] = ";".join(quality_issues)
                else:
                    output["angle_usable"] = "True"
                    output["angle_quality"] = "ok"

                writer.writerow(output)

    print(f"Hip-proxy CSV saved to {args.output}")
    print(f"Minimum axis span used:  {min_axis_span_px} pixels")

if __name__ == "__main__":
    main()