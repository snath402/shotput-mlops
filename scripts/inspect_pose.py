import argparse
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw pose landmarks and identify the block and throwing sides."
    )
    parser.add_argument("--image", type=Path, required=True, help="Input image path")
    parser.add_argument("--output", type=Path, required=True, help="Output overlay path")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pose_landmarker_lite.task"),
        help="Pose Landmarker model path",
    )
    parser.add_argument(
        "--throwing-hand",
        choices=("right", "left"),
        required=True,
        help="Athlete's throwing hand",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    if args.output.exists():
        raise FileExistsError(
            f"Output already exists; choose a new filename to protect it: {args.output}"
        )

    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(args.model)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    image = mp.Image.create_from_file(str(args.image))

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(image)

    print(f"Poses found: {len(result.pose_landmarks)}")

    if not result.pose_landmarks:
        return

    landmarks = result.pose_landmarks[0]
    key_points = [
        (11, "left_shoulder"),
        (12, "right_shoulder"),
        (13, "left_elbow"),
        (14, "right_elbow"),
        (15, "left_wrist"),
        (16, "right_wrist"),
        (23, "left_hip"),
        (24, "right_hip"),
        (25, "left_knee"),
        (26, "right_knee"),
        (27, "left_ankle"),
        (28, "right_ankle"),
    ]

    for index, name in key_points:
        point = landmarks[index]
        print(
            f"{name}: x={point.x:.3f}, y={point.y:.3f}, "
            f"visibility={point.visibility:.3f}"
        )

    frame = cv2.imread(str(args.image))
    if frame is None:
        raise ValueError(f"Could not read image: {args.image}")

    height, width = frame.shape[:2]

    def pixelpoint(point):
        return int(point.x * width), int(point.y * height)

    def midpoint(point_a, point_b):
        return (
            (point_a[0] + point_b[0]) // 2,
            (point_a[1] + point_b[1]) // 2,
        )

    limb_connections = [
        (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
        (23, 24),
        (23, 25), (25, 27),
        (24, 26), (26, 28),
    ]

    for start, end in limb_connections:
        first = landmarks[start]
        second = landmarks[end]

        if min(first.visibility, second.visibility) >= 0.5:
            cv2.line(
                frame,
                pixelpoint(first),
                pixelpoint(second),
                (0, 255, 0),
                4,
            )

    throwing_hand = args.throwing_hand
    block_side = "left" if throwing_hand == "right" else "right"
    power_side = throwing_hand

    side_points = {
        "left": {
            "ankle": pixelpoint(landmarks[27]),
            "knee": pixelpoint(landmarks[25]),
            "hip": pixelpoint(landmarks[23]),
            "shoulder": pixelpoint(landmarks[11]),
            "elbow": pixelpoint(landmarks[13]),
            "wrist": pixelpoint(landmarks[15]),
        },
        "right": {
            "ankle": pixelpoint(landmarks[28]),
            "knee": pixelpoint(landmarks[26]),
            "hip": pixelpoint(landmarks[24]),
            "shoulder": pixelpoint(landmarks[12]),
            "elbow": pixelpoint(landmarks[14]),
            "wrist": pixelpoint(landmarks[16]),
        },
    }

    role_colors = {
        block_side: (255, 255, 0),  # Cyan: block side
        power_side: (255, 0, 255),  # Magenta: throwing side
    }

    for side, points in side_points.items():
        color = role_colors[side]
        oblique_proxy = midpoint(points["hip"], points["shoulder"])

        chain = [
            points["ankle"],
            points["knee"],
            points["hip"],
            oblique_proxy,
            points["shoulder"],
        ]

        if side == power_side:
            chain.extend([points["elbow"], points["wrist"]])

        for start, end in zip(chain, chain[1:]):
            cv2.line(frame, start, end, color, 4)

        for point in chain:
            cv2.circle(frame, point, 7, color, -1)

        role = "BLOCK" if side == block_side else "THROW"
        label = f"{side.upper()} - {role}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.65
        thickness = 2


        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness,
        )

        label_x = points["shoulder"][0] + 10
        if label_x + text_width > width - 5:
            label_x = points["shoulder"][0] - text_width - 10
        label_x = max(5, min(label_x, width - text_width - 5))

        label_y = points["shoulder"][1]
        label_y = max(
            text_height + 5,
            min(label_y, height- baseline-5)
        )

        cv2.putText(
            frame,
            label,
            (label_x, label_y),
            font,
            font_scale,
            color,
            thickness,
        )

    for index, _ in key_points:
        point = landmarks[index]

        if point.visibility >= 0.5:
            cv2.circle(frame, pixelpoint(point), 8, (0, 0, 255), -1)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(args.output), frame):
        raise OSError(f"Could not save overlay: {args.output}")

    print(f"Throwing hand: {throwing_hand}")
    print(f"Block side: {block_side}")
    print(f"Overlay saved to {args.output}")


if __name__ == "__main__":
    main()