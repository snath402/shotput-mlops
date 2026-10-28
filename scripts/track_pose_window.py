import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


LANDMARKS = {
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track pose landmarks over a selected video-frame window."
    )
    parser.add_argument("--video", type=Path, required=True, help="Input video path")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/pose_landmarker_lite.task"),
        help="Pose Landmarker model path",
    )
    parser.add_argument("--start-frame", type=int, default=330)
    parser.add_argument("--end-frame", type=int, default=420)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.video.is_file():
        raise FileNotFoundError(f"Video not found: {args.video}")

    if not args.model.is_file():
        raise FileNotFoundError(f"Model not found: {args.model}")

    if args.start_frame < 0 or args.end_frame < args.start_frame:
        raise ValueError("Check that the frame window starts at zero or later and end is after start.")

    capture = cv2.VideoCapture(str(args.video))

    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {args.video}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0:
            raise ValueError("Video FPS is unavailable.")

        if frame_count <= 0:
            raise ValueError("Video frame count is unavailable.")

        if args.end_frame >= frame_count:
            raise ValueError(
                f"END_FRAME {args.end_frame} is past the last video frame "
                f"({frame_count - 1})."
            )

        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=str(args.model)
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
        )

        fieldnames = [
            "frame_index",
            "timestamp_seconds",
            "pose_detected",
        ]

        for name in LANDMARKS.values():
            fieldnames.extend(
                [
                    f"{name}_x",
                    f"{name}_y",
                    f"{name}_z",
                    f"{name}_visibility",
                ]
            )

        args.output.parent.mkdir(parents=True, exist_ok=True)
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            with args.output.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()

                for frame_index in range(args.start_frame, args.end_frame + 1):
                    success, frame = capture.read()

                    if not success:
                        raise RuntimeError(f"Could not read frame {frame_index}.")

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )
                    result = landmarker.detect(mp_image)

                    row = {
                        "frame_index": frame_index,
                        "timestamp_seconds": f"{frame_index / fps:.3f}",
                        "pose_detected": bool(result.pose_landmarks),
                    }

                    if result.pose_landmarks:
                        pose = result.pose_landmarks[0]

                        for landmark_index, name in LANDMARKS.items():
                            point = pose[landmark_index]
                            row[f"{name}_x"] = f"{point.x:.6f}"
                            row[f"{name}_y"] = f"{point.y:.6f}"
                            row[f"{name}_z"] = f"{point.z:.6f}"
                            row[f"{name}_visibility"] = f"{point.visibility:.6f}"
                    else:
                        for name in LANDMARKS.values():
                            row[f"{name}_x"] = ""
                            row[f"{name}_y"] = ""
                            row[f"{name}_z"] = ""
                            row[f"{name}_visibility"] = ""

                    writer.writerow(row)
                    print(
                        f"Processed frame {frame_index}: "
                        f"pose detected={row['pose_detected']}"
                    )

        print(f"CSV saved to {args.output}")

    finally:
        capture.release()


if __name__ == "__main__":
    main()