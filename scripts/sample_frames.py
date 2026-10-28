import argparse
from pathlib import Path

import cv2


def sample_frames(
    video_path: Path,
    output_dir: Path,
    interval_seconds: float,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> int:
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be greater than zero")

    capture = cv2.VideoCapture(str(video_path))

    try:
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or frame_count <= 0:
            raise ValueError(
                "Video does not contain valid frame timing information."
            )

        last_frame = frame_count - 1

        if start_frame < 0:
            raise ValueError("start_frame cannot be negative")

        if start_frame > last_frame:
            raise ValueError("start_frame is beyond the end of the video")

        if end_frame is None:
            end_frame = last_frame
        elif end_frame > last_frame:
            raise ValueError(
                f"end_frame {end_frame} is past the last video frame "
                f"({last_frame})"
            )

        if end_frame < start_frame:
            raise ValueError(
                "end_frame must be greater than or equal to start_frame"
            )

        frame_step = max(1, round(fps * interval_seconds))
        frame_indices = range(start_frame, end_frame + 1, frame_step)

        output_dir.mkdir(parents=True, exist_ok=True)

        existing_outputs = [
            output_dir / f"frame_{frame_index:05d}.jpg"
            for frame_index in frame_indices
            if (output_dir / f"frame_{frame_index:05d}.jpg").exists()
        ]

        if existing_outputs:
            raise FileExistsError(
                "Some sampled frames already exist in "
                f"{output_dir}. Choose a new output folder to protect them."
            )

        saved_count = 0

        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()

            if not success:
                continue

            timestamp = frame_index / fps
            label = f"frame {frame_index} | {timestamp:.2f} seconds"

            cv2.putText(
                frame,
                label,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            output_path = output_dir / f"frame_{frame_index:05d}.jpg"

            if not cv2.imwrite(str(output_path), frame):
                raise OSError(f"Could not save sampled frame: {output_path}")

            saved_count += 1

        return saved_count

    finally:
        capture.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Save timestamped sample frames from a video."
    )
    parser.add_argument("video_path", type=Path, help="Input video path")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output folder for this video's sampled frames",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=0.5,
        help="Time between sampled frames; default is 0.5 seconds",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame to sample; default is 0",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        help="Last frame to sample; omit to use the video's final frame",
    )

    args = parser.parse_args()

    saved_count = sample_frames(
        args.video_path,
        args.output_dir,
        args.interval_seconds,
        args.start_frame,
        args.end_frame,
    )

    print(f"Saved {saved_count} frame(s) to {args.output_dir}")


if __name__ == "__main__":
    main()