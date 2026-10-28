"""Make matched-time pose snapshots for visual kinetic-chain review."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def get_point(row, name, width, height, excluded):
    if name in excluded:
        return None
    try:
        if float(row[f"{name}_visibility"]) < 0.5:
            return None
        return int(float(row[f"{name}_x"]) * width), int(float(row[f"{name}_y"]) * height)
    except (KeyError, ValueError):
        return None


def midpoint(first, second):
    return ((first[0] + second[0]) // 2, (first[1] + second[1]) // 2)


def draw_marker(frame, point, shape, color):
    if point is None:
        return
    x, y = point
    r = 10
    if shape == "circle":
        cv2.circle(frame, point, r, color, -1, cv2.LINE_AA)
    elif shape == "square":
        cv2.rectangle(frame, (x-r, y-r), (x+r, y+r), color, 3, cv2.LINE_AA)
    elif shape == "diamond":
        vertices = ((x, y-r), (x+r, y), (x, y+r), (x-r, y))
        for i in range(4):
            cv2.line(frame, vertices[i], vertices[(i+1) % 4], color, 3, cv2.LINE_AA)
    elif shape == "triangle":
        vertices = ((x, y-r), (x+r, y+r), (x-r, y+r))
        for i in range(3):
            cv2.line(frame, vertices[i], vertices[(i+1) % 3], color, 3, cv2.LINE_AA)


def draw_frame(frame, row, hand, excluded, is_release):
    height, width = frame.shape[:2]
    throw = hand
    support = "right" if throw == "left" else "left"
    colors = {
        "leg": (0, 220, 0),
        "trunk": (0, 200, 255),
        "arm": (255, 0, 255),
        "support": (255, 255, 0),
    }
    points = {}
    for side in ("left", "right"):
        for joint in ("foot_index", "heel", "ankle", "knee", "hip", "shoulder", "elbow", "wrist"):
            name = f"{side}_{joint}"
            points[name] = get_point(row, name, width, height, excluded)

    def line(first, second, color, thickness=4):
        a, b = points.get(first), points.get(second)
        if a is not None and b is not None:
            cv2.line(frame, a, b, color, thickness, cv2.LINE_AA)

    # Throwing-side chain is grouped by visual phase, without claiming success.
    line(f"{throw}_foot_index", f"{throw}_ankle", colors["leg"])
    line(f"{throw}_ankle", f"{throw}_knee", colors["leg"])
    line(f"{throw}_knee", f"{throw}_hip", colors["leg"])
    line(f"{throw}_shoulder", f"{throw}_elbow", colors["arm"])
    line(f"{throw}_elbow", f"{throw}_wrist", colors["arm"])
    line("left_shoulder", "right_shoulder", colors["trunk"], 3)
    line("left_hip", "right_hip", colors["trunk"], 3)

    throw_hip = points.get(f"{throw}_hip")
    throw_shoulder = points.get(f"{throw}_shoulder")
    left_shoulder = points.get("left_shoulder")
    right_shoulder = points.get("right_shoulder")
    oblique_proxy = (
        midpoint(throw_hip, throw_shoulder)
        if throw_hip is not None and throw_shoulder is not None
        else None
    )
    chest_proxy = (
        midpoint(left_shoulder, right_shoulder)
        if left_shoulder is not None and right_shoulder is not None
        else None
    )
    trunk_chain = (throw_hip, oblique_proxy, chest_proxy, throw_shoulder)
    for first, second in zip(trunk_chain, trunk_chain[1:]):
        if first is not None and second is not None:
            cv2.line(frame, first, second, colors["trunk"], 4, cv2.LINE_AA)

    for joint in ("foot_index", "ankle", "knee"):
        draw_marker(frame, points.get(f"{throw}_{joint}"), "circle", colors["leg"])
    for joint in ("hip", "shoulder"):
        draw_marker(frame, points.get(f"{throw}_{joint}"), "circle", colors["trunk"])
    draw_marker(frame, oblique_proxy, "circle", colors["trunk"])
    draw_marker(frame, chest_proxy, "circle", colors["trunk"])
    for joint in ("elbow", "wrist"):
        draw_marker(frame, points.get(f"{throw}_{joint}"), "circle", colors["arm"])

    # Support-side symbols are distinct; absent joints are removed by profile.
    draw_marker(frame, points.get(f"{support}_shoulder"), "square", colors["support"])
    for joint in ("foot_index", "ankle", "knee", "hip"):
        draw_marker(frame, points.get(f"{support}_{joint}"), "diamond", colors["support"])
    for joint in ("elbow", "wrist"):
        draw_marker(frame, points.get(f"{support}_{joint}"), "triangle", colors["support"])

    if is_release:
        cv2.putText(
            frame, "ANNOTATED RELEASE", (12, 32), cv2.FONT_HERSHEY_SIMPLEX,
            0.7, (0, 0, 255), 2, cv2.LINE_AA,
        )


def main():
    parser = argparse.ArgumentParser(description="Plot matched-time pose frames for both athletes.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--athlete-id", required=True, help="Create a frame panel for one athlete only.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-0.5, -0.25, 0.0, 0.2])
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output = args.output if args.output.is_absolute() else root / args.output
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new filename: {output}")
    selected_athletes = [
        athlete for athlete in config["athletes"]
        if athlete["athlete_id"] == args.athlete_id
    ]
    if len(selected_athletes) != 1:
        raise ValueError(f"Athlete ID not found or duplicated in config: {args.athlete_id}")
    if len(args.offsets) != 4:
        raise ValueError("Use exactly four offsets to keep the familiar 2x4 layout.")

    athlete_frames = []
    for athlete in selected_athletes:
        proxy = Path(athlete["hip_proxy_csv"])
        if not proxy.is_absolute():
            proxy = root / proxy
        attempt_dir = proxy.resolve().parent.parent
        name = proxy.name.split("_hip_proxy_")[0]
        video_path = attempt_dir / "raw" / f"{name}.mp4"
        pose_files = sorted((attempt_dir / "processed").glob(f"{name}_pose_window_*_with_feet.csv"))
        if not pose_files:
            raise FileNotFoundError(f"Foot-aware pose CSV missing for {athlete['athlete_id']}")
        pose_file = max(pose_files, key=lambda p: len(read_csv(p)))
        pose_by_frame = {int(row["frame_index"]): row for row in read_csv(pose_file)}
        release_frame = int(athlete["release_frame"])
        capture = cv2.VideoCapture(str(video_path))
        try:
            if not capture.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0:
                raise ValueError(f"FPS unavailable: {video_path}")
            snapshots = []
            for offset in args.offsets:
                frame_index = release_frame + round(offset * fps)
                if frame_index not in pose_by_frame or not 0 <= frame_index < frame_count:
                    raise ValueError(f"{athlete['athlete_id']}: no tracked pose at frame {frame_index}")
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = capture.read()
                if not ok:
                    raise ValueError(f"Could not read frame {frame_index} from {video_path}")
                row = pose_by_frame[frame_index]
                draw_frame(
                    frame, row, str(athlete["throwing_hand"]),
                    set(athlete.get("exclude_landmarks", [])),
                    is_release=(frame_index == release_frame),
                )
                actual_offset = (frame_index - release_frame) / fps
                snapshots.append((actual_offset, frame_index, frame))
            athlete_frames.append((athlete, snapshots))
        finally:
            capture.release()

    athlete, snapshots = athlete_frames[0]
    fig, axes = plt.subplots(1, 4, figsize=(18, 9))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.19, wspace=0.12)
    fig.suptitle(
        f"{athlete['athlete_id']} — individual kinetic-chain visual review\n"
        f"{athlete.get('limb_note', '')}",
        fontsize=15,
    )
    for column, (actual_offset, frame_index, frame) in enumerate(snapshots):
        ax = axes[column]
        ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{actual_offset:+.2f} s | frame {frame_index}", fontsize=10)
        ax.axis("off")

    legend = [
        Line2D([0], [0], color="limegreen", marker="o", label="Throw-side foot/leg"),
        Line2D([0], [0], color="goldenrod", marker="o", label="Throw-side hip/trunk proxies"),
        Line2D([0], [0], color="magenta", marker="o", label="Throwing arm"),
        Line2D([0], [0], color="cyan", marker="s", label="Support shoulder"),
        Line2D([0], [0], color="cyan", marker="D", label="Support lower body"),
        Line2D([0], [0], color="cyan", marker="^", label="Support arm if present"),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.065),
        ncol=3,
        fontsize=9,
    )
    fig.text(
        0.5, 0.012,
        "Colored points identify visible landmarks for review; they do not mark a phase as successful. "
        "Foot index is a toe-area proxy. Release is human-annotated.",
        ha="center", fontsize=9,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Pose frame panel saved to {output}")


if __name__ == "__main__":
    main()
