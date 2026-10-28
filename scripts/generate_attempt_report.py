"""Generate a plain-language HTML review for one standing-throw attempt."""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import cv2


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(value: str, root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def number(value: str | None) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an athlete-friendly HTML report for one attempt."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--athlete-id",
        default=None,
        help="Select one athlete when config is a multi-athlete comparison config.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = project_root()
    config_path = resolve_path(str(args.config), root)
    source_config = json.loads(config_path.read_text(encoding="utf-8"))

    if "video_path" in source_config:
        config = source_config
        video = resolve_path(config["video_path"], root)
        athlete_id = str(config["athlete_id"])
        hand = str(config["throwing_hand"])
        release = int(config["release_frame"])
        technique = str(config.get("protocol", "standing_power_position"))
        attempt_id = str(config.get("attempt_id", video.stem))
        limb_note = str(config.get("limb_notes", ""))
        configured_proxy = None
    else:
        athletes = source_config.get("athletes", [])
        matches = [item for item in athletes if item["athlete_id"] == args.athlete_id]
        if len(matches) != 1:
            raise ValueError(
                "For a multi-athlete config, pass --athlete-id matching exactly one entry."
            )
        athlete = matches[0]
        proxy = resolve_path(athlete["hip_proxy_csv"], root)
        attempt_dir = proxy.parent.parent
        attempt_name = proxy.name.split("_hip_proxy_")[0]
        video = attempt_dir / "raw" / f"{attempt_name}.mp4"
        athlete_id = str(athlete["athlete_id"])
        hand = str(athlete["throwing_hand"])
        release = int(athlete["release_frame"])
        technique = "standing_power_position"
        attempt_id = f"{athlete_id}_session_001_{attempt_name}"
        limb_note = str(athlete.get("limb_note", ""))
        configured_proxy = proxy

    if not video.is_file():
        raise FileNotFoundError(f"Attempt video not found: {video}")
    attempt_dir = video.parent.parent
    processed = attempt_dir / "processed"
    stem = video.stem
    signals_csv = processed / f"{stem}_kinetic_chain_signals_v3.csv"
    frame_panel = processed / f"{stem}_kinetic_chain_frames_individual.png"
    signal_plot = processed / f"{stem}_kinetic_chain_individual_review_v2.png"

    manifest_path = processed / f"{stem}_pipeline_manifest.json"
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        start = int(manifest["tracked_frame_window"]["start_frame"])
        end = int(manifest["tracked_frame_window"]["end_frame"])
        window = f"{start:05d}_{end:05d}"
        hip_csv = processed / f"{stem}_hip_proxy_{window}.csv"
        knee_candidates = sorted(processed.glob(f"{stem}_knee_angles_*.png"))
    else:
        if configured_proxy is None:
            raise FileNotFoundError(f"Pipeline manifest not found: {manifest_path}")
        hip_csv = configured_proxy
        knee_candidates = sorted(
            path for path in processed.glob(f"{stem}_knee_angles_*.png")
            if "release_zoom" not in path.name
        )

    capture = cv2.VideoCapture(str(video))
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if fps <= 0:
        raise ValueError(f"FPS unavailable for {video}")
    release_time = release / fps

    required_files = [signals_csv, hip_csv, frame_panel, signal_plot]
    if not knee_candidates:
        raise FileNotFoundError(f"No regular knee-angle plot found in {processed}")
    knee_plot = knee_candidates[0]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing report inputs:\n" + "\n".join(missing))

    signals = read_csv(signals_csv)
    hip_rows = read_csv(hip_csv)
    by_frame = {int(row["frame_index"]): row for row in signals}
    current = by_frame.get(release, {})
    prior = by_frame.get(release - 3, {})
    elbow_before = number(prior.get("throw_elbow_angle_deg"))
    elbow_release = number(current.get("throw_elbow_angle_deg"))
    if elbow_before is not None and elbow_release is not None:
        arm_text = (
            f"The 2D throwing-elbow estimate changed from {elbow_before:.0f} degrees "
            f"at frame {release - 3} to {elbow_release:.0f} degrees at release. "
            "Use it as a movement cue to check against the video, not as a score."
        )
    else:
        arm_text = "The throwing-elbow estimate is incomplete near release; review the frames."

    hip_row = next(
        (row for row in hip_rows if int(row["frame_index"]) == release), None
    )
    if hip_row and hip_row.get("angle_usable", "").lower() == "true":
        hip_text = (
            "The pelvis-axis proxy passed its image visibility/span checks at release. "
            "It is still only a 2D image-plane estimate."
        )
    elif hip_row:
        span = number(hip_row.get("hip_span_px"))
        span_text = f" The projected hip span was {span:.1f} pixels." if span is not None else ""
        hip_text = (
            "The pelvis-axis proxy was flagged as unreliable at release. "
            f"Quality flag: {hip_row.get('angle_quality', 'unavailable')}."
            f"{span_text} Do not infer hip rotation from this frame."
        )
    else:
        hip_text = "No pelvis-axis proxy was available at the release frame."

    misses = [
        row["frame_index"] for row in signals
        if row.get("pose_detected", "").lower() == "false"
    ]
    if misses:
        tracking_text = (
            f"Pose was not detected at {len(misses)} tracked frame(s): "
            f"{', '.join(misses)}. Estimates can be missing near those frames."
        )
    else:
        tracking_text = "Pose was detected throughout the tracked window."

    output = args.output
    if output is None:
        output = processed / f"{stem}_athlete_review.html"
    elif not output.is_absolute():
        output = (root / output).resolve()
    if output.exists():
        raise FileExistsError(f"Report already exists; choose another path: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    athlete_safe = html.escape(athlete_id)
    hand_safe = html.escape(hand.title())
    technique_safe = html.escape(technique.replace("_", " ").title())
    attempt_safe = html.escape(attempt_id)
    note_safe = html.escape(limb_note.strip())
    note_html = f"<p class='muted'>{note_safe}</p>" if note_safe else ""
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{athlete_safe} attempt review</title>
<style>
body {{ max-width: 1080px; margin: 28px auto; padding: 0 20px 34px; background: #f3f6fa; color: #1d2936; font: 16px Segoe UI, Arial, sans-serif; }}
header, section {{ background: white; border: 1px solid #dce4ed; border-radius: 14px; padding: 22px; margin: 16px 0; }}
h1 {{ margin: 4px 0 8px; font-size: 28px; }} h2 {{ margin: 0 0 12px; font-size: 20px; }}
.muted {{ color: #586779; }} .cards {{ display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; }}
.card {{ background: #f3f7fc; border-radius: 10px; padding: 14px; }}
.label {{ display: block; color: #586779; font-size: 13px; margin-bottom: 5px; }}
.value {{ font-size: 19px; font-weight: 650; }} img {{ width: 100%; height: auto; border-radius: 8px; }}
li {{ line-height: 1.55; margin: 10px 0; }} summary {{ cursor: pointer; font-weight: 600; }}
.links a {{ display: inline-block; margin: 10px 16px 0 0; }}
footer {{ text-align: center; color: #586779; font-size: 13px; padding: 12px; }}
@media(max-width:700px) {{ .cards {{ grid-template-columns: 1fr; }} body {{ margin-top: 10px; padding: 0 10px; }} }}
</style>
</head>
<body>
<header>
<p class="muted">Individual attempt review</p>
<h1>{athlete_safe} - {technique_safe}</h1>
<p class="muted">Attempt {attempt_safe}. This is an individual review, not a ranking or technique grade.</p>
<div class="cards">
<div class="card"><span class="label">Throwing hand</span><span class="value">{hand_safe}</span></div>
<div class="card"><span class="label">Human-confirmed release</span><span class="value">Frame {release}</span></div>
<div class="card"><span class="label">Video time at release</span><span class="value">{release_time:.2f} s</span></div>
</div>{note_html}
</header>
<section>
<h2>Movement snapshots</h2>
<p class="muted">Only the release frame is confirmed. The other snapshots are checkpoints for review.</p>
<img src="{html.escape(frame_panel.name)}" alt="Four annotated pose snapshots around the release">
</section>
<section>
<h2>Plain-language review</h2>
<ul>
<li><strong>Throwing arm:</strong> {html.escape(arm_text)}</li>
<li><strong>Hip view:</strong> {html.escape(hip_text)}</li>
<li><strong>Tracking:</strong> {html.escape(tracking_text)}</li>
</ul>
</section>
<section>
<h2>How to use this</h2>
<p>Check these observations against the original video. Pose estimates are 2D image-plane proxies; they do not measure force or prove that a phase worked. Speed spikes can come from landmark jitter.</p>
<details><summary>Technical plots and source data</summary><div class="links">
<a href="{html.escape(signal_plot.name)}">Movement-signal plots</a>
<a href="{html.escape(knee_plot.name)}">Knee-angle plot</a>
<a href="{html.escape(signals_csv.name)}">Movement-signal CSV</a>
<a href="{html.escape(hip_csv.name)}">Hip-proxy CSV</a>
</div></details>
</section>
<footer>Generated from this athlete's standing-throw analysis. Review alongside the video before drawing coaching conclusions.</footer>
</body>
</html>
"""
    output.write_text(page, encoding="utf-8")
    print(f"Athlete-friendly report saved to {output}")


if __name__ == "__main__":
    main()
