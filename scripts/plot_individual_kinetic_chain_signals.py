"""Plot one athlete's kinetic-chain review signals per image, without ranking."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


SIGNALS = [
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


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an independent motion-signal plot for each athlete."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--before-seconds", type=float, default=None)
    parser.add_argument("--after-seconds", type=float, default=None)
    args = parser.parse_args()

    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    root = config_path.parent.parent
    config = json.loads(config_path.read_text(encoding="utf-8"))
    before = (
        float(config["window_before_seconds"])
        if args.before_seconds is None else args.before_seconds
    )
    after = (
        float(config["window_after_seconds"])
        if args.after_seconds is None else args.after_seconds
    )

    for athlete in config["athletes"]:
        proxy = Path(athlete["hip_proxy_csv"])
        if not proxy.is_absolute():
            proxy = root / proxy
        attempt_dir = proxy.resolve().parent.parent
        prefix = proxy.name.split("_hip_proxy_")[0]
        source_csv = attempt_dir / "processed" / f"{prefix}_kinetic_chain_signals_v3.csv"
        if not source_csv.is_file():
            raise FileNotFoundError(f"Signal CSV not found: {source_csv}")
        output = attempt_dir / "processed" / f"{prefix}_kinetic_chain_individual_review_v2.png"
        if output.exists():
            raise FileExistsError(f"Output exists; choose a new name: {output}")

        rows = [
            row for row in read_csv(source_csv)
            if -before <= float(row["seconds_from_release"]) <= after
        ]
        if not rows:
            raise ValueError(f"No rows in release window for {athlete['athlete_id']}")

        fig, axes = plt.subplots(
            len(SIGNALS), 1, figsize=(14, 2.0 * len(SIGNALS)), sharex=True
        )
        fig.suptitle(
            f"{athlete['athlete_id']} — individual kinetic-chain review\n"
            f"{athlete.get('limb_note', '')}",
            fontsize=15,
            y=0.995,
        )
        fig.subplots_adjust(left=0.13, right=0.98, top=0.965, bottom=0.045, hspace=0.30)

        for ax, (field, label, limits) in zip(axes, SIGNALS):
            xs, ys = [], []
            for row in rows:
                value = row.get(field, "")
                if value:
                    xs.append(float(row["seconds_from_release"]))
                    ys.append(float(value))
            if xs:
                ax.plot(xs, ys, marker=".", markersize=4, linewidth=1.2)
            else:
                ax.text(
                    0.5, 0.5, "not available / not applicable",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9,
                )
            ax.axvline(0, color="red", linestyle="--", linewidth=1)
            ax.set_ylabel(label, fontsize=9)
            if limits:
                ax.set_ylim(*limits)
            ax.set_xlim(-before, after)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Seconds from this athlete's annotated release")
        fig.text(
            0.5, 0.006,
            "Movement proxies for visual review only. Speed spikes may be pose noise; "
            "video pose does not measure ground force or technique success.",
            ha="center", fontsize=9,
        )
        fig.savefig(output, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"{athlete['athlete_id']} individual review saved to {output}")


if __name__ == "__main__":
    main()
