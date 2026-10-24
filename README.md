# Shotput MLOps

A video-first training analysis project for shot-put athletes. The first goal is to turn a throw video and its session context into a traceable, coach-reviewable analysis, then compare evidence across sessions.

## Product principle

Show what the video supports, link each observation to frames or measurements, and expose uncertainty. This is a training aid, not an automated replacement for a qualified coach or a medical assessment.

## Initial scope

- Event: shot put only
- Input: one athlete, one throw per clip, with session metadata
- Initial analysis: video quality checks, key event timestamps, pose/keypoint extraction, and a reviewable report
- Longitudinal view: compare measured features and outcomes across throws and sessions
- Out of scope for the first version: automatic workout programming, injury prediction, all-sports support, and claims of accurate 3D biomechanics from a single 2D camera

## Phases

See [`docs/roadmap.md`](docs/roadmap.md) for milestones and exit criteria.

## Repository map

```text
configs/       Versioned analysis and annotation schemas
data/          Local-only data folders (ignored by Git)
docs/          Roadmap, data notes, and decisions
models/        Local model artifacts (ignored by Git)
src/           Python package and analysis pipeline
```

## Data and privacy

Never commit athlete videos, identifying metadata, or model artifacts. Keep raw clips under `data/raw/` and generated outputs under `data/processed/`; both are ignored by Git. Use participant IDs rather than names in analysis records. Before using anyone else's footage, get their permission and document the allowed use.

## Development status

Phase 0: repository and data contract established. The next task is a data audit of a small, representative set of historical clips and a first manual annotation pass.

## Run the video audit

Install the local package, place permitted clips under `data/raw/`, then run:

```bash
python -m pip install -e .
shotput-audit data/raw --output data/processed/video_inventory.csv
```

The inventory includes hashes and media properties. It does not upload or alter the clips. Review the CSV before sharing it; filenames can contain identifying information.
