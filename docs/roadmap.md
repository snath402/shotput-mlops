# Project roadmap

## North-star workflow

Capture or upload a throw → attach session context → extract measurable evidence → review it with the athlete/coach → compare against prior throws → record the training decision and outcome.

## Phase 0 — Define the experiment and audit data

**Goal:** establish what the available footage can support before choosing a model.

- Inventory 10–20 representative clips: view, frame rate, resolution, lighting, occlusion, throw style, and whether distance/result is known.
- Agree on a minimal annotation guide: throw style, event timestamps, outcome, and coach-observed notes.
- Identify one primary evaluation question. Initial candidate: can the pipeline consistently locate release and selected movement phases in the available video?
- Keep a record of unusable clips and reasons; do not silently exclude failures.

**Exit:** clips can be described with the schema in `configs/session.schema.yaml`; annotation definitions and an initial split strategy are documented.

## Phase 1 — Reproducible video intake and baseline

**Goal:** create a repeatable path from a clip to a versioned analysis record.

- Validate input metadata and video readability.
- Record source hash, tool/model versions, run configuration, and output locations.
- Produce a human-reviewable clip report with sampled frames and timestamps.
- Establish simple baselines before introducing learned event detection.

**Exit:** the same input and configuration produce a traceable run record; failures are reported clearly.

## Phase 2 — Pose and event measurements

**Goal:** extract a small set of useful, defensible measurements.

- Evaluate pose/keypoint models on the actual footage and camera views.
- Detect candidate phases (for example, power position, release, recovery) with confidence and review controls.
- Calculate only measurements supported by the camera setup; distinguish image-space measurements from calibrated physical measurements.
- Compare automated outputs with manual annotations and coach review.

**Exit:** report error by event/feature and by recording condition; known limitations are visible in the report.

## Phase 3 — Longitudinal athlete analysis

**Goal:** compare throws and sessions without confusing recording variation for performance change.

- Group by athlete, session, throw style, and camera view.
- Trend throw outcomes alongside technique features and athlete notes.
- Compare like-for-like clips and display uncertainty/sample counts.
- Track coach/athlete feedback on whether findings were useful.

**Exit:** a coach or athlete can inspect a trend, trace it to source clips, and correct a misleading comparison.

## Phase 4 — MLOps and deployment

**Goal:** make the pipeline maintainable and safe to use beyond a laptop demo.

- Add automated data validation, model/config versioning, experiment tracking, and evaluation reports.
- Package inference as a service or batch job; select cloud/storage only after privacy and cost needs are clear.
- Add access control, retention/deletion behavior, monitoring, and rollback for model changes.
- Keep human review and an explicit “insufficient evidence” outcome.

**Exit:** a fresh environment can deploy a pinned version and reproduce a known analysis from a permitted sample clip.

## Phase 5 — Expand to training sessions

Only after the shot-put loop works: add gym and plyometric exercise logs, then connect wearable/cardio data where available. Treat each activity as its own task with its own labels and evaluation; do not assume one model understands every workout.

## Working cadence

Each phase should produce a reviewable artifact and an evaluation result before the next phase expands scope. Keep model metrics separate from product usefulness: a technically accurate event detector may still fail to help an athlete make a better training decision.
