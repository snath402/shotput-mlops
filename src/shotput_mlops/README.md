# Pipeline modules

The initial implementation will add small, independently reviewable components here:

- `intake`: validate metadata, inspect video properties, and calculate source hashes.
- `sampling`: extract contact sheets/key frames for manual review.
- `pose`: wrap a selected pose estimator behind a stable interface.
- `events`: store/manual-label event timestamps and later evaluate detector output.
- `reports`: render evidence-linked per-throw and session summaries.

Model choice is intentionally deferred until the footage audit in Phase 0.
