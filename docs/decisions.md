# Decisions

## 2026-10-24 — Start with shot put

The project starts with shot-put throw analysis because that is the most specific and differentiating part of the original problem. Gym, plyometrics, and cardio are later extensions with separate data and evaluation needs.

## 2026-10-24 — Evidence-linked feedback

Every generated observation should point to its source timestamps, measurements, and confidence. If the video does not support a claim, the system should say so instead of guessing.

## 2026-10-24 — Data audit before model selection

Camera views, frame rates, occlusion, and clip quality determine which approach is viable. The first milestone audits available footage before committing to a pose model or cloud platform.
