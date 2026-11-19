# Changelog

All notable changes to Eidos are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A note on how we version this product, since it's part of the point:

- **MAJOR** (e.g. 1.x.x → 2.0.0) — a new historical-year CV milestone, or any
  change that breaks the existing API contract.
- **MINOR** (e.g. 1.0.0 → 1.1.0) — a backward-compatible feature added to the
  current milestone. Auto face detection will land here as **1.1.0**: it extends
  the 1991 feature without breaking it, so it is *not* a 2.0.0.
- **PATCH** (e.g. 1.0.0 → 1.0.1) — backward-compatible bug fixes only.

## [Unreleased]

### Planned
- **1.1.0** — Automatic face detection with the paper's distance-from-face-space
  "face map" (backward-compatible enhancement of the 1991 milestone).
- **2.0.0** — The 1992 milestone.

## [1.0.0] - 2026-11-19

The 1991 milestone: **Eigenfaces** (M. Turk and A. Pentland, "Eigenfaces for
Recognition", *Journal of Cognitive Neuroscience* 3(1):71–86, 1991).

### Added
- `eidos_core`: a pure-numpy implementation that follows the paper equation by
  equation: mean face, the M×M eigenvector trick (Eqs. 1–6), projection (Eq. 7),
  distance to each person (Eq. 8), distance from face space (Eq. 9), and the
  paper's four outcomes (known person, unknown face, not a face).
- Face preprocessing: crop a hand-drawn face box, grey, resize to 92×112, and an
  optional Gaussian background window (on by default).
- Data-driven threshold suggestion (leave-one-out, 95th percentile), adjustable
  in the UI.
- ORL Database of Faces demo: downloaded on request, 30 people enrolled, held-out
  and stranger test photos. Credit: Olivetti Research Laboratory (AT&T
  Laboratories Cambridge).
- API: people and face samples, training, model inspection (mean face,
  eigenfaces, explained variance), thresholds, recognition, reconstruction,
  and the ORL demo.
- Frontend with three views: **People** (enroll and draw face boxes),
  **How it works** (mean face, eigenfaces, variance curve, reconstruction
  slider) and **Recognize** (outcome, both distances, four-quadrant plot with
  draggable thresholds).
- Tests: 59 unit and API tests, plus an opt-in ORL replication test
  (`pytest -m dataset`, 86.7% accuracy at M′ = 20).

### Changed
- Backend requires Python 3.12.

### Notes
- The database schema changed since the scaffold and there are no migrations
  yet: delete `backend/data/eidos.db` before the first run of 1.0.0.
- Known limits of the method (documented, not bugs): strangers are often matched
  to an enrolled person; smooth non-face images can land close to face space;
  accuracy depends heavily on face alignment.

## [0.0.0] - 2026-06-28

### Added
- Project scaffold: FastAPI backend, SQLite via SQLAlchemy, React + Vite + TypeScript
  frontend, health endpoint, and a passing end-to-end status check.
- Tooling: ruff (lint + format) and mypy (type checking) on the backend.
