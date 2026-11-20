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
- **3.0.0** — The 1993 milestone.

## [2.0.1] - 2026-11-20

### Fixed
- **Strong settings looked blurry.** Chambolle's 2004 solver needs far more than
  100 iterations when λ is small: at λ = 2 an edge came out about 9 px wide
  instead of 1 px, and the energy was 12% above its minimum. The blur was the
  unfinished solver, not the ROF model. Eidos now uses Chambolle & Pock's
  accelerated primal-dual algorithm (2011). Its error falls like 1/(λ·k²), so the
  number of iterations now grows like 1/√λ: 100 for λ ≥ 4, up to 200 at λ = 1.
  Across the whole slider the result is within 3 grey levels of the converged
  one (tested). Speed at λ = 8 is about the same (preview ≈ 1.7 s, 2048 px
  download ≈ 23 s); the strongest setting takes up to twice as long (≈ 2.9 s and
  ≈ 44 s).
- **The "settled / still moving" note was misleading.** A small change over the
  last iterations does not prove a slow solver is near the minimum. The note now
  states what the tests check instead.

## [2.0.0] - 2026-11-19

The 1992 milestone: total variation denoising, from L. I. Rudin, S. Osher and
E. Fatemi, "Nonlinear total variation based noise removal algorithms",
*Physica D* 60:259–268, 1992. A new year, so a new MAJOR version; the 1991
features and API are unchanged.

### Added
- `eidos_core.denoise`: the ROF model, minimizing E(u) = Σ|∇u| + (λ/2)·Σ(u − f)².
  Removes noise but keeps edges sharp, which blurring cannot do (tested against
  Gaussian blur at equal noise reduction).
- **Model 1992 · solver 2004** (replaced in 2.0.1). The energy is the 1992 contribution; to find its
  minimum Eidos uses Chambolle's dual algorithm (2004), simpler and more stable
  than the paper's gradient flow. The tests check it against that gradient flow.
- API: `POST /tv/denoise` (preview, up to 640 px, with the removed noise and the
  energy per iteration), `POST /tv/denoise.png` (download, up to 2048 px on the
  longer side), `GET /tv/sample.png` (a synthetic test scene with adjustable noise).
  The strength λ must lie between 1e-6 and 1e9 (otherwise 422).
- Frontend: a timeline with one entry per milestone year, and the 1992 tool: a
  strength slider, a before/after comparison, the removed noise, and the energy
  curve with a "settled / still moving" note.

### Known limitations
- **Staircasing:** strong settings turn gentle gradients into flat steps.
- **Contrast loss:** ROF lowers the contrast of small or thin features and shaves
  a little height off every edge.
- **Speed:** a 640 px preview takes about 1.3 s; a 2048 px download about 20 s
  (CPU, NumPy). Larger photos are scaled down to 2048 px.
- Colour channels are denoised independently (no colour TV).

## [1.1.0] - 2026-11-19

Completes the 1991 milestone with the paper's second idea: locating faces with
"face space" (Turk & Pentland 1991, Eqs. 10–15, Figure 7).

### Added
- `eidos_core.detection`: the face map ε(x, y), the distance from face space at
  every position, computed with the paper's correlation formula (M′ + 2
  correlations per map, about 340× faster than cropping every position, and tested
  against that slow version), and a multi-size search for the best face box.
- Brightness/contrast normalization of each box before measuring ε, the default
  method. **This is not in the 1991 paper.** Measured on ORL faces, the paper's raw
  ε rates flat walls and smooth gradients as more face-like than real faces;
  normalization fixes that. The paper's raw ε remains available for comparison
  (`method=paper`).
- A detector face space trained once from ORL, independent of your people and of
  retraining. API: `GET /detector`, `POST /detector`, `POST /detect`.
- Frontend: uploading a photo pre-fills a suggested face box (People and
  Recognize), with "Enable face detection", a "Show face map" overlay, and a
  method selector to compare the normalized and paper methods.

### Changed
- `prepare` can standardize a face (`standardize_face=True`); recognition does not
  use it, so 1.0.0 behavior is unchanged.
- Photos the browser cannot open (e.g. HEIC) now show a message.

### Notes
- Detection searches portrait-sized faces only (at least 35% of the photo's
  shorter side): ε is not comparable across sizes, so small background boxes
  would otherwise win. Group photos with small faces get no useful suggestion;
  draw the box yourself.
- Known limitation: when a face fills most of the photo (a close selfie), the
  suggestion often covers only part of the face, because smaller boxes score as
  more face-like. Measured on held-out ORL faces in a portrait: 13/14 found at 42%
  of the photo's width, 11/14 at 56%, 4/14 at 70%. Redraw the box when that
  happens. Pinned by `pytest -m dataset` (the 70% case is an expected failure).
- A suggestion is a best guess. "As face-like as real faces" means ε is within the
  range of 95% of held-out real faces; a wrong box can score that well too, so
  always check it.

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
- Tests: 70 unit and API tests, plus an opt-in ORL replication test
  (`pytest -m dataset`, 86.7% accuracy at M′ = 20).

### Security
- The API refuses data-changing requests sent by other websites (checked with the
  browser's Origin header), so a page you visit cannot write into a local Eidos.

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
