# Eidos

A computer vision product that grows one historical milestone per year.

Each major version of Eidos adds a working feature built on an important computer vision development from a successive year: version 1 brings 1991, version 2 brings 1992, version 3 brings 1993, and so on. Every feature is implemented from the original idea and explains itself in the interface, so you can see *how* it works, not only *that* it works. Eidos runs entirely on your own machine.

The interface is a timeline: pick a year, use that year's milestone.

| Year | Milestone | Since |
|------|-----------|-------|
| 1991 | Eigenfaces: recognize and find faces | 1.0.0 |
| 1992 | Total variation: remove noise, keep edges | 2.0.0 |
| 1993 | Robust optical flow: see how every pixel moves | 3.0.0 |

## 1991: Eigenfaces

Based on M. Turk and A. Pentland, "Eigenfaces for Recognition", *Journal of Cognitive Neuroscience* 3(1):71–86, 1991.

- **People**: enroll people by uploading photos (Eidos suggests the face box with the paper's face map; you check or redraw it), or load the ORL demo dataset.
- **How it works**: train the model, then see the mean face, the eigenfaces, how much variation they explain, and a face rebuilt from k eigenfaces.
- **Recognize**: get one of the paper's four outcomes (known person, unknown face, not a face) with both distances, and move the two thresholds to see the trade-off.
- **Face detection** (1.1.0): the face map ε(x, y) suggests where the face is. By default each box is normalized for brightness and contrast first (not in the 1991 paper; the paper's raw ε is selectable for comparison).

## 1992: Total variation denoising

Based on L. I. Rudin, S. Osher and E. Fatemi, "Nonlinear total variation based noise removal algorithms", *Physica D* 60:259–268, 1992.

- Of all images close to the noisy photo, keep the one whose brightness changes least overall: noise goes, edges stay sharp.
- Upload a photo or use the synthetic sample (with adjustable noise), move the strength slider, and compare before/after with a draggable divider. See the noise that was removed and the energy falling per iteration.
- Download the cleaned photo as PNG (up to 2048 px on the longer side, about 23 s; up to 45 s at the strongest settings).
- Model 1992 · solver 2011: the minimum is found with Chambolle & Pock's accelerated primal-dual algorithm, checked in the tests against the paper's gradient flow.
- Limits: strong settings cause "staircasing" (gradients become flat steps) and lower the contrast of small details.

## 1993: Robust optical flow

Based on M. J. Black and P. Anandan, "A framework for the robust estimation of optical flow", *ICCV* 1993 (expanded in *CVIU* 63(1):75–104, 1996).

- Find how every pixel moves between two frames. Squared errors (Horn & Schunck, 1981) let one bad measurement or a moving object's edge pull everything; a robust penalty stops trusting large errors, so motion boundaries stay sharp.
- Use the paper's test scene (a textured square moving over a still surface, with adjustable motion and noise, scored against the true motion) or two photos of your own taken a moment apart.
- Compare robust and quadratic side by side, with arrows, the data outliers ("brightness constancy fails here") and the motion discontinuities, and watch outliers drop out over the 6 graduated non-convexity stages.
- Model 1993 · solver 1993: the paper's over-relaxation, graduated non-convexity and coarse-to-fine warping. The CHANGELOG lists where Eidos departs from the paper's numbers, and why (measured).
- Limits: grey only, 240 px previews (about 2 s), motions of a few pixels; brightness changes and textureless areas break the assumptions.

## Architecture

```
backend/
  eidos_core/   Pure-Python CV library (numpy). No web code — importable on its own.
  app/          FastAPI web adapter. A thin layer over eidos_core.
  tests/        Unit tests (core) + integration tests (API).
frontend/       React + Vite + TypeScript single-page app.
```

The guiding rule: **`eidos_core` never imports the web framework.** The API depends on the core, never the reverse. This keeps the computer vision logic independently testable and reusable for the product's whole lifetime.

## Requirements

- Python 3.12+
- Node.js 20+

Works on Linux, macOS and Windows.

## Running it

### Backend

```bash
cd backend
python3.12 -m venv .venv          # Windows: py -3.12 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
pip install -r requirements.txt
fastapi dev app/main.py
```

The API runs at http://127.0.0.1:8000. Interactive docs at http://127.0.0.1:8000/docs.

Upgrading from the scaffold (0.0.0)? Delete `backend/data/eidos.db` first: the schema changed and there are no migrations yet.

### Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173.

### First run

1. **People** → **Load ORL demo (30 people)**. This downloads the dataset once (about 4 MB) into `backend/data/datasets/`.
2. **How it works** → **Train**.
3. **Recognize** → click any ORL test photo, or upload your own photo.
4. When you upload a photo, click **Enable face detection** once: from then on Eidos suggests the face box (check it, or drag to redraw). Tick **Show face map** to see ε at every position.

For 1992, choose **1992 · Total variation** in the timeline and click **Use the sample**. For 1993, choose **1993 · Robust optical flow**: the paper's test runs at once; try more noise, or two burst photos of your own.

### Tests and checks

```bash
cd backend
pytest                       # unit + API tests
pytest -m dataset -s         # opt-in: replicate on real ORL data (downloads it)
ruff check . && ruff format --check . && mypy app eidos_core
cd ../frontend && npm run build
```

## Configuration

Settings can be overridden with `EIDOS_*` environment variables or `backend/.env`, for example `EIDOS_GAUSSIAN_WINDOW=false` or `EIDOS_DEFAULT_COMPONENTS=30`. See `backend/app/config.py`.

## Data and credits

- Your photos, models and the downloaded demo dataset live in `backend/data/` on your machine and are never committed or uploaded.
- Demo face images: ORL Database of Faces. Credit: Olivetti Research Laboratory (AT&T Laboratories Cambridge).

## Roadmap

- **1.0.0 (1991)** — Eigenfaces recognition, with manual face boxes. ✅
- **1.1.0 (1991)** — Suggested face boxes with the paper's distance-from-face-space map. ✅
- **2.0.0 (1992)** — Total variation denoising (Rudin, Osher & Fatemi). ✅
- **3.0.0 (1993)** — Robust optical flow (Black & Anandan). ✅
- **4.0.0 (1994)** — The next year's milestone.

See `CHANGELOG.md` for version history.

## License

MIT — see `LICENSE`.
