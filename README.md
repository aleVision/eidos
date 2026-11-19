# Eidos

A computer vision product that grows one historical milestone per year.

Each major version of Eidos adds a working feature built on an important computer vision development from a successive year: version 1 brings 1991, version 2 brings 1992, and so on. Every feature is implemented from the original idea and explains itself in the interface, so you can see *how* it works, not only *that* it works. Eidos runs entirely on your own machine.

## Version 1.0.0 — 1991: Eigenfaces

Based on M. Turk and A. Pentland, "Eigenfaces for Recognition", *Journal of Cognitive Neuroscience* 3(1):71–86, 1991.

- **People**: enroll people by uploading photos and drawing a box around each face, or load the ORL demo dataset.
- **How it works**: train the model, then see the mean face, the eigenfaces, how much variation they explain, and a face rebuilt from k eigenfaces.
- **Recognize**: get one of the paper's four outcomes (known person, unknown face, not a face) with both distances, and move the two thresholds to see the trade-off.

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
3. **Recognize** → click any ORL test photo, or upload your own and draw a box around the face.

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
- **1.1.0** — Automatic face detection with the paper's distance-from-face-space map.
- **2.0.0 (1992)** — The next year's milestone.

See `CHANGELOG.md` for version history.

## License

MIT — see `LICENSE`.
