# Eidos

A privacy-first visual recognition product built on classical, explainable computer vision — and grown one historical CV milestone per year.

Each version of Eidos adds a feature inspired by an important computer vision development from a successive year, starting with **Eigenfaces (Turk & Pentland, 1991)** in V1. The software runs entirely on your own machine: no cloud, no uploads.

## Project status

**Scaffold** — full-stack skeleton runs end to end (React → FastAPI → SQLite). No CV features yet; those begin in the next development step.

## Architecture

```
backend/
  eidos_core/   Pure-Python CV library. No web code — importable on its own.
  app/          FastAPI web adapter. A thin layer over eidos_core.
  tests/        Unit tests (core) + integration tests (api).
frontend/       React + Vite + TypeScript single-page app.
```

The guiding rule: **`eidos_core` never imports the web framework.** The API depends on the core, never the reverse. This keeps the computer-vision logic independently testable and reusable for the product's whole lifetime.

## Requirements

- Python 3.12+
- Node.js 20+ (for the frontend)

## Running it

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
fastapi dev app/main.py
```

The API runs at http://127.0.0.1:8000. Interactive docs at http://127.0.0.1:8000/docs.

### Frontend

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

The app runs at http://127.0.0.1:5173 and will show the live backend status.

## Roadmap

- **V1 (1991)** — Eigenfaces face recognition.
- V2 (1992) onward — a new CV milestone each year.

See `CHANGELOG.md` for version history.