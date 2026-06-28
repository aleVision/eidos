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

### Added
- Project scaffold: FastAPI backend, SQLite via SQLAlchemy, React + Vite + TypeScript
  frontend, health endpoint, and a passing end-to-end status check.
- Tooling: ruff (lint + format) and mypy (type checking) on the backend.

### Planned
- **1.0.0** — Eigenfaces face recognition (Turk & Pentland, 1991) with manual
  face marking by the user.
- **1.1.0** — Automatic face detection (backward-compatible enhancement).