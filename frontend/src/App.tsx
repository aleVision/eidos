import { useEffect, useState } from "react";
import { getHealth } from "./api/client";
import ModelView from "./views/ModelView";
import PeopleView from "./views/PeopleView";
import RecognizeView from "./views/RecognizeView";
import FlowView from "./views/FlowView";
import TotalVariationView from "./views/TotalVariationView";

type Status =
  | { kind: "checking" }
  | { kind: "ok"; version: string }
  | { kind: "error"; message: string };

// The product grows one historical milestone per year: one entry per year.
const MILESTONES = [
  {
    year: 1991,
    title: "Eigenfaces",
    description: "Recognize and find faces in face space (Turk & Pentland)",
  },
  {
    year: 1992,
    title: "Total variation",
    description: "Remove noise while keeping edges sharp (Rudin, Osher & Fatemi)",
  },
  {
    year: 1993,
    title: "Robust optical flow",
    description: "See how every pixel moves, and where motion breaks (Black & Anandan)",
  },
] as const;
type Year = (typeof MILESTONES)[number]["year"];

const EIGENFACES_TABS = [
  { id: "people", label: "People" },
  { id: "model", label: "How it works" },
  { id: "recognize", label: "Recognize" },
] as const;
type EigenfacesTab = (typeof EIGENFACES_TABS)[number]["id"];

export default function App() {
  const [status, setStatus] = useState<Status>({ kind: "checking" });
  const [year, setYear] = useState<Year>(1991);
  const [tab, setTab] = useState<EigenfacesTab>("people");

  useEffect(() => {
    getHealth()
      .then((h) => setStatus({ kind: "ok", version: h.version }))
      .catch((e: unknown) =>
        setStatus({ kind: "error", message: e instanceof Error ? e.message : "unreachable" })
      );
  }, []);

  return (
    <main className="app">
      <header className="topbar">
        <div className="brand">
          <h1>
            Eid<span>os</span>
          </h1>
          <span className="small muted">
            A computer vision product that grows one milestone per year
            {status.kind === "ok" && ` · v${status.version}`}
          </span>
        </div>
      </header>

      <nav className="timeline" aria-label="Milestones by year">
        {MILESTONES.map((m) => (
          <button
            key={m.year}
            aria-current={year === m.year ? "page" : undefined}
            onClick={() => setYear(m.year)}
          >
            <span className="year">{m.year}</span>
            <span className="title">{m.title}</span>
            <span className="small muted">{m.description}</span>
          </button>
        ))}
      </nav>

      {status.kind === "error" && (
        <p className="notice error" role="alert">
          Backend unreachable ({status.message}). Start it with <code>fastapi dev app/main.py</code>{" "}
          in the backend folder.
        </p>
      )}

      {year === 1991 && (
        <>
          <nav className="tabs" aria-label="1991 · Eigenfaces">
            {EIGENFACES_TABS.map((t) => (
              <button
                key={t.id}
                aria-current={tab === t.id ? "page" : undefined}
                onClick={() => setTab(t.id)}
              >
                {t.label}
              </button>
            ))}
          </nav>
          {tab === "people" && <PeopleView />}
          {tab === "model" && <ModelView />}
          {tab === "recognize" && <RecognizeView />}
        </>
      )}
      {year === 1992 && <TotalVariationView />}
      {year === 1993 && <FlowView />}
    </main>
  );
}
