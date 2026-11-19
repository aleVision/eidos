import { useEffect, useState } from "react";
import { getHealth } from "./api/client";
import ModelView from "./views/ModelView";
import PeopleView from "./views/PeopleView";
import RecognizeView from "./views/RecognizeView";

type Status =
  | { kind: "checking" }
  | { kind: "ok"; version: string }
  | { kind: "error"; message: string };

const TABS = [
  { id: "people", label: "People" },
  { id: "model", label: "How it works" },
  { id: "recognize", label: "Recognize" },
] as const;
type Tab = (typeof TABS)[number]["id"];

export default function App() {
  const [status, setStatus] = useState<Status>({ kind: "checking" });
  const [tab, setTab] = useState<Tab>("people");

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
            Eigenfaces · Turk &amp; Pentland, 1991
            {status.kind === "ok" && ` · v${status.version}`}
          </span>
        </div>
        <nav className="tabs" aria-label="Sections">
          {TABS.map((t) => (
            <button key={t.id} aria-current={tab === t.id ? "page" : undefined} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          ))}
        </nav>
      </header>

      {status.kind === "error" && (
        <p className="notice error" role="alert">
          Backend unreachable ({status.message}). Start it with <code>fastapi dev app/main.py</code>{" "}
          in the backend folder.
        </p>
      )}

      {tab === "people" && <PeopleView />}
      {tab === "model" && <ModelView />}
      {tab === "recognize" && <RecognizeView />}
    </main>
  );
}
