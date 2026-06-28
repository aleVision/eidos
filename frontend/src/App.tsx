import { useEffect, useState } from "react";
import { getHealth } from "./api/client";

type Status =
  | { kind: "checking" }
  | { kind: "ok"; app: string; version: string }
  | { kind: "error"; message: string };

export default function App() {
  const [status, setStatus] = useState<Status>({ kind: "checking" });

  useEffect(() => {
    getHealth()
      .then((h) => setStatus({ kind: "ok", app: h.app, version: h.version }))
      .catch((e: unknown) =>
        setStatus({
          kind: "error",
          message: e instanceof Error ? e.message : "Could not reach the backend",
        })
      );
  }, []);

  return (
    <main
      style={{
        fontFamily: "system-ui, sans-serif",
        maxWidth: 560,
        margin: "10vh auto",
        padding: "0 1.5rem",
        lineHeight: 1.6,
      }}
    >
      <h1 style={{ marginBottom: "0.25rem" }}>Eidos</h1>
      <p style={{ marginTop: 0, color: "#555" }}>
        Classical computer vision, grown one milestone per year.
      </p>

      <section
        style={{
          marginTop: "2rem",
          padding: "1rem 1.25rem",
          border: "1px solid #ddd",
          borderRadius: 8,
        }}
      >
        {status.kind === "checking" && <p>Checking backend…</p>}

        {status.kind === "ok" && (
          <p>
            Backend connected — {status.app} v{status.version}.
          </p>
        )}

        {status.kind === "error" && (
          <p style={{ color: "#b00020" }}>
            Backend unreachable: {status.message}. Start it with{" "}
            <code>fastapi dev app/main.py</code> in the backend folder.
          </p>
        )}
      </section>
    </main>
  );
}