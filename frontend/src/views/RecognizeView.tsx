import { useEffect, useState } from "react";
import {
  listTestPhotos,
  recognize,
  setThresholds,
  type Box,
  type Recognition,
  type TestPhoto,
  type Thresholds,
} from "../api/client";
import BoxDrawer from "../components/BoxDrawer";
import DistanceScatter, { quadrantOf } from "../components/DistanceScatter";

const ORL_BOX: Box = { x: 0, y: 0, w: 92, h: 112 }; // ORL photos are already face crops

// Turk & Pentland's four cases, in their words where possible.
const QUADRANTS = {
  1: { title: "Known person", cls: "known", text: "Near face space and near a known face class: recognized and identified." },
  2: { title: "Unknown face", cls: "unknown_face", text: "Near face space but not near any known face class: a face, but someone not enrolled." },
  3: { title: "Not a face", cls: "not_a_face", text: "Distant from face space but near a face class: the classic false positive, caught by ε." },
  4: { title: "Not a face", cls: "not_a_face", text: "Distant from face space and from every face class." },
} as const;

const TOP_N = 5;

export default function RecognizeView() {
  const [file, setFile] = useState<File | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  const [photos, setPhotos] = useState<TestPhoto[]>([]);
  const [result, setResult] = useState<Recognition | null>(null);
  const [thresholds, setLocal] = useState<Thresholds | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listTestPhotos().then(setPhotos, () => setPhotos([]));
  }, []);

  async function run(image: Blob, faceBox: Box) {
    setBusy(true);
    setError(null);
    try {
      const r = await recognize(image, faceBox);
      setResult(r);
      setLocal(r.thresholds);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function runTestPhoto(photo: TestPhoto) {
    const response = await fetch(photo.url);
    await run(await response.blob(), ORL_BOX);
  }

  async function commit(t: Thresholds) {
    try {
      await setThresholds(t);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  const q = result && thresholds ? quadrantOf(result.nearest.distance, result.face_space_distance, thresholds) : null;
  const info = q ? QUADRANTS[q] : null;
  const top = result?.distances.slice(0, TOP_N) ?? [];
  const barMax = Math.max(...top.map((d) => d.distance), thresholds?.person ?? 0) * 1.05;

  return (
    <div className="view">
      <div className="cols-2">
        <section className="card">
          <h2>Your photo</h2>
          <input id="query-file" type="file" accept="image/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
          {file && <BoxDrawer file={file} onChange={setBox} />}
          <button className="btn primary" disabled={busy || !file || !box}
            onClick={() => file && box && run(file, box)}>
            {busy ? "Recognizing…" : "Recognize"}
          </button>
        </section>

        <section className="card">
          <h2>Or an ORL test photo</h2>
          {photos.length === 0 ? (
            <p className="muted small">Load the ORL demo on the People tab to get test photos.</p>
          ) : (
            <>
              <p className="small muted">
                None of these were used for training. “Enrolled” photos show people the model knows;
                “stranger” photos show 10 people it has never seen.
              </p>
              <div className="faces" style={{ maxHeight: 320, overflowY: "auto" }}>
                {photos.map((p) => (
                  <figure key={p.url}>
                    <button className="btn" style={{ padding: 0, border: 0 }} disabled={busy}
                      onClick={() => runTestPhoto(p)} aria-label={`Recognize ORL s${p.subject} photo ${p.index}`}>
                      <img className="face" src={p.url} alt="" loading="lazy" />
                    </button>
                    <figcaption>
                      <span>s{p.subject}/{p.index}</span>
                      <span>{p.enrolled ? "enrolled" : "stranger"}</span>
                    </figcaption>
                  </figure>
                ))}
              </div>
            </>
          )}
        </section>
      </div>

      {error && <p className="error" role="alert">{error}</p>}

      {result && thresholds && info && (
        <section className="card" aria-live="polite">
          <div className="cols-2">
            <div style={{ display: "grid", gap: 14 }}>
              <div className={`outcome ${info.cls}`}>
                <span className="small muted">Quadrant {q}</span>
                <strong>{q === 1 ? result.nearest.name : info.title}</strong>
                <span className="small">{info.text}</span>
              </div>
              {result.stale && (
                <p className="notice small">The model is older than your latest changes. Train again to include them.</p>
              )}
              <div className="cols-2">
                <figure style={{ margin: 0 }}>
                  <img className="face" src={result.input_png} alt="Prepared input face" />
                  <figcaption className="small muted">input Γ (cropped, windowed)</figcaption>
                </figure>
                <figure style={{ margin: 0 }}>
                  <img className="face" src={result.projection_png} alt="Projection onto face space" />
                  <figcaption className="small muted">projection Ψ + Φ_f</figcaption>
                </figure>
              </div>
              <table>
                <thead>
                  <tr><th>nearest people</th><th>ε_k (Eq. 8)</th><th style={{ width: "40%" }} /></tr>
                </thead>
                <tbody>
                  {top.map((d) => (
                    <tr key={d.person_id}>
                      <td>{d.name}</td>
                      <td>{d.distance.toFixed(0)}</td>
                      <td style={{ position: "relative" }}>
                        <div className="bar" style={{ width: `${(d.distance / barMax) * 100}%` }} />
                        <div style={{ position: "absolute", top: 0, bottom: 0, left: `${(thresholds.person / barMax) * 100}%`,
                          borderLeft: "2px dashed var(--ink)" }} title="θ person" />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="small">
                ε = <span className="mono">{result.face_space_distance.toFixed(0)}</span> (distance from face space, Eq. 9),
                θ face space = <span className="mono">{thresholds.face_space.toFixed(0)}</span>.
              </p>
            </div>
            <DistanceScatter
              personDistance={result.nearest.distance}
              faceDistance={result.face_space_distance}
              thresholds={thresholds}
              axisThresholds={result.thresholds}
              onThresholdsChange={setLocal}
              onThresholdsCommit={commit}
            />
          </div>
        </section>
      )}
    </div>
  );
}
