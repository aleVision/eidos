import { useEffect, useState } from "react";
import {
  eigenfaceUrl,
  getModel,
  listPeople,
  listSamples,
  meanFaceUrl,
  reconstructionUrl,
  sampleImageUrl,
  trainModel,
  type ModelInfo,
  type Person,
  type Sample,
} from "../api/client";
import LineChart from "../components/LineChart";

const SHOWN_EIGENFACES = 16;

/** Returns `value` once it has stopped changing for `ms` milliseconds. */
function useDebounced<T>(value: T, ms: number): T {
  const [settled, setSettled] = useState(value);
  useEffect(() => {
    const timer = setTimeout(() => setSettled(value), ms);
    return () => clearTimeout(timer);
  }, [value, ms]);
  return settled;
}

export default function ModelView() {
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [components, setComponents] = useState(20);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [people, setPeople] = useState<Person[]>([]);
  const [personId, setPersonId] = useState<number | null>(null);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [sampleId, setSampleId] = useState<number | null>(null);
  const [k, setK] = useState(7);
  const settledK = useDebounced(k, 150);

  useEffect(() => {
    getModel().then(setModel, () => setModel(null)); // 404 simply means "not trained yet"
    listPeople().then((list) => setPeople(list.filter((p) => p.n_samples > 0)), () => undefined);
  }, []);

  useEffect(() => {
    if (personId === null) return;
    listSamples(personId).then((list) => {
      setSamples(list);
      setSampleId(list[0]?.id ?? null);
    }, () => undefined);
  }, [personId]);

  async function train() {
    setBusy(true);
    setError(null);
    try {
      const trained = await trainModel(components);
      setModel(trained);
      setK(Math.min(k, trained.n_components));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const variance = model?.explained_variance.map((v, i) => [i + 1, v] as [number, number]) ?? [];
  const kept = model ? model.explained_variance[model.n_components - 1] : 0;

  return (
    <div className="view">
      <section className="card">
        <h2>Train face space</h2>
        <p className="muted">
          Training averages every enrolled face into the mean face Ψ, then finds the eigenvectors of
          the faces' covariance using Turk &amp; Pentland's trick: solve the small M×M matrix
          L = AᵀA instead of the huge N²×N² one (Eqs. 3–6). The top M′ eigenvectors are the
          eigenfaces.
        </p>
        <div className="row">
          <label htmlFor="components" className="small">Eigenfaces to keep (M′)</label>
          <input id="components" type="number" min={1} max={500} value={components} style={{ width: 90 }}
            onChange={(e) => setComponents(Math.max(1, Number(e.target.value) || 1))} />
          <button className="btn primary" disabled={busy} onClick={train}>
            {busy ? "Training…" : "Train"}
          </button>
          {model && (
            <span className="small muted">
              Model #{model.id}: {model.n_people} people, {model.n_samples} photos, M′ ={" "}
              {model.n_components}
            </span>
          )}
        </div>
        {model?.stale && (
          <p className="notice small">People or photos changed since this model was trained. Train again to include them.</p>
        )}
        {error && <p className="error" role="alert">{error}</p>}
      </section>

      {!model ? (
        <p className="muted">No model yet. Enroll at least 2 people (4+ photos in total) and train.</p>
      ) : (
        <>
          <div className="cols">
            <section className="card">
              <h3>Mean face Ψ</h3>
              <div className="plate">
                <img className="face" src={meanFaceUrl(model.id)} alt="Mean face" />
              </div>
              <p className="small muted">The average of all {model.n_samples} training faces, after the Gaussian background window.</p>
            </section>
            <section className="card">
              <h3>Eigenfaces u₁…u{Math.min(SHOWN_EIGENFACES, model.n_components)}</h3>
              <div className="plate faces">
                {Array.from({ length: Math.min(SHOWN_EIGENFACES, model.n_components) }, (_, i) => (
                  <figure key={i}>
                    <img className="face" src={eigenfaceUrl(model.id, i + 1)} alt={`Eigenface ${i + 1}`} />
                    <figcaption style={{ color: "#9aa4b1" }}><span>u{i + 1}</span></figcaption>
                  </figure>
                ))}
              </div>
              <p className="small muted">
                Each eigenface is a direction in which the training faces differ. Grey is zero;
                light and dark show where pixels move together. They need not look like eyes or noses.
              </p>
            </section>
          </div>

          <div className="cols-2">
            <section className="card">
              <h3>Variance explained</h3>
              <LineChart points={variance} xLabel="k (eigenfaces)" yLabel="cumulative variance"
                marker={model.n_components} markerLabel={`M′ = ${model.n_components}`} />
              <p className="small muted">
                The kept {model.n_components} eigenfaces explain {(kept * 100).toFixed(1)}% of the
                variation among the training faces (λ_k, Eq. 1).
              </p>
            </section>

            <section className="card">
              <h3>Rebuild a face from k eigenfaces</h3>
              <div className="row">
                <label htmlFor="recon-person" className="small">Person</label>
                <select id="recon-person" value={personId ?? ""}
                  onChange={(e) => setPersonId(e.target.value ? Number(e.target.value) : null)}>
                  <option value="">Choose…</option>
                  {people.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
                </select>
                {samples.map((s) => (
                  <button key={s.id} className="btn small" aria-pressed={s.id === sampleId}
                    style={s.id === sampleId ? { borderColor: "var(--accent)" } : undefined}
                    onClick={() => setSampleId(s.id)}>
                    #{s.id}
                  </button>
                ))}
              </div>
              {sampleId !== null && (
                <>
                  <div className="cols-2">
                    <figure style={{ margin: 0 }}>
                      <img className="face" src={sampleImageUrl(sampleId)} alt="Original photo" style={{ objectFit: "contain" }} />
                      <figcaption className="small muted">original photo</figcaption>
                    </figure>
                    <figure style={{ margin: 0 }}>
                      <img className="face" src={reconstructionUrl(model.id, sampleId, settledK)} alt={`Rebuilt from ${settledK} eigenfaces`} />
                      <figcaption className="small muted">Ψ + Σ ω_i u_i, i ≤ {settledK}</figcaption>
                    </figure>
                  </div>
                  <label htmlFor="k" className="small">k = {k}</label>
                  <input id="k" type="range" min={0} max={model.n_components} value={k}
                    onChange={(e) => setK(Number(e.target.value))} />
                  <p className="small muted">
                    k = 0 gives the mean face. Each eigenface adds weight ω_k = u_kᵀ(Γ − Ψ) (Eq. 7).
                  </p>
                </>
              )}
            </section>
          </div>
        </>
      )}
    </div>
  );
}
