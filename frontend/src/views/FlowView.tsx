import { useEffect, useRef, useState } from "react";
import { estimateFlow, flowSample, type FlowMethod, type FlowResult, type FlowSample } from "../api/client";
import LineChart from "../components/LineChart";

// Smoothness slider (0..100) on a log scale from 0.3 to 10, the measured useful range:
// below 0.3 the problem is ill-posed and noise invents motion.
const S_MIN = 0.3;
const S_MAX = 10;
const smoothnessFor = (p: number) => S_MIN * (S_MAX / S_MIN) ** (p / 100);
const DEFAULT_POSITION = Math.round((100 * Math.log(1 / S_MIN)) / Math.log(S_MAX / S_MIN)); // ≈ 1
const DEBOUNCE_MS = 250;

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

type Source = { kind: "sample"; motion: number; noise: number } | { kind: "photos"; frame1: Blob; frame2: Blob };

/** The colour key: direction = hue (right = red, clockwise like the image's y axis), speed = saturation. */
function Wheel({ maxMagnitude }: { maxMagnitude: number }) {
  return (
    <div className="row small muted">
      <span className="flow-wheel" aria-hidden="true" />
      <span>
        Colour = direction, strength = speed. Full colour = {maxMagnitude.toFixed(2)} px; white = no motion.
      </span>
    </div>
  );
}

function FlowPanel({ title, result, arrows }: { title: string; result: FlowResult; arrows: boolean }) {
  const scale = 16 / Math.max(result.max_magnitude, 1e-6) / 2; // the fastest arrow spans half a cell
  return (
    <figure className="flow-figure">
      <figcaption className="small">
        <strong>{title}</strong>
      </figcaption>
      <div className="flow-stack">
        <img src={result.flow_png} alt={`${title}: flow colour map`} />
        {arrows && (
          <svg viewBox={`0 0 ${result.width} ${result.height}`} aria-hidden="true">
            {result.arrows.map(([x, y, u, v]) => (
              <line key={`${x}-${y}`} x1={x} y1={y} x2={x + u * scale} y2={y + v * scale} />
            ))}
          </svg>
        )}
      </div>
      <div className="flow-masks">
        <figure>
          <img src={result.outliers_png} alt={`${title}: data outliers`} />
          <figcaption className="small muted">
            Data outliers ({pct(result.stages[result.stages.length - 1].outlier_fraction)})
          </figcaption>
        </figure>
        <figure>
          <img src={result.discontinuities_png} alt={`${title}: motion discontinuities`} />
          <figcaption className="small muted">Motion discontinuities (noise adds false marks)</figcaption>
        </figure>
      </div>
    </figure>
  );
}

export default function FlowView() {
  const [motion, setMotion] = useState(2);
  const [noise, setNoise] = useState(0.03);
  const [photos, setPhotos] = useState<{ frame1?: File; frame2?: File }>({});
  const [usePhotos, setUsePhotos] = useState(false);
  const [position, setPosition] = useState(DEFAULT_POSITION);
  const [compare, setCompare] = useState(true);
  const [arrows, setArrows] = useState(false);
  const [robust, setRobust] = useState<FlowResult | FlowSample | null>(null);
  const [quadratic, setQuadratic] = useState<FlowResult | FlowSample | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const latest = useRef(0); // sequence number of the newest request

  const smoothness = smoothnessFor(position);
  const source: Source | null = !usePhotos
    ? { kind: "sample", motion, noise }
    : photos.frame1 && photos.frame2
      ? { kind: "photos", frame1: photos.frame1, frame2: photos.frame2 }
      : null;

  // Debounced; the sequence number moves on as soon as any input changes, so an
  // answer still in flight for old inputs is ignored (the 1992 lesson).
  useEffect(() => {
    const id = ++latest.current;
    if (!source) return;
    const timer = setTimeout(() => {
      const run = (method: FlowMethod) =>
        source.kind === "sample"
          ? flowSample(source.motion, source.noise, smoothness, method)
          : estimateFlow(source.frame1, source.frame2, smoothness, method);
      setBusy(true);
      Promise.all([run("robust"), compare ? run("quadratic") : Promise.resolve(null)]).then(
        ([r, q]) => {
          if (id !== latest.current) return;
          setRobust(r);
          setQuadratic(q);
          setError(null);
          setBusy(false);
        },
        (e: unknown) => {
          if (id !== latest.current) return;
          setError(errorText(e));
          setBusy(false);
        }
      );
    }, DEBOUNCE_MS);
    return () => clearTimeout(timer);
    // `source` is rebuilt every render from exactly these inputs, so they are the dependencies.
  }, [usePhotos, motion, noise, photos, smoothness, compare]);

  const sample = robust && "truth_png" in robust ? robust : null;
  const sampleQuadratic = quadratic && "epe" in quadratic ? quadratic : null;
  const stagePoints = (robust?.stages ?? []).map((s, i) => [i + 1, s.outlier_fraction] as [number, number]);

  function chooseSource(photos: boolean) {
    setUsePhotos(photos);
    setRobust(null); // never show one source's results next to another
    setQuadratic(null);
  }

  function choosePhoto(which: "frame1" | "frame2", file: File | undefined) {
    setPhotos((p) => ({ ...p, [which]: file }));
    setRobust(null);
    setQuadratic(null);
  }

  return (
    <div className="view">
      <section className="card">
        <h2>Robust optical flow</h2>
        <p className="muted">
          Black &amp; Anandan (1993): find how every pixel moves between two frames. The classic
          method (Horn &amp; Schunck, 1981) squares every error, so one bad measurement or the edge of
          a moving object pulls everything around it. A robust penalty stops trusting large errors:
          outliers are ignored and motion boundaries stay sharp.
        </p>
        <p className="mono">
          E = Σ λ_D·ρ(I_x·u + I_y·v + I_t, σ_D) + λ_S·Σ [ρ(u_s − u_n, σ_S) + ρ(v_s − v_n, σ_S)],
          ρ(x, σ) = log(1 + ½(x/σ)²)
        </p>
        <p className="small muted">
          Model 1993 · solver 1993 (SOR + graduated non-convexity + coarse-to-fine).
        </p>
      </section>

      <div className="cols-2">
        <section className="card">
          <h3>Frames</h3>
          <div className="row" role="radiogroup" aria-label="Frames source">
            <label className="row small">
              <input type="radio" checked={!usePhotos} onChange={() => chooseSource(false)} /> The paper's test
            </label>
            <label className="row small">
              <input type="radio" checked={usePhotos} onChange={() => chooseSource(true)} /> Two of your photos
            </label>
          </div>
          {!usePhotos ? (
            <div key="sample" className="flow-inputs">
              <p className="small muted">
                A textured square moving left over a still textured surface, with noise on frame 2.
                The true motion is known, so Eidos can score itself.
              </p>
              <label htmlFor="flow-motion" className="small">
                Motion: {motion} px
              </label>
              <input
                id="flow-motion"
                type="range"
                min={0}
                max={8}
                step={0.5}
                value={motion}
                onChange={(e) => setMotion(Number(e.target.value))}
              />
              <label htmlFor="flow-noise" className="small">
                Noise: ±{Math.round(noise * 255)} grey levels ({Math.round(noise * 100)}%)
              </label>
              <input
                id="flow-noise"
                type="range"
                min={0}
                max={0.3}
                step={0.01}
                value={noise}
                onChange={(e) => setNoise(Number(e.target.value))}
              />
            </div>
          ) : (
            <div key="photos" className="flow-inputs">
              <p className="small muted">
                Two photos of the same size, taken a moment apart (e.g. a phone burst). Small motions
                work best; they are scaled to 240 px.
              </p>
              <label className="small" htmlFor="flow-frame1">Frame 1</label>
              <input id="flow-frame1" type="file" accept="image/*" onChange={(e) => choosePhoto("frame1", e.target.files?.[0])} />
              <label className="small" htmlFor="flow-frame2">Frame 2</label>
              <input id="flow-frame2" type="file" accept="image/*" onChange={(e) => choosePhoto("frame2", e.target.files?.[0])} />
            </div>
          )}
        </section>

        <section className="card">
          <h3>Smoothness</h3>
          <label htmlFor="flow-smoothness" className="small">
            λ_S = {smoothness.toFixed(2)} (the paper: 1)
          </label>
          <input
            id="flow-smoothness"
            type="range"
            min={0}
            max={100}
            value={position}
            onChange={(e) => setPosition(Number(e.target.value))}
          />
          <p className="small muted">
            More smoothness makes neighbours move alike. Too little, and noise invents motion; too much,
            and motion boundaries blur (less so with the robust penalty).
          </p>
          <label className="row small" htmlFor="flow-compare">
            <input id="flow-compare" type="checkbox" checked={compare} onChange={(e) => setCompare(e.target.checked)} />
            Compare with quadratic (Horn &amp; Schunck)
          </label>
          <label className="row small" htmlFor="flow-arrows">
            <input id="flow-arrows" type="checkbox" checked={arrows} onChange={(e) => setArrows(e.target.checked)} />
            Show arrows
          </label>
        </section>
      </div>

      {error && <p className="error" role="alert">{error}</p>}
      {!source && <p className="muted">Choose two photos to see their motion.</p>}

      {source && (
        <section className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3>Motion</h3>
            <span className="small muted">
              {busy ? "Estimating… (about 2 s)" : robust && `Preview ${robust.width}×${robust.height}`}
            </span>
          </div>
          {robust ? (
            <>
              <Wheel maxMagnitude={robust.max_magnitude} />
              <div className="flow-grid">
                {sample && (
                  <figure className="flow-figure">
                    <figcaption className="small">
                      <strong>True motion</strong>
                    </figcaption>
                    <img src={sample.truth_png} alt="True motion" />
                    <div className="flow-masks">
                      <figure>
                        <img src={sample.frame1_png} alt="Frame 1" />
                        <figcaption className="small muted">Frame 1</figcaption>
                      </figure>
                      <figure>
                        <img src={sample.frame2_png} alt="Frame 2" />
                        <figcaption className="small muted">Frame 2</figcaption>
                      </figure>
                    </div>
                  </figure>
                )}
                <FlowPanel title="Robust (Lorentzian)" result={robust} arrows={arrows} />
                {quadratic && <FlowPanel title="Quadratic (Horn & Schunck)" result={quadratic} arrows={arrows} />}
              </div>
              {sample && (
                <p className="small">
                  Error against the true motion: robust <strong>{sample.epe.toFixed(3)} px</strong> (
                  {sample.angular_error.toFixed(2)}°)
                  {sampleQuadratic &&
                    ` · quadratic ${sampleQuadratic.epe.toFixed(3)} px (${sampleQuadratic.angular_error.toFixed(2)}°)`}
                  . Mean endpoint error, and the paper's angular error.
                </p>
              )}
              {!sample && quadratic && (
                <p className="small muted">Each map is coloured against its own top speed (99th percentile, so a few wild pixels cannot wash it out); the key shows the robust one.</p>
              )}
            </>
          ) : (
            <p className="muted">Estimating the motion…</p>
          )}
        </section>
      )}

      {robust && (
        <section className="card">
          <h3>Graduated non-convexity: outliers per stage</h3>
          <LineChart
            points={stagePoints}
            xLabel="stage"
            yLabel="data outliers"
            xName="stage"
            yMax={Math.max(0.01, ...stagePoints.map(([, y]) => y)) * 1.2}
          />
          <p className="small muted">
            The robust solve starts convex (σ large: every error trusted, like least squares) and lowers
            σ over 6 stages, letting outliers drop out gradually. The outlier threshold is |error| ≥ √2·σ_D.
          </p>
        </section>
      )}
    </div>
  );
}
