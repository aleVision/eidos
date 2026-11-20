import { useEffect, useRef, useState } from "react";
import { denoiseDownload, denoisePreview, sampleUrl, type DenoisePreview } from "../api/client";
import CompareSlider from "../components/CompareSlider";
import LineChart from "../components/LineChart";

// Strength slider (0 = gentle, 100 = strong) on a log scale of λ from 64 to 1:
// across this whole range, the default 100-200 solver iterations land near the minimum (tested). Larger λ stays closer
// to the photo; smaller λ removes more.
const LAM_GENTLE = 64;
const LAM_STRONG = 1;
const lambdaFor = (s: number) => LAM_GENTLE * (LAM_STRONG / LAM_GENTLE) ** (s / 100);
const DEBOUNCE_MS = 200;

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

export default function TotalVariationView() {
  const [source, setSource] = useState<Blob | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [noise, setNoise] = useState(25);
  const [strength, setStrength] = useState(50);
  const [result, setResult] = useState<DenoisePreview | null>(null);
  const [busy, setBusy] = useState(false);
  const [showNoise, setShowNoise] = useState(false);
  const [downloading, setDownloading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const latest = useRef(0); // sequence number of the newest preview request

  const lam = lambdaFor(strength);

  function chooseSource(blob: Blob) {
    setSource(blob);
    setResult(null);
    setError(null);
  }

  useEffect(() => {
    if (!source) return setSourceUrl(null);
    const url = URL.createObjectURL(source);
    setSourceUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [source]);

  // Preview: debounced, and only the newest request may update the screen.
  // The sequence number moves on as soon as the photo or λ changes, so an answer
  // still in flight for the old settings is ignored even before the new request.
  useEffect(() => {
    const id = ++latest.current;
    if (!source) return;
    const timer = setTimeout(() => {
      setBusy(true);
      denoisePreview(source, lam).then(
        (preview) => {
          if (id !== latest.current) return; // a newer request is on its way
          setResult(preview);
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
  }, [source, lam]);

  async function loadSample() {
    try {
      const response = await fetch(sampleUrl(noise));
      chooseSource(await response.blob());
    } catch (e) {
      setError(errorText(e));
    }
  }

  async function download() {
    if (!source) return;
    setDownloading(true);
    setError(null);
    try {
      const url = URL.createObjectURL(await denoiseDownload(source, lam));
      const link = document.createElement("a");
      link.href = url;
      link.download = "eidos-denoised.png";
      link.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setError(errorText(e));
    } finally {
      setDownloading(false);
    }
  }

  const energies = result?.energies ?? [];
  const energyPoints = energies.map((e, i) => [i + 1, e] as [number, number]);

  return (
    <div className="view">
      <section className="card">
        <h2>Total variation denoising</h2>
        <p className="muted">
          Rudin, Osher &amp; Fatemi (1992): of all images close to the noisy photo f, keep the one
          whose brightness changes least overall. Noise changes everywhere and costs a lot; an edge
          costs only its height. So noise goes and edges stay sharp, which blurring cannot do.
        </p>
        <p className="mono">E(u) = Σ|∇u| + (λ/2)·Σ(u − f)²</p>
        <p className="small muted">Model 1992 · solver 2011 (Chambolle &amp; Pock's accelerated primal-dual algorithm).</p>
      </section>

      <div className="cols-2">
        <section className="card">
          <h3>Your photo</h3>
          <input
            id="tv-file"
            type="file"
            accept="image/*"
            onChange={(e) => e.target.files?.[0] && chooseSource(e.target.files[0])}
          />
          <div className="row">
            <button className="btn" onClick={loadSample}>
              Use the sample
            </button>
            <label htmlFor="tv-noise" className="small">
              with noise σ = {noise}
            </label>
            <input
              id="tv-noise"
              type="range"
              min={0}
              max={80}
              value={noise}
              style={{ width: 140 }}
              onChange={(e) => setNoise(Number(e.target.value))}
            />
          </div>
          <p className="small muted">The sample is a synthetic scene: flat shapes, a gradient, thin lines and text.</p>
        </section>

        <section className="card">
          <h3>Strength</h3>
          <label htmlFor="tv-strength" className="small">
            {strength < 34 ? "Gentle" : strength < 67 ? "Medium" : "Strong"} · λ = {lam.toFixed(1)}
          </label>
          <input
            id="tv-strength"
            type="range"
            min={0}
            max={100}
            value={strength}
            onChange={(e) => setStrength(Number(e.target.value))}
          />
          <p className="small muted">
            Smaller λ removes more noise; very strong settings turn gentle gradients into flat steps
            ("staircasing", TV's typical artifact).
          </p>
          <div className="row">
            <button className="btn primary" disabled={!source || downloading} onClick={download}>
              {downloading ? "Working… (up to ~45 s)" : "Download cleaned photo"}
            </button>
            <span className="small muted">PNG, up to 2048 px on the longer side.</span>
          </div>
        </section>
      </div>

      {error && <p className="error" role="alert">{error}</p>}

      {source && sourceUrl && (
        <div className="cols-2">
          <section className="card">
            <div className="row" style={{ justifyContent: "space-between" }}>
              <h3>Before / after</h3>
              <span className="small muted">{busy ? "Updating…" : result && `Preview ${result.width}×${result.height}`}</span>
            </div>
            {result ? (
              <CompareSlider before={sourceUrl} after={result.image_png} beforeLabel="Noisy" afterLabel="Cleaned" />
            ) : (
              <p className="muted">Cleaning the preview…</p>
            )}
            <label className="small row" htmlFor="tv-show-noise">
              <input id="tv-show-noise" type="checkbox" checked={showNoise} onChange={(e) => setShowNoise(e.target.checked)} />
              Show removed noise (f − u, amplified ×4; mid-grey = nothing removed)
            </label>
            {showNoise && result && <img className="tv-noise" src={result.noise_png} alt="Removed noise" />}
          </section>

          <section className="card">
            <h3>Energy per iteration</h3>
            {energyPoints.length > 0 && (
              <LineChart
                points={energyPoints}
                xLabel="iteration"
                yLabel="energy E(u)"
                xName="iteration"
                yMax={Math.max(...energies) * 1.05}
                yFormat={(v) => (v >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 0 : 1)}k` : v.toFixed(0))}
              />
            )}
            {energies.length > 0 && (
              <p className="small muted">
                The solver lowers E(u) overall; an accelerated method may wobble slightly on the way.
                Eidos stops after {energies.length} iterations (more for stronger settings); the tests
                check that this lands within a few grey levels (at most about 7 of 255) of the true
                minimum across the whole strength range. A small change per step alone would not
                prove that.
              </p>
            )}
          </section>
        </div>
      )}
    </div>
  );
}
