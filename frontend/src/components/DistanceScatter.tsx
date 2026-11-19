import type { Thresholds } from "../api/client";

interface Props {
  personDistance: number; // ε_k to the nearest person
  faceDistance: number; // ε, distance from face space
  thresholds: Thresholds;
  axisThresholds: Thresholds; // fixes the axis range, so it does not move while dragging
  onThresholdsChange: (t: Thresholds) => void;
  onThresholdsCommit: (t: Thresholds) => void;
}

const W = 520;
const H = 300;
const M = { l: 50, r: 12, t: 12, b: 36 };

/** The paper's four quadrants (Eq. 8 against Eq. 9), shared by the plot and the result card. */
export function quadrantOf(personDistance: number, faceDistance: number, t: Thresholds): 1 | 2 | 3 | 4 {
  const nearFace = faceDistance <= t.face_space;
  const nearPerson = personDistance <= t.person;
  if (nearFace) return nearPerson ? 1 : 2;
  return nearPerson ? 3 : 4;
}

function logScale(lo: number, hi: number, a: number, b: number) {
  return (v: number) => a + ((Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * (b - a);
}

/**
 * Where this image lands relative to the two thresholds, on log axes.
 * Sliders move the thresholds live; releasing a slider saves them.
 */
export default function DistanceScatter(props: Props) {
  const { personDistance, faceDistance, thresholds: t } = props;
  const a = props.axisThresholds;
  const xLo = Math.min(personDistance, a.person) / 4;
  const xHi = Math.max(personDistance, a.person) * 4;
  const yLo = Math.min(faceDistance, a.face_space) / 4;
  const yHi = Math.max(faceDistance, a.face_space) * 4;
  const xs = logScale(xLo, xHi, M.l, W - M.r);
  const ys = logScale(yLo, yHi, H - M.b, M.t);
  const X = xs(t.person);
  const Y = ys(t.face_space);
  const q = quadrantOf(personDistance, faceDistance, t);

  // Sliders run in log space so both small and large distances are reachable.
  const toSlider = (v: number, lo: number, hi: number) => (Math.log(v / lo) / Math.log(hi / lo)) * 1000;
  const fromSlider = (s: number, lo: number, hi: number) => lo * Math.exp((s / 1000) * Math.log(hi / lo));

  const corner = [
    { n: 1, x: M.l + 6, y: H - M.b - 6, anchor: "start" },
    { n: 2, x: W - M.r - 6, y: H - M.b - 6, anchor: "end" },
    { n: 3, x: M.l + 6, y: M.t + 14, anchor: "start" },
    { n: 4, x: W - M.r - 6, y: M.t + 14, anchor: "end" },
  ] as const;

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img"
        aria-label={`Distance to nearest person against distance from face space: quadrant ${q}`}>
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={Math.max(0, Y - M.t)} fill="var(--bad)" opacity={0.07} />
        <rect x={M.l} y={Y} width={Math.max(0, X - M.l)} height={Math.max(0, H - M.b - Y)} fill="var(--ok)" opacity={0.09} />
        <rect x={X} y={Y} width={Math.max(0, W - M.r - X)} height={Math.max(0, H - M.b - Y)} fill="var(--warn)" opacity={0.07} />
        <line className="axis" x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} />
        <line className="axis" x1={M.l} x2={M.l} y1={M.t} y2={H - M.b} />
        <line x1={X} x2={X} y1={Y} y2={H - M.b} stroke="var(--ink)" strokeDasharray="5 4" />
        <line x1={M.l} x2={W - M.r} y1={Y} y2={Y} stroke="var(--ink)" strokeDasharray="5 4" />
        {corner.map((c) => (
          <text key={c.n} x={c.x} y={c.y} textAnchor={c.anchor}
            style={{ fill: c.n === q ? "var(--ink)" : "var(--ink-3)", fontWeight: c.n === q ? 700 : 400 }}>
            {c.n}
          </text>
        ))}
        <text x={X + 4} y={H - M.b - 6}>θ person</text>
        <text x={W - M.r - 4} y={Y - 5} textAnchor="end">θ face space</text>
        <circle cx={xs(personDistance)} cy={ys(faceDistance)} r={7} fill="var(--accent)"
          stroke="var(--panel)" strokeWidth={2} />
        <text x={(M.l + W - M.r) / 2} y={H - 4} textAnchor="middle">ε_k · distance to nearest person (log) →</text>
        <text x={12} y={(M.t + H - M.b) / 2} textAnchor="middle"
          transform={`rotate(-90 12 ${(M.t + H - M.b) / 2})`}>ε · distance from face space (log) →</text>
      </svg>
      <div className="cols-2">
        <label className="small">
          θ person = {t.person.toFixed(0)}
          <input type="range" min={0} max={1000} value={toSlider(t.person, xLo, xHi)}
            onChange={(e) => props.onThresholdsChange({ ...t, person: fromSlider(+e.target.value, xLo, xHi) })}
            onPointerUp={() => props.onThresholdsCommit(t)} onKeyUp={() => props.onThresholdsCommit(t)} />
        </label>
        <label className="small">
          θ face space = {t.face_space.toFixed(0)}
          <input type="range" min={0} max={1000} value={toSlider(t.face_space, yLo, yHi)}
            onChange={(e) => props.onThresholdsChange({ ...t, face_space: fromSlider(+e.target.value, yLo, yHi) })}
            onPointerUp={() => props.onThresholdsCommit(t)} onKeyUp={() => props.onThresholdsCommit(t)} />
        </label>
      </div>
    </div>
  );
}
