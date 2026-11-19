import { useState } from "react";

interface Props {
  points: [number, number][]; // x, y with y in 0..1
  xLabel: string;
  yLabel: string;
  marker?: number; // vertical line at this x
  markerLabel?: string;
}

const W = 520;
const H = 240;
const M = { l: 44, r: 12, t: 12, b: 34 };

/** A single-series line chart (y as a percentage) with a hover crosshair. */
export default function LineChart({ points, xLabel, yLabel, marker, markerLabel }: Props) {
  const [hover, setHover] = useState<[number, number] | null>(null);
  if (points.length === 0) return null;

  const xMax = Math.max(...points.map(([x]) => x));
  const xs = (x: number) => M.l + ((x - 1) / Math.max(1, xMax - 1)) * (W - M.l - M.r);
  const ys = (y: number) => H - M.b - y * (H - M.t - M.b);
  const xTicks = Array.from(new Set([1, ...[0.25, 0.5, 0.75, 1].map((f) => Math.round(f * xMax))]));
  const path = points.map(([x, y], i) => `${i ? "L" : "M"}${xs(x)} ${ys(y)}`).join("");

  function onMove(event: React.PointerEvent<SVGRectElement>) {
    const svg = event.currentTarget.ownerSVGElement;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const sx = ((event.clientX - rect.left) * W) / rect.width;
    let best = points[0];
    for (const p of points) if (Math.abs(xs(p[0]) - sx) < Math.abs(xs(best[0]) - sx)) best = p;
    setHover(best);
  }

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${yLabel} by ${xLabel}`}>
        {[0, 0.25, 0.5, 0.75, 1].map((v) => (
          <g key={v}>
            <line className="grid" x1={M.l} x2={W - M.r} y1={ys(v)} y2={ys(v)} />
            <text x={M.l - 6} y={ys(v) + 3} textAnchor="end">{`${v * 100}%`}</text>
          </g>
        ))}
        {xTicks.map((v) => (
          <text key={v} x={xs(v)} y={H - M.b + 14} textAnchor="middle">{v}</text>
        ))}
        <text x={(M.l + W - M.r) / 2} y={H - 2} textAnchor="middle">{xLabel}</text>
        <text x={12} y={(M.t + H - M.b) / 2} textAnchor="middle"
          transform={`rotate(-90 12 ${(M.t + H - M.b) / 2})`}>{yLabel}</text>
        {marker !== undefined && (
          <g>
            <line x1={xs(marker)} x2={xs(marker)} y1={M.t} y2={H - M.b}
              stroke="var(--accent)" strokeDasharray="3 3" />
            <text x={xs(marker) + 4} y={M.t + 10} style={{ fill: "var(--accent)" }}>{markerLabel}</text>
          </g>
        )}
        <path d={path} fill="none" stroke="var(--series)" strokeWidth={2} strokeLinejoin="round" />
        {hover && (
          <g>
            <line className="axis" x1={xs(hover[0])} x2={xs(hover[0])} y1={M.t} y2={H - M.b} />
            <circle cx={xs(hover[0])} cy={ys(hover[1])} r={5} fill="var(--series)"
              stroke="var(--panel)" strokeWidth={2} />
            <text x={Math.min(xs(hover[0]) + 8, W - 110)} y={ys(hover[1]) - 8}
              style={{ fill: "var(--ink)" }}>
              {`k = ${hover[0]} · ${(hover[1] * 100).toFixed(1)}%`}
            </text>
          </g>
        )}
        <rect x={M.l} y={M.t} width={W - M.l - M.r} height={H - M.t - M.b} fill="transparent"
          onPointerMove={onMove} onPointerLeave={() => setHover(null)} />
      </svg>
    </div>
  );
}
