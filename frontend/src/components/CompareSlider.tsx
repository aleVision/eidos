import { useState } from "react";

interface Props {
  before: string;
  after: string;
  beforeLabel?: string;
  afterLabel?: string;
}

/**
 * Two images of the same scene stacked; drag the divider (or use the slider,
 * also with the keyboard) to reveal "before" on the left and "after" on the right.
 */
export default function CompareSlider({
  before,
  after,
  beforeLabel = "Before",
  afterLabel = "After",
}: Props) {
  const [position, setPosition] = useState(50); // percent of the width showing "before"

  function dragTo(event: React.PointerEvent<HTMLDivElement>) {
    if (event.buttons !== 1) return;
    const rect = event.currentTarget.getBoundingClientRect();
    setPosition(Math.min(100, Math.max(0, ((event.clientX - rect.left) / rect.width) * 100)));
  }

  return (
    <div style={{ display: "grid", gap: 6 }}>
      <div
        className="compare"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          dragTo(e);
        }}
        onPointerMove={dragTo}
      >
        <img src={after} alt={afterLabel} draggable={false} />
        <img
          src={before}
          alt={beforeLabel}
          draggable={false}
          style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
        />
        <div className="compare-divider" style={{ left: `${position}%` }} />
        <span className="compare-label" style={{ left: 8 }}>{beforeLabel}</span>
        <span className="compare-label" style={{ right: 8 }}>{afterLabel}</span>
      </div>
      <input
        id="compare-position"
        type="range"
        min={0}
        max={100}
        value={position}
        aria-label="Divider position between before and after"
        onChange={(e) => setPosition(Number(e.target.value))}
      />
    </div>
  );
}
