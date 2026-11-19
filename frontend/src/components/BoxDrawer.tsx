import { useEffect, useRef, useState } from "react";
import type { Box } from "../api/client";

interface Props {
  file: Blob;
  onChange: (box: Box | null) => void;
}

/**
 * Shows an image and lets the user drag a face box over it.
 *
 * The box is reported in the image's natural pixels, which is what the backend
 * crops. Browsers draw EXIF-rotated photos upright, and the backend rotates
 * them the same way before cropping, so the two always agree.
 */
export default function BoxDrawer({ file, onChange }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [image, setImage] = useState<ImageBitmap | null>(null);
  const [box, setBox] = useState<Box | null>(null);
  const start = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => {
    let cancelled = false;
    setImage(null);
    setBox(null);
    onChange(null);
    // createImageBitmap applies the EXIF orientation, like an <img> would.
    createImageBitmap(file).then((bitmap) => {
      if (!cancelled) setImage(bitmap);
    }, () => !cancelled && onChange(null));
    return () => {
      cancelled = true;
    };
    // onChange is intentionally not a dependency: a new file resets the box once.
  }, [file]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !image) return;
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(image, 0, 0);
    if (box) {
      ctx.fillStyle = "rgba(0,0,0,0.45)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(image, box.x, box.y, box.w, box.h, box.x, box.y, box.w, box.h);
      ctx.strokeStyle = "#f0a43a";
      ctx.lineWidth = Math.max(2, canvas.width / 300);
      ctx.strokeRect(box.x, box.y, box.w, box.h);
    }
  }, [image, box]);

  /** Pointer position in natural image pixels, clamped to the image. */
  function toImage(event: React.PointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const scale = event.currentTarget.width / rect.width;
    const clamp = (v: number, max: number) => Math.min(Math.max(0, v), max);
    return {
      x: clamp((event.clientX - rect.left) * scale, event.currentTarget.width),
      y: clamp((event.clientY - rect.top) * scale, event.currentTarget.height),
    };
  }

  function update(event: React.PointerEvent<HTMLCanvasElement>, done: boolean) {
    if (!start.current) return;
    const p = toImage(event);
    const next = {
      x: Math.round(Math.min(start.current.x, p.x)),
      y: Math.round(Math.min(start.current.y, p.y)),
      w: Math.round(Math.abs(p.x - start.current.x)),
      h: Math.round(Math.abs(p.y - start.current.y)),
    };
    const valid = next.w >= 8 && next.h >= 8;
    setBox(valid ? next : null);
    if (done) {
      start.current = null;
      onChange(valid ? next : null);
    }
  }

  return (
    <div className="drawer">
      <canvas
        ref={canvasRef}
        aria-label="Photo: drag to draw a box around the face"
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          start.current = toImage(e);
        }}
        onPointerMove={(e) => update(e, false)}
        onPointerUp={(e) => update(e, true)}
      />
      <p className="small muted">
        {box
          ? `Face box: ${box.w} × ${box.h} px at (${box.x}, ${box.y}). Drag again to redo.`
          : "Drag a box from the top of the forehead to the chin, ear to ear."}
      </p>
    </div>
  );
}
