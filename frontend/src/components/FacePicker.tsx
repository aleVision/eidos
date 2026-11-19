import { useEffect, useState } from "react";
import {
  ApiError,
  detectFace,
  getDetector,
  setupDetector,
  type Box,
  type DetectMethod,
  type FaceSuggestion,
} from "../api/client";
import BoxDrawer from "./BoxDrawer";

interface Props {
  file: File;
  onChange: (box: Box | null) => void;
}

type Status = "detecting" | "ready" | "not-set-up" | "enabling" | "failed";

const errorText = (e: unknown) => (e instanceof Error ? e.message : String(e));

/**
 * A photo with a suggested face box from the face map (Turk & Pentland 1991).
 * The suggestion is a best guess: the user can always drag to redraw it.
 */
export default function FacePicker({ file, onChange }: Props) {
  const [method, setMethod] = useState<DetectMethod>("normalized");
  const [suggestion, setSuggestion] = useState<FaceSuggestion | null>(null);
  const [status, setStatus] = useState<Status>("detecting");
  const [message, setMessage] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setSuggestion(null);
    setStatus("detecting");
    setMessage(null);
    getDetector()
      .then(({ available }) => {
        if (!available) throw new ApiError("Face detection is not set up yet", 409);
        return detectFace(file, method);
      })
      .then(
        (found) => {
          if (cancelled) return;
          setSuggestion(found);
          setStatus("ready");
        },
        (e: unknown) => {
          if (cancelled) return;
          if (e instanceof ApiError && e.status === 409) return setStatus("not-set-up");
          setStatus("failed");
          setMessage(errorText(e));
        }
      );
    return () => {
      cancelled = true;
    };
  }, [file, method, attempt]);

  async function enable() {
    setStatus("enabling");
    try {
      await setupDetector();
      setAttempt((n) => n + 1); // detect again, now that the detector exists
    } catch (e) {
      setStatus("failed");
      setMessage(errorText(e));
    }
  }

  return (
    <div style={{ display: "grid", gap: 8 }}>
      <BoxDrawer
        file={file}
        onChange={onChange}
        initialBox={suggestion?.box ?? null}
        overlay={suggestion?.map_png ?? null}
      />
      <div className="row small">
        <label htmlFor="detect-method">Face detection</label>
        <select
          id="detect-method"
          value={method}
          onChange={(e) => setMethod(e.target.value as DetectMethod)}
        >
          <option value="normalized">Normalized (default; not in the 1991 paper)</option>
          <option value="paper">Paper's raw ε (1991)</option>
        </select>
      </div>
      {status === "detecting" && <p className="small muted">Looking for a face…</p>}
      {status === "ready" && suggestion && (
        <p className="small muted">
          Best guess from the face map: ε = {suggestion.distance.toFixed(1)}
          {suggestion.confident
            ? ", as face-like as real faces."
            : ", less face-like than 95% of real faces."}{" "}
          Please check it; drag on the photo to redraw.
        </p>
      )}
      {status === "not-set-up" && (
        <div className="row small">
          <button className="btn" onClick={enable}>
            Enable face detection
          </button>
          <span className="muted">Downloads the ORL dataset once (about 4 MB). Or draw the box yourself.</span>
        </div>
      )}
      {status === "enabling" && <p className="small muted">Setting up face detection…</p>}
      {status === "failed" && (
        <p className="small muted">
          No suggestion ({message}). Drag a box around the face yourself.
        </p>
      )}
    </div>
  );
}
