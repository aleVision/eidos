// All calls to the backend live in this module. Components never call fetch
// directly — they call these functions. That keeps the network layer in one
// place and the API contract typed.

export interface Health {
  status: string;
  app: string;
  version: string;
}

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface Person {
  id: number;
  name: string;
  is_demo: boolean;
  n_samples: number;
}

export interface Sample {
  id: number;
  person_id: number;
  box: Box;
}

export interface ModelInfo {
  id: number;
  created_at: string;
  n_people: number;
  n_samples: number;
  n_components: number;
  theta_person: number;
  theta_face: number;
  explained_variance: number[];
  stale: boolean;
}

export interface PersonDistance {
  person_id: number;
  name: string;
  distance: number;
}

export type Outcome = "known" | "unknown_face" | "not_a_face";

export interface Thresholds {
  person: number;
  face_space: number;
}

export interface Recognition {
  outcome: Outcome;
  quadrant: 1 | 2 | 3 | 4;
  nearest: PersonDistance;
  face_space_distance: number;
  distances: PersonDistance[];
  thresholds: Thresholds;
  weights: number[];
  input_png: string;
  projection_png: string;
  stale: boolean;
}

export interface TestPhoto {
  subject: number;
  index: number;
  enrolled: boolean;
  url: string;
}

export type DetectMethod = "normalized" | "paper";

export interface FaceSuggestion {
  box: Box;
  distance: number;
  confident: boolean;
  map_png: string;
}

/** An HTTP error from the API, with FastAPI's {detail} as the message. */
export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number
  ) {
    super(message);
  }
}

/** fetch wrapper: parses JSON and turns FastAPI's {detail} into an ApiError. */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { detail?: unknown };
      if (typeof body.detail === "string") message = body.detail;
    } catch {
      // not JSON: keep the generic message
    }
    throw new ApiError(message, response.status);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function json(method: string, body: unknown): RequestInit {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

function imageForm(image: Blob, box: Box): FormData {
  const form = new FormData();
  form.append("image", image, "image");
  for (const [key, value] of Object.entries(box)) form.append(key, String(Math.round(value)));
  return form;
}

export const getHealth = () => request<Health>("/health");

export const listPeople = () => request<Person[]>("/people");
export const createPerson = (name: string) => request<Person>("/people", json("POST", { name }));
export const deletePerson = (id: number) => request<void>(`/people/${id}`, { method: "DELETE" });

export const listSamples = (personId: number) => request<Sample[]>(`/people/${personId}/samples`);
export const addSample = (personId: number, image: Blob, box: Box) =>
  request<Sample>(`/people/${personId}/samples`, { method: "POST", body: imageForm(image, box) });
export const deleteSample = (id: number) => request<void>(`/samples/${id}`, { method: "DELETE" });
export const sampleImageUrl = (id: number) => `/samples/${id}/image`;

export const getModel = () => request<ModelInfo>("/model");
export const trainModel = (nComponents: number) =>
  request<ModelInfo>("/model/train", json("POST", { n_components: nComponents }));
export const setThresholds = (t: Thresholds) =>
  request<ModelInfo>("/model/thresholds", json("PATCH", t));
export const meanFaceUrl = (modelId: number) => `/model/mean.png?v=${modelId}`;
export const eigenfaceUrl = (modelId: number, k: number) => `/model/eigenfaces/${k}.png?v=${modelId}`;
export const reconstructionUrl = (modelId: number, sampleId: number, k: number) =>
  `/model/reconstruct/${sampleId}?k=${k}&v=${modelId}`;

export const recognize = (image: Blob, box: Box) =>
  request<Recognition>("/recognize", { method: "POST", body: imageForm(image, box) });

export const loadDemo = () => request<{ people_added: number }>("/demo/orl", { method: "POST" });
export const removeDemo = () => request<void>("/demo/orl", { method: "DELETE" });
export const listTestPhotos = () => request<TestPhoto[]>("/demo/orl/test-photos");

export const getDetector = () => request<{ available: boolean }>("/detector");
export const setupDetector = () => request<unknown>("/detector", { method: "POST" });
export function detectFace(image: Blob, method: DetectMethod): Promise<FaceSuggestion> {
  const form = new FormData();
  form.append("image", image, "image");
  form.append("method", method);
  return request<FaceSuggestion>("/detect", { method: "POST", body: form });
}

// 1992 · Total variation (stateless: nothing is stored)

export interface DenoisePreview {
  image_png: string;
  noise_png: string;
  energies: number[];
  lam: number;
  width: number;
  height: number;
}

function strengthForm(image: Blob, strength: number): FormData {
  const form = new FormData();
  form.append("image", image, "image");
  form.append("strength", String(strength));
  return form;
}

export const denoisePreview = (image: Blob, strength: number) =>
  request<DenoisePreview>("/tv/denoise", { method: "POST", body: strengthForm(image, strength) });

/** The cleaned photo as a PNG file (up to 2048 px on the longer side). */
export async function denoiseDownload(image: Blob, strength: number): Promise<Blob> {
  const response = await fetch("/tv/denoise.png", {
    method: "POST",
    body: strengthForm(image, strength),
  });
  if (!response.ok) throw new ApiError(`Download failed (${response.status})`, response.status);
  return response.blob();
}

export const sampleUrl = (noise: number) => `/tv/sample.png?noise=${noise}`;
