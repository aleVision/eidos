import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies API calls to the backend, so the frontend can call
// "/health" without hard-coding the backend's host or worrying about CORS in
// development. (CORS is still configured on the backend as a safety net and for
// non-proxied setups.)
// EIDOS_API_URL points the proxy at another backend (default: the local one on 8000).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: Object.fromEntries(
      ["/health", "/people", "/samples", "/model", "/recognize", "/demo", "/detect", "/tv"].map((path) => [
        path,
        process.env.EIDOS_API_URL ?? "http://127.0.0.1:8000",
      ])
    ),
  },
});