import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dev server proxies API calls to the backend, so the frontend can call
// "/health" without hard-coding the backend's host or worrying about CORS in
// development. (CORS is still configured on the backend as a safety net and for
// non-proxied setups.)
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": "http://127.0.0.1:8000",
    },
  },
});