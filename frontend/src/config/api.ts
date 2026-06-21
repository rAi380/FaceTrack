/**
 * Base URL for the Flask backend.
 * - In Vite dev, defaults to `/api` (proxied to Flask) so HTTPS UI avoids mixed-content blocks.
 * - Override with `VITE_API_URL` for production or LAN (e.g. http://host:5000).
 */
function apiBase(): string {
  const fromEnv = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "");
  if (fromEnv) {
    // HTTPS page + http:// API in .env → browser blocks or misbehaves; Vite proxy avoids that in dev.
    if (
      import.meta.env.DEV &&
      typeof window !== "undefined" &&
      window.location.protocol === "https:" &&
      fromEnv.startsWith("http:")
    ) {
      return "/api";
    }
    return fromEnv;
  }
  // Dev over HTTPS cannot call http://127.0.0.1:5000 (mixed content). Use Vite proxy /api → Flask.
  if (import.meta.env.DEV) {
    return "/api";
  }
  if (typeof window !== "undefined") {
    const h = window.location.hostname;
    if (h && h !== "localhost" && h !== "127.0.0.1") {
      return `http://${h}:5000`;
    }
  }
  return "http://127.0.0.1:5000";
}

export const API_URL = apiBase();
