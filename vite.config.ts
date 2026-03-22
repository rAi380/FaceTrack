import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";
import { componentTagger } from "lovable-tagger";

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  server: {
    // Listen on all interfaces so http://<LAN-IP>:8080 works; HMR uses the same host as the page.
    host: true,
    port: 8080,
    strictPort: true,
    hmr: {
      overlay: false,
      clientPort: 8080,
    },
    // Optional: set VITE_API_URL=/api in .env to use proxy instead of direct Flask URL.
    proxy: {
      "/api": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, "") || "/",
      },
    },
  },
  plugins: [react(), mode === "development" && componentTagger()].filter(Boolean),
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
}));
