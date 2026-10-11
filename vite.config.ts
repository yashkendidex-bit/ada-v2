import { defineConfig } from "vite";

// Security headers for every page the dev server and preview serve. The
// window is frameless and loads these pages directly, so framing must stay
// blocked; the rest are standard hardening.
const SECURITY_HEADERS = {
  "X-Frame-Options": "DENY",
  "Content-Security-Policy": "frame-ancestors 'none'",
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "no-referrer",
};

export default defineConfig({
  // Relative asset paths: the packaged Electron build loads dist/index.html
  // from disk (file://), where absolute paths would not resolve.
  base: "./",
  server: {
    // 5174 is what electron/main.js loads in development.
    port: 5174,
    strictPort: true,
    headers: SECURITY_HEADERS,
  },
  preview: {
    headers: SECURITY_HEADERS,
  },
  build: {
    outDir: "dist",
  },
});
