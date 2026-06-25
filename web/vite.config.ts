import path from "path"
import { defineConfig } from "vite"
import react from "@vitejs/plugin-react"
import tailwindcss from "@tailwindcss/vite"

// base: "./" -> relative Asset-Pfade, damit dist/index.html auch per file://
// (im pywebview-Fenster, python -m f1_rl.desktop) lädt, nicht nur über HTTP.
// Das Frontend spricht über die pywebview-Bridge mit Python (kein Server/Proxy).
export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
})
