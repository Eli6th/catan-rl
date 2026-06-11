import { defineConfig } from "vite";

export default defineConfig({
  server: {
    proxy: {
      "/rustapi": {
        target: "http://127.0.0.1:5050",
        changeOrigin: true,
        secure: false,
      },
      "/api": {
        target: "http://127.0.0.1:5000",
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: "dist",
  },
});
