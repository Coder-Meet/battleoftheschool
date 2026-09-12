import { defineConfig } from "vite";

export default defineConfig({
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        // explorer.py rejects POSTs whose Origin differs from Host, so drop it on the way through
        configure: (proxy) =>
          proxy.on("proxyReq", (req) => req.removeHeader("origin")),
      },
    },
  },
});
