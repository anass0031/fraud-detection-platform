import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// Inside Docker the backend is reachable as "backend"; on the host use localhost.
const apiTarget = process.env.VITE_API_PROXY_TARGET ?? 'http://localhost:8000'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    // Windows bind mounts send no file events into Linux containers: poll instead.
    watch: { usePolling: true, interval: 300 },
    proxy: {
      '/api': { target: apiTarget, changeOrigin: true },
    },
  },
})
