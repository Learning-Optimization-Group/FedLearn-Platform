import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    'global': {},
  },
  build: {
    minify: 'esbuild',
    sourcemap: true,
    esbuild: {
      drop: [],
    },
  },
  server: {
    proxy: {
      // This tells Vite to proxy any request starting with /api
      '/api': {
        // to your backend server
        target: 'http://18.191.187.132:8081',
        // This is important for virtual hosts
        changeOrigin: true,
        // You can rewrite the path if needed, but this is usually not necessary
        // rewrite: (path) => path.replace(/^\/api/, ''), 
      }
    }
  }
})
