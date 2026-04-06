import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  define: {
    'global': {},
  },
  build: {
    minify: 'esbuild',
    sourcemap: true,
  },
  server: {
    port: 5173,
    strictPort: false,
    host: true,
  },
});
