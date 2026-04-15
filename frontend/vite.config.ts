import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
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
