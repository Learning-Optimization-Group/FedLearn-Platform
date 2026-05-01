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
    // Sourcemaps are NOT shipped to the CDN. The previous setting (`true`)
    // produced ./dist/assets/index-*.js.map alongside the bundle, which any
    // visitor could fetch and de-minify back to readable TypeScript —
    // exposing internal API paths, auth handling, and business logic.
    //
    // If you later wire up Sentry / Datadog RUM and want server-side
    // de-minification, switch to `'hidden'`: it generates the maps locally
    // (so they can be uploaded to the error tracker) but omits the
    // //# sourceMappingURL comment so the bundle never references them.
    sourcemap: false,
  },
  server: {
    port: 5173,
    strictPort: false,
    host: true,
    proxy: {
      '/api': {
        target: 'http://3.137.147.240:8081',
        changeOrigin: true,
      },
      '/ws-logs': {
        target: 'ws://3.137.147.240:8081',
        ws: true,
        changeOrigin: true,
      }
    }
  },
});

/*
 * AWS CloudFront Deployment Routing Rule (React Router 404 Fix)
 * -------------------------------------------------------------
 * When deploying a single page application (SPA) to S3 + CloudFront,
 * direct navigation to routes (e.g., /projects, /login) will return a 403/404
 * from S3 because these objects don't exist. React Router handles them on the client.
 * 
 * To fix this, create a Custom Error Response in your CloudFront Distribution:
 * 1. Go to Error Pages in your CloudFront Distribution.
 * 2. Create Custom Error Response.
 * 3. HTTP Error Code: 404 (Not Found)
 * 4. Customize Error Response: Yes
 * 5. Response Page Path: /index.html
 * 6. HTTP Response Code: 200 (OK)
 * 
 * Also ensure your S3 Bucket Policy is configured for public read if needed.
 */
