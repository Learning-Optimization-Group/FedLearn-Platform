import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => {
  // Load .env.[mode] so the proxy target can vary by mode without editing this file.
  // The third arg '' loads ALL vars (not just VITE_* — proxy-only vars don't need
  // client exposure, but using the VITE_ prefix keeps intent self-documenting).
  const env = loadEnv(mode, process.cwd(), '');
  const proxyTarget = env.VITE_PROXY_TARGET;

  // Only configure the dev proxy when a target is set (i.e. ec2demo mode).
  // Full-local dev calls the backend directly via VITE_FEDLEARN_API_URL, so
  // the proxy block is dead weight there.
  const proxy = proxyTarget
    ? {
        '/api': {
          target: proxyTarget,
          changeOrigin: true,
          secure: true,
        },
        '/ws-logs': {
          target: proxyTarget.replace(/^http/, 'ws'),
          ws: true,
          changeOrigin: true,
          secure: true,
        },
      }
    : undefined;

  return {
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
      // strictPort: true — refuse to shift to a different port if 5173 is busy.
      // The backend's CORS allowlist only includes :5173, so a silent shift to
      // :5174 produces confusing "Access-Control-Allow-Credentials missing"
      // errors. Failing fast surfaces the stuck-process root cause immediately.
      strictPort: true,
      host: true,
      proxy,
    },
  };
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
