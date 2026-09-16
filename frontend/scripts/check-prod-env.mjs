// FE-7: prebuild sanity check.
//
// The committed `.env.production` ships a placeholder API host, so a naive
// `npm run build` would happily bake a dead origin into the bundle. This script
// runs automatically via the npm `prebuild` hook (before `vite build`) and
// aborts the build if the resolved API host is still the placeholder.
//
// Dependency-free: it reuses Vite's own `loadEnv` (already a devDependency) so
// the values it inspects match EXACTLY what the bundle would inline, including
// Vite's precedence — a real host injected via a gitignored `.env.local` or an
// exported CI env var (which override the committed `.env.production`) makes
// this pass. See `frontend/.env.example`.
import { loadEnv } from 'vite';

const PLACEHOLDER_API_HOST = 'REPLACE_WITH_YOUR_API_HOST';

// Build mode (Vite's `build` defaults to "production"). Passed by the npm
// script so this stays reusable for other modes if ever needed.
const mode = process.argv[2] || 'production';

// Only VITE_-prefixed vars are exposed to the client bundle; that's all we check.
const env = loadEnv(mode, process.cwd(), 'VITE_');

const CHECKED_VARS = ['VITE_FEDLEARN_API_URL', 'VITE_SERVER_ROOT_URL'];

const offenders = CHECKED_VARS.filter(
    (name) => typeof env[name] === 'string' && env[name].includes(PLACEHOLDER_API_HOST)
).map((name) => `  - ${name} = ${env[name]}`);

if (offenders.length > 0) {
    console.error(
        `\n[prebuild] Refusing to build in "${mode}" mode: placeholder API host ` +
        `"${PLACEHOLDER_API_HOST}" is still present in:\n` +
        offenders.join('\n') +
        '\n\nInject your real https:// API origin before building — either:\n' +
        '  - a gitignored frontend/.env.local (highest Vite precedence), or\n' +
        '  - exported VITE_FEDLEARN_API_URL / VITE_SERVER_ROOT_URL in CI.\n' +
        'See frontend/.env.example for the full variable reference.\n'
    );
    process.exit(1);
}
