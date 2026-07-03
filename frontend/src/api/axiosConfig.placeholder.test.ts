import { describe, it, expect, vi, afterEach } from 'vitest';

// FE-7: axiosConfig's production host guard runs at module-load time, reading
// import.meta.env.{PROD, VITE_FEDLEARN_API_URL}. To exercise it we stub those
// values, drop the module from the registry, then dynamically re-import so the
// top-level guard re-evaluates against the stubbed env. A static top-level
// import would evaluate the module once (in the test/DEV env, where PROD is
// false) and never see the stubbed values.
async function loadWithProdEnv(url: string) {
    vi.resetModules();
    vi.stubEnv('PROD', true);
    vi.stubEnv('VITE_FEDLEARN_API_URL', url);
    return import('./axiosConfig');
}

describe('axiosConfig production host guard (FE-7)', () => {
    afterEach(() => {
        vi.unstubAllEnvs();
        vi.resetModules();
    });

    it('throws when VITE_FEDLEARN_API_URL is still the placeholder host', async () => {
        await expect(loadWithProdEnv('https://REPLACE_WITH_YOUR_API_HOST/api')).rejects.toThrow(
            /REPLACE_WITH_YOUR_API_HOST/
        );
    });

    it('throws when VITE_FEDLEARN_API_URL is empty in production', async () => {
        await expect(loadWithProdEnv('')).rejects.toThrow(/must be set for production/i);
    });

    it('throws when VITE_FEDLEARN_API_URL is a non-https origin in production', async () => {
        await expect(loadWithProdEnv('http://api.example.com/api')).rejects.toThrow(/https/i);
    });

    it('loads cleanly with a real https:// host', async () => {
        const mod = await loadWithProdEnv('https://api.example.com/api');
        expect(mod.baseURL).toBe('https://api.example.com/api');
    });
});
