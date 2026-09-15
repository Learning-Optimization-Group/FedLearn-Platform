import { describe, it, expect, vi, beforeEach } from 'vitest';

vi.mock('../api/axiosConfig', () => ({
  default: { post: vi.fn(), get: vi.fn(), put: vi.fn(), delete: vi.fn() },
}));

import api from '../api/axiosConfig';
import { startProjectServer } from './apiServices';

// The start request body is built field by field, so a field the builder does not copy never reaches
// the backend -- silently. These pin the robust-aggregation fields through that seam.
describe('startProjectServer — request body', () => {
  beforeEach(() => {
    vi.mocked(api.post).mockReset().mockResolvedValue({ data: {} } as never);
  });

  it('passes the robust rule and its attacker share through', async () => {
    await startProjectServer('p1', {
      strategy: 'Robust', numRounds: 5, minClients: 20, robustMethod: 'BULYAN', byzantineFraction: 0.2,
    });
    expect(api.post).toHaveBeenCalledWith('/projects/p1/start', {
      strategy: 'Robust', numRounds: 5, minClients: 20, robustMethod: 'BULYAN', byzantineFraction: 0.2,
    });
  });

  it('keeps a trim ratio of zero instead of dropping it', async () => {
    // Zero is a real choice here: trimming nothing is the plain mean. Dropping it lets the server
    // fall back to its default of 0.1 -- a different rule from the one selected.
    await startProjectServer('p1', { strategy: 'Robust', robustMethod: 'TRIMMED_MEAN', trimRatio: 0 });
    expect(api.post).toHaveBeenCalledWith('/projects/p1/start', {
      strategy: 'Robust', robustMethod: 'TRIMMED_MEAN', trimRatio: 0,
    });
  });

  it('passes the clipping radius through', async () => {
    await startProjectServer('p1', { strategy: 'Robust', robustMethod: 'CENTERED_CLIP', centeredClipTau: 1.5 });
    expect(api.post).toHaveBeenCalledWith('/projects/p1/start', {
      strategy: 'Robust', robustMethod: 'CENTERED_CLIP', centeredClipTau: 1.5,
    });
  });

  it('sends nothing robust for a plain start', async () => {
    await startProjectServer('p1', { strategy: 'FedAvg', numRounds: 5, minClients: 2 });
    expect(api.post).toHaveBeenCalledWith('/projects/p1/start', {
      strategy: 'FedAvg', numRounds: 5, minClients: 2,
    });
  });

  it('passes secure aggregation and its threshold through', async () => {
    await startProjectServer('p1', { strategy: 'DeComFL', secureAggregation: true, secureAggThreshold: 2 });
    expect(api.post).toHaveBeenCalledWith('/projects/p1/start', {
      strategy: 'DeComFL', secureAggregation: true, secureAggThreshold: 2,
    });
  });

  it('sends nothing secure when secure aggregation is off', async () => {
    await startProjectServer('p1', { strategy: 'DeComFL', secureAggregation: false });
    expect(api.post).toHaveBeenCalledWith('/projects/p1/start', { strategy: 'DeComFL' });
  });
});
