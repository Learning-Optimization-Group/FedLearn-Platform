import { beforeEach, describe, expect, it, vi } from 'vitest';

vi.mock('../api/axiosConfig', () => ({ default: { get: vi.fn() } }));
import api from '../api/axiosConfig';
import { fetchProjectBenchmark } from './apiServices';

describe('benchmark run selection', () => {
  beforeEach(() => vi.mocked(api.get).mockReset().mockResolvedValue({ data: {} } as never));

  it('requests the recorded run instead of whichever run is currently active', async () => {
    await fetchProjectBenchmark('project-1', 'recorded-run');
    expect(api.get).toHaveBeenCalledWith('/admin/benchmarks/projects/project-1', { params: { runId: 'recorded-run' } });
  });

  it('selects legacy records explicitly when their run is unknown', async () => {
    await fetchProjectBenchmark('project-1', null);
    expect(api.get).toHaveBeenCalledWith('/admin/benchmarks/projects/project-1', { params: { legacy: true } });
  });

  it('retains the default current-run request for existing callers', async () => {
    await fetchProjectBenchmark('project-1');
    expect(api.get).toHaveBeenCalledWith('/admin/benchmarks/projects/project-1');
  });
});
