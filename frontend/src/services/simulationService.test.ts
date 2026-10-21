import { afterEach, describe, expect, it } from 'vitest';
import type { InternalAxiosRequestConfig } from 'axios';
import api from '../api/axiosConfig';
import * as simulations from './simulationService';

const originalAdapter = api.defaults.adapter;
afterEach(() => { api.defaults.adapter = originalAdapter; });
const id = '11111111-1111-4111-8111-111111111111';

describe('simulation HTTP boundary', () => {
  it('sends only the choices wrapper over the existing cookie transport', async () => {
    let observed: InternalAxiosRequestConfig | undefined;
    api.defaults.adapter = async (config) => {
      observed = config;
      return { data: { id, status: 'STARTING' }, status: 202, statusText: 'Accepted', headers: {}, config };
    };
    const result = await simulations.startSimulation({ seed: 42 });
    expect(result.status).toBe('STARTING');
    expect(observed?.url).toBe('/simulations/runs');
    expect(observed?.method).toBe('post');
    expect(JSON.parse(observed?.data as string)).toEqual({ choices: { seed: 42 } });
    expect(observed?.withCredentials).toBe(true);
    expect(observed?.timeout).toBe(30000);
    expect(observed?.headers.get('Authorization')).toBeUndefined();
  });

  it('bounds history pagination and retains the request cancellation signal', async () => {
    let observed: InternalAxiosRequestConfig | undefined;
    api.defaults.adapter = async (config) => {
      observed = config;
      return { data: { content: [], totalElements: 0, page: 0, size: 20 }, status: 200, statusText: 'OK', headers: {}, config };
    };
    const controller = new AbortController();
    await simulations.listSimulations(3, controller.signal);
    expect(observed?.params).toEqual({ page: 3, size: 20, scope: 'mine' });
    expect(observed?.signal).toBe(controller.signal);
    expect(observed?.timeout).toBe(30000);
  });

  it('encodes run identifiers so query text cannot become another endpoint', async () => {
    const paths: string[] = [];
    api.defaults.adapter = async (config) => {
      paths.push(config.url!);
      return { data: {}, status: 200, statusText: 'OK', headers: {}, config };
    };
    await simulations.getSimulation('a/b?x=1');
    await simulations.stopSimulation(id);
    expect(paths).toEqual(['/simulations/runs/a%2Fb%3Fx%3D1', `/simulations/runs/${id}/stop`]);
  });

  it('obtains diagnostic downloads as blobs without a public-file URL', async () => {
    let responseType: string | undefined;
    api.defaults.adapter = async (config) => {
      responseType = config.responseType;
      return { data: new Blob(['{"researchRecordDpSafe":false}']), status: 200, statusText: 'OK', headers: {}, config };
    };
    const data = await simulations.downloadSimulationRecord(id);
    expect(data).toBeInstanceOf(Blob);
    expect(responseType).toBe('blob');
  });
});
