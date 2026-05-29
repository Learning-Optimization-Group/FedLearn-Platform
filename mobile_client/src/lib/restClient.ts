// REST client for the control plane (/api/*). Cookie-based auth contract (withCredentials),
// matching the platform's HttpOnly-cookie model (04 §4 / README §1.1). The base URL comes from
// app config (FEDLEARN_API_URL, 15-LLD §8) — call configureApi(...) once at startup.
import axios, { type AxiosInstance } from 'axios';

export const api: AxiosInstance = axios.create({
  withCredentials: true,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

export function configureApi(baseUrl: string): void {
  if (!baseUrl) throw new Error('configureApi: FEDLEARN_API_URL is required');
  api.defaults.baseURL = baseUrl;
}
