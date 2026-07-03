// =============================================================================
// FedLearn Desktop — shared backend HTTP client (Main process)
// =============================================================================
// Single axios instance for every backend REST call. The backend scopes
// `Authorization: Bearer` acceptance to native clients (SE-9): a Bearer token
// is honored only when the request also carries the X-FedLearn-Client marker
// header — browsers stay strictly cookie-only. The marker is a plain client
// identifier (an intent signal), NOT a secret.
//
// All main-process services must call the backend through this instance so the
// marker rides on every request as a shared default, never per-call.
// =============================================================================

import axios, { AxiosInstance } from 'axios';

export const NATIVE_CLIENT_HEADER = 'X-FedLearn-Client';
export const NATIVE_CLIENT_VALUE = 'fedlearn-desktop';

export const http: AxiosInstance = axios.create();
http.defaults.headers.common[NATIVE_CLIENT_HEADER] = NATIVE_CLIENT_VALUE;
