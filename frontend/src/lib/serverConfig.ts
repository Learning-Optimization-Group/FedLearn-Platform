// =============================================================================
// FedLearn Frontend — server / broker origin (FE-9)
// =============================================================================
// Single source of truth for the backend's HTTP origin and the STOMP-over-WS
// broker URL derived from it. Before this module existed, the derivation was
// copy-pasted across PlaygroundView, OwnerDashboard, the (now removed)
// DashboardV2, and read again ad hoc in SettingsView — any one of them
// drifting (e.g. a fixed protocol-swap regex) would silently desync a WS
// surface from the REST origin it's supposed to share.

/**
 * The backend's HTTP origin. `VITE_SERVER_ROOT_URL` is set per Vite mode
 * (development/ec2demo/production — see `frontend/.env.*`); the fallback only
 * ever fires when the env var is unset (e.g. a stray local run) and assumes
 * the backend is reachable on the same host, port 8081.
 */
export const SERVER_ROOT_URL: string =
    import.meta.env.VITE_SERVER_ROOT_URL || `http://${window.location.hostname}:8081`;

/**
 * The STOMP-over-WebSocket broker endpoint, derived from {@link SERVER_ROOT_URL}
 * by swapping the URL scheme (http -> ws, https -> wss) and appending the
 * backend's `/ws-logs` STOMP endpoint. The `JwtHandshakeInterceptor` on the
 * backend authenticates the handshake via the same HttpOnly cookie used for
 * REST calls — nothing token-related is attached here or anywhere upstream.
 */
export const WS_BROKER_URL: string = `${SERVER_ROOT_URL.replace(/^http/, 'ws')}/ws-logs`;
