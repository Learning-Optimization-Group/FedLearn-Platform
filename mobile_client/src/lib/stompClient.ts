// STOMP-over-WebSocket client for live server streams (training logs + LLM generation tokens).
// Mirrors the web/desktop STOMP usage but authenticates with the mobile Bearer token instead of the
// HttpOnly cookie: the backend JwtHandshakeInterceptor + JwtChannelInterceptor accept
// `Authorization: Bearer <jwt>` on BOTH the WebSocket upgrade and the STOMP CONNECT frame. RN's
// WebSocket (unlike the browser's) supports custom upgrade headers, so we set the token on both.
import { Client, type IFrame, type IMessage } from '@stomp/stompjs';
import { getToken } from './authStore';
import { getServerBaseUrl } from './serverConfig';
import { NATIVE_CLIENT_HEADER, NATIVE_CLIENT_VALUE } from './restClient';

/** REST base (http/https) → the ws/wss origin for the /ws-logs STOMP endpoint. */
function toWsUrl(base: string): string {
  return base.replace(/\/+$/, '').replace(/^http/i, 'ws') + '/ws-logs';
}

/**
 * Headers for both the WS upgrade and the STOMP CONNECT frame: the Bearer token (when
 * present) plus the native-client marker (SE-9) that scoped Bearer acceptance requires.
 */
export function stompAuthHeaders(token: string | null): Record<string, string> {
  const headers: Record<string, string> = { [NATIVE_CLIENT_HEADER]: NATIVE_CLIENT_VALUE };
  if (token) headers.Authorization = `Bearer ${token}`;
  return headers;
}

export interface StompHandle {
  /** Subscribe to a destination; returns an unsubscribe fn. Message body is delivered as a string. */
  subscribe(destination: string, onMessage: (body: string) => void): () => void;
  deactivate(): void;
}

/**
 * Open a STOMP connection to /ws-logs and resolve once CONNECTED. Rejects if the socket fails before
 * connecting. `onError` receives later transport/protocol errors (after a successful connect).
 */
export async function connectStomp(onError?: (msg: string) => void): Promise<StompHandle> {
  const [token, base] = await Promise.all([getToken(), getServerBaseUrl()]);
  const url = toWsUrl(base);
  const auth = stompAuthHeaders(token);

  const client = new Client({
    // 3rd arg (options.headers) is React Native's WebSocket extension — sets the upgrade headers.
    webSocketFactory: () =>
      new WebSocket(url, [], { headers: auth }) as unknown as WebSocket,
    connectHeaders: auth,
    reconnectDelay: 0, // caller owns lifecycle; no silent auto-reconnect storms
    heartbeatIncoming: 10000,
    heartbeatOutgoing: 10000,
  });

  return await new Promise<StompHandle>((resolve, reject) => {
    let settled = false;
    client.onConnect = () => {
      if (settled) return;
      settled = true;
      resolve({
        subscribe(destination, onMessage) {
          const sub = client.subscribe(destination, (m: IMessage) => onMessage(m.body));
          return () => {
            try {
              sub.unsubscribe();
            } catch {
              /* already torn down */
            }
          };
        },
        deactivate() {
          void client.deactivate();
        },
      });
    };
    client.onStompError = (frame: IFrame) => {
      const msg = frame.headers['message'] ?? 'STOMP protocol error';
      if (settled) onError?.(msg);
      else {
        settled = true;
        reject(new Error(msg));
      }
    };
    client.onWebSocketError = () => {
      if (settled) onError?.('WebSocket connection error');
      else {
        settled = true;
        reject(new Error('Could not open the WebSocket — check the server URL and that you are signed in.'));
      }
    };
    client.activate();
  });
}
