// =============================================================================
// FedLearn Desktop — Inference Stream Service
// =============================================================================
// Main-process STOMP bridge: subscribes to /topic/inference/{projectId} and
// forwards each streaming token to the renderer via inference:token IPC push,
// then fires the POST /api/inference/{projectId}/generate HTTP call. The
// subscription is opened before the POST so no tokens are missed.
//
// The JWT Authorization header is built entirely inside this service and never
// leaves the Main process.
// =============================================================================

import { BrowserWindow } from 'electron';
import axios from 'axios';
import { Client as StompClient } from '@stomp/stompjs';
import WS from 'ws';
import log from 'electron-log';
import { AuthService } from './auth.service';

interface GenerationResult {
  modelType: string;
  prompt: string;
  generatedText: string;
  tokenCount: number;
  finishReason: string;
}

/** Streams generation tokens from the backend STOMP topic to the renderer, and runs the generate POST. */
export class InferenceStreamService {
  constructor(private readonly auth: AuthService, private readonly mainWindow: BrowserWindow) {}

  private wsBrokerUrl(): string {
    // getApiUrl() → http(s)://host:port/api  ⇒  ws(s)://host:port/ws-logs
    const root = this.auth.getApiUrl().replace(/\/api\/?$/, '');
    return root.replace(/^http/, 'ws') + '/ws-logs';
  }

  async runGeneration(
    projectId: string,
    payload: { prompt: string; maxNewTokens: number; temperature: number },
  ): Promise<{ success: boolean; result?: GenerationResult; error?: string }> {
    const header = this.auth.getAuthHeader();
    if (!header) return { success: false, error: 'Not authenticated' };

    const client = new StompClient({
      webSocketFactory: () =>
        new WS(this.wsBrokerUrl(), { headers: { Authorization: header } }) as unknown as WebSocket,
      reconnectDelay: 0, // single-shot; don't reconnect after the request ends
    });

    const send = (token: string) => {
      if (!this.mainWindow.isDestroyed()) {
        this.mainWindow.webContents.send('inference:token', token);
      }
    };

    await new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, 8000);
      client.onConnect = () => {
        clearTimeout(timer);
        client.subscribe(`/topic/inference/${projectId}`, (msg) => {
          try {
            const { token } = JSON.parse(msg.body) as { token?: unknown };
            if (typeof token === 'string') send(token);
          } catch {
            // ignore non-token frames
          }
        });
        resolve();
      };
      client.onStompError = () => resolve(); // proceed even if stream fails; HTTP result is backstop
      client.onWebSocketError = () => resolve(); // guard against transport-level stalls
      client.activate();
    });

    try {
      const res = await axios.post(
        `${this.auth.getApiUrl()}/inference/${projectId}/generate`,
        payload,
        {
          headers: { Authorization: header, 'Content-Type': 'application/json' },
          validateStatus: (s) => s < 600,
        },
      );
      if (res.status !== 200) {
        const msg =
          (res.data && (res.data.message as string)) ||
          `Generation failed (HTTP ${res.status})`;
        return { success: false, error: msg };
      }
      return { success: true, result: res.data as GenerationResult };
    } catch (err: unknown) {
      log.error(
        `[InferenceStreamService] generate failed: ${err instanceof Error ? err.message : err}`,
      );
      return { success: false, error: 'Could not reach the backend.' };
    } finally {
      if (client.active) client.deactivate();
    }
  }
}
