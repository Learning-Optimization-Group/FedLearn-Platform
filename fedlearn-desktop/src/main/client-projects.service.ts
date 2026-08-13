// =============================================================================
// FedLearn Desktop — Client Projects Service (Main process)
// =============================================================================
// Drives the "models I can train" flow. Instead of the user typing a project
// id / server address / partition id by hand, the desktop asks the backend
// which projects the logged-in user may train (GET /api/client/projects) and,
// when they pick one, fetches the live connection details — gRPC address and
// the server-assigned partition id — from GET /api/client/projects/{id}/connection.
//
// The JWT stays in Main (never exposed to the Renderer); the renderer reaches
// this only through IPC.
// =============================================================================

import log from 'electron-log';
import { AuthService } from './auth.service';
import { http } from './http';

export interface ClientProject {
  projectId: string;
  name: string;
  modelType: string;
  status: string;
  visibility: string | null;
  requirements?: import('../shared/deviceCapabilities.types').DeviceRequirements;
}

export interface ProjectConnection {
  projectId: string;
  name: string;
  modelType: string;
  serverAddress: string;
  partitionId: number;
  status: string;
  // Backend-minted FL connection token (ClientConnectionDto.connectionToken).
  connectionToken?: string;
  // The active run's aggregation strategy (ClientConnectionDto.strategy); forwarded to the client
  // as --strategy so a non-MLP DeComFL project runs the DeComFL path, not the default FedAvg path.
  strategy?: string;
  // The project's training arm; forwarded to the client as --training-arm so a FROZEN_HEAD project
  // federates the head subset the server expects instead of the full state dict.
  trainingArm?: string;
}

export class ClientProjectService {
  constructor(private readonly auth: AuthService) {}

  /** Projects the authenticated user owns or is an approved CLIENT of. */
  async listProjects(): Promise<{ success: boolean; projects?: ClientProject[]; error?: string }> {
    const header = this.auth.getAuthHeader();
    if (!header) {
      return { success: false, error: 'Not authenticated' };
    }
    try {
      const res = await http.get(`${this.auth.getApiUrl()}/client/projects`, {
        headers: { Authorization: header },
        validateStatus: (s) => s < 500,
      });
      if (res.status !== 200) {
        return { success: false, error: `Failed to load your projects (HTTP ${res.status})` };
      }
      return { success: true, projects: res.data as ClientProject[] };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[ClientProjectService] listProjects failed: ${message}`);
      return { success: false, error: 'Could not reach the backend.' };
    }
  }

  /**
   * Live connection details for one project. Requires the project's FL server
   * to be RUNNING (the owner starts it) — the backend returns 4xx with a clear
   * message otherwise, which we surface verbatim.
   */
  async getConnection(
    projectId: string,
  ): Promise<{ success: boolean; connection?: ProjectConnection; error?: string }> {
    const header = this.auth.getAuthHeader();
    if (!header) {
      return { success: false, error: 'Not authenticated' };
    }
    try {
      // Join first (idempotent). The backend's /connection endpoint enrolls only owner-or-CLIENT
      // members, so a project the user only *discovered* (a PUBLIC one they haven't joined) 403s
      // "Access denied". POST /join is a no-op when the user is already a CLIENT and auto-joins a
      // PUBLIC project — mirroring the mobile client, which joins before it connects. Without this
      // the desktop client can never reach a project it only discovered.
      const joinRes = await http.post(
        `${this.auth.getApiUrl()}/client/projects/${projectId}/join`,
        {},
        { headers: { Authorization: header }, validateStatus: (s) => s < 600 },
      );
      if (joinRes.status !== 200) {
        const jmsg = (joinRes.data && (joinRes.data.message as string))
          || `Could not join the project (HTTP ${joinRes.status})`;
        return { success: false, error: jmsg };
      }
      const res = await http.get(
        `${this.auth.getApiUrl()}/client/projects/${projectId}/connection`,
        {
          headers: { Authorization: header },
          // Accept 4xx so we can surface the backend's message (e.g. "not running").
          validateStatus: (s) => s < 600,
        },
      );
      if (res.status !== 200) {
        const msg = (res.data && (res.data.message as string))
          || `Could not get connection details (HTTP ${res.status})`;
        return { success: false, error: msg };
      }
      return { success: true, connection: res.data as ProjectConnection };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[ClientProjectService] getConnection failed: ${message}`);
      return { success: false, error: 'Could not reach the backend.' };
    }
  }
}
