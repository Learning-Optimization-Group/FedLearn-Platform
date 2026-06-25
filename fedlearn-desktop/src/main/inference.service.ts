// =============================================================================
// FedLearn Desktop — Inference Service (Main process)
// =============================================================================
// Calls the backend "Use a model" endpoints with the JWT held in Main (never
// exposed to the Renderer). The backend runs the real PyTorch model and returns
// class probabilities. The renderer reaches this only through IPC.
// =============================================================================

import axios from 'axios';
import log from 'electron-log';
import { AuthService } from './auth.service';

export interface InferableModel {
  projectId: string;
  name: string;
  modelType: string;
  modelName: string;
  status: string;
  inputKind: 'image' | 'vector' | 'text' | null;
  classes: string[];
  supported: boolean;
}

export interface InferenceResult {
  modelType: string;
  predictedIndex: number;
  predictedLabel: string;
  classes: string[];
  probabilities: number[];
  logits: number[];
}

export interface InferencePayload {
  imageBase64?: string;
  values?: number[];
  text?: string;
}

export class InferenceService {
  constructor(private readonly auth: AuthService) {}

  /** Lists the authenticated user's runnable trained models. */
  async listModels(): Promise<{ success: boolean; models?: InferableModel[]; error?: string }> {
    const header = this.auth.getAuthHeader();
    if (!header) {
      return { success: false, error: 'Not authenticated' };
    }
    try {
      const res = await axios.get(`${this.auth.getApiUrl()}/inference/models`, {
        headers: { Authorization: header },
        validateStatus: (s) => s < 500,
      });
      if (res.status !== 200) {
        return { success: false, error: `Failed to load models (HTTP ${res.status})` };
      }
      return { success: true, models: res.data as InferableModel[] };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[InferenceService] listModels failed: ${message}`);
      return { success: false, error: 'Could not reach the backend.' };
    }
  }

  /** Runs one inference against a project's model. */
  async runInference(
    projectId: string,
    payload: InferencePayload,
  ): Promise<{ success: boolean; result?: InferenceResult; error?: string }> {
    const header = this.auth.getAuthHeader();
    if (!header) {
      return { success: false, error: 'Not authenticated' };
    }
    try {
      const res = await axios.post(`${this.auth.getApiUrl()}/inference/${projectId}`, payload, {
        headers: { Authorization: header, 'Content-Type': 'application/json' },
        // Accept 4xx so we can surface the backend's message instead of throwing.
        validateStatus: (s) => s < 600,
      });
      if (res.status !== 200) {
        const msg = (res.data && (res.data.message as string)) || `Inference failed (HTTP ${res.status})`;
        return { success: false, error: msg };
      }
      return { success: true, result: res.data as InferenceResult };
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      log.error(`[InferenceService] runInference failed: ${message}`);
      return { success: false, error: 'Could not reach the backend.' };
    }
  }
}
