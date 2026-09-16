// Server-side inference + LLM generation REST surface (parity with the desktop "Use a Model" tab).
// All calls go through the Bearer-authenticated axios client. Token streaming for generation arrives
// out-of-band on the STOMP topic /topic/inference/{projectId} (see stompClient.ts + PlaygroundScreen).
import { api } from './restClient';

export interface InferableModel {
  projectId: string;
  name: string;
  modelType: string;
  modelName: string;
  status: string;
  inputKind: 'image' | 'vector' | 'text' | null; // null → a generative (chat) model, not a classifier
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

export type ChatRole = 'user' | 'assistant';
export interface ChatTurn {
  role: ChatRole;
  content: string;
}

export interface GenerationRequest {
  prompt: string;
  maxNewTokens: number; // 1..2048
  temperature: number; // 0..2
  history?: ChatTurn[]; // prior turns, max 100
}

export interface GenerationResult {
  modelType: string;
  prompt: string;
  generatedText: string;
  tokenCount: number;
  finishReason: string; // 'stop' | 'length' | 'stopped'
}

/** GET /api/inference/models — models the signed-in user may run inference on. */
export async function listInferableModels(): Promise<InferableModel[]> {
  const { data } = await api.get<InferableModel[]>('/api/inference/models');
  return Array.isArray(data) ? data : [];
}

/** POST /api/inference/{projectId} — classify one input (exactly one of text/values/imageBase64). */
export async function runInference(
  projectId: string,
  input: { text?: string; values?: number[]; imageBase64?: string },
): Promise<InferenceResult> {
  const { data } = await api.post<InferenceResult>(`/api/inference/${projectId}`, input);
  return data;
}

/** POST /api/inference/{projectId}/generate — kick off generation; tokens stream to the STOMP topic,
 *  and this resolves with the accumulated result when generation finishes. */
export async function startGeneration(
  projectId: string,
  req: GenerationRequest,
): Promise<GenerationResult> {
  // Generation can take much longer than a control-plane call; override the 15s default.
  const { data } = await api.post<GenerationResult>(`/api/inference/${projectId}/generate`, req, {
    timeout: 120_000,
  });
  return data;
}

/** POST /api/inference/{projectId}/generate/stop — cancel an in-flight generation. */
export async function stopGeneration(projectId: string): Promise<boolean> {
  const { data } = await api.post<{ stopped: boolean }>(
    `/api/inference/${projectId}/generate/stop`,
    {},
  );
  return Boolean(data?.stopped);
}
