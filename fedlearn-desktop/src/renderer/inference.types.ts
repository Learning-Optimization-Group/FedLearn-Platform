// Shared inference types for the renderer (mirror the backend DTOs).

export interface InferableModel {
  projectId: string;
  name: string;
  modelType: string;
  modelName: string;
  status: string;
  inputKind: 'image' | 'vector' | null;
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
