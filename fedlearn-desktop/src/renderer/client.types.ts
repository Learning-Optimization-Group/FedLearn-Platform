// Shared renderer-side types for the "models I can train" discovery flow.
// Mirror of the main-process ClientProjectService DTOs.

import type { DeviceRequirements } from '../shared/deviceCapabilities.types';

export type { DeviceRequirements };

export interface ClientProject {
  projectId: string;
  name: string;
  modelType: string;
  status: string;
  visibility: string | null;
  requirements?: DeviceRequirements;
}

export interface ProjectConnection {
  projectId: string;
  name: string;
  modelType: string;
  serverAddress: string;
  partitionId: number;
  status: string;
  connectionToken?: string;
}
