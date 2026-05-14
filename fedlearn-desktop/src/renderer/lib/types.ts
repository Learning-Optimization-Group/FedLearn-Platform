export type ProjectStatus = 'CREATED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'STOPPED';
export type Visibility = 'PUBLIC' | 'PRIVATE';
export type RequestStatus = 'PENDING' | 'APPROVED' | 'DENIED';
export type DiscoverStatus = 'NONE' | 'PENDING' | 'APPROVED' | 'DENIED';

export interface ClientProject {
  projectId: string;
  name: string;
  modelType: string;
  modelName: string;
  status: ProjectStatus;
  visibility: Visibility;
}

export interface DiscoverProject {
  id: string;
  name: string;
  visibility: Visibility;
  ownerUsername: string;
  modelType: string;
  description?: string | null;
  lastAccuracy?: number | null;
  myRequestStatus: DiscoverStatus;
}

export interface MyAccessRequest {
  id: number;
  projectId: string;
  projectName: string;
  status: RequestStatus;
  message?: string | null;
  requestedAt: string;
  decidedAt?: string | null;
  decidedByUsername?: string | null;
}

export interface ClientConnection {
  projectId: string;
  name: string;
  modelType: string;
  modelName: string;
  serverAddress: string;
  partitionId: number;
  status: ProjectStatus;
}

export interface TrainResult {
  success: boolean;
  error?: string;
}
