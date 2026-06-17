import api from '../api/axiosConfig';
import { AxiosResponse } from 'axios';

// Type definitions
export interface LoginCredentials {
    username: string;
    password: string;
}

export interface RegisterData {
    username: string;
    email: string;
    password: string;
}

export interface ProjectData {
    name: string;
    modelType: string;
    modelName: string;
    optimizer: string;
    pretrainEpochs: number;
}

export interface StartServerData {
    strategy?: string;
    numRounds?: number;
    minClients?: number;
}

export interface Project {
    id: string;
    name: string;
    modelType: string;
    modelName: string;
    optimizer: string;
    status: 'RUNNING' | 'STOPPED' | 'COMPLETED' | 'FAILED';
    serverPort?: number;
}

export interface ProjectResult {
    id: string;
    serverRound: number;
    loss: number;
    accuracy: number;
    gpuUtilization?: number;
}

// ─── Authentication ──────────────────────────────────────────────────────
//
// Auth contract is cookie-only: the backend sets an HttpOnly `jwtToken`
// cookie on /auth/login and clears it on /auth/logout. JS never sees the
// token. To answer "am I logged in?" the SPA calls /auth/me.

export interface AuthIdentity {
    username: string;
    email: string;
    role: 'USER' | 'ADMIN';
}

export const loginUser = (credentials: LoginCredentials): Promise<AxiosResponse<AuthIdentity>> => {
    return api.post<AuthIdentity>('/auth/login', credentials);
};

export const registerUser = (userData: RegisterData): Promise<AxiosResponse> => {
    return api.post('/auth/register', userData);
};

/**
 * Returns the current user's identity if the session cookie is still valid;
 * otherwise the request fails with 401 (which the axios interceptor ignores
 * for this specific URL — see api/axiosConfig.ts).
 */
export const fetchCurrentUser = (): Promise<AxiosResponse<AuthIdentity>> => {
    return api.get<AuthIdentity>('/auth/me');
};

/**
 * Clears the auth cookie server-side. The frontend should also clear its
 * in-memory user state immediately; don't wait on this call's success.
 */
export const logoutUser = (): Promise<AxiosResponse<void>> => {
    return api.post<void>('/auth/logout');
};

// ─── Model recipe catalog ───────────────────────────────────────────────
//
// The backend owns the catalog of trainable model types (architectures,
// their base models and optimizers). The project modals fetch this at open
// time so the picker stays in sync with what the framework actually supports.

export interface ModelRecipe {
    key: string;
    displayName: string;
    inputKind: string;
    classes: string[];
    baseModels: string[];
    optimizers: string[];
}

/** Lists the model recipes the platform can train. */
export const fetchModelRecipes = (): Promise<AxiosResponse<ModelRecipe[]>> => {
    return api.get<ModelRecipe[]>('/model-recipes');
};

// Project Management Endpoints
export const fetchProjects = async (): Promise<AxiosResponse<Project[]>> => {
    return api.get<Project[]>('/projects');
};

export const createProject = (projectData: ProjectData): Promise<AxiosResponse<Project>> => {
    return api.post<Project>('/projects', projectData);
};

export const startProjectServer = (projectId: string, startData: StartServerData): Promise<AxiosResponse<Project>> => {
    const body: StartServerData = {};
    if (startData?.strategy) {
        body.strategy = startData.strategy;
    }
    if (startData?.numRounds) {
        body.numRounds = startData.numRounds;
    }
    if (startData?.minClients) {
        body.minClients = startData.minClients;
    }
    return api.post<Project>(`/projects/${projectId}/start`, body);
};

export const stopProjectServer = (projectId: string): Promise<AxiosResponse<Project>> => {
    return api.post<Project>(`/projects/${projectId}/stop`, {});
};

export const updateProject = (projectId: string, updateData: Partial<Project>): Promise<AxiosResponse<Project>> => {
    return api.put<Project>(`/projects/${projectId}`, updateData);
};

export const fetchProjectResults = (projectId: string): Promise<AxiosResponse<ProjectResult[]>> => {
    return api.get<ProjectResult[]>(`/projects/${projectId}/results`);
};

export const fetchProjectLogs = (projectId: string): Promise<AxiosResponse<any[]>> => {
    return api.get<any[]>(`/projects/${projectId}/logs`);
};

export const deleteProject = (projectId: string): Promise<AxiosResponse<{ projectId: string; message: string }>> => {
    // Canonical DELETE; the backend keeps POST /projects/{id}/delete around
    // as a deprecated alias for older Electron builds.
    return api.delete<{ projectId: string; message: string }>(`/projects/${projectId}`);
};

// ─── Inference ("Use a model") ───────────────────────────────────────────
//
// Both the web app and the desktop app call these endpoints. The backend runs
// the real PyTorch model server-side and returns class probabilities.

export interface InferableModel {
    projectId: string;
    name: string;
    modelType: string;
    modelName: string;
    status: string;
    /** "image" → collect an image; "vector" → collect a numeric vector; null → not runnable. */
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

export interface InferencePayload {
    /** Base64 image (may include a data: URL prefix). For image models. */
    imageBase64?: string;
    /** Numeric feature vector. For tabular (MLP) models. */
    values?: number[];
}

/** Lists the current user's trained models that can be run interactively. */
export const fetchInferableModels = (): Promise<AxiosResponse<InferableModel[]>> => {
    return api.get<InferableModel[]>('/inference/models');
};

/** Runs one inference against a project's trained model. */
export const runInference = (
    projectId: string,
    payload: InferencePayload,
): Promise<AxiosResponse<InferenceResult>> => {
    return api.post<InferenceResult>(`/inference/${projectId}`, payload);
};

// User / Client Management Endpoints
export interface User {
    id: number;
    username: string;
    email: string;
    role?: 'USER' | 'ADMIN';
    createdAt?: string;
}

/**
 * Lists every user. Admin-only on the backend — non-admins receive 403,
 * which callers should display as "permission denied" rather than a hard
 * logout (the axios interceptor already enforces this distinction).
 */
export const fetchUsers = (): Promise<AxiosResponse<User[]>> => {
    return api.get<User[]>('/users');
};

export const createUser = (userData: RegisterData): Promise<AxiosResponse<User>> => {
    return api.post<User>('/users', userData);
};

export const deleteUser = (userId: number): Promise<AxiosResponse<any>> => {
    return api.delete(`/users/${userId}`);
};
