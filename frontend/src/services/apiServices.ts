import api from '../api/axiosConfig';
import { AxiosError, AxiosResponse, isAxiosError } from 'axios';

// ─── Error helpers ───────────────────────────────────────────────────────
//
// The backend's GlobalExceptionHandler returns {message} (and sometimes
// {error}); 403 means "not allowed" and 409 means "conflict" (e.g. demoting
// the last admin, or an owner request that already exists). These helpers let
// callers render those failures inline instead of guessing at the shape.

/** The HTTP status of an axios error, or undefined for non-HTTP failures. */
export function errorStatus(err: unknown): number | undefined {
    return isAxiosError(err) ? err.response?.status : undefined;
}

/** Pull a human-readable message out of a backend error response. */
export function errorMessage(err: unknown, fallback = 'Something went wrong. Please try again.'): string {
    if (isAxiosError(err)) {
        const data = (err as AxiosError<{ message?: string; error?: string }>).response?.data;
        if (data?.message) return data.message;
        if (data?.error) return data.error;
    }
    return fallback;
}

/** A 204 / empty-body response carries no resource. */
export function isEmptyBody(data: unknown): boolean {
    return data === '' || data == null;
}

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
    /** "SEQ_CLASSIFICATION" (default) | "CAUSAL_LM". Only meaningful for LLM_LORA. */
    taskType?: string;
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

export type Role = 'USER' | 'PROJECT_OWNER' | 'PLATFORM_ADMIN';

export interface AuthIdentity {
    username: string;
    email: string;
    role: Role;
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
    /** "image" → collect an image; "vector" → collect a numeric vector; "text" → collect raw text; "generation" → prompt + sliders; null → not runnable. */
    inputKind: 'image' | 'vector' | 'text' | 'generation' | null;
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
    /** Raw text. For text models (LLM_LORA, TRANSFORMER). */
    text?: string;
    /** Prompt for generation models. */
    prompt?: string;
    /** Maximum number of tokens to generate. */
    maxNewTokens?: number;
    /** Sampling temperature (0–2). */
    temperature?: number;
    /** Prior conversation turns for multi-turn generation. */
    history?: { role: 'user' | 'assistant'; content: string }[];
}

export interface GenerationResult {
    modelType: string;
    prompt: string;
    generatedText: string;
    tokenCount: number;
    finishReason: string;
}

/** Runs streaming generation; tokens arrive over /topic/inference/{projectId}, this resolves with the final text. */
export const runGeneration = (
    projectId: string,
    payload: {
        prompt: string;
        maxNewTokens: number;
        temperature: number;
        history?: { role: 'user' | 'assistant'; content: string }[];
    },
): Promise<AxiosResponse<GenerationResult>> => {
    return api.post<GenerationResult>(`/inference/${projectId}/generate`, payload);
};

/** Cancels an in-flight generation for the project. */
export const stopGeneration = (
    projectId: string,
): Promise<AxiosResponse<{ stopped: boolean }>> => {
    return api.post<{ stopped: boolean }>(`/inference/${projectId}/generate/stop`);
};

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
    role?: Role;
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

// ─── Project visibility (3 tiers) ────────────────────────────────────────
//
// PUBLIC     — anyone can join & train.
// RESTRICTED — discoverable; the owner approves join requests.
// PRIVATE    — hidden; invite-only.

export type Visibility = 'PUBLIC' | 'RESTRICTED' | 'PRIVATE';

/** Plain-language copy for each visibility tier (reused by selectors). */
export const VISIBILITY_HELP: Record<Visibility, string> = {
    PUBLIC: 'Anyone can join and train.',
    RESTRICTED: 'Discoverable — you approve join requests.',
    PRIVATE: 'Hidden — invite-only.',
};

type Decision = 'APPROVED' | 'DENIED';
type RequestStatus = 'PENDING' | 'APPROVED' | 'DENIED';

// ─── Admin (PLATFORM_ADMIN only — non-admins get 403) ────────────────────
//
// 403 here means "not allowed", NOT "session expired": the axios interceptor
// only logs out on 401, so callers should render these failures inline.

export interface AdminOverview {
    totalUsers: number;
    owners: number;
    admins: number;
    totalProjects: number;
    runningProjects: number;
    pendingOwnerRequests: number;
    pendingDeletionRequests: number;
    pendingAccessRequests: number;
}

export interface AdminUser {
    id: number;
    username: string;
    email: string;
    role: Role;
    projectsOwned: number;
    memberships: number;
    createdAt: string;
}

export interface AdminProject {
    id: string;
    name: string;
    modelType: string;
    status: string;
    visibility: Visibility;
    ownerUsername: string;
    participantCount: number;
}

export interface OwnerRequest {
    id: number;
    userId: number;
    username: string;
    email: string;
    status: RequestStatus;
    message?: string;
    requestedAt: string;
    decidedAt?: string;
    decidedByUsername?: string;
}

export interface DeletionRequest {
    id: number;
    projectId: string;
    projectName: string;
    requestedByUsername: string;
    status: RequestStatus;
    reason?: string;
    requestedAt: string;
    decidedAt?: string;
    decidedByUsername?: string;
}

/** Platform-wide counts for the admin overview tiles. */
export const fetchAdminOverview = (): Promise<AxiosResponse<AdminOverview>> => {
    return api.get<AdminOverview>('/admin/overview');
};

/** Every user with role + counts (admin users table). */
export const fetchAdminUsers = (): Promise<AxiosResponse<AdminUser[]>> => {
    return api.get<AdminUser[]>('/admin/users');
};

/**
 * Change a user's platform role. The backend guards the last admin — demoting
 * it returns 409, which callers should surface (don't swallow).
 */
export const updateUserRole = (userId: number, role: Role): Promise<AxiosResponse<AdminUser>> => {
    return api.put<AdminUser>(`/admin/users/${userId}/role`, { role });
};

/** All projects across the platform (admin all-projects table). */
export const fetchAdminProjects = (): Promise<AxiosResponse<AdminProject[]>> => {
    return api.get<AdminProject[]>('/admin/projects');
};

/** Owner-promotion requests. Omit `status` for all; pass PENDING for the queue. */
export const fetchOwnerRequests = (status?: RequestStatus): Promise<AxiosResponse<OwnerRequest[]>> => {
    return api.get<OwnerRequest[]>('/admin/owner-requests', { params: status ? { status } : undefined });
};

/** Approve or deny an owner-promotion request. */
export const decideOwnerRequest = (id: number, decision: Decision): Promise<AxiosResponse<OwnerRequest>> => {
    return api.put<OwnerRequest>(`/admin/owner-requests/${id}`, { decision });
};

/** Project-deletion requests. Omit `status` for all; pass PENDING for the queue. */
export const fetchDeletionRequests = (status?: RequestStatus): Promise<AxiosResponse<DeletionRequest[]>> => {
    return api.get<DeletionRequest[]>('/admin/deletion-requests', { params: status ? { status } : undefined });
};

/** Approve (permanently deletes the project) or deny a deletion request. */
export const decideDeletionRequest = (id: number, decision: Decision): Promise<AxiosResponse<DeletionRequest>> => {
    return api.put<DeletionRequest>(`/admin/deletion-requests/${id}`, { decision });
};

// ─── Owner promotion (any authenticated user) ────────────────────────────

/** Submit a request to be promoted to PROJECT_OWNER. 409 if already pending/owner. */
export const submitOwnerRequest = (message?: string): Promise<AxiosResponse<OwnerRequest>> => {
    return api.post<OwnerRequest>('/owner-requests', { message });
};

/**
 * The caller's own owner-promotion request, or HTTP 204 (empty body) if none.
 * Callers should treat a 204 / null body as "no request yet".
 */
export const fetchMyOwnerRequest = (): Promise<AxiosResponse<OwnerRequest | ''>> => {
    return api.get<OwnerRequest | ''>('/owner-requests/mine');
};

// ─── Owner-scoped project endpoints ──────────────────────────────────────
//
// These extend the existing /projects surface with the owner's relationship,
// visibility, and the access-request / membership / deletion-request flows.

export interface OwnedProject {
    id: string;
    name: string;
    modelType: string;
    modelName: string;
    status: Project['status'];
    visibility: Visibility;
    myRelationship: string;
    serverPort?: number;
    optimizer: string;
}

export interface AccessRequest {
    id: number;
    projectId: string;
    projectName: string;
    userId: number;
    username: string;
    status: RequestStatus;
    message?: string;
    requestedAt: string;
}

export interface Membership {
    projectId: string;
    userId: number;
    username: string;
    role: 'CLIENT' | 'MEMBER';
    partitionId?: number;
    joinedVia: string;
    addedAt: string;
}

export interface DiscoverableProject {
    id: string;
    name: string;
    visibility: Visibility;
    ownerUsername: string;
    modelType: string;
    description?: string;
    myRequestStatus: 'NONE' | 'PENDING' | 'APPROVED' | 'DENIED';
}

export interface ProjectVisibilityUpdate {
    visibility: Visibility;
    name?: string;
    description?: string;
}

/** The caller's owned projects (richer shape than the base /projects list). */
export const fetchOwnedProjects = (): Promise<AxiosResponse<OwnedProject[]>> => {
    return api.get<OwnedProject[]>('/projects');
};

/** Update a project's visibility (and optionally name/description). */
export const updateProjectVisibility = (
    projectId: string,
    update: ProjectVisibilityUpdate,
): Promise<AxiosResponse<OwnedProject>> => {
    return api.patch<OwnedProject>(`/projects/${projectId}`, update);
};

/** Pending (or all) join requests for a project the caller owns. */
export const fetchAccessRequests = (
    projectId: string,
    status?: RequestStatus,
): Promise<AxiosResponse<AccessRequest[]>> => {
    return api.get<AccessRequest[]>(`/projects/${projectId}/access-requests`, {
        params: status ? { status } : undefined,
    });
};

/** Approve or deny a join request on a project the caller owns. */
export const decideAccessRequest = (
    projectId: string,
    requestId: number,
    decision: Decision,
): Promise<AxiosResponse<AccessRequest>> => {
    return api.put<AccessRequest>(`/projects/${projectId}/access-requests/${requestId}`, { decision });
};

/** Members of a project the caller owns. */
export const fetchMemberships = (projectId: string): Promise<AxiosResponse<Membership[]>> => {
    return api.get<Membership[]>(`/projects/${projectId}/memberships`);
};

/** Add a user to a project by username. */
export const addMembership = (
    projectId: string,
    username: string,
    role: 'CLIENT' | 'MEMBER',
): Promise<AxiosResponse<Membership>> => {
    return api.post<Membership>(`/projects/${projectId}/memberships`, { username, role });
};

/** Remove a member from a project. */
export const removeMembership = (projectId: string, userId: number): Promise<AxiosResponse<void>> => {
    return api.delete<void>(`/projects/${projectId}/memberships/${userId}`);
};

/** Owner asks an admin to permanently delete a project. */
export const submitDeletionRequest = (
    projectId: string,
    reason?: string,
): Promise<AxiosResponse<DeletionRequest>> => {
    return api.post<DeletionRequest>(`/projects/${projectId}/deletion-request`, { reason });
};

/** The project's pending deletion request, or HTTP 204 (empty body) if none. */
export const fetchProjectDeletionRequest = (
    projectId: string,
): Promise<AxiosResponse<DeletionRequest | ''>> => {
    return api.get<DeletionRequest | ''>(`/projects/${projectId}/deletion-request`);
};

// ─── Discovery (any authenticated user) ──────────────────────────────────

/** Projects the caller can discover and request access to / join. */
export const fetchDiscoverableProjects = (): Promise<AxiosResponse<DiscoverableProject[]>> => {
    return api.get<DiscoverableProject[]>('/projects/discover');
};

/**
 * Request access to a project. PUBLIC auto-joins (returns a membership);
 * RESTRICTED creates a pending request; PRIVATE returns 403.
 */
export const requestProjectAccess = (
    projectId: string,
    message?: string,
): Promise<AxiosResponse<Membership | AccessRequest>> => {
    return api.post<Membership | AccessRequest>(`/projects/${projectId}/access-requests`, { message });
};

// ─── Benchmarking & observability (PLATFORM_ADMIN) ───────────────────────
//
// Mirrors the Java DTOs (camelCase 1:1). Every metric is nullable — a field is
// simply absent when the recipe task_type (classification vs generative) or the
// available data doesn't produce it.

/** Per-project benchmark rollup row (the runs table + drilldown header). */
export interface BenchmarkRun {
    projectId: string;
    projectName: string | null;
    modelType: string | null;
    taskType: string | null;
    roundsCompleted: number | null;
    finalLoss: number | null;
    finalAccuracy: number | null;
    bestAccuracy: number | null;
    bestRound: number | null;
    finalF1Macro: number | null;
    finalPerplexity: number | null;
    bestPerplexity: number | null;
    finalEce: number | null;
    targetAccuracy: number | null;
    roundsToTarget: number | null;
    msToTarget: number | null;
    totalRoundMs: number | null;
    avgRoundMs: number | null;
    modelSizeMb: number | null;
    paramCount: number | null;
    clientCount: number | null;
    firstRecordedAt: string | null;
    lastRecordedAt: string | null;
    primaryMetricName: string | null;
    primaryMetricValue: number | null;
}

/** Platform-wide aggregates + the full runs table (one fetch). */
export interface BenchmarkOverview {
    benchmarkedProjects: number;
    totalRoundsRecorded: number;
    classificationRuns: number;
    generativeRuns: number;
    avgFinalAccuracy: number | null;
    avgFinalF1Macro: number | null;
    bestAccuracy: number | null;
    bestAccuracyProject: string | null;
    avgRoundDurationMs: number | null;
    avgModelSizeMb: number | null;
    runs: BenchmarkRun[];
}

/** One round's scalar metrics — the unit of the per-run time series. */
export interface BenchmarkRoundPoint {
    serverRound: number;
    loss: number | null;
    accuracy: number | null;
    balancedAccuracy: number | null;
    precisionMacro: number | null;
    recallMacro: number | null;
    f1Macro: number | null;
    f1Micro: number | null;
    f1Weighted: number | null;
    mcc: number | null;
    cohenKappa: number | null;
    rocAuc: number | null;
    logLoss: number | null;
    ece: number | null;
    brier: number | null;
    perplexity: number | null;
    tokenAccuracy: number | null;
    roundDurationMs: number | null;
    evalDurationMs: number | null;
    modelSizeMb: number | null;
    paramCount: number | null;
    clientCount: number | null;
    samplesEvaluated: number | null;
}

export interface PerClassMetric {
    label: string;
    precision: number | null;
    recall: number | null;
    f1: number | null;
    support: number | null;
}

/** Full per-project drilldown. */
export interface ProjectBenchmark {
    summary: BenchmarkRun | null;
    rounds: BenchmarkRoundPoint[];
    taskType: string | null;
    classLabels: string[] | null;
    latestPerClass: PerClassMetric[] | null;
    latestConfusionMatrix: number[][] | null;
}

export const fetchBenchmarkOverview = (): Promise<AxiosResponse<BenchmarkOverview>> => {
    return api.get<BenchmarkOverview>('/admin/benchmarks/overview');
};

export const fetchProjectBenchmark = (
    projectId: string,
): Promise<AxiosResponse<ProjectBenchmark>> => {
    return api.get<ProjectBenchmark>(`/admin/benchmarks/projects/${projectId}`);
};
