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
    visibility?: 'PUBLIC' | 'PRIVATE';
    myRelationship?: 'OWNER' | 'MEMBER' | 'CLIENT' | null;
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

export const fetchProject = (projectId: string): Promise<AxiosResponse<Project>> =>
    api.get<Project>(`/projects/${projectId}`);

export const patchProject = (
    projectId: string,
    data: { name?: string; description?: string; visibility?: 'PUBLIC' | 'PRIVATE' }
): Promise<AxiosResponse<Project>> =>
    api.patch<Project>(`/projects/${projectId}`, data);

// ─── Discover ───────────────────────────────────────────────────────────────
export const fetchDiscover = (): Promise<AxiosResponse<DiscoverProject[]>> =>
    api.get<DiscoverProject[]>('/projects/discover');

// ─── Memberships ────────────────────────────────────────────────────────────
export const fetchMemberships = (projectId: string): Promise<AxiosResponse<Membership[]>> =>
    api.get<Membership[]>(`/projects/${projectId}/memberships`);

export const addMembership = (
    projectId: string,
    body: { username: string; role: 'MEMBER' | 'CLIENT' }
): Promise<AxiosResponse<Membership>> =>
    api.post<Membership>(`/projects/${projectId}/memberships`, body);

export const removeMembership = (projectId: string, userId: number): Promise<AxiosResponse<void>> =>
    api.delete<void>(`/projects/${projectId}/memberships/${userId}`);

// ─── Access Requests (project-scoped) ───────────────────────────────────────
export const fetchProjectAccessRequests = (projectId: string): Promise<AxiosResponse<AccessRequest[]>> =>
    api.get<AccessRequest[]>(`/projects/${projectId}/access-requests`);

export const createAccessRequest = (projectId: string, message?: string): Promise<AxiosResponse<AccessRequest>> =>
    api.post<AccessRequest>(`/projects/${projectId}/access-requests`, message ? { message } : {});

export const decideAccessRequest = (
    projectId: string,
    reqId: number,
    decision: 'APPROVED' | 'DENIED'
): Promise<AxiosResponse<AccessRequest>> =>
    api.put<AccessRequest>(`/projects/${projectId}/access-requests/${reqId}`, { decision });

// ─── My Access Requests ──────────────────────────────────────────────────────
export const fetchMyAccessRequests = (): Promise<AxiosResponse<AccessRequest[]>> =>
    api.get<AccessRequest[]>('/my/access-requests');

// ─── Admin ──────────────────────────────────────────────────────────────────
export const fetchAdminUsers = (): Promise<AxiosResponse<AdminUser[]>> =>
    api.get<AdminUser[]>('/admin/users');

export const updateUserRole = (userId: number, role: 'USER' | 'ADMIN'): Promise<AxiosResponse<AdminUser>> =>
    api.put<AdminUser>(`/admin/users/${userId}/role`, { role });

export const fetchAdminProjects = (): Promise<AxiosResponse<Project[]>> =>
    api.get<Project[]>('/admin/projects');

// ─── User Search ─────────────────────────────────────────────────────────────
export const searchUsers = (q: string): Promise<AxiosResponse<UserSearchResult[]>> =>
    api.get<UserSearchResult[]>(`/users/search?q=${encodeURIComponent(q)}`);

// User / Client Management Endpoints
export interface User {
    id: number;
    username: string;
    email: string;
    role?: 'USER' | 'ADMIN';
    createdAt?: string;
}

// ─── RBAC types ─────────────────────────────────────────────────────────────

export interface Membership {
    projectId: string;
    userId: number;
    username: string;
    role: 'MEMBER' | 'CLIENT' | 'OWNER';
    partitionId?: number | null;
    joinedVia: string;
    addedAt: string;
}

export interface AccessRequest {
    id: number;
    projectId: string;
    projectName: string;
    userId: number;
    username: string;
    status: 'PENDING' | 'APPROVED' | 'DENIED';
    message?: string;
    requestedAt: string;
    decidedAt?: string;
    decidedByUsername?: string;
}

export interface DiscoverProject {
    id: string;
    name: string;
    visibility: 'PUBLIC' | 'PRIVATE';
    ownerUsername: string;
    modelType: string;
    myRequestStatus: 'NONE' | 'PENDING' | 'APPROVED' | 'DENIED';
    lastAccuracy?: number;
    description?: string;
}

export interface AdminUser {
    id: number;
    username: string;
    email: string;
    role: 'USER' | 'ADMIN';
    projectsOwned: number;
    memberships: number;
    createdAt: string;
}

export interface UserSearchResult {
    id: number;
    username: string;
}

export interface AppNotification {
    id: string;
    type: 'ACCESS_REQUEST_CREATED' | 'ACCESS_REQUEST_DECIDED' | 'MEMBERSHIP_ADDED' | 'MEMBERSHIP_REMOVED' | 'PROJECT_VISIBILITY_CHANGED';
    projectId: string;
    projectName: string;
    actorId: number;
    actorUsername: string;
    subjectId?: number;
    subjectUsername?: string;
    decision?: string;
    role?: string;
    timestamp: string;
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
