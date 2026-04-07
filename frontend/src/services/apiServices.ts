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

// Authentication Endpoints
export const loginUser = (credentials: LoginCredentials): Promise<AxiosResponse> => {
    return api.post('/auth/login', credentials);
};

export const registerUser = (userData: RegisterData): Promise<AxiosResponse> => {
    return api.post('/auth/register', userData);
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

export const deleteProject = (projectId: string): Promise<AxiosResponse<string>> => {
    return api.post<string>(`/projects/${projectId}/delete`);
};
