import type { ClientProject, DiscoverProject, MyAccessRequest, TrainResult } from './types';

export async function fetchMyProjects(): Promise<ClientProject[]> {
  const r = await window.fedLearnAPI.listClientProjects();
  return r.success && Array.isArray(r.projects) ? r.projects : [];
}

export async function fetchDiscoverProjects(): Promise<DiscoverProject[]> {
  const r = await window.fedLearnAPI.listDiscover();
  return r.success && Array.isArray(r.projects) ? r.projects : [];
}

export async function fetchMyRequests(): Promise<MyAccessRequest[]> {
  const r = await window.fedLearnAPI.listMyRequests();
  return r.success && Array.isArray(r.requests) ? r.requests : [];
}

export async function requestAccess(
  projectId: string,
  message?: string,
): Promise<{ success: boolean; status?: 'JOINED' | 'PENDING'; error?: string }> {
  const r = await window.fedLearnAPI.requestAccess(projectId, message ?? '');
  return { ...r, status: r.status as 'JOINED' | 'PENDING' | undefined };
}

export async function trainProject(projectId: string, datasetPath: string): Promise<TrainResult> {
  return window.fedLearnAPI.trainProject(projectId, datasetPath);
}

export async function getLastDatasetPath(projectId: string): Promise<string> {
  const r = await window.fedLearnAPI.getLastDatasetPath(projectId);
  return r.success && typeof r.path === 'string' ? r.path : '';
}

export async function setLastDatasetPath(projectId: string, path: string): Promise<void> {
  await window.fedLearnAPI.setLastDatasetPath(projectId, path);
}
