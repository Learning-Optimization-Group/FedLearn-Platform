import { api } from './restClient';
import { evaluateEligibility } from './evaluateEligibility';
import type { DeviceCapabilities, DeviceRequirements, EligibilityResult } from './deviceCapabilities.types';

export interface ClientProject {
  projectId: string;
  name: string;
  modelType: string;
  status: string;
  visibility: string | null;
  joined?: boolean;
  recipeKey?: string;
  requirements?: DeviceRequirements;
}

export async function listProjects(): Promise<ClientProject[]> {
  const res = await api.get('/api/client/projects');
  return (res.data ?? []) as ClientProject[];
}

export async function joinProject(projectId: string): Promise<void> {
  await api.post(`/api/client/projects/${projectId}/join`);
}

/** Pairs each project with its eligibility verdict against the device. Pure. */
export function annotateEligibility(
  projects: ClientProject[],
  caps: DeviceCapabilities,
): Array<{ project: ClientProject; result: EligibilityResult }> {
  return projects.map((project) => ({
    project,
    result: evaluateEligibility(caps, project.requirements),
  }));
}
