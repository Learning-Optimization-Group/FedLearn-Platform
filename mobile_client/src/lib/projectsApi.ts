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

/**
 * Pairs each project with its eligibility verdict against the device. Pure.
 *
 * A model whose catalog entry is not yet cleared for phones (mobileSafe false) is refused by a release build. A
 * development build may join it, with a warning, because clearing it takes a measurement on a phone; every other
 * requirement still applies. The shared rule in evaluateEligibility (kept in sync with the desktop app) is unchanged.
 */
export function annotateEligibility(
  projects: ClientProject[],
  caps: DeviceCapabilities,
): Array<{ project: ClientProject; result: EligibilityResult }> {
  return projects.map((project) => ({ project, result: eligibility(caps, project.requirements) }));
}

const NOT_CLEARED_WARNING = 'Not cleared for phones yet: a development build may train it to measure it';

function eligibility(caps: DeviceCapabilities, req: DeviceRequirements | undefined): EligibilityResult {
  if (!__DEV__ || req?.mobileSafe !== false) return evaluateEligibility(caps, req);
  const result = evaluateEligibility(caps, { ...req, mobileSafe: undefined });
  return { ...result, softWarnings: [NOT_CLEARED_WARNING, ...result.softWarnings] };
}
