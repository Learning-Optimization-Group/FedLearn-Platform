// Pure decision table for ProjectDetail's single action slot (one primary action per view).
//
// Precedence, most-specific first:
//   1. this project IS the joined run            → Leave
//   2. some other project is the joined run      → explain, no action (one run per device)
//   3. native FL core missing (iOS scaffold)     → explain, no action (MO-14)
//   4. device fails the project's hard gates     → explain, no action
//   5. non-member of a non-PUBLIC project        → needs owner approval (visibility tiers)
//   6. otherwise                                 → Join (with the REST membership join first
//                                                  when the device isn't a member yet)
import type { ClientProject } from './projectsApi';

export type JoinDecision =
  | { kind: 'join'; needsMembership: boolean }
  | { kind: 'leave' }
  | { kind: 'busyElsewhere'; otherProjectName: string }
  | { kind: 'unavailable' }
  | { kind: 'ineligible' }
  | { kind: 'needsApproval' };

export interface JoinDecisionInput {
  project: ClientProject;
  /** Device passes the project's hard eligibility gates. */
  eligible: boolean;
  /** The native FL core is compiled into this build. */
  nativeAvailable: boolean;
  /** Project id of the run this device is currently joined to (null when idle). */
  activeRunProjectId: string | null;
  /** Human name of that project (falls back to its id upstream). */
  activeRunProjectName: string | null;
}

export function decideJoinAction(input: JoinDecisionInput): JoinDecision {
  const { project, eligible, nativeAvailable, activeRunProjectId, activeRunProjectName } = input;
  if (activeRunProjectId === project.projectId) return { kind: 'leave' };
  if (activeRunProjectId != null) {
    return { kind: 'busyElsewhere', otherProjectName: activeRunProjectName ?? activeRunProjectId };
  }
  if (!nativeAvailable) return { kind: 'unavailable' };
  if (!eligible) return { kind: 'ineligible' };
  const member = project.joined === true;
  if (!member && project.visibility !== 'PUBLIC') return { kind: 'needsApproval' };
  return { kind: 'join', needsMembership: !member };
}
