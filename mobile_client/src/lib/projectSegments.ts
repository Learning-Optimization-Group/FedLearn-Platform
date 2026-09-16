// Pure presentation logic for the Projects tab (stage 2): segment partitioning, the
// per-project contribution fold the "Joined" rows render, and the one-tap-join rule.
//
// Kept free of React/native imports so the segmented flow is unit-testable without a renderer.
import type { ClientProject } from './projectsApi';
import type { EligibilityResult } from './deviceCapabilities.types';
import type { ContributionEntry } from './contributionLedger';

/** A project row paired with its device-eligibility verdict (annotateEligibility output). */
export interface AnnotatedProject {
  project: ClientProject;
  result: EligibilityResult;
}

/** Per-project ledger summary for a "Joined" row: how much and how recently. */
export interface ProjectLedgerInfo {
  rounds: number;
  /** ISO timestamp of the newest contribution (entries are stored newest-first). */
  lastAt: string;
}

export type ProjectLedgerMap = Record<string, ProjectLedgerInfo>;

/**
 * Split the annotated projects into the two segments. Only an explicit `joined: true` counts
 * as joined — an absent flag means the backend didn't say, which must not hide the project
 * from Discover.
 */
export function splitProjects(rows: AnnotatedProject[]): {
  joined: AnnotatedProject[];
  discover: AnnotatedProject[];
} {
  const joined: AnnotatedProject[] = [];
  const discover: AnnotatedProject[] = [];
  for (const row of rows) {
    (row.project.joined === true ? joined : discover).push(row);
  }
  return { joined, discover };
}

/**
 * Fold the (newest-first) ledger into per-project round counts + last-contribution time.
 * The first entry seen per project is the newest, so its `at` wins.
 */
export function foldLedgerByProject(entries: ContributionEntry[]): ProjectLedgerMap {
  const map: ProjectLedgerMap = {};
  for (const e of entries) {
    const existing = map[e.projectId];
    if (existing) {
      existing.rounds += 1;
    } else {
      map[e.projectId] = { rounds: 1, lastAt: e.at };
    }
  }
  return map;
}

/** One-line "what has this phone done here" caption for a Joined row. */
export function formatLastContribution(info: ProjectLedgerInfo | undefined): string {
  if (!info || info.rounds === 0) return 'No contributions from this device yet';
  const noun = info.rounds === 1 ? 'round' : 'rounds';
  return `${info.rounds} ${noun} contributed · last ${new Date(info.lastAt).toLocaleString()}`;
}

/**
 * PUBLIC projects auto-join on request (RESTRICTED needs owner approval, PRIVATE is
 * invite-only), so only a PUBLIC project the device hasn't joined gets the one-tap Join
 * affordance. The affordance still routes through ProjectDetail's privacy label — the join
 * itself executes there, keeping the label as the single interstitial.
 */
export function canOneTapJoin(p: ClientProject): boolean {
  return p.visibility === 'PUBLIC' && p.joined === false;
}
