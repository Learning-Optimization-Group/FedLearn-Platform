// Pure logic behind the segmented Projects tab: segment split, the per-project ledger fold
// the Joined rows render, the last-contribution caption, and the one-tap-join rule.
import {
  canOneTapJoin,
  foldLedgerByProject,
  formatLastContribution,
  splitProjects,
  type AnnotatedProject,
} from '../lib/projectSegments';
import type { ClientProject } from '../lib/projectsApi';
import type { ContributionEntry } from '../lib/contributionLedger';

const ok = { eligible: true, hardFailures: [], softWarnings: [] };

function proj(over: Partial<ClientProject>): ClientProject {
  return {
    projectId: 'p1',
    name: 'Project',
    modelType: 'CNN',
    status: 'READY',
    visibility: 'PUBLIC',
    ...over,
  };
}

function entry(over: Partial<ContributionEntry>): ContributionEntry {
  return {
    projectId: 'p1',
    projectName: 'Project',
    round: 1,
    wallClockMs: 1000,
    bytesUp: 100,
    bytesDown: 200,
    at: '2026-07-01T10:00:00.000Z',
    ...over,
  };
}

describe('splitProjects', () => {
  it('routes only explicit joined:true rows to the Joined segment', () => {
    const rows: AnnotatedProject[] = [
      { project: proj({ projectId: 'a', joined: true }), result: ok },
      { project: proj({ projectId: 'b', joined: false }), result: ok },
      { project: proj({ projectId: 'c' }), result: ok }, // flag absent → Discover, never hidden
    ];
    const { joined, discover } = splitProjects(rows);
    expect(joined.map((r) => r.project.projectId)).toEqual(['a']);
    expect(discover.map((r) => r.project.projectId)).toEqual(['b', 'c']);
  });

  it('handles an empty list', () => {
    expect(splitProjects([])).toEqual({ joined: [], discover: [] });
  });
});

describe('foldLedgerByProject', () => {
  it('counts rounds per project and keeps the newest timestamp (entries are newest-first)', () => {
    const fold = foldLedgerByProject([
      entry({ projectId: 'a', round: 9, at: '2026-07-03T00:00:00.000Z' }),
      entry({ projectId: 'b', round: 2, at: '2026-07-02T00:00:00.000Z' }),
      entry({ projectId: 'a', round: 8, at: '2026-07-01T00:00:00.000Z' }),
    ]);
    expect(fold.a).toEqual({ rounds: 2, lastAt: '2026-07-03T00:00:00.000Z' });
    expect(fold.b).toEqual({ rounds: 1, lastAt: '2026-07-02T00:00:00.000Z' });
  });

  it('is empty for an empty ledger', () => {
    expect(foldLedgerByProject([])).toEqual({});
  });
});

describe('formatLastContribution', () => {
  it('reads honestly when the device has never contributed', () => {
    expect(formatLastContribution(undefined)).toBe('No contributions from this device yet');
  });

  it('singular/plural round wording with the last-contribution time', () => {
    const one = formatLastContribution({ rounds: 1, lastAt: '2026-07-03T00:00:00.000Z' });
    expect(one).toContain('1 round contributed');
    expect(one).not.toContain('rounds');
    const many = formatLastContribution({ rounds: 3, lastAt: '2026-07-03T00:00:00.000Z' });
    expect(many).toContain('3 rounds contributed');
    expect(many).toContain('last ');
  });
});

describe('canOneTapJoin', () => {
  it('only a PUBLIC project the device has not joined qualifies', () => {
    expect(canOneTapJoin(proj({ visibility: 'PUBLIC', joined: false }))).toBe(true);
    expect(canOneTapJoin(proj({ visibility: 'PUBLIC', joined: true }))).toBe(false);
    expect(canOneTapJoin(proj({ visibility: 'RESTRICTED', joined: false }))).toBe(false);
    expect(canOneTapJoin(proj({ visibility: 'PRIVATE', joined: false }))).toBe(false);
    // joined flag absent → membership unknown → no one-tap join
    expect(canOneTapJoin(proj({ visibility: 'PUBLIC' }))).toBe(false);
    expect(canOneTapJoin(proj({ visibility: null, joined: false }))).toBe(false);
  });
});
