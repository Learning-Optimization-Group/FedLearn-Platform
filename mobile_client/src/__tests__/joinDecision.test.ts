// Decision table for ProjectDetail's single action slot — full precedence matrix.
import { decideJoinAction, type JoinDecisionInput } from '../lib/joinDecision';
import type { ClientProject } from '../lib/projectsApi';

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

function input(over: Partial<JoinDecisionInput>): JoinDecisionInput {
  return {
    project: proj({}),
    eligible: true,
    nativeAvailable: true,
    activeRunProjectId: null,
    activeRunProjectName: null,
    ...over,
  };
}

describe('decideJoinAction', () => {
  it('this project being the joined run wins over everything → leave', () => {
    // Even with eligibility/native flags down, an active run here must stay leavable.
    expect(
      decideJoinAction(
        input({ activeRunProjectId: 'p1', eligible: false, nativeAvailable: false }),
      ),
    ).toEqual({ kind: 'leave' });
  });

  it('a run on another project blocks joining (one run per device), named for the user', () => {
    expect(
      decideJoinAction(input({ activeRunProjectId: 'other', activeRunProjectName: 'ECG Study' })),
    ).toEqual({ kind: 'busyElsewhere', otherProjectName: 'ECG Study' });
  });

  it('falls back to the other project id when its name is unknown', () => {
    expect(decideJoinAction(input({ activeRunProjectId: 'other' }))).toEqual({
      kind: 'busyElsewhere',
      otherProjectName: 'other',
    });
  });

  it('missing native core (iOS scaffold) → unavailable', () => {
    expect(decideJoinAction(input({ nativeAvailable: false }))).toEqual({ kind: 'unavailable' });
  });

  it('hard eligibility failure → ineligible', () => {
    expect(decideJoinAction(input({ eligible: false }))).toEqual({ kind: 'ineligible' });
  });

  it('non-member of a non-PUBLIC project → needsApproval', () => {
    for (const visibility of ['RESTRICTED', 'PRIVATE', null]) {
      expect(
        decideJoinAction(input({ project: proj({ visibility, joined: false }) })),
      ).toEqual({ kind: 'needsApproval' });
    }
  });

  it('PUBLIC non-member → join with the REST membership join first', () => {
    expect(decideJoinAction(input({ project: proj({ joined: false }) }))).toEqual({
      kind: 'join',
      needsMembership: true,
    });
  });

  it('existing member → join straight into the run (any visibility)', () => {
    for (const visibility of ['PUBLIC', 'RESTRICTED', 'PRIVATE']) {
      expect(
        decideJoinAction(input({ project: proj({ visibility, joined: true }) })),
      ).toEqual({ kind: 'join', needsMembership: false });
    }
  });
});
