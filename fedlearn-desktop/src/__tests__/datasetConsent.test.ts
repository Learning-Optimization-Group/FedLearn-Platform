// Client-audit: docker:start-training bind-mounts the renderer-supplied datasetPath into the training
// container. sanitizeDatasetPath proves the path is an existing absolute dir but does NOT prove the USER
// chose it — a compromised renderer could pass an arbitrary host dir (e.g. ~/.ssh). The consent allowlist
// closes that: only a path returned by the native "Select dataset" dialog (a physical user selection) may
// be mounted.
import path from 'path';
import {
  recordConsentedDatasetPath,
  isDatasetPathConsented,
  _clearConsentedDatasetPaths,
} from '../main/dataset-consent';

describe('dataset-path consent allowlist', () => {
  beforeEach(() => _clearConsentedDatasetPaths());

  it('consents a path the user selected via the dialog, normalized (trailing slash)', () => {
    recordConsentedDatasetPath('/Users/me/datasets/xray/');
    expect(isDatasetPathConsented(path.resolve('/Users/me/datasets/xray'))).toBe(true);
  });

  it('does NOT consent a path never returned by the dialog (the attacker case)', () => {
    recordConsentedDatasetPath('/Users/me/datasets/xray');
    expect(isDatasetPathConsented('/Users/victim/.ssh')).toBe(false);
  });

  it('nothing is consented until the dialog records a selection', () => {
    expect(isDatasetPathConsented('/anything')).toBe(false);
  });

  it('ignores empty / non-string inputs without recording them', () => {
    recordConsentedDatasetPath('');
    recordConsentedDatasetPath(undefined as unknown as string);
    expect(isDatasetPathConsented('')).toBe(false);
  });
});
