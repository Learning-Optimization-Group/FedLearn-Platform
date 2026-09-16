// =============================================================================
// Dataset-path consent allowlist
// =============================================================================
// docker:start-training bind-mounts the renderer-supplied datasetPath into the
// training container. Even after sanitizeDatasetPath proves the path is an
// existing absolute directory, that does NOT prove the USER chose it: a
// compromised renderer (threat model b) could pass an arbitrary host directory
// (e.g. ~/.ssh) to mount into the container. This module records the paths the
// user explicitly selected via the native "Select dataset" dialog
// (dialog:open-directory), and docker:start-training only mounts a path that is
// on this list — so an arbitrary, non-user-selected path is refused. Getting a
// path onto the list requires a real dialog invocation and a physical user
// selection, which a headless renderer compromise cannot forge.
//
// The set is in-memory (per app run). The renderer always re-selects the dataset
// via the dialog within the same session (HardwareSelector holds datasetPath in
// component state, not persisted), so no cross-restart persistence is needed.
import path from 'path';

const consented = new Set<string>();

/** Record a directory the user picked via the native dialog as consented for mounting. */
export function recordConsentedDatasetPath(p: string): void {
  if (typeof p !== 'string' || p.length === 0) {
    return;
  }
  try {
    consented.add(path.resolve(p));
  } catch {
    /* unresolvable path — never consent it */
  }
}

/** True iff `resolvedPath` (already path.resolve'd, as sanitizeDatasetPath returns) was user-selected. */
export function isDatasetPathConsented(resolvedPath: string): boolean {
  return consented.has(resolvedPath);
}

/** Test seam: clear all recorded consent. */
export function _clearConsentedDatasetPaths(): void {
  consented.clear();
}
