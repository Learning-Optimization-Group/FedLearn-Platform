import { NativeModules } from 'react-native';
import { getToken } from './authStore';
import { captureAuthAdmission, requireCurrentAuthAdmission } from './backendScope';
import { requestOrigin } from './backendOrigin';

// Stage 3 artifact delivery (wikis/mobile/11, slice A3). The Kotlin ArtifactService streams a contract artifact
// into app-private storage, refusing bytes unless their size and SHA-256 match the contract. Authentication is
// request-specific and bound to the captured server; native downloads never inherit a shared cookie jar.
interface ArtifactServiceModule {
  fetchArtifact(url: string, sha256: string, byteSize: number, credentialOrigin: string, token: string): Promise<string>;
}

const ArtifactService = NativeModules.ArtifactService as ArtifactServiceModule | undefined;

/** Download (or re-use) one contract artifact; resolves to its verified local path. */
export async function fetchArtifact(url: string, sha256: string, byteSize: number): Promise<string> {
  if (!ArtifactService) {
    throw new Error('ArtifactService is not available in this build');
  }
  const admission = captureAuthAdmission();
  const origin = requestOrigin(admission.scope.origin, url);
  if (!origin) throw new Error('The artifact URL has no valid server origin.');
  const token = origin === admission.scope.origin ? await getToken(origin) : null;
  requireCurrentAuthAdmission(admission);
  const absolute = /^https?:\/\//i.test(url) ? url : url.startsWith('//')
    ? (admission.scope.origin.startsWith('https:') ? 'https:' : 'http:') + url : admission.scope.origin + url;
  return ArtifactService.fetchArtifact(absolute, sha256, byteSize, admission.scope.origin, token ?? '');
}
