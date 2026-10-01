import { NativeModules } from 'react-native';

// Stage 3 artifact delivery (wikis/mobile/11, slice A3). The Kotlin ArtifactService streams a contract artifact
// straight into app-private storage with React Native's shared HTTP client (so the session cookie applies),
// refusing it unless its size and SHA-256 match what the execution contract declares. Nothing crosses the JS
// bridge except the URL, the digest, the size and the resulting local path.
interface ArtifactServiceModule {
  fetchArtifact(url: string, sha256: string, byteSize: number): Promise<string>;
}

const ArtifactService = NativeModules.ArtifactService as ArtifactServiceModule | undefined;

/** Download (or re-use) one contract artifact; resolves to its verified local path. */
export async function fetchArtifact(url: string, sha256: string, byteSize: number): Promise<string> {
  if (!ArtifactService) {
    throw new Error('ArtifactService is not available in this build');
  }
  return ArtifactService.fetchArtifact(url, sha256, byteSize);
}
