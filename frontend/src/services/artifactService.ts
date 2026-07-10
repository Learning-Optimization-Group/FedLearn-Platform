// =============================================================================
// FedLearn Frontend — Artifact registry service (FE-11)
// =============================================================================
// Typed client for the content-addressed model-artifact registry. Reuses the
// app's single shared axios instance (cookie auth, `withCredentials: true`;
// there is no Bearer token and nothing in localStorage) — the SAME singleton
// that `apiServices.ts` imports from `../api/axiosConfig`.

import api from '../api/axiosConfig';

/** One versioned, content-addressed model artifact. */
export interface ArtifactDto {
    id: string;
    orgId: string;
    projectId: string;
    runId: string | null;
    /** e.g. FULL_CHECKPOINT | ADAPTER | BASE_REF. */
    kind: string;
    /** 64-hex content address of the immutable weights blob. */
    blobSha256: string;
    recipeKey: string | null;
    baseModelRef: string | null;
    licenseTag: string | null;
    /** RAW JSON string — parse defensively (try/catch); may be null or invalid. */
    evalCardJson: string | null;
    createdBy: number | null;
    /** ISO-8601 instant. */
    createdAt: string;
}

/** One ancestor in an artifact's lineage chain (GET /artifacts/{id}/lineage). */
export interface LineageNode {
    id: string;
    kind: string;
    sha256: string;
    baseModelRef: string | null;
    licenseTag: string | null;
    createdAt: string | null;
}

/** Lists a project's artifacts, newest-first. Cross-org rows are filtered server-side. */
export async function listArtifacts(projectId: string): Promise<ArtifactDto[]> {
    const res = await api.get<ArtifactDto[]>('/artifacts', { params: { projectId } });
    return Array.isArray(res.data) ? res.data : [];
}

/** Fetches a single artifact by id. */
export async function getArtifact(id: string): Promise<ArtifactDto> {
    const res = await api.get<ArtifactDto>(`/artifacts/${id}`);
    return res.data;
}

/** Fetches an artifact's ancestor chain (root-most first as returned by the backend). */
export async function getLineage(id: string): Promise<LineageNode[]> {
    const res = await api.get<LineageNode[]>(`/artifacts/${id}/lineage`);
    return Array.isArray(res.data) ? res.data : [];
}

/** Downloads the immutable weights blob (octet-stream) as a Blob for a local object URL. */
export async function downloadBlob(id: string): Promise<Blob> {
    const res = await api.get<Blob>(`/artifacts/${id}/blob`, { responseType: 'blob' });
    return res.data;
}
