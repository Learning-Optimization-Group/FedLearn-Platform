import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import type { AxiosResponse } from 'axios';
import { RegistryView } from './RegistryView';
import * as api from '../../services/apiServices';
import * as registry from '../../services/artifactService';
import type { ArtifactDto, LineageNode } from '../../services/artifactService';

// Keep the REAL error helpers (errorMessage/errorStatus) so the publish-failure
// path renders the backend's message rather than a mocked stub; only the data
// fetch (fetchProjects) is stubbed.
vi.mock('../../services/apiServices', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../../services/apiServices')>();
  return { ...actual, fetchProjects: vi.fn() };
});
vi.mock('../../services/artifactService');

/** Minimal AxiosResponse wrapper — the view only ever reads `.data`. */
function resp<T>(data: T): AxiosResponse<T> {
  return { data, status: 200, statusText: 'OK', headers: {}, config: {} } as unknown as AxiosResponse<T>;
}

const PROJECT: api.Project = {
  id: 'proj-1',
  name: 'Pneumonia CNN',
  modelType: 'PNEUMONIA_CNN',
  modelName: 'pneumonia-cnn',
  optimizer: 'SGD',
  status: 'COMPLETED',
};

/** Build an ArtifactDto, overriding only the fields a test cares about. */
function artifact(overrides: Partial<ArtifactDto> = {}): ArtifactDto {
  return {
    id: 'art-1',
    orgId: 'org-1',
    projectId: 'proj-1',
    runId: 'run-9',
    kind: 'FULL_CHECKPOINT',
    blobSha256: 'a'.repeat(64),
    recipeKey: 'CNN',
    baseModelRef: null,
    licenseTag: 'apache-2.0',
    evalCardJson: null,
    createdBy: 7,
    createdAt: '2026-07-01T12:00:00Z',
    published: false,
    publishedAt: null,
    ...overrides,
  };
}

const LINEAGE_NODE: LineageNode = {
  id: 'anc-1',
  kind: 'BASE_REF',
  sha256: 'b'.repeat(64),
  baseModelRef: null,
  licenseTag: 'mit',
  createdAt: '2026-06-01T00:00:00Z',
};

beforeEach(() => {
  vi.mocked(api.fetchProjects).mockResolvedValue(resp([PROJECT]));
  vi.mocked(registry.getLineage).mockResolvedValue([]);
});

// FE-11: the registry lists a project's content-addressed artifacts. The first
// project is auto-selected, so its artifacts load without extra interaction.
describe('RegistryView — artifact list (FE-11)', () => {
  it('renders the selected project\'s artifacts', async () => {
    vi.mocked(registry.listArtifacts).mockResolvedValue([
      artifact({ id: 'art-1', recipeKey: 'CNN', kind: 'FULL_CHECKPOINT' }),
      artifact({ id: 'art-2', recipeKey: 'MLP', kind: 'ADAPTER', blobSha256: 'c'.repeat(64) }),
    ]);

    render(<RegistryView />);

    expect(await screen.findByText('CNN')).toBeInTheDocument();
    expect(screen.getByText('MLP')).toBeInTheDocument();
    expect(screen.getByText('FULL_CHECKPOINT')).toBeInTheDocument();
    expect(screen.getByText('ADAPTER')).toBeInTheDocument();
    // The list was scoped to the auto-selected project.
    expect(registry.listArtifacts).toHaveBeenCalledWith('proj-1');
  });
});

// FE-11: clicking an artifact surfaces its parsed eval card, its provenance,
// and its lineage chain.
describe('RegistryView — artifact detail (FE-11)', () => {
  it('parses and renders the eval card, provenance, and lineage on click', async () => {
    vi.mocked(registry.listArtifacts).mockResolvedValue([
      artifact({
        evalCardJson: JSON.stringify({ accuracy: 0.912, loss: 0.08 }),
        baseModelRef: 'resnet18',
      }),
    ]);
    vi.mocked(registry.getLineage).mockResolvedValue([LINEAGE_NODE]);

    render(<RegistryView />);
    fireEvent.click(await screen.findByRole('button', { name: /view artifact FULL_CHECKPOINT/i }));

    // Eval card key/values parsed from the raw JSON string.
    expect(await screen.findByText('accuracy')).toBeInTheDocument();
    expect(screen.getByText('0.912')).toBeInTheDocument();
    expect(screen.getByText('loss')).toBeInTheDocument();

    // Provenance carries the base-model ref.
    expect(screen.getByText('resnet18')).toBeInTheDocument();

    // Lineage renders the ancestor's short sha (from getLineage).
    expect(await screen.findByText('bbbbbbbbbbbb')).toBeInTheDocument();
    expect(registry.getLineage).toHaveBeenCalledWith('art-1');
  });

  it('shows "No eval card" for a null eval card without crashing', async () => {
    vi.mocked(registry.listArtifacts).mockResolvedValue([artifact({ evalCardJson: null })]);

    render(<RegistryView />);
    fireEvent.click(await screen.findByRole('button', { name: /view artifact/i }));

    expect(await screen.findByText('No eval card')).toBeInTheDocument();
  });

  it('shows "No eval card" for malformed eval-card JSON without crashing', async () => {
    vi.mocked(registry.listArtifacts).mockResolvedValue([
      artifact({ evalCardJson: '{ this is not: valid json' }),
    ]);

    render(<RegistryView />);
    fireEvent.click(await screen.findByRole('button', { name: /view artifact/i }));

    expect(await screen.findByText('No eval card')).toBeInTheDocument();
  });
});

// FE-11: an honest empty state — a project with zero artifacts must not read as
// an error or a blank screen.
describe('RegistryView — empty state (FE-11)', () => {
  it('renders an honest empty state when the project has no artifacts', async () => {
    vi.mocked(registry.listArtifacts).mockResolvedValue([]);

    render(<RegistryView />);

    expect(await screen.findByText('No artifacts yet')).toBeInTheDocument();
  });
});

// FE-12: an owner can publish/unpublish a LORA_ADAPTER to the marketplace from
// the artifact detail panel. The toggle only appears for LORA_ADAPTER kinds; the
// server is the source of truth for authorization, so a 403 must surface a
// readable message rather than a crash.
describe('RegistryView — marketplace publish toggle (FE-12)', () => {
  it('publishes a LORA_ADAPTER and reflects the published state', async () => {
    const lora = artifact({
      id: 'art-lora',
      kind: 'LORA_ADAPTER',
      recipeKey: 'LORA',
      published: false,
      publishedAt: null,
    });
    vi.mocked(registry.listArtifacts).mockResolvedValue([lora]);
    vi.mocked(registry.publishAdapter).mockResolvedValue({
      ...lora,
      published: true,
      publishedAt: '2026-07-10T00:00:00Z',
    });

    render(<RegistryView />);
    fireEvent.click(await screen.findByRole('button', { name: /view artifact LORA_ADAPTER/i }));

    fireEvent.click(await screen.findByRole('button', { name: /publish to marketplace/i }));

    // The row's published state flips; the button becomes "Unpublish".
    expect(await screen.findByRole('button', { name: /unpublish/i })).toBeInTheDocument();
    expect(screen.getByText('Listed on the marketplace')).toBeInTheDocument();
    expect(registry.publishAdapter).toHaveBeenCalledWith('art-lora');
  });

  it('unpublishes a listed LORA_ADAPTER and reflects the withdrawn state', async () => {
    const listed = artifact({
      id: 'art-lora',
      kind: 'LORA_ADAPTER',
      recipeKey: 'LORA',
      published: true,
      publishedAt: '2026-07-10T00:00:00Z',
    });
    vi.mocked(registry.listArtifacts).mockResolvedValue([listed]);
    // The unpublish button must call the DISTINCT unpublishAdapter method, not publishAdapter.
    vi.mocked(registry.unpublishAdapter).mockResolvedValue({
      ...listed,
      published: false,
      publishedAt: null,
    });

    render(<RegistryView />);
    fireEvent.click(await screen.findByRole('button', { name: /view artifact LORA_ADAPTER/i }));

    // Starts listed → click Unpublish.
    expect(await screen.findByText('Listed on the marketplace')).toBeInTheDocument();
    fireEvent.click(await screen.findByRole('button', { name: /unpublish/i }));

    // State flips back; the button returns to "Publish to marketplace"; the right method was called.
    expect(await screen.findByRole('button', { name: /publish to marketplace/i })).toBeInTheDocument();
    expect(screen.getByText('Not published')).toBeInTheDocument();
    expect(registry.unpublishAdapter).toHaveBeenCalledWith('art-lora');
    expect(registry.publishAdapter).not.toHaveBeenCalled();
  });

  it('surfaces a readable error when publishing is forbidden (403)', async () => {
    const lora = artifact({ id: 'art-lora', kind: 'LORA_ADAPTER', recipeKey: 'LORA' });
    vi.mocked(registry.listArtifacts).mockResolvedValue([lora]);
    // A non-owner gets a 403 with a backend message; errorMessage must extract it.
    vi.mocked(registry.publishAdapter).mockRejectedValue({
      isAxiosError: true,
      response: { status: 403, data: { message: 'You are not the owner of this adapter.' } },
    });

    render(<RegistryView />);
    fireEvent.click(await screen.findByRole('button', { name: /view artifact LORA_ADAPTER/i }));
    fireEvent.click(await screen.findByRole('button', { name: /publish to marketplace/i }));

    // The backend message renders — not "[object Object]" and not a crash.
    expect(await screen.findByText('You are not the owner of this adapter.')).toBeInTheDocument();
    // Still unpublished — the toggle stays on "Publish to marketplace".
    expect(screen.getByRole('button', { name: /publish to marketplace/i })).toBeInTheDocument();
  });

  it('does not show the publish toggle for a non-adapter artifact', async () => {
    vi.mocked(registry.listArtifacts).mockResolvedValue([
      artifact({ id: 'art-ckpt', kind: 'FULL_CHECKPOINT' }),
    ]);

    render(<RegistryView />);
    fireEvent.click(await screen.findByRole('button', { name: /view artifact FULL_CHECKPOINT/i }));

    // Detail panel is up (Download is present) but no marketplace publish control.
    expect(await screen.findByRole('button', { name: /download/i })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: /publish to marketplace/i })).not.toBeInTheDocument();
  });
});
