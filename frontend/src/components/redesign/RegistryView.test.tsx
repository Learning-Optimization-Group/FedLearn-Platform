import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import type { AxiosResponse } from 'axios';
import { RegistryView } from './RegistryView';
import * as api from '../../services/apiServices';
import * as registry from '../../services/artifactService';
import type { ArtifactDto, LineageNode } from '../../services/artifactService';

vi.mock('../../services/apiServices');
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
