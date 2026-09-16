import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MarketplaceView } from './MarketplaceView';
import * as registry from '../../services/artifactService';
import type { ArtifactDto } from '../../services/artifactService';

vi.mock('../../services/artifactService');

/** Build a published LORA_ADAPTER ArtifactDto, overriding only the fields a test cares about. */
function adapter(overrides: Partial<ArtifactDto> = {}): ArtifactDto {
  return {
    id: 'ad-1',
    orgId: 'org-1',
    projectId: 'proj-1',
    runId: 'run-1',
    kind: 'LORA_ADAPTER',
    blobSha256: 'a'.repeat(64),
    recipeKey: 'LORA_SST2',
    baseModelRef: 'bert-base',
    licenseTag: 'apache-2.0',
    evalCardJson: null,
    createdBy: 3,
    createdAt: '2026-07-01T00:00:00Z',
    published: true,
    publishedAt: '2026-07-05T00:00:00Z',
    ...overrides,
  };
}

// FE-12: the marketplace lists PUBLISHED LoRA adapters visible to the caller's
// org. Each entry surfaces its recipe, its license (prominently), and its
// parsed eval card.
describe('MarketplaceView — published feed (FE-12)', () => {
  it('renders the published adapters with license and eval card', async () => {
    vi.mocked(registry.listMarketplace).mockResolvedValue([
      adapter({
        id: 'ad-1',
        recipeKey: 'LORA_SST2',
        licenseTag: 'apache-2.0',
        baseModelRef: 'bert-base',
        evalCardJson: JSON.stringify({ accuracy: 0.93 }),
      }),
      adapter({
        id: 'ad-2',
        recipeKey: 'LORA_QNLI',
        licenseTag: 'mit',
        baseModelRef: 'roberta-base',
        blobSha256: 'c'.repeat(64),
      }),
    ]);

    render(<MarketplaceView />);

    // Both adapters show their recipe names.
    expect(await screen.findByText('LORA_SST2')).toBeInTheDocument();
    expect(screen.getByText('LORA_QNLI')).toBeInTheDocument();

    // The kind chip identifies them as adapters (one per card).
    expect(screen.getAllByText('LORA_ADAPTER')).toHaveLength(2);

    // Each adapter's license is surfaced prominently.
    expect(screen.getByText('apache-2.0')).toBeInTheDocument();
    expect(screen.getByText('mit')).toBeInTheDocument();

    // Provenance base model + parsed eval card.
    expect(screen.getByText('bert-base')).toBeInTheDocument();
    expect(screen.getByText('accuracy')).toBeInTheDocument();
    expect(screen.getByText('0.93')).toBeInTheDocument();
  });
});

// FE-12: org-scoped visibility means an empty feed is a normal result — it must
// read as "nothing published yet", not as an error or a blank screen.
describe('MarketplaceView — empty state (FE-12)', () => {
  it('renders an honest empty state when no adapters are published', async () => {
    vi.mocked(registry.listMarketplace).mockResolvedValue([]);

    render(<MarketplaceView />);

    expect(await screen.findByText('No published adapters yet')).toBeInTheDocument();
  });
});

// FE-12: a failed load must surface a readable banner, not crash the page.
describe('MarketplaceView — load error (FE-12)', () => {
  it('renders a load-error banner when the marketplace fails to load', async () => {
    vi.mocked(registry.listMarketplace).mockRejectedValue(new Error('network down'));

    render(<MarketplaceView />);

    expect(await screen.findByText('Failed to load the adapter marketplace.')).toBeInTheDocument();
  });
});
