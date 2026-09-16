import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { Client as StompClient } from '@stomp/stompjs';
import * as api from '../../services/apiServices';
import { ClientDashboard } from './ClientDashboard';
import { OwnerDashboard } from './OwnerDashboard';
import { AdminDashboard } from './AdminDashboard';

// Smoke coverage: each role's dashboard mounts and renders its own console against a mocked API
// layer — catches a mount-time crash (e.g. a null-deref on empty data) per role (FE-5).
vi.mock('../../services/apiServices');
vi.mock('@stomp/stompjs', () => ({ Client: vi.fn() }));

const ok = <T,>(data: T) => ({ data }) as { data: T };

// AdminDashboard deep-links into the directories with router <Link>s, so every
// dashboard mounts inside a MemoryRouter (harmless for the other two).
const renderWithRouter = (ui: React.ReactElement) => render(<MemoryRouter>{ui}</MemoryRouter>);

describe('role dashboards mount and render for their role', () => {
  beforeEach(() => {
    // Re-establish the STOMP client stub after restoreMocks resets factory implementations, so
    // OwnerDashboard's `new StompClient(...)` returns an inert client that never opens a socket.
    vi.mocked(StompClient).mockImplementation(() => ({
      onConnect: null,
      active: false,
      activate: vi.fn(),
      deactivate: vi.fn(),
      subscribe: vi.fn(),
    }) as unknown as InstanceType<typeof StompClient>);
  });

  it('ClientDashboard renders the client overview', async () => {
    vi.mocked(api.fetchMyOwnerRequest).mockResolvedValue(ok(null) as never);
    vi.mocked(api.fetchDiscoverableProjects).mockResolvedValue(ok([]) as never);
    renderWithRouter(<ClientDashboard />);
    expect(await screen.findByText('Overview')).toBeInTheDocument();
  });

  it('OwnerDashboard renders the owner project list', async () => {
    vi.mocked(api.fetchOwnedProjects).mockResolvedValue(ok([]) as never);
    vi.mocked(api.fetchProjectResults).mockResolvedValue(ok([]) as never);
    renderWithRouter(<OwnerDashboard />);
    expect(await screen.findByText('My projects')).toBeInTheDocument();
  });

  it('AdminDashboard renders the admin console', async () => {
    const emptyPage = { items: [], page: 0, size: 5, total: 0 };
    vi.mocked(api.fetchAdminOverview).mockResolvedValue(ok({}) as never);
    vi.mocked(api.searchAdminUsers).mockResolvedValue(ok(emptyPage) as never);
    vi.mocked(api.searchAdminProjects).mockResolvedValue(ok(emptyPage) as never);
    vi.mocked(api.fetchOwnerRequests).mockResolvedValue(ok([]) as never);
    vi.mocked(api.fetchDeletionRequests).mockResolvedValue(ok([]) as never);
    renderWithRouter(<AdminDashboard />);
    expect(await screen.findByText('Admin')).toBeInTheDocument();
  });
});
