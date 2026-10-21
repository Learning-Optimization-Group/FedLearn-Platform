import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { makeAuth } from '../../test/authFixtures';
import { Sidebar } from './Sidebar';
vi.mock('../../context/AuthContext', async (original) => ({ ...await original<typeof import('../../context/AuthContext')>(), useAuth: vi.fn() }));
it.each(['PROJECT_OWNER', 'PLATFORM_ADMIN'] as const)('shows the Lab navigation to %s', role => {
  vi.mocked(useAuth).mockReturnValue(makeAuth({ currentUser: { username: 'owner', email: 'o@example.com', role } }));
  render(<MemoryRouter><Sidebar /></MemoryRouter>);
  expect(screen.getByRole('link', { name: 'Simulation Lab' })).toHaveAttribute('href', '/simulations');
});
describe('member navigation', () => {
  it('does not offer an unauthorized Lab link', () => {
    vi.mocked(useAuth).mockReturnValue(makeAuth({ currentUser: { username: 'member', email: 'm@example.com', role: 'USER' } }));
    render(<MemoryRouter><Sidebar /></MemoryRouter>);
    expect(screen.queryByRole('link', { name: 'Simulation Lab' })).not.toBeInTheDocument();
  });
});
