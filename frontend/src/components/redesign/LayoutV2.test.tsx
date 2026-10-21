import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { beforeEach, afterEach, expect, it, vi } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { makeAuth } from '../../test/authFixtures';
import { LayoutV2 } from './LayoutV2';
vi.mock('../../context/AuthContext', async (original) => ({ ...await original<typeof import('../../context/AuthContext')>(), useAuth: vi.fn() }));
let resize: ((event: MediaQueryListEvent) => void) | undefined;
beforeEach(() => {
  vi.mocked(useAuth).mockReturnValue(makeAuth({ currentUser: { username: 'owner', email: 'owner@example.com', role: 'PROJECT_OWNER' }, isOwner: true }));
  vi.spyOn(window, 'matchMedia').mockReturnValue({ matches: false, addEventListener: (_: string, listener: EventListener) => { resize = listener as unknown as typeof resize; }, removeEventListener: () => {} } as unknown as MediaQueryList);
});
afterEach(() => { vi.restoreAllMocks(); resize = undefined; });
function show() { return render(<MemoryRouter initialEntries={['/simulations']}><Routes><Route element={<LayoutV2 />}><Route path="/simulations" element={<main>Training workspace</main>} /><Route path="/models" element={<main>Model catalog</main>} /></Route></Routes></MemoryRouter>); }
it('opens named mobile navigation and returns keyboard focus after Escape', () => {
  show(); const trigger = screen.getByRole('button', { name: 'Open navigation' }); trigger.focus(); fireEvent.click(trigger);
  expect(screen.getByRole('dialog', { name: 'Navigation' })).toContainElement(document.activeElement as HTMLElement);
  fireEvent.keyDown(document.activeElement!, { key: 'Escape' });
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument(); expect(trigger).toHaveFocus();
});
it('closes the mobile menu after navigation without replacing the routed workspace shell', () => {
  show(); fireEvent.click(screen.getByRole('button', { name: 'Open navigation' }));
  fireEvent.click(within(screen.getByRole('dialog', { name: 'Navigation' })).getByRole('link', { name: 'Models' }));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument(); expect(screen.getByText('Model catalog')).toBeInTheDocument();
});
it('releases a mobile navigation trap when the viewport becomes desktop width', () => {
  show(); fireEvent.click(screen.getByRole('button', { name: 'Open navigation' }));
  expect(screen.getByRole('dialog', { name: 'Navigation' })).toBeInTheDocument();
  act(() => resize?.({ matches: true } as MediaQueryListEvent));
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
});
