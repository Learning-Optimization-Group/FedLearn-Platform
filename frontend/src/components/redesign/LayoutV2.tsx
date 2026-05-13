// =============================================================================
// FedLearn Frontend — LayoutV2 (Sidebar + Outlet)
// =============================================================================
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';

export function LayoutV2() {
  return (
    <div
      className="flex h-screen w-screen overflow-hidden font-sans transition-colors duration-300"
      style={{
        backgroundColor: 'var(--background-primary)',
        color: 'var(--text-primary)',
      }}
    >
      <Sidebar />
      <Outlet />
    </div>
  );
}
