// =============================================================================
// FedLearn Frontend — LayoutV2 (Sidebar + Outlet)
// =============================================================================
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';

export function LayoutV2() {
  return (
    <div className="flex h-screen w-screen bg-canvas text-fg overflow-hidden font-sans">
      <Sidebar />
      <Outlet />
    </div>
  );
}
