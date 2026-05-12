// =============================================================================
// FedLearn Frontend — LayoutV2 (Sidebar + Outlet)
// =============================================================================
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';

export function LayoutV2() {
  return (
    <div className="flex h-screen w-screen bg-black text-[#f5f5f7] overflow-hidden font-sans">
      <Sidebar />
      <Outlet />
    </div>
  );
}
