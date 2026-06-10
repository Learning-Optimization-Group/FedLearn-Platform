// =============================================================================
// FedLearn Frontend — LayoutV2 (Sidebar + Outlet)
// =============================================================================
// Each routed page is a self-contained flex child that owns its own scroll.
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';

export function LayoutV2() {
    return (
        <div className="flex h-screen w-screen overflow-hidden bg-canvas font-sans text-fg">
            <Sidebar />
            <Outlet />
        </div>
    );
}
