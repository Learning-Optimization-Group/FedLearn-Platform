import { useEffect, useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Menu } from 'lucide-react';
import { Sidebar } from './Sidebar';
import { Wordmark } from '../brand';
import { Button, Modal } from '../ui';

export function LayoutV2() {
    const [navigationOpen, setNavigationOpen] = useState(false);
    useEffect(() => {
        const desktop = window.matchMedia('(min-width: 768px)');
        const changed = (event: MediaQueryListEvent) => { if (event.matches) setNavigationOpen(false); };
        desktop.addEventListener('change', changed);
        return () => desktop.removeEventListener('change', changed);
    }, []);
    return (
        <div className="flex h-dvh w-full flex-col overflow-hidden bg-canvas font-sans text-fg md:flex-row">
            <div className="hidden md:flex"><Sidebar /></div>
            <div className="flex h-14 shrink-0 items-center justify-between border-b border-hairline px-4 md:hidden">
                <Wordmark size={24} />
                <Button variant="ghost" aria-label="Open navigation" aria-haspopup="dialog" aria-expanded={navigationOpen} onClick={() => setNavigationOpen(true)}>
                    <Menu className="h-5 w-5" aria-hidden="true" />
                </Button>
            </div>
            <div className="flex min-h-0 min-w-0 flex-1"><Outlet /></div>
            <Modal open={navigationOpen} onClose={() => setNavigationOpen(false)} title="Navigation" size="sm" className="max-h-[calc(100dvh-2rem)] overflow-y-auto">
                <Sidebar mobile onNavigate={() => setNavigationOpen(false)} />
            </Modal>
        </div>
    );
}
