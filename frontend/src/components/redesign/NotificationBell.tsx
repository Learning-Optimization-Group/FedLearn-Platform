import { useEffect, useRef, useState } from 'react';
import { Bell } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useNotifications } from '../../context/NotificationContext';
import type { AppNotification } from '../../services/apiServices';

function notificationLabel(n: AppNotification): string {
    switch (n.type) {
        case 'ACCESS_REQUEST_CREATED':
            return `${n.actorUsername} requested access to ${n.projectName}`;
        case 'ACCESS_REQUEST_DECIDED':
            return `Your request for ${n.projectName} was ${n.decision?.toLowerCase()}`;
        case 'MEMBERSHIP_ADDED':
            return `${n.actorUsername} added you to ${n.projectName} as ${n.role?.toLowerCase()}`;
        case 'MEMBERSHIP_REMOVED':
            return `You were removed from ${n.projectName}`;
        case 'PROJECT_VISIBILITY_CHANGED':
            return `${n.projectName} visibility changed`;
        default:
            return `Update on ${n.projectName}`;
    }
}

export function NotificationBell() {
    const { notifications, unreadCount, markAllRead } = useNotifications();
    const [open, setOpen] = useState(false);
    const ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
        function handleClick(e: MouseEvent) {
            if (ref.current && !ref.current.contains(e.target as Node)) {
                setOpen(false);
            }
        }
        document.addEventListener('mousedown', handleClick);
        return () => document.removeEventListener('mousedown', handleClick);
    }, []);

    const toggle = () => {
        setOpen((v) => !v);
        if (!open) markAllRead();
    };

    return (
        <div className="relative" ref={ref}>
            <button
                onClick={toggle}
                className="relative w-8 h-8 flex items-center justify-center rounded-xl text-(--text-secondary) hover:text-(--text-primary) hover:bg-(--background-card) transition-all"
                title="Notifications"
            >
                <Bell className="w-4 h-4" />
                {unreadCount > 0 && (
                    <span className="absolute -top-0.5 -right-0.5 min-w-[16px] h-4 rounded-full bg-(--accent-primary) text-white text-[10px] font-bold flex items-center justify-center px-0.5">
                        {unreadCount > 99 ? '99+' : unreadCount}
                    </span>
                )}
            </button>

            {open && (
                <div
                    className="absolute left-full ml-2 top-0 z-50 w-80 rounded-2xl shadow-lg overflow-hidden"
                    style={{
                        backgroundColor: 'var(--background-card)',
                        border: '1px solid var(--border-color)',
                        boxShadow: 'var(--shadow-strong)',
                    }}
                >
                    <div className="px-4 py-3 border-b text-[13px] font-semibold text-(--text-primary)" style={{ borderColor: 'var(--border-color)' }}>
                        Notifications
                    </div>
                    <div className="max-h-80 overflow-y-auto">
                        {notifications.length === 0 ? (
                            <div className="px-4 py-6 text-center text-[13px] text-(--text-secondary)">
                                No notifications yet
                            </div>
                        ) : (
                            notifications.map((n) => (
                                <Link
                                    key={n.id}
                                    to={`/projects/${n.projectId}`}
                                    onClick={() => setOpen(false)}
                                    className="block px-4 py-3 text-[13px] text-(--text-primary) border-b last:border-b-0 hover:bg-(--background-secondary) transition-colors no-underline"
                                    style={{ borderColor: 'var(--border-color)' }}
                                >
                                    <div>{notificationLabel(n)}</div>
                                    <div className="text-[11px] text-(--text-secondary) mt-0.5">
                                        {new Date(n.timestamp).toLocaleString()}
                                    </div>
                                </Link>
                            ))
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
