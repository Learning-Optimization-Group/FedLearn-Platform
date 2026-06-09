import { createContext, useContext, useEffect, useRef, useState, ReactNode } from 'react';
import { Client as StompClient } from '@stomp/stompjs';
import { useAuth } from './AuthContext';
import type { AppNotification } from '../services/apiServices';

interface NotificationContextType {
    notifications: AppNotification[];
    unreadCount: number;
    markAllRead: () => void;
}

const NotificationContext = createContext<NotificationContextType | null>(null);

const SERVER_ROOT_URL = import.meta.env.VITE_SERVER_ROOT_URL || `http://${window.location.hostname}:8081`;
const WEBSOCKET_URL_BASE = SERVER_ROOT_URL.replace(/^http/, 'ws');

export function NotificationProvider({ children }: { children: ReactNode }) {
    const { currentUser } = useAuth();
    const [notifications, setNotifications] = useState<AppNotification[]>([]);
    const [unreadCount, setUnreadCount] = useState(0);
    const clientRef = useRef<StompClient | null>(null);

    useEffect(() => {
        if (!currentUser) return;

        const client = new StompClient({
            brokerURL: `${WEBSOCKET_URL_BASE}/ws-logs`,
            reconnectDelay: 5000,
        });

        client.onConnect = () => {
            client.subscribe('/user/queue/notifications', (msg) => {
                try {
                    const notif: AppNotification = JSON.parse(msg.body);
                    setNotifications((prev) => [notif, ...prev].slice(0, 50));
                    setUnreadCount((c) => c + 1);
                } catch {
                    // ignore malformed payload
                }
            });
        };

        client.activate();
        clientRef.current = client;

        return () => {
            if (clientRef.current?.active) clientRef.current.deactivate();
            clientRef.current = null;
        };
    }, [currentUser]);

    const markAllRead = () => setUnreadCount(0);

    return (
        <NotificationContext.Provider value={{ notifications, unreadCount, markAllRead }}>
            {children}
        </NotificationContext.Provider>
    );
}

export function useNotifications(): NotificationContextType {
    const ctx = useContext(NotificationContext);
    if (!ctx) throw new Error('useNotifications must be used within NotificationProvider');
    return ctx;
}
