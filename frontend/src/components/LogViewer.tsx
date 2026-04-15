import React, { useState, useEffect, useRef } from 'react';
import { Client, StompSubscription } from '@stomp/stompjs';
import * as api from '../services/apiServices';
import { logStore, StoredLogEntry } from '../services/logStore';
import '../styles/LogViewer.css';

interface LogViewerProps {
    projectId: string;
    serverUrl: string;
    onClose: () => void;
}

const LogViewer: React.FC<LogViewerProps> = ({ projectId, serverUrl, onClose }) => {
    const [logs, setLogs] = useState<StoredLogEntry[]>(() => logStore.get(projectId));
    const [isConnected, setIsConnected] = useState(false);
    const logContainerRef = useRef<HTMLDivElement>(null);
    const subscriptionRef = useRef<StompSubscription | null>(null);
    const clientRef = useRef<Client | null>(null);

    useEffect(() => {
        if (!projectId) return;

        // Hydrate from cache first so reopening never flashes an empty pane.
        setLogs(logStore.get(projectId));

        // Subscribe to store updates — this keeps the UI reactive when either
        // the live STOMP subscription here or any other code path appends logs.
        const unsubscribe = logStore.subscribe(projectId, (next) => setLogs(next));

        // Fetch historical logs only once per project; on subsequent re-opens
        // the cache already has them.
        if (!logStore.hasLoadedHistorical(projectId)) {
            api.fetchProjectLogs(projectId)
                .then((response) => {
                    if (Array.isArray(response.data) && response.data.length > 0) {
                        const formatted: StoredLogEntry[] = response.data.map((log: any) => ({
                            level: log.level,
                            message: log.message,
                            timestamp: log.timestamp,
                            stackTrace: log.stackTrace,
                        }));
                        logStore.mergeHistorical(projectId, formatted);
                    } else {
                        logStore.markHistoricalLoaded(projectId);
                    }
                })
                .catch((err) => {
                    console.error('Failed to fetch historical logs:', err);
                });
        }

        const brokerURL = serverUrl.replace(/^http/, 'ws') + '/ws-logs';
        const stompClient = new Client({
            brokerURL,
            connectHeaders: {},
            reconnectDelay: 5000,
        });

        stompClient.onConnect = () => {
            setIsConnected(true);
            logStore.append(projectId, {
                level: 'INFO',
                message: '--- WebSocket Connection Established ---',
                timestamp: new Date().toISOString(),
            });

            subscriptionRef.current = stompClient.subscribe(`/topic/logs/${projectId}`, (message) => {
                try {
                    const parsed = JSON.parse(message.body);
                    logStore.append(projectId, {
                        level: parsed.level ?? 'INFO',
                        message: parsed.message ?? message.body,
                        timestamp: parsed.timestamp,
                        stackTrace: parsed.stackTrace,
                    });
                } catch {
                    logStore.append(projectId, { level: 'INFO', message: message.body });
                }
            });
        };

        stompClient.onDisconnect = () => {
            setIsConnected(false);
            logStore.append(projectId, {
                level: 'WARN',
                message: '--- WebSocket Connection Lost ---',
                timestamp: new Date().toISOString(),
            });
        };

        stompClient.onStompError = (frame) => {
            console.error('Broker reported error:', frame.headers['message']);
            setIsConnected(false);
        };

        stompClient.activate();
        clientRef.current = stompClient;

        return () => {
            unsubscribe();
            if (subscriptionRef.current) subscriptionRef.current.unsubscribe();
            if (clientRef.current?.active) clientRef.current.deactivate();
        };
    }, [projectId, serverUrl]);

    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs]);

    useEffect(() => {
        const onKey = (e: KeyboardEvent) => {
            if (e.key === 'Escape') onClose();
        };
        window.addEventListener('keydown', onKey);
        return () => window.removeEventListener('keydown', onKey);
    }, [onClose]);

    const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (e.target === e.currentTarget) onClose();
    };

    const handleClearLogs = () => {
        logStore.clear(projectId);
    };

    return (
        <div className="log-viewer-backdrop" onClick={handleBackdropClick}>
            <div className="log-viewer" onClick={(e) => e.stopPropagation()}>
                <div className="log-header">
                    <h4>Live Server Logs</h4>
                    <div className="header-controls">
                        <span className="log-count" aria-label="Log entry count">
                            {logs.length} entries
                        </span>
                        <button onClick={handleClearLogs} className="clear-btn" title="Clear cached logs">
                            Clear
                        </button>
                        <div className={`connection-status ${isConnected ? 'connected' : ''}`}>
                            {isConnected ? '● Connected' : '○ Disconnected'}
                        </div>
                        <button
                            onClick={onClose}
                            className="close-btn"
                            title="Close log viewer"
                            aria-label="Close"
                        >
                            &times;
                        </button>
                    </div>
                </div>
                <div className="logs-container" ref={logContainerRef}>
                    {logs.length > 0 ? (
                        logs.map((log, i) => (
                            <div key={i} className={`log-entry log-${log.level?.toLowerCase() ?? 'info'}`}>
                                <span className="log-timestamp">{log.timestamp}</span>
                                <span className={`log-level level-${log.level?.toLowerCase() ?? 'info'}`}>
                                    [{log.level ?? 'INFO'}]
                                </span>
                                <span className="log-message">{log.message}</span>
                                {log.stackTrace && <pre className="log-stacktrace">{log.stackTrace}</pre>}
                            </div>
                        ))
                    ) : (
                        <div className="log-entry log-info">
                            <span className="log-message">Waiting for server logs...</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};

export default LogViewer;
