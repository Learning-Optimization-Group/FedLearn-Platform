import React, { useState, useEffect, useRef } from 'react';
import { Client, StompSubscription } from '@stomp/stompjs';
import '../styles/LogViewer.css';

interface LogEntry {
    level: string;
    message: string;
    timestamp?: string;
    stackTrace?: string;
}

interface LogViewerProps {
    projectId: string;
    serverUrl: string;
    onClose: () => void;
}

const MAX_LOGS = 1000; // Prevent memory issues

const LogViewer: React.FC<LogViewerProps> = ({ projectId, serverUrl, onClose }) => {
    const [logs, setLogs] = useState<LogEntry[]>([]);
    const [isConnected, setIsConnected] = useState(false);
    const logContainerRef = useRef<HTMLDivElement>(null);
    const subscriptionRef = useRef<StompSubscription | null>(null);
    const clientRef = useRef<Client | null>(null);

    useEffect(() => {
        if (!projectId) return;

        const brokerURL = serverUrl.replace(/^http/, 'ws') + '/ws-logs';
        const stompClient = new Client({
            brokerURL: brokerURL,
            connectHeaders: {},
            reconnectDelay: 5000,
        });

        stompClient.onConnect = () => {
            setIsConnected(true);
            setLogs(prev => [
                ...prev.slice(-MAX_LOGS + 1),
                {
                    level: 'INFO',
                    message: '--- WebSocket Connection Established ---',
                    timestamp: new Date().toISOString()
                }
            ]);

            // Store subscription reference for cleanup
            const subscription = stompClient.subscribe(`/topic/logs/${projectId}`, (message) => {
                try {
                    const logObj: LogEntry = JSON.parse(message.body);
                    setLogs(prevLogs => [...prevLogs.slice(-MAX_LOGS + 1), logObj]);
                } catch (e) {
                    setLogs(prevLogs => [
                        ...prevLogs.slice(-MAX_LOGS + 1),
                        { level: 'INFO', message: message.body }
                    ]);
                }
            });

            subscriptionRef.current = subscription;
        };

        stompClient.onDisconnect = () => {
            setIsConnected(false);
            setLogs(prev => [
                ...prev.slice(-MAX_LOGS + 1),
                {
                    level: 'ERROR',
                    message: '--- WebSocket Connection Lost ---',
                    timestamp: new Date().toISOString()
                }
            ]);
        };

        stompClient.onStompError = (frame) => {
            console.error('Broker reported error:', frame.headers['message']);
            setIsConnected(false);
        };

        stompClient.activate();
        clientRef.current = stompClient;

        return () => {
            if (subscriptionRef.current) {
                subscriptionRef.current.unsubscribe();
            }
            if (clientRef.current?.active) {
                clientRef.current.deactivate();
            }
        };
    }, [projectId, serverUrl]);

    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs]);

    const handleBackdropClick = (e: React.MouseEvent<HTMLDivElement>) => {
        if (e.target === e.currentTarget) {
            onClose();
        }
    };

    const handleClearLogs = () => {
        setLogs([]);
    };

    return (
        <div className="log-viewer-backdrop" onClick={handleBackdropClick}>
            <div className="log-viewer" onClick={e => e.stopPropagation()}>
                <div className="log-header">
                    <h4>Live Server Logs</h4>
                    <div className="header-controls">
                        <button
                            onClick={handleClearLogs}
                            className="clear-btn"
                            title="Clear logs"
                        >
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
                            <div key={i} className={`log-entry log-${log.level?.toLowerCase()}`}>
                                <span className="log-timestamp">{log.timestamp}</span>
                                <span className={`log-level level-${log.level?.toLowerCase()}`}>
                                    [{log.level}]
                                </span>
                                <span className="log-message">{log.message}</span>
                                {log.stackTrace && (
                                    <pre className="log-stacktrace">{log.stackTrace}</pre>
                                )}
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
