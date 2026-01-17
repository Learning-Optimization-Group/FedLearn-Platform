import React, { useState, useEffect, useRef } from 'react';
import { Client } from '@stomp/stompjs';
import '../styles/LogViewer.css';

const LogViewer = ({ projectId, serverUrl, onClose }) => {
    const [logs, setLogs] = useState([]);
    const [isConnected, setIsConnected] = useState(false);
    const logContainerRef = useRef(null);

    useEffect(() => {
        if (!projectId) return;

        const brokerURL = serverUrl.replace(/^http/, 'ws') + '/ws-logs';
        // 1. Create a STOMP client
        const stompClient = new Client({
            brokerURL: brokerURL, // Use the direct WebSocket URL
            connectHeaders: {},
            debug: (str) => console.log('STOMP: ' + str),
            reconnectDelay: 5000,
        });

        // 2. Define connection handlers
        stompClient.onConnect = (frame) => {
            setIsConnected(true);
            setLogs(prev => [...prev, '--- WebSocket Connection Established ---']);
            stompClient.subscribe(`/topic/logs/${projectId}`, (message) => {
                setLogs(prevLogs => [...prevLogs, message.body]);
            });
        };

        stompClient.onDisconnect = () => {
            setIsConnected(false);
            setLogs(prev => [...prev, '--- WebSocket Connection Lost ---']);
        };

        stompClient.onStompError = (frame) => {
            console.error('Broker reported error: ' + frame.headers['message']);
            setIsConnected(false);
        };

        // 3. Activate the client
        stompClient.activate();

        // 4. Define cleanup logic for when the component is unmounted
        return () => {
            if (stompClient.active) {
                stompClient.deactivate();
            }
        };
    }, [projectId, serverUrl]);

    // Effect to auto-scroll to the bottom
    useEffect(() => {
        if (logContainerRef.current) {
            logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
        }
    }, [logs]);

    return (
        <div className="log-viewer-backdrop" onClick={onClose}>
            <div className="log-viewer" onClick={e => e.stopPropagation()}>
                <div className="log-header">
                    <h4>Live Server Logs</h4>
                    <div className="header-controls">
                        <div className={`connection-status ${isConnected ? 'connected' : ''}`}>
                            {isConnected ? '● Connected' : '○ Disconnected'}
                        </div>
                        <button onClick={onClose} className="close-btn" title="Close log viewer">&times;</button>
                    </div>
                </div>
                <pre className="log-container" ref={logContainerRef}>
                    {logs.length > 0 ? logs.join('\n') : 'Waiting for server logs...'}
                </pre>
            </div>
        </div>
    );
};

export default LogViewer;