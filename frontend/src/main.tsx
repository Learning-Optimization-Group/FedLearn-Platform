import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import App from './App';
import './index.css';
import './styles/tailwind.css';
import './styles/fonts.css';
import { AuthProvider } from './context/AuthContext';
import { NotificationProvider } from './context/NotificationContext';
import { ThemeProvider } from './context/ThemeContext';
import { ErrorBoundary } from './components/ErrorBoundary';

if (typeof window !== 'undefined') {
    (window as any).global = window;
}

const rootElement = document.getElementById('root');

if (!rootElement) {
    throw new Error('Root element not found');
}

ReactDOM.createRoot(rootElement).render(
    <React.StrictMode>
        <ErrorBoundary>
            <ThemeProvider>
                <BrowserRouter>
                    <AuthProvider>
                        <NotificationProvider>
                            <App />
                        </NotificationProvider>
                    </AuthProvider>
                </BrowserRouter>
            </ThemeProvider>
        </ErrorBoundary>
    </React.StrictMode>
);
