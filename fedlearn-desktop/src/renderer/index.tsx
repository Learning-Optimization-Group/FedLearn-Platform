// =============================================================================
// FedLearn Desktop — Renderer Entry Point
// =============================================================================

import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';

const rootElement = document.getElementById('root');

if (!rootElement) {
  throw new Error('Failed to find root element. Ensure index.html contains <div id="root"></div>');
}

const root = createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
