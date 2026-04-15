// =============================================================================
// FedLearn Desktop — LogPanel Component
// =============================================================================
// SECURITY: Renders plain text only — no HTML from container output.
// No dangerouslySetInnerHTML. No innerHTML. All log lines are rendered
// as React text nodes inside <pre> elements to guarantee XSS safety.
// =============================================================================

import React, { useEffect, useRef } from 'react';

interface LogPanelProps {
  logs: string[];
}

const LogPanel: React.FC<LogPanelProps> = ({ logs }) => {
  const logEndRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const isAutoScrollRef = useRef(true);

  // Detect if user has scrolled up (disable auto-scroll)
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = container;
      // User is "at bottom" if within 50px of the end
      isAutoScrollRef.current = scrollHeight - scrollTop - clientHeight < 50;
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    return () => container.removeEventListener('scroll', handleScroll);
  }, []);

  // Auto-scroll to bottom when new logs arrive (only if user hasn't scrolled up)
  useEffect(() => {
    if (isAutoScrollRef.current && logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [logs]);

  if (logs.length === 0) {
    return (
      <div 
        className="log-panel log-panel-empty" 
        ref={containerRef}
        style={{ overflowY: 'auto', height: '400px' }}
      >
        <div className="log-empty-state">
          <span className="log-empty-icon">📋</span>
          <p className="log-empty-text">
            No container output yet. Start a training session to see real-time logs.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div 
      className="log-panel" 
      ref={containerRef}
      style={{ overflowY: 'auto', height: '400px' }}
    >
      <pre className="log-content">
        {/*
          SECURITY: Each log line is rendered as a plain text node.
          React's default behavior escapes all content — no HTML is interpreted.
          This prevents any XSS payload from container output from executing.
          
          PERFORMANCE: Using a single join to prevent DOM explosion from mapping 
          thousands of spans, eliminating lag and scroll thrashing.
        */}
        {logs.join('')}
      </pre>
      <div ref={logEndRef} />
    </div>
  );
};

export default LogPanel;
