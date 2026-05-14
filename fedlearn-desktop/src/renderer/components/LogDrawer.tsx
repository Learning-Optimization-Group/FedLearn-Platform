import React, { useEffect, useRef, useState } from 'react';

interface LogDrawerProps {
  logs: string[];
  autoOpen: boolean;
}

const LogDrawer: React.FC<LogDrawerProps> = ({ logs, autoOpen }) => {
  const [open, setOpen] = useState<boolean>(false);
  const bodyRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoOpen) setOpen(true);
  }, [autoOpen]);

  useEffect(() => {
    if (open && bodyRef.current) {
      bodyRef.current.scrollTop = bodyRef.current.scrollHeight;
    }
  }, [logs, open]);

  return (
    <div className={`log-drawer${open ? '' : ' log-drawer--collapsed'}`}>
      <div className="log-drawer__bar" onClick={() => setOpen((v) => !v)}>
        <span>Training Output · {logs.length} lines</span>
        <span>{open ? '▾' : '▴'}</span>
      </div>
      {open && (
        <div className="log-drawer__body" ref={bodyRef}>
          {logs.length === 0 && <div className="log-drawer__empty">No output yet.</div>}
          {logs.map((line, idx) => (
            <div key={idx} className="log-drawer__line">{line}</div>
          ))}
        </div>
      )}
    </div>
  );
};

export default LogDrawer;
