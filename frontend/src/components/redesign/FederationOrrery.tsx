import React, { useState, useEffect, useMemo } from 'react';

export interface Client {
  id?: string;
  name?: string;
  status: 'training' | 'uploading' | 'idle' | 'offline';
  contribution: number;
  ring?: 0 | 1 | 2;
}

interface FederationOrreryProps {
  clients?: Client[];
  round?: number;
  totalRounds?: number;
  size?: number;
  showRoundPulse?: boolean;
  spin?: boolean;
}

export const FederationOrrery: React.FC<FederationOrreryProps> = ({
  clients = [],
  round = 12,
  totalRounds = 50,
  size = 360,
  showRoundPulse = true,
  spin = true,
}) => {
  const [t, setT] = useState(0);

  useEffect(() => {
    let id: number;
    const tick = () => {
      setT(performance.now() / 1000);
      id = requestAnimationFrame(tick);
    };
    if (spin) id = requestAnimationFrame(tick);
    return () => {
      if (id) cancelAnimationFrame(id);
    };
  }, [spin]);

  const c = size / 2;
  const rings = [size * 0.42, size * 0.32, size * 0.22];

  // distribute clients across 3 rings
  const distributed = useMemo(() => {
    const ringCounts = [0, 0, 0];
    return clients.map((cl) => {
      const ring = cl.ring ?? (cl.contribution > 0.7 ? 2 : cl.contribution > 0.4 ? 1 : 0);
      const idxInRing = ringCounts[ring]++;
      const ringSize = clients.filter(
        (x) => (x.ring ?? (x.contribution > 0.7 ? 2 : x.contribution > 0.4 ? 1 : 0)) === ring
      ).length;
      const baseAngle = (idxInRing / Math.max(1, ringSize)) * Math.PI * 2;
      const speed = ring === 0 ? 0.04 : ring === 1 ? 0.08 : 0.13;
      const dir = ring % 2 === 0 ? 1 : -1;
      const phase = ring * 0.6;
      return { ...cl, ring, baseAngle, speed, dir, phase };
    });
  }, [clients]);

  return (
    <div style={{ position: 'relative', width: size, height: size }}>
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ overflow: 'visible' }}>
        <defs>
          <radialGradient id="orr-core" cx="50%" cy="50%">
            <stop offset="0%" stopColor="var(--accent-primary)" stopOpacity="1" />
            <stop offset="55%" stopColor="var(--accent-primary)" stopOpacity="0.35" />
            <stop offset="100%" stopColor="var(--accent-primary)" stopOpacity="0" />
          </radialGradient>
          <linearGradient id="orr-trail" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="var(--accent-primary)" stopOpacity="0" />
            <stop offset="100%" stopColor="var(--accent-primary)" stopOpacity="0.9" />
          </linearGradient>
          <filter id="orr-glow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* rings */}
        {rings.map((r, i) => (
          <circle
            key={i}
            cx={c}
            cy={c}
            r={r}
            fill="none"
            stroke="var(--border-color)"
            strokeOpacity={0.45 + i * 0.1}
            strokeWidth="1"
            strokeDasharray={i === 0 ? '2 4' : '0'}
          />
        ))}

        {/* round pulse — expands every few seconds */}
        {showRoundPulse && (
          <circle
            cx={c}
            cy={c}
            r="10"
            fill="none"
            stroke="var(--accent-primary)"
            strokeWidth="1"
            style={{ transformOrigin: `${c}px ${c}px`, animation: 'pulse-ring 2.6s ease-out infinite' }}
          />
        )}
        {showRoundPulse && (
          <circle
            cx={c}
            cy={c}
            r="10"
            fill="none"
            stroke="var(--accent-primary)"
            strokeWidth="1"
            style={{ transformOrigin: `${c}px ${c}px`, animation: 'pulse-ring 2.6s ease-out 1.3s infinite' }}
          />
        )}

        {/* core */}
        <circle cx={c} cy={c} r={size * 0.13} fill="url(#orr-core)" />
        <circle cx={c} cy={c} r={size * 0.055} fill="var(--accent-primary)" filter="url(#orr-glow)" />
        <circle cx={c} cy={c} r={size * 0.022} fill="var(--background-primary)" />

        {/* clients */}
        {distributed.map((cl, i) => {
          const r = rings[cl.ring];
          const angle = cl.baseAngle + cl.dir * t * cl.speed + cl.phase;
          const x = c + Math.cos(angle) * r;
          const y = c + Math.sin(angle) * r;
          const isUp = cl.status === 'uploading';
          const isTraining = cl.status === 'training';
          const isOffline = cl.status === 'offline';
          const color = isOffline
            ? 'var(--text-secondary)'
            : isUp
            ? 'var(--accent-primary)'
            : isTraining
            ? 'oklch(0.52 0.16 220)' // info color
            : 'var(--text-primary)';

          return (
            <g key={cl.id || i}>
              {isUp && (
                <line
                  x1={x}
                  y1={y}
                  x2={c}
                  y2={c}
                  stroke="var(--accent-primary)"
                  strokeWidth="1.2"
                  strokeOpacity="0.5"
                  strokeDasharray="2 4"
                  style={{ animation: 'flicker 1.2s linear infinite' }}
                />
              )}
              <circle
                cx={x}
                cy={y}
                r={isUp ? 6.5 : 4.5}
                fill={color}
                style={{ filter: isUp ? 'drop-shadow(0 0 6px var(--accent-primary))' : 'none' }}
              />
              {isTraining && (
                <circle
                  cx={x}
                  cy={y}
                  r="9"
                  fill="none"
                  stroke={color}
                  strokeWidth="1"
                  opacity="0.4"
                  style={{ transformOrigin: `${x}px ${y}px`, animation: 'pulse-ring 1.8s ease-out infinite' }}
                />
              )}
              {cl.name && (
                <text
                  x={x + 9}
                  y={y + 3.5}
                  fontSize="9"
                  fontFamily="var(--font-mono)"
                  fill="var(--text-secondary)"
                  style={{ userSelect: 'none' }}
                >
                  {cl.name}
                </text>
              )}
            </g>
          );
        })}
      </svg>

      {/* HUD overlays */}
      <div
        style={{
          position: 'absolute',
          top: 10,
          left: 10,
          display: 'flex',
          flexDirection: 'column',
          gap: 4,
        }}
      >
        <span className="font-mono text-[10px] tracking-widest text-(--text-secondary) font-bold">FEDERATION</span>
        <span className="font-mono text-[12px] text-(--text-primary)">
          {clients.length} nodes
        </span>
      </div>
      <div
        style={{
          position: 'absolute',
          top: 10,
          right: 10,
          textAlign: 'right',
        }}
      >
        <div className="font-mono text-[10px] tracking-widest text-(--text-secondary) font-bold">ROUND</div>
        <div className="font-mono text-[22px] text-(--text-primary) font-medium">
          {String(round).padStart(2, '0')}
          <span style={{ color: 'var(--text-secondary)' }}>/{totalRounds}</span>
        </div>
      </div>

      <style>{`
        @keyframes pulse-ring {
          0% { transform: scale(1); opacity: 1; stroke-width: 2px; }
          100% { transform: scale(4); opacity: 0; stroke-width: 0.5px; }
        }
        @keyframes flicker {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>
    </div>
  );
};
