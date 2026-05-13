import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { LogOut, Copy, Check, User as UserIcon, Server, Info, ShieldCheck } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { cn } from '../../lib/utils';

const SERVER_ROOT_URL =
  import.meta.env.VITE_SERVER_ROOT_URL || `http://${window.location.hostname}:8081`;

function Row({
  label,
  value,
  copyable,
}: {
  label: string;
  value: string;
  copyable?: boolean;
}) {
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (!copied) return;
    const t = setTimeout(() => setCopied(false), 1500);
    return () => clearTimeout(t);
  }, [copied]);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
    } catch {
      // Ignore clipboard failures.
    }
  };

  return (
    <div className="flex items-center justify-between gap-4 py-3 border-b last:border-b-0" style={{ borderColor: 'var(--border-color)' }}>
      <span className="text-[13px] uppercase tracking-wider font-semibold text-(--text-secondary)">
        {label}
      </span>
      <div className="flex items-center gap-2 min-w-0">
        <code className="font-mono text-[13px] px-3 py-1 rounded-md truncate max-w-[340px]" style={{ color: 'var(--text-primary)', backgroundColor: 'var(--background-secondary)' }}>
          {value}
        </code>
        {copyable && (
          <button
            onClick={handleCopy}
            className={cn(
              'flex items-center gap-1 text-[12px] px-2 py-1 rounded-md transition-colors',
              copied
                ? 'text-emerald-500'
                : 'text-(--text-secondary) hover:text-(--text-primary)'
            )}
            title="Copy value"
            style={{ backgroundColor: copied ? 'color-mix(in srgb, #22c55e 16%, transparent)' : 'var(--background-secondary)' }}
          >
            {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            {copied ? 'Copied' : 'Copy'}
          </button>
        )}
      </div>
    </div>
  );
}

export function SettingsView() {
  const { currentUser, logout } = useAuth();

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden">
      <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px)' }}>
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}>
          <h1 className="font-display text-4xl font-semibold tracking-tight text-(--text-primary)">System Settings</h1>
          <p className="text-sm text-(--text-secondary) mt-1">Session controls and orchestrator metadata.</p>
        </motion.div>
      </div>

      <div className="flex-1 overflow-y-auto px-8 py-8">
        <div className="max-w-4xl mx-auto flex flex-col gap-6">
          <section className="rounded-3xl p-6" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}>
            <div className="flex items-center gap-3 mb-2">
              <UserIcon className="w-5 h-5 text-(--accent-primary)" />
              <h2 className="text-[19px] font-semibold tracking-tight text-(--text-primary)">Account</h2>
            </div>
            <Row label="Username" value={currentUser?.username || '—'} />
            <Row label="Email" value={currentUser?.email || '—'} />
            <div className="flex justify-end pt-4">
              <button
                onClick={logout}
                className="flex items-center gap-2 px-5 py-2.5 rounded-full text-[15px] font-medium"
                style={{ backgroundColor: 'color-mix(in srgb, #ef4444 12%, transparent)', color: '#ef4444', border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)' }}
              >
                <LogOut className="w-4 h-4" />
                Log out
              </button>
            </div>
          </section>

          <section className="rounded-3xl p-6" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}>
            <div className="flex items-center gap-3 mb-2">
              <Server className="w-5 h-5 text-emerald-500" />
              <h2 className="text-[19px] font-semibold tracking-tight text-(--text-primary)">Orchestrator</h2>
            </div>
            <Row label="Server URL" value={SERVER_ROOT_URL} copyable />
            <Row label="Client bootstrap" value={`--server ${SERVER_ROOT_URL}`} copyable />
          </section>

          <section className="rounded-3xl p-6" style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}>
            <div className="flex items-center gap-3 mb-3">
              <Info className="w-5 h-5 text-violet-500" />
              <h2 className="text-[19px] font-semibold tracking-tight text-(--text-primary)">Platform Notes</h2>
            </div>
            <p className="text-[15px] leading-relaxed text-(--text-secondary)">
              FedLearn coordinates federated training across heterogeneous edge environments. Provision and rotate node accounts from the Node Network view, then assign projects and monitor their convergence from the dashboard.
            </p>
            <div className="mt-4 inline-flex items-center gap-2 px-3 py-2 rounded-xl text-sm" style={{ backgroundColor: 'var(--background-secondary)', border: '1px solid var(--border-color)' }}>
              <ShieldCheck className="w-4 h-4 text-(--accent-primary)" />
              Cookie-based auth is active for all API and WebSocket traffic.
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
