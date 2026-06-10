// =============================================================================
// FedLearn Frontend — Settings View (Ember design system)
// =============================================================================
// Session + connection details, with a sign-out action.

import { useEffect, useState } from 'react';
import { LogOut, Copy, Check, User as UserIcon, Server, ShieldCheck } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { cn } from '../../lib/utils';
import { Button, Card } from '../ui';
import { PageHeader } from './PageHeader';

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
      /* no-op */
    }
  };

  return (
    <div className="flex items-center justify-between gap-4 py-3.5 border-b border-hairline last:border-b-0">
      <span className="text-label font-medium text-fg-muted">{label}</span>
      <div className="flex items-center gap-2 min-w-0">
        <code className="font-mono text-label text-fg bg-surface-2 px-3 py-1 rounded-md truncate max-w-[340px]">
          {value}
        </code>
        {copyable && (
          <button
            onClick={handleCopy}
            className={cn(
              'flex items-center gap-1 text-caption px-2 py-1 rounded-md transition-colors duration-[120ms]',
              copied ? 'text-success bg-surface-2' : 'text-fg-muted hover:text-fg hover:bg-surface-2'
            )}
            title="Copy value"
          >
            {copied ? <Check strokeWidth={1.5} className="w-3.5 h-3.5" /> : <Copy strokeWidth={1.5} className="w-3.5 h-3.5" />}
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
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
      <PageHeader title="Settings" subtitle="Your account and connection details." />

      <div className="flex-1 overflow-y-auto px-6 md:px-10 py-8 bg-canvas">
        <div className="max-w-3xl mx-auto flex flex-col gap-6">
          {/* Account */}
          <Card padding="lg">
            <div className="flex items-center gap-2.5 mb-2">
              <UserIcon strokeWidth={1.5} className="w-5 h-5 text-accent" />
              <h2 className="text-h4 font-display text-fg">Account</h2>
            </div>
            <Row label="Username" value={currentUser?.username || '—'} />
            <Row label="Email" value={currentUser?.email || '—'} />
            <div className="flex justify-end pt-4">
              <Button variant="danger" onClick={logout}>
                <LogOut strokeWidth={1.5} className="w-4 h-4" />
                Sign out
              </Button>
            </div>
          </Card>

          {/* Server */}
          <Card padding="lg">
            <div className="flex items-center gap-2.5 mb-2">
              <Server strokeWidth={1.5} className="w-5 h-5 text-accent" />
              <h2 className="text-h4 font-display text-fg">Server</h2>
            </div>
            <Row label="Server address" value={SERVER_ROOT_URL} copyable />
            <Row label="Connect a device" value={`--server ${SERVER_ROOT_URL}`} copyable />
          </Card>

          {/* About */}
          <Card padding="lg">
            <div className="flex items-center gap-2.5 mb-3">
              <ShieldCheck strokeWidth={1.5} className="w-5 h-5 text-accent" />
              <h2 className="text-h4 font-display text-fg">About</h2>
            </div>
            <p className="text-body leading-relaxed text-fg-muted">
              FedLearn trains AI together across many kinds of devices — laptops, servers,
              NVIDIA Jetson, Apple Silicon, and Android phones — without any of them sharing
              private data. Add the devices that will help train from the{' '}
              <span className="text-accent font-medium">Devices</span> page.
            </p>
          </Card>
        </div>
      </div>
    </div>
  );
}
