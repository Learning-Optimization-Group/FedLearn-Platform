// =============================================================================
// FedLearn Frontend — Settings View (Ledger design system)
// =============================================================================
// Session + connection details, with a sign-out action.

import { useEffect, useState } from 'react';
import { LogOut, Copy, Check } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { cn } from '../../lib/utils';
import { Button, Card } from '../ui';
import { PageHeader } from './PageHeader';
import { SERVER_ROOT_URL } from '../../lib/serverConfig';

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
    <div className="flex items-center justify-between gap-4 py-2.5 border-b border-hairline last:border-b-0">
      <span className="text-label text-fg-muted">{label}</span>
      <div className="flex items-center gap-1 min-w-0">
        <span className="font-mono text-label text-fg truncate max-w-[340px]">{value}</span>
        {copyable && (
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopy}
            className={cn(copied && 'text-success hover:text-success')}
            title="Copy value"
          >
            {copied ? <Check strokeWidth={1.5} className="w-3.5 h-3.5" /> : <Copy strokeWidth={1.5} className="w-3.5 h-3.5" />}
            {copied ? 'Copied' : 'Copy'}
          </Button>
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

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[1400px] px-6 py-6 md:px-10">
          <div className="max-w-3xl flex flex-col gap-6">
            {/* Account + connection details — one card, hairline-divided rows */}
            <Card padding="lg">
              <Row label="Username" value={currentUser?.username || '—'} />
              <Row label="Email" value={currentUser?.email || '—'} />
              <Row label="Server address" value={SERVER_ROOT_URL} copyable />
              <Row label="Connect a device" value={`--server ${SERVER_ROOT_URL}`} copyable />
              <div className="flex justify-end pt-4">
                <Button variant="secondary" onClick={logout}>
                  <LogOut strokeWidth={1.5} className="w-4 h-4" />
                  Sign out
                </Button>
              </div>
            </Card>

            {/* About */}
            <Card padding="lg">
              <h2 className="text-h4 font-semibold text-fg mb-3">About</h2>
              <p className="text-body leading-relaxed text-fg-muted">
                FedLearn trains AI together across many kinds of devices — laptops, servers,
                NVIDIA Jetson, Apple Silicon, and Android phones — without any of them sharing
                private data. Devices join training from the FedLearn desktop or mobile app:
                install it, sign in, and choose a project to train.
              </p>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
