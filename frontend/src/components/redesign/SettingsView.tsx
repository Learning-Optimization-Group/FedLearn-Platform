// =============================================================================
// FedLearn Frontend — V2 Settings View
// =============================================================================
// Session + orchestrator metadata, with a logout action.

import { useEffect, useState } from 'react';
import { LogOut, Copy, Check, User as UserIcon, Server, Info } from 'lucide-react';
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
      /* no-op */
    }
  };

  return (
    <div className="flex items-center justify-between gap-4 py-3 border-b border-[rgba(255,255,255,0.05)] last:border-b-0">
      <span className="text-[13px] uppercase tracking-wider font-semibold text-[#86868b]">
        {label}
      </span>
      <div className="flex items-center gap-2 min-w-0">
        <code className="font-mono text-[13px] text-[#f5f5f7] bg-[#2c2c2e] px-3 py-1 rounded-md truncate max-w-[340px]">
          {value}
        </code>
        {copyable && (
          <button
            onClick={handleCopy}
            className={cn(
              'flex items-center gap-1 text-[12px] px-2 py-1 rounded-md transition-colors',
              copied
                ? 'text-[#32d74b] bg-[#32d74b]/10'
                : 'text-[#86868b] hover:text-[#f5f5f7] hover:bg-[rgba(255,255,255,0.05)]'
            )}
            title="Copy value"
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
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-black text-[#f5f5f7] font-sans">
      <div className="h-24 flex items-center justify-between px-10 border-b border-[#2c2c2e] bg-[rgba(0,0,0,0.65)] backdrop-blur-3xl saturate-[1.8] sticky top-0 z-20">
        <div>
          <h1 className="text-[28px] font-semibold tracking-tight">Settings</h1>
          <p className="text-[15px] text-[#86868b] mt-0.5 tracking-tight">
            Session and orchestrator preferences.
          </p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-10 py-10 bg-black">
        <div className="max-w-3xl mx-auto flex flex-col gap-8">
          {/* Account */}
          <section className="bg-[#1c1c1e] rounded-[24px] p-6 border border-[rgba(255,255,255,0.05)]">
            <div className="flex items-center gap-3 mb-2">
              <UserIcon className="w-5 h-5 text-[#0a84ff]" />
              <h2 className="text-[18px] font-semibold tracking-tight">Account</h2>
            </div>
            <Row label="Username" value={currentUser?.username || '—'} />
            <Row label="Email" value={currentUser?.email || '—'} />
            <div className="flex justify-end pt-4">
              <button
                onClick={logout}
                className="flex items-center gap-2 bg-[#ff453a]/10 text-[#ff453a] hover:bg-[#ff453a]/20 px-5 py-2.5 rounded-full text-[15px] font-medium transition-colors"
              >
                <LogOut className="w-4 h-4" />
                Log out
              </button>
            </div>
          </section>

          {/* Server */}
          <section className="bg-[#1c1c1e] rounded-[24px] p-6 border border-[rgba(255,255,255,0.05)]">
            <div className="flex items-center gap-3 mb-2">
              <Server className="w-5 h-5 text-[#32d74b]" />
              <h2 className="text-[18px] font-semibold tracking-tight">Orchestrator</h2>
            </div>
            <Row label="Server URL" value={SERVER_ROOT_URL} copyable />
            <Row
              label="Client bootstrap"
              value={`--server ${SERVER_ROOT_URL}`}
              copyable
            />
          </section>

          {/* About */}
          <section className="bg-[#1c1c1e] rounded-[24px] p-6 border border-[rgba(255,255,255,0.05)]">
            <div className="flex items-center gap-3 mb-3">
              <Info className="w-5 h-5 text-[#bf5af2]" />
              <h2 className="text-[18px] font-semibold tracking-tight">About</h2>
            </div>
            <p className="text-[15px] leading-relaxed text-[#86868b]">
              FedLearn-Platform coordinates federated training across heterogeneous edge
              devices (Jetson ARM64, Apple Silicon, x86/CUDA). Provision client credentials
              from the <span className="text-[#0a84ff] font-medium">Node Network</span> view.
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
