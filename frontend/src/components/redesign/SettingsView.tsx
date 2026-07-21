// =============================================================================
// FedLearn Frontend — Settings View (Ledger design system)
// =============================================================================
// Profile management (identity + password + read-only account meta), then
// session + connection details, with a sign-out action.

import { useEffect, useState, type FormEvent } from 'react';
import { LogOut, Copy, Check, Loader2 } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import { cn } from '../../lib/utils';
import { Button, Card, Input, FormField } from '../ui';
import { PageHeader } from './PageHeader';
import { SERVER_ROOT_URL } from '../../lib/serverConfig';
import {
  fetchMyProfile,
  updateMyProfile,
  errorMessage,
  errorStatus,
  type UserProfile,
  type UpdateProfileRequest,
} from '../../services/apiServices';

const MIN_PASSWORD_LENGTH = 8;
const MAX_DISPLAY_NAME_LENGTH = 80;

/** Same strength rule as registration: 8+ chars, upper, lower, number. */
function validatePassword(password: string): string | null {
  if (password.length < MIN_PASSWORD_LENGTH) {
    return `Password must be at least ${MIN_PASSWORD_LENGTH} characters long`;
  }
  if (!/[A-Z]/.test(password)) {
    return 'Password must contain at least one uppercase letter';
  }
  if (!/[a-z]/.test(password)) {
    return 'Password must contain at least one lowercase letter';
  }
  if (!/[0-9]/.test(password)) {
    return 'Password must contain at least one number';
  }
  return null;
}

function formatDate(iso?: string): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleDateString();
}

function formatDateTime(iso?: string): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
}

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

/** Quiet neutral chip — the email-verification state. */
function QuietChip({ children }: { children: React.ReactNode }) {
  return (
    <span className="text-caption font-medium px-2.5 py-0.5 rounded-pill bg-surface-2 border border-hairline text-fg-muted">
      {children}
    </span>
  );
}

/** Identity form + password change + read-only meta, driven by /users/me/profile. */
function ProfileSection() {
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loadError, setLoadError] = useState('');
  const [loading, setLoading] = useState(true);

  // Identity form
  const [displayName, setDisplayName] = useState('');
  const [email, setEmail] = useState('');
  const [emailError, setEmailError] = useState('');
  const [profileError, setProfileError] = useState('');
  const [savingProfile, setSavingProfile] = useState(false);
  const [profileSaved, setProfileSaved] = useState(false);

  // Password form
  const [currentPassword, setCurrentPassword] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [currentPasswordError, setCurrentPasswordError] = useState('');
  const [newPasswordError, setNewPasswordError] = useState('');
  const [confirmError, setConfirmError] = useState('');
  const [passwordError, setPasswordError] = useState('');
  const [changingPassword, setChangingPassword] = useState(false);
  const [passwordSaved, setPasswordSaved] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetchMyProfile()
      .then((res) => {
        if (cancelled) return;
        setProfile(res.data);
        setDisplayName(res.data.displayName ?? '');
        setEmail(res.data.email);
      })
      .catch((err) => {
        if (!cancelled) setLoadError(errorMessage(err, 'Could not load your profile.'));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const dirty =
    profile !== null &&
    (displayName.trim() !== (profile.displayName ?? '') || email.trim() !== profile.email);

  const handleSaveProfile = async (e: FormEvent) => {
    e.preventDefault();
    if (!profile || !dirty || savingProfile) return;
    setEmailError('');
    setProfileError('');
    setProfileSaved(false);
    setSavingProfile(true);
    const body: UpdateProfileRequest = {};
    if (displayName.trim() !== (profile.displayName ?? '')) body.displayName = displayName.trim();
    if (email.trim() !== profile.email) body.email = email.trim();
    try {
      const res = await updateMyProfile(body);
      setProfile(res.data);
      setDisplayName(res.data.displayName ?? '');
      setEmail(res.data.email);
      setProfileSaved(true);
    } catch (err) {
      if (errorStatus(err) === 409) {
        setEmailError(errorMessage(err, 'That email address is already in use.'));
      } else {
        setProfileError(errorMessage(err, 'Could not save your profile.'));
      }
    } finally {
      setSavingProfile(false);
    }
  };

  const handleChangePassword = async (e: FormEvent) => {
    e.preventDefault();
    if (changingPassword) return;
    setCurrentPasswordError('');
    setNewPasswordError('');
    setConfirmError('');
    setPasswordError('');
    setPasswordSaved(false);

    let invalid = false;
    if (!currentPassword) {
      setCurrentPasswordError('Enter your current password');
      invalid = true;
    }
    const strengthError = validatePassword(newPassword);
    if (strengthError) {
      setNewPasswordError(strengthError);
      invalid = true;
    }
    if (newPassword !== confirmPassword) {
      setConfirmError('Passwords do not match');
      invalid = true;
    }
    if (invalid) return;

    setChangingPassword(true);
    try {
      const res = await updateMyProfile({ currentPassword, newPassword });
      setProfile(res.data);
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      setPasswordSaved(true);
    } catch (err) {
      if (errorStatus(err) === 403) {
        setCurrentPasswordError(errorMessage(err, 'Current password is incorrect.'));
      } else {
        setPasswordError(errorMessage(err, 'Could not change your password.'));
      }
    } finally {
      setChangingPassword(false);
    }
  };

  if (loading) {
    return (
      <Card padding="lg" className="flex flex-col gap-4">
        <h2 className="text-h4 font-semibold text-fg">Profile</h2>
        <div className="flex flex-col gap-3">
          <div className="h-9 rounded-md bg-surface-2 animate-pulse" />
          <div className="h-9 rounded-md bg-surface-2 animate-pulse" />
          <div className="h-9 w-32 rounded-md bg-surface-2 animate-pulse" />
        </div>
      </Card>
    );
  }

  if (loadError || !profile) {
    return (
      <Card padding="lg">
        <h2 className="text-h4 font-semibold text-fg mb-3">Profile</h2>
        <p className="text-body text-danger">{loadError || 'Could not load your profile.'}</p>
      </Card>
    );
  }

  return (
    <Card padding="lg">
      <h2 className="text-h4 font-semibold text-fg mb-4">Profile</h2>

      {/* Identity */}
      <form onSubmit={handleSaveProfile} className="flex flex-col gap-4" noValidate>
        <FormField label="Display name">
          <Input
            value={displayName}
            onChange={(e) => {
              setDisplayName(e.target.value);
              setProfileSaved(false);
            }}
            maxLength={MAX_DISPLAY_NAME_LENGTH}
            placeholder="How your name appears to others"
            autoComplete="name"
          />
        </FormField>
        <FormField
          label="Email"
          help="Changing your email requires re-verification."
          error={emailError || undefined}
        >
          <Input
            type="email"
            value={email}
            onChange={(e) => {
              setEmail(e.target.value);
              setEmailError('');
              setProfileSaved(false);
            }}
            autoComplete="email"
          />
        </FormField>
        <div className="flex items-center gap-3">
          <Button type="submit" variant="primary" disabled={!dirty || savingProfile}>
            {savingProfile && <Loader2 className="w-4 h-4 animate-spin" strokeWidth={2} />}
            {savingProfile ? 'Saving…' : 'Save profile'}
          </Button>
          {profileSaved && <span className="text-caption text-success">Profile saved.</span>}
          {profileError && <span className="text-caption text-danger">{profileError}</span>}
        </div>
      </form>

      {/* Change password */}
      <div className="mt-6 border-t border-hairline pt-6">
        <h3 className="text-body font-semibold text-fg mb-4">Change password</h3>
        <form onSubmit={handleChangePassword} className="flex flex-col gap-4" noValidate>
          <FormField label="Current password" error={currentPasswordError || undefined}>
            <Input
              type="password"
              value={currentPassword}
              onChange={(e) => {
                setCurrentPassword(e.target.value);
                setCurrentPasswordError('');
                setPasswordSaved(false);
              }}
              autoComplete="current-password"
            />
          </FormField>
          <FormField
            label="New password"
            help={`At least ${MIN_PASSWORD_LENGTH} characters, with an uppercase letter, a lowercase letter, and a number.`}
            error={newPasswordError || undefined}
          >
            <Input
              type="password"
              value={newPassword}
              onChange={(e) => {
                setNewPassword(e.target.value);
                setNewPasswordError('');
                setPasswordSaved(false);
              }}
              autoComplete="new-password"
            />
          </FormField>
          <FormField label="Confirm new password" error={confirmError || undefined}>
            <Input
              type="password"
              value={confirmPassword}
              onChange={(e) => {
                setConfirmPassword(e.target.value);
                setConfirmError('');
                setPasswordSaved(false);
              }}
              autoComplete="new-password"
            />
          </FormField>
          <div className="flex items-center gap-3">
            <Button
              type="submit"
              variant="secondary"
              disabled={changingPassword || !currentPassword || !newPassword || !confirmPassword}
            >
              {changingPassword && <Loader2 className="w-4 h-4 animate-spin" strokeWidth={2} />}
              {changingPassword ? 'Changing…' : 'Change password'}
            </Button>
            {passwordSaved && <span className="text-caption text-success">Password changed.</span>}
            {passwordError && <span className="text-caption text-danger">{passwordError}</span>}
          </div>
        </form>
      </div>

      {/* Read-only account meta */}
      <div className="mt-6 border-t border-hairline pt-4 flex flex-wrap items-center gap-x-6 gap-y-2">
        <span className="text-caption text-fg-muted">
          Member since <span className="text-fg">{formatDate(profile.createdAt)}</span>
        </span>
        <span className="text-caption text-fg-muted">
          Last sign-in <span className="text-fg">{formatDateTime(profile.lastLoginAt)}</span>
        </span>
        <QuietChip>{profile.emailVerified ? 'Verified' : 'Unverified'}</QuietChip>
      </div>
    </Card>
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
            {/* Profile — identity, password, account meta */}
            <ProfileSection />

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
