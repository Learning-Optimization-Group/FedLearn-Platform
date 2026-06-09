import React, { useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { Brain } from 'lucide-react';
import { registerUser } from '../services/apiServices';
import { createLogger } from '../lib/logger';
import { Button, Card, Input } from '../components/ui';

const log = createLogger('RegisterPage');

const MIN_PASSWORD_LENGTH = 8;

const RegisterPage: React.FC = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [successMessage, setSuccessMessage] = useState('');
  const navigate = useNavigate();

  // Track the post-success redirect timer so we can cancel it on unmount —
  // otherwise the navigate() call fires after the component is gone and
  // React warns "state update on unmounted component".
  const redirectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    return () => {
      if (redirectTimerRef.current) {
        clearTimeout(redirectTimerRef.current);
      }
    };
  }, []);

  const validatePassword = (password: string): string | null => {
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
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setSuccessMessage('');
    setError('');
    setIsLoading(true);

    if (!username || !email || !password || !confirmPassword) {
      setError('All fields are mandatory');
      setIsLoading(false);
      return;
    }

    const passwordError = validatePassword(password);
    if (passwordError) {
      setError(passwordError);
      setIsLoading(false);
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match!');
      setIsLoading(false);
      return;
    }

    try {
      // Routed through the shared axios instance so it picks up
      // baseURL, withCredentials, and the 401/403 interceptor — same
      // contract as every other API call in the app.
      const response = await registerUser({ username, email, password });
      setSuccessMessage(response.data?.message || 'Registration successful! Please login.');

      setUsername('');
      setEmail('');
      setPassword('');
      setConfirmPassword('');

      redirectTimerRef.current = setTimeout(() => {
        navigate('/login');
      }, 2000);
    } catch (err: any) {
      log.error('register failed', err);
      const data = err?.response?.data;
      // GlobalExceptionHandler returns either {message, fieldErrors:{...}}
      // (validation), {message} (generic), or — for legacy paths —
      // {error}. Try them in order without losing the field-level detail.
      let displayError = 'An error occurred during registration. Please try again later.';
      if (data?.fieldErrors && typeof data.fieldErrors === 'object') {
        displayError = `Validation failed: ${Object.values(data.fieldErrors).join(', ')}`;
      } else if (data?.message) {
        displayError = data.message;
      } else if (data?.error) {
        displayError = data.error;
      }
      setError(displayError);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-canvas text-fg font-sans px-4 py-12">
      <Card padding="lg" className="w-full max-w-sm">
        <Link to="/" className="flex items-center gap-3 mb-8">
          <div className="w-8 h-8 rounded-md bg-surface-2 border border-hairline flex items-center justify-center">
            <Brain className="w-5 h-5 text-fg" strokeWidth={1.5} />
          </div>
          <span className="text-h4 tracking-tight text-fg">FedLearn</span>
        </Link>

        <h2 className="text-h3 text-fg mb-1">Create account</h2>
        <p className="text-body text-fg-muted mb-6">Start training on distributed data.</p>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <label htmlFor="username" className="text-label font-medium text-fg-muted">
              Username
            </label>
            <Input
              type="text"
              id="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              autoComplete="username"
            />
          </div>

          <div className="flex flex-col gap-2">
            <label htmlFor="email" className="text-label font-medium text-fg-muted">
              Email
            </label>
            <Input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              autoComplete="email"
            />
          </div>

          <div className="flex flex-col gap-2">
            <label htmlFor="password" className="text-label font-medium text-fg-muted">
              Password
            </label>
            <Input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="new-password"
              minLength={MIN_PASSWORD_LENGTH}
            />
            <span className="text-caption text-fg-subtle">
              Minimum {MIN_PASSWORD_LENGTH} characters with uppercase, lowercase, and number
            </span>
          </div>

          <div className="flex flex-col gap-2">
            <label htmlFor="confirmPassword" className="text-label font-medium text-fg-muted">
              Confirm Password
            </label>
            <Input
              type="password"
              id="confirmPassword"
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              required
              autoComplete="new-password"
            />
          </div>

          {error && (
            <p
              className="text-label text-danger bg-surface-2 border border-hairline rounded-sm px-3 py-2"
              role="alert"
            >
              {error}
            </p>
          )}
          {successMessage && (
            <p
              className="text-label text-success bg-surface-2 border border-hairline rounded-sm px-3 py-2"
              role="status"
            >
              {successMessage}
            </p>
          )}

          <Button type="submit" disabled={isLoading} className="w-full mt-2">
            {isLoading ? 'Registering…' : 'Register'}
          </Button>
        </form>

        <p className="text-label text-fg-muted text-center mt-6">
          Already have an account?{' '}
          <Link to="/login" className="text-accent hover:text-accent-hover transition-colors">
            Login here
          </Link>
        </p>
      </Card>
    </div>
  );
};

export default RegisterPage;
