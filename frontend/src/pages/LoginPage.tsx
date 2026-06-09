import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { Brain } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { loginUser } from '../services/apiServices';
import { Button, Card, Input } from '../components/ui';

const LoginPage: React.FC = () => {
  const [identifier, setIdentifier] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const auth = useAuth();

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    setError('');
    setIsLoading(true);

    if (!identifier || !password) {
      setError('Email/Username and Password are required.');
      setIsLoading(false);
      return;
    }

    try {
      // The JWT lands in an HttpOnly cookie; the response body only
      // gives us the identity needed to render the shell.
      const response = await loginUser({ username: identifier, password });
      const { username, email, role } = response.data;
      auth.setSession({ username, email, role });

      const from = (location.state as any)?.from?.pathname || '/dashboard';
      navigate(from, { replace: true });
    } catch (err: any) {
      const responseData = err?.response?.data;
      setError(
        responseData?.message ||
          responseData?.error ||
          'An error occurred during login. Please try again later.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-canvas text-fg font-sans px-4">
      <Card padding="lg" className="w-full max-w-sm">
        <Link to="/" className="flex items-center gap-3 mb-8">
          <div className="w-8 h-8 rounded-md bg-surface-2 border border-hairline flex items-center justify-center">
            <Brain className="w-5 h-5 text-fg" strokeWidth={1.5} />
          </div>
          <span className="text-h4 tracking-tight text-fg">FedLearn</span>
        </Link>

        <h2 className="text-h3 text-fg mb-1">Sign in</h2>
        <p className="text-body text-fg-muted mb-6">Welcome back to the platform.</p>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-2">
            <label htmlFor="identifier" className="text-label font-medium text-fg-muted">
              Email or Username
            </label>
            <Input
              type="text"
              id="identifier"
              value={identifier}
              onChange={(e) => setIdentifier(e.target.value)}
              required
              autoComplete="username"
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
              autoComplete="current-password"
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

          <Button type="submit" disabled={isLoading} className="w-full mt-2">
            {isLoading ? 'Signing in…' : 'Sign in'}
          </Button>
        </form>

        <p className="text-label text-fg-muted text-center mt-6">
          Don't have an account?{' '}
          <Link to="/register" className="text-accent hover:text-accent-hover transition-colors">
            Register here
          </Link>
        </p>
      </Card>
    </div>
  );
};

export default LoginPage;
