import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { loginUser } from '../services/apiServices';
import { Activity } from 'lucide-react';

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
    <div
      className="min-h-screen flex items-center justify-center p-6 relative overflow-hidden font-sans"
      style={{ backgroundColor: 'var(--background-primary)' }}
    >
      <div
        className="absolute inset-0 pointer-events-none"
        style={{
          background: 'radial-gradient(circle at 50% 0%, var(--glow-accent), transparent 60%)',
          opacity: 0.8,
        }}
      />

      <div
        className="w-full max-w-[400px] relative z-10 rounded-2xl p-8 shadow-2xl"
        style={{
          background: 'var(--background-card)',
          border: '1px solid var(--border-color)',
        }}
      >
        <div className="flex flex-col items-center mb-8">
          <div
            className="w-12 h-12 rounded-xl flex items-center justify-center mb-4"
            style={{ background: 'var(--accent-primary)' }}
          >
            <Activity className="w-6 h-6 text-white" />
          </div>
          <h2 className="text-[24px] font-display font-medium tracking-tight text-(--text-primary) m-0">
            Sign in to FedLearn
          </h2>
          <p className="text-[14px] text-(--text-secondary) mt-2 font-medium">
            Welcome back to the platform.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <label
              htmlFor="identifier"
              className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)"
            >
              Email or Username
            </label>
            <input
              type="text"
              id="identifier"
              value={identifier}
              onChange={(e) => setIdentifier(e.target.value)}
              required
              autoComplete="username"
              className="w-full px-4 py-3 rounded-lg text-[14px] outline-none transition-colors"
              style={{
                backgroundColor: 'var(--input-background)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
              }}
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label
              htmlFor="password"
              className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)"
            >
              Password
            </label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              autoComplete="current-password"
              className="w-full px-4 py-3 rounded-lg text-[14px] outline-none transition-colors"
              style={{
                backgroundColor: 'var(--input-background)',
                color: 'var(--text-primary)',
                border: '1px solid var(--border-color)',
              }}
            />
          </div>

          {error && (
            <div
              className="p-3 mt-2 rounded-lg text-[13px] font-medium text-center border"
              style={{
                backgroundColor: 'color-mix(in srgb, var(--destructive) 10%, transparent)',
                color: 'var(--destructive)',
                borderColor: 'color-mix(in srgb, var(--destructive) 30%, transparent)',
              }}
            >
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-3 px-4 rounded-lg text-[14px] font-semibold mt-4 transition-all hover:brightness-110 disabled:cursor-not-allowed disabled:hover:brightness-100"
            style={{
              backgroundColor: isLoading ? 'var(--muted)' : 'var(--accent-primary)',
              color: isLoading ? 'var(--muted-foreground)' : 'var(--primary-foreground)',
            }}
          >
            {isLoading ? 'Authenticating…' : 'Sign In'}
          </button>
        </form>

        <p className="text-center mt-6 text-[13px] text-(--text-secondary) font-medium">
          Don't have an account?{' '}
          <Link to="/register" className="text-(--accent-primary) hover:underline">
            Register here
          </Link>
        </p>
      </div>
    </div>
  );
};

export default LoginPage;
