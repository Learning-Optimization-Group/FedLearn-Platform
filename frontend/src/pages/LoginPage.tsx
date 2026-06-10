import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { ShieldCheck, AlertCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { loginUser } from '../services/apiServices';
import { Button, Card, Input } from '../components/ui';
import { Wordmark } from '../components/brand';

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
        <div className="bg-ambient relative flex min-h-screen items-center justify-center bg-canvas px-4 py-12 font-sans text-fg">
            <div className="bg-grid pointer-events-none absolute inset-0 opacity-50" />
            <div className="relative w-full max-w-md">
                <div className="mb-8 flex justify-center">
                    <Link to="/" aria-label="FedLearn home">
                        <Wordmark size={32} />
                    </Link>
                </div>

                <Card padding="lg" className="p-7">
                    <h1 className="font-display text-h2 text-fg">Welcome back</h1>
                    <p className="mt-1.5 text-body text-fg-muted">
                        Sign in to keep training your models.
                    </p>

                    <form onSubmit={handleSubmit} className="mt-7 flex flex-col gap-4">
                        <div className="flex flex-col gap-1.5">
                            <label htmlFor="identifier" className="text-label font-medium text-fg">
                                Email or username
                            </label>
                            <Input
                                type="text"
                                id="identifier"
                                className="h-11"
                                value={identifier}
                                onChange={(e) => setIdentifier(e.target.value)}
                                required
                                autoComplete="username"
                                placeholder="you@example.com"
                            />
                        </div>

                        <div className="flex flex-col gap-1.5">
                            <label htmlFor="password" className="text-label font-medium text-fg">
                                Password
                            </label>
                            <Input
                                type="password"
                                id="password"
                                className="h-11"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                                autoComplete="current-password"
                                placeholder="••••••••"
                            />
                        </div>

                        {error && (
                            <p
                                className="flex items-start gap-2 rounded-md border border-danger/30 bg-danger/10 px-3 py-2.5 text-label text-danger"
                                role="alert"
                            >
                                <AlertCircle className="mt-px h-4 w-4 flex-shrink-0" strokeWidth={1.5} />
                                {error}
                            </p>
                        )}

                        <Button
                            type="submit"
                            size="lg"
                            disabled={isLoading}
                            className="mt-2 w-full"
                        >
                            {isLoading ? 'Signing in…' : 'Sign in'}
                        </Button>
                    </form>

                    <p className="mt-6 text-center text-label text-fg-muted">
                        New here?{' '}
                        <Link
                            to="/register"
                            className="font-medium text-accent transition-colors hover:text-accent-hover"
                        >
                            Create a free account
                        </Link>
                    </p>
                </Card>

                <p className="mt-6 flex items-center justify-center gap-2 text-caption text-fg-subtle">
                    <ShieldCheck className="h-3.5 w-3.5" strokeWidth={1.5} />
                    Private by design — your data stays on your devices.
                </p>
            </div>
        </div>
    );
};

export default LoginPage;
