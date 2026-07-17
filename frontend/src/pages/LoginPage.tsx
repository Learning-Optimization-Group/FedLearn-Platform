import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { ShieldCheck, AlertCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { loginUser, errorMessage } from '../services/apiServices';
import { Button, Card, Input, FormField } from '../components/ui';
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

            const from =
                (location.state as { from?: { pathname?: string } } | null)?.from?.pathname ||
                '/dashboard';
            navigate(from, { replace: true });
        } catch (err: unknown) {
            setError(errorMessage(err, 'An error occurred during login. Please try again later.'));
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex min-h-screen items-center justify-center bg-canvas px-4 py-12 font-sans text-fg">
            <div className="reveal w-full max-w-md">
                <div className="mb-8 flex justify-center">
                    <Link to="/" aria-label="FedLearn home">
                        <Wordmark size={32} />
                    </Link>
                </div>

                <Card padding="lg" className="p-7">
                    <h1 className="text-h3 text-fg">Welcome back</h1>
                    <p className="mt-1.5 text-body text-fg-muted">
                        Sign in to keep training your models.
                    </p>

                    <form onSubmit={handleSubmit} className="mt-6 flex flex-col gap-4">
                        <FormField label="Email or username">
                            <Input
                                type="text"
                                id="identifier"
                                value={identifier}
                                onChange={(e) => setIdentifier(e.target.value)}
                                required
                                autoComplete="username"
                                placeholder="you@example.com"
                            />
                        </FormField>

                        <FormField label="Password">
                            <Input
                                type="password"
                                id="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                                autoComplete="current-password"
                                placeholder="••••••••"
                            />
                        </FormField>

                        {error && (
                            <p
                                className="flex items-start gap-2 rounded-md border border-danger/30 bg-danger/10 px-3 py-2.5 text-label text-danger"
                                role="alert"
                            >
                                <AlertCircle className="mt-px h-4 w-4 flex-shrink-0" strokeWidth={1.5} />
                                {error}
                            </p>
                        )}

                        <Button type="submit" disabled={isLoading} className="mt-2 w-full">
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
