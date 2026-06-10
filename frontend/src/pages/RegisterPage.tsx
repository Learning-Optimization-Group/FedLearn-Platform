import React, { useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { ShieldCheck, AlertCircle, CheckCircle2 } from 'lucide-react';
import { registerUser } from '../services/apiServices';
import { createLogger } from '../lib/logger';
import { Button, Card, Input } from '../components/ui';
import { Wordmark } from '../components/brand';

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
        <div className="bg-ambient relative flex min-h-screen items-center justify-center bg-canvas px-4 py-12 font-sans text-fg">
            <div className="bg-grid pointer-events-none absolute inset-0 opacity-50" />
            <div className="relative w-full max-w-md">
                <div className="mb-8 flex justify-center">
                    <Link to="/" aria-label="FedLearn home">
                        <Wordmark size={32} />
                    </Link>
                </div>

                <Card padding="lg" className="p-7">
                    <h1 className="font-display text-h2 text-fg">Create your account</h1>
                    <p className="mt-1.5 text-body text-fg-muted">
                        Start training AI together — it's free.
                    </p>

                    <form onSubmit={handleSubmit} className="mt-7 flex flex-col gap-4">
                        <div className="flex flex-col gap-1.5">
                            <label htmlFor="username" className="text-label font-medium text-fg">
                                Username
                            </label>
                            <Input
                                type="text"
                                id="username"
                                className="h-11"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                required
                                autoComplete="username"
                                placeholder="yourname"
                            />
                        </div>

                        <div className="flex flex-col gap-1.5">
                            <label htmlFor="email" className="text-label font-medium text-fg">
                                Email
                            </label>
                            <Input
                                type="email"
                                id="email"
                                className="h-11"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                                autoComplete="email"
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
                                autoComplete="new-password"
                                minLength={MIN_PASSWORD_LENGTH}
                                placeholder="••••••••"
                            />
                            <span className="text-caption text-fg-subtle">
                                At least {MIN_PASSWORD_LENGTH} characters, with an uppercase letter, a
                                lowercase letter, and a number.
                            </span>
                        </div>

                        <div className="flex flex-col gap-1.5">
                            <label htmlFor="confirmPassword" className="text-label font-medium text-fg">
                                Confirm password
                            </label>
                            <Input
                                type="password"
                                id="confirmPassword"
                                className="h-11"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                                required
                                autoComplete="new-password"
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
                        {successMessage && (
                            <p
                                className="flex items-start gap-2 rounded-md border border-success/30 bg-success/10 px-3 py-2.5 text-label text-success"
                                role="status"
                            >
                                <CheckCircle2 className="mt-px h-4 w-4 flex-shrink-0" strokeWidth={1.5} />
                                {successMessage}
                            </p>
                        )}

                        <Button
                            type="submit"
                            size="lg"
                            disabled={isLoading}
                            className="mt-2 w-full"
                        >
                            {isLoading ? 'Creating account…' : 'Create account'}
                        </Button>
                    </form>

                    <p className="mt-6 text-center text-label text-fg-muted">
                        Already have an account?{' '}
                        <Link
                            to="/login"
                            className="font-medium text-accent transition-colors hover:text-accent-hover"
                        >
                            Sign in
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

export default RegisterPage;
