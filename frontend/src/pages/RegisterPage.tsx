import React, { useEffect, useRef, useState } from 'react';
import { isAxiosError } from 'axios';
import { Link, useNavigate } from 'react-router-dom';
import { ShieldCheck, AlertCircle, CheckCircle2 } from 'lucide-react';
import { registerUser } from '../services/apiServices';
import { createLogger } from '../lib/logger';
import { Button, Card, Input, FormField } from '../components/ui';
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
        } catch (err: unknown) {
            log.error('register failed', err);
            // GlobalExceptionHandler returns either {message, fieldErrors:{...}}
            // (validation), {message} (generic), or — for legacy paths —
            // {error}. Try them in order without losing the field-level detail.
            let displayError = 'An error occurred during registration. Please try again later.';
            if (isAxiosError(err)) {
                const data = err.response?.data as
                    | { message?: string; error?: string; fieldErrors?: Record<string, string> }
                    | undefined;
                if (data?.fieldErrors && typeof data.fieldErrors === 'object') {
                    displayError = `Validation failed: ${Object.values(data.fieldErrors).join(', ')}`;
                } else if (data?.message) {
                    displayError = data.message;
                } else if (data?.error) {
                    displayError = data.error;
                }
            }
            setError(displayError);
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
                    <h1 className="text-h3 text-fg">Create your account</h1>
                    <p className="mt-1.5 text-body text-fg-muted">
                        Start training AI together — it's free.
                    </p>

                    <form onSubmit={handleSubmit} className="mt-6 flex flex-col gap-4">
                        <FormField label="Username">
                            <Input
                                type="text"
                                id="username"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                required
                                autoComplete="username"
                                placeholder="yourname"
                            />
                        </FormField>

                        <FormField label="Email">
                            <Input
                                type="email"
                                id="email"
                                value={email}
                                onChange={(e) => setEmail(e.target.value)}
                                required
                                autoComplete="email"
                                placeholder="you@example.com"
                            />
                        </FormField>

                        <FormField
                            label="Password"
                            help={`At least ${MIN_PASSWORD_LENGTH} characters, with an uppercase letter, a lowercase letter, and a number.`}
                        >
                            <Input
                                type="password"
                                id="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                required
                                autoComplete="new-password"
                                minLength={MIN_PASSWORD_LENGTH}
                                placeholder="••••••••"
                            />
                        </FormField>

                        <FormField label="Confirm password">
                            <Input
                                type="password"
                                id="confirmPassword"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                                required
                                autoComplete="new-password"
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
                        {successMessage && (
                            <p
                                className="flex items-start gap-2 rounded-md border border-success/30 bg-success/10 px-3 py-2.5 text-label text-success"
                                role="status"
                            >
                                <CheckCircle2 className="mt-px h-4 w-4 flex-shrink-0" strokeWidth={1.5} />
                                {successMessage}
                            </p>
                        )}

                        <Button type="submit" disabled={isLoading} className="mt-2 w-full">
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
