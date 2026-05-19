import React, { useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { registerUser } from '../services/apiServices';
import { createLogger } from '../lib/logger';
import { Activity } from 'lucide-react';

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
        <div className="min-h-screen flex items-center justify-center p-6 relative overflow-hidden font-sans" style={{ backgroundColor: 'var(--background-primary)' }}>
            <div className="absolute inset-0 pointer-events-none" style={{
                background: 'radial-gradient(circle at 50% 0%, var(--glow-accent), transparent 60%)',
                opacity: 0.8
            }} />
            
            <div className="w-full max-w-[440px] relative z-10 rounded-2xl p-8 shadow-2xl" style={{
                background: 'var(--background-card)',
                border: '1px solid var(--border-color)',
            }}>
                <div className="flex flex-col items-center mb-8">
                    <div className="w-12 h-12 rounded-xl flex items-center justify-center mb-4" style={{ background: 'var(--accent-primary)' }}>
                        <Activity className="w-6 h-6 text-white" />
                    </div>
                    <h2 className="text-[24px] font-display font-medium tracking-tight text-(--text-primary) m-0">Create an account</h2>
                    <p className="text-[14px] text-(--text-secondary) mt-2 font-medium">Join the FedLearn platform.</p>
                </div>

                <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                    <div className="flex flex-col gap-1.5">
                        <label htmlFor="username" className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)">Username</label>
                        <input
                            type="text"
                            id="username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                            autoComplete="username"
                            className="w-full px-4 py-3 rounded-lg text-[14px] outline-none transition-colors"
                            style={{ backgroundColor: 'var(--input-background)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
                            onFocus={(e) => e.target.style.borderColor = 'var(--accent-primary)'}
                            onBlur={(e) => e.target.style.borderColor = 'var(--border-color)'}
                        />
                    </div>

                    <div className="flex flex-col gap-1.5">
                        <label htmlFor="email" className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)">Email</label>
                        <input
                            type="email"
                            id="email"
                            value={email}
                            onChange={(e) => setEmail(e.target.value)}
                            required
                            autoComplete="email"
                            className="w-full px-4 py-3 rounded-lg text-[14px] outline-none transition-colors"
                            style={{ backgroundColor: 'var(--input-background)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
                            onFocus={(e) => e.target.style.borderColor = 'var(--accent-primary)'}
                            onBlur={(e) => e.target.style.borderColor = 'var(--border-color)'}
                        />
                    </div>

                    <div className="flex flex-col gap-1.5">
                        <label htmlFor="password" className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)">Password</label>
                        <input
                            type="password"
                            id="password"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                            autoComplete="new-password"
                            minLength={MIN_PASSWORD_LENGTH}
                            className="w-full px-4 py-3 rounded-lg text-[14px] outline-none transition-colors"
                            style={{ backgroundColor: 'var(--input-background)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
                            onFocus={(e) => e.target.style.borderColor = 'var(--accent-primary)'}
                            onBlur={(e) => e.target.style.borderColor = 'var(--border-color)'}
                        />
                        <p className="text-[11px] text-(--text-secondary) mt-0.5">
                            Minimum {MIN_PASSWORD_LENGTH} characters with uppercase, lowercase, and number
                        </p>
                    </div>

                    <div className="flex flex-col gap-1.5">
                        <label htmlFor="confirmPassword" className="text-[12px] font-medium uppercase tracking-wider text-(--text-secondary)">Confirm Password</label>
                        <input
                            type="password"
                            id="confirmPassword"
                            value={confirmPassword}
                            onChange={(e) => setConfirmPassword(e.target.value)}
                            required
                            autoComplete="new-password"
                            className="w-full px-4 py-3 rounded-lg text-[14px] outline-none transition-colors"
                            style={{ backgroundColor: 'var(--input-background)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
                            onFocus={(e) => e.target.style.borderColor = 'var(--accent-primary)'}
                            onBlur={(e) => e.target.style.borderColor = 'var(--border-color)'}
                        />
                    </div>

                    {error && (
                        <div className="p-3 mt-2 rounded-lg text-[13px] font-medium text-center border" role="alert" style={{ backgroundColor: 'color-mix(in srgb, var(--destructive) 10%, transparent)', color: 'var(--destructive)', borderColor: 'color-mix(in srgb, var(--destructive) 30%, transparent)' }}>
                            {error}
                        </div>
                    )}
                    {successMessage && (
                        <div className="p-3 mt-2 rounded-lg text-[13px] font-medium text-center border" role="status" style={{ backgroundColor: 'color-mix(in srgb, var(--success) 10%, transparent)', color: 'var(--success)', borderColor: 'color-mix(in srgb, var(--success) 30%, transparent)' }}>
                            {successMessage}
                        </div>
                    )}

                    <button 
                        type="submit" 
                        disabled={isLoading}
                        className="w-full py-3 px-4 rounded-lg text-[14px] font-semibold text-white mt-4 transition-all hover:brightness-110 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:brightness-100"
                        style={{ backgroundColor: 'var(--accent-primary)' }}
                    >
                        {isLoading ? 'Registering...' : 'Create Account'}
                    </button>
                </form>

                <p className="text-center mt-6 text-[13px] text-(--text-secondary) font-medium">
                    Already have an account?{' '}
                    <Link to="/login" className="text-(--accent-primary) hover:underline">
                        Sign in
                    </Link>
                </p>
            </div>
        </div>
    );
};

export default RegisterPage;
