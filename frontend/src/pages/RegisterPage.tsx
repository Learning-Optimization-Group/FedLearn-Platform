import React, { useEffect, useRef, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { registerUser } from '../services/apiServices';
import { createLogger } from '../lib/logger';
import '../styles/AuthStyles.css';

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
        <div className="auth-container">
            <h2>Register</h2>
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label htmlFor="username">Username</label>
                    <input
                        type="text"
                        id="username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        required
                        autoComplete="username"
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="email">Email</label>
                    <input
                        type="email"
                        id="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                        autoComplete="email"
                    />
                </div>

                <div className="form-group">
                    <label htmlFor="password">Password</label>
                    <input
                        type="password"
                        id="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                        autoComplete="new-password"
                        minLength={MIN_PASSWORD_LENGTH}
                    />
                    <small>Minimum {MIN_PASSWORD_LENGTH} characters with uppercase, lowercase, and number</small>
                </div>

                <div className="form-group">
                    <label htmlFor="confirmPassword">Confirm Password</label>
                    <input
                        type="password"
                        id="confirmPassword"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        required
                        autoComplete="new-password"
                    />
                </div>

                {error && <p className="error-message" role="alert">{error}</p>}
                {successMessage && <p className="success-message" role="status">{successMessage}</p>}

                <button type="submit" disabled={isLoading} className="auth-button">
                    {isLoading ? 'Registering...' : 'Register'}
                </button>
            </form>
            <p className="auth-switch">
                Already have an account? <Link to="/login">Login here</Link>
            </p>
        </div>
    );
};

export default RegisterPage;
