import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import '../styles/AuthStyles.css';

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

    const API_SERVER_ROOT = import.meta.env.VITE_API_BASE_URL || `http://${window.location.hostname}:8081/api`;
    const endpointPath = '/auth/register';
    const fullApiUrl = `${API_SERVER_ROOT}${endpointPath}`;

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

        const registrationData = {
            username,
            email,
            password
        };

        try {
            const response = await fetch(fullApiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify(registrationData),
            });

            const responseData = await response.json();

            if (response.ok) {
                setSuccessMessage(responseData.message || 'Registration successful! Please login.');

                setUsername('');
                setEmail('');
                setPassword('');
                setConfirmPassword('');

                setTimeout(() => {
                    navigate('/login');
                }, 2000);
            } else {
                let displayError = `Registration failed: ${response.statusText}`;
                if (responseData.message) {
                    displayError = responseData.message;
                }
                if (responseData.errors) {
                    const fieldErrors = Object.values(responseData.errors).join(', ');
                    displayError = `Validation failed: ${fieldErrors}`;
                } else if (responseData.error && responseData.message) {
                    displayError = responseData.message;
                }
                setError(displayError);
            }
        } catch (err) {
            console.error('Registration API call failed:', err);
            setError('An error occurred during registration. Please try again later.');
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
