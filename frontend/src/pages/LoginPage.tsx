import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import '../styles/AuthStyles.css';

const LoginPage: React.FC = () => {
    const [identifier, setIdentifier] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();
    const location = useLocation();
    const auth = useAuth();

    const API_SERVER_ROOT = import.meta.env.VITE_API_BASE_URL || `http://${window.location.hostname}:8081/api`;
    const endpointPath = '/auth/login';
    const fullApiUrl = `${API_SERVER_ROOT}${endpointPath}`;

    const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        setError('');
        setIsLoading(true);

        if (!identifier || !password) {
            setError('Email/Username and Password are required.');
            setIsLoading(false);
            return;
        }

        const loginData = {
            username: identifier,
            password: password,
        };

        try {
            const response = await fetch(fullApiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                },
                credentials: 'include',
                body: JSON.stringify(loginData)
            });

            if (response.ok) {
                const responseData = await response.json();

                localStorage.setItem('jwtToken', responseData.accessToken);
                auth.login({ username: responseData.username }, responseData.accessToken);

                const from = (location.state as any)?.from?.pathname || "/dashboard";
                navigate(from, { replace: true });
            } else {
                const responseData = await response.json();
                setError(responseData.error || responseData.message || `Login failed: ${response.statusText}`);
            }
        } catch (err) {
            console.error('Login API call failed:', err);
            setError('An error occurred during login. Please try again later.');
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="auth-container">
            <h2>Login</h2>
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label htmlFor="identifier">Email or Username</label>
                    <input
                        type="text"
                        id="identifier"
                        value={identifier}
                        onChange={(e) => setIdentifier(e.target.value)}
                        required
                        autoComplete="username"
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
                        autoComplete="current-password"
                    />
                </div>

                {error && <p className="error-message">{error}</p>}

                <button type="submit" disabled={isLoading} className="auth-button">
                    {isLoading ? 'Logging in...' : 'Login'}
                </button>
            </form>
            <p className="auth-switch">
                Don't have an account? <Link to="/register">Register here</Link>
            </p>
        </div>
    );
};

export default LoginPage;
