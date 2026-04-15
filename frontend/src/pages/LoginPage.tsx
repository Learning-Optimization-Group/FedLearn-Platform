import React, { useState } from 'react';
import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import api from '../api/axiosConfig';
import '../styles/AuthStyles.css';

interface LoginResponse {
    username: string;
    email?: string;
    accessToken: string;
}

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
            const response = await api.post<LoginResponse>('/auth/login', {
                username: identifier,
                password: password,
            });

            const { accessToken, username, email } = response.data;

            if (!accessToken) {
                setError('Login response is missing the access token. Please contact support.');
                return;
            }

            auth.login({ username, email }, accessToken);

            const from = (location.state as any)?.from?.pathname || '/dashboard';
            navigate(from, { replace: true });
        } catch (err: any) {
            const responseData = err?.response?.data;
            setError(
                responseData?.error ||
                    responseData?.message ||
                    'An error occurred during login. Please try again later.'
            );
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
