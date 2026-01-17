import React, { useState } from 'react'
import { Link, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext.jsx';

const LoginPage = () => {

    const [identifier, setIdentifier] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const navigate = useNavigate();
    const auth = useAuth();
    const API_SERVER_ROOT = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8081/api';
    const endpointPath = '/auth/login';
    const fullApiUrl = `${API_SERVER_ROOT}${endpointPath}`;


    const handleSubmit = async (event) => {
        console.log('import.meta.env.VITE_API_BASE_URL - ', import.meta.env.VITE_API_BASE_URL)
        console.log('API_SERVER_ROOT - ', API_SERVER_ROOT);
        event.preventDefault();

        setError('');
        setIsLoading(true);

        if (!identifier || !password) {
            setError('Email/ Username and Password are required.');
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
                    'ngrok-skip-browser-warning': 'true',
                    'Accept': 'application/json',
                },
                body: JSON.stringify(loginData)
            });

            const responseData = response.json;
            console.log('responseData - ', responseData)

            if (response.ok) {
                const responseData = await response.json();
                console.log('Login successfull', responseData);

                localStorage.setItem('jwtToken', responseData.accessToken);

                auth.login({ username: responseData.username }, responseData.accessToken);


                const from = location.state?.from?.pathname || "/dashboard";
                navigate(from, { replace: true });
            } else {
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
                        type="text" // Could be 'email' if you only allow email login initially
                        id="identifier"
                        value={identifier}
                        onChange={(e) => setIdentifier(e.target.value)}
                        required
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
                    />
                </div>

                {error && <p className="error-message">{error}</p>}
                {/* Success message usually not needed on login, but could be added */}

                <button type="submit" disabled={isLoading} className="auth-button">
                    {isLoading ? 'Logging in...' : 'Login'}
                </button>
            </form>
            <p className="auth-switch">
                Don't have an account? <Link to="/register">Register here</Link>
            </p>
        </div>
    );
}

export default LoginPage