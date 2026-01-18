import React, { useState } from 'react'
import "../styles/AuthStyles.css"
import { Link, useNavigate } from 'react-router-dom';


const RegisterPage = () => {
    const [username, setUsername] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [error, setError] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [successMessage, setSuccessMessage] = useState('');
    const navigate = useNavigate();
    const API_SERVER_ROOT = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8081/api';
    const endpointPath = '/auth/register';
    const fullApiUrl = `${API_SERVER_ROOT}${endpointPath}`;

    const handleSubmit = async (event) => {
        console.log(' 1 fullApiUrl - ', fullApiUrl);
        event.preventDefault();
        setSuccessMessage('');
        setError('');
        setIsLoading(true);

        if (!username || !email || !password || !confirmPassword) {
            setError('All Fields are mandatory');
            setIsLoading(false);
            return;
        }

        if (password !== confirmPassword) { // Corrected: !== for comparison
            setError('Passwords do not match!');
            setIsLoading(false);
            return;
        }

        const registrationData = {
            username: username,
            email: email,
            password: password
        };

        try {
            console.log('API_SERVER_ROOT - ', API_SERVER_ROOT);
            console.log(' 2 fullApiUrl - ', fullApiUrl);
            const response = await fetch(fullApiUrl, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    'ngrok-skip-browser-warning': 'true', // Good practice to specify what you accept
                },
                body: JSON.stringify(registrationData),
            });

            const responseData = await response.json(); // Attempt to parse JSON for all responses

            if (response.ok) { // Status 200-299 (e.g., 201 Created)
                setSuccessMessage(responseData.message || 'Registration successful! Please login.');
                // console.log("User ID:", responseData.userId); // If you send userId back

                setUsername('');
                setEmail('');
                setPassword('');
                setConfirmPassword('');

                setTimeout(() => {
                    navigate('/login');
                }, 2000);

            } else {

                let displayError = `Registration failed: ${response.statusText}`; // Fallback
                if (responseData.message) {
                    displayError = responseData.message;
                }
                if (responseData.errors) { // If there's a map of field errors
                    // You could format this nicely, e.g., join them or pick the first one
                    const fieldErrors = Object.values(responseData.errors).join(', ');
                    displayError = `Validation failed: ${fieldErrors}`;
                } else if (responseData.error && responseData.message) { // For UserAlreadyExists or generic
                    displayError = responseData.message;
                }
                setError(displayError);
            }
            // ...
        } catch (err) {
            // This catch block handles network errors (e.g., server down, DNS issues)
            // or if response.json() fails for a non-JSON response (less likely if backend is consistent)
            console.error('fullApiUrl - ', fullApiUrl);
            console.error('Registration API call failed:', err);
            setError('An error occurred during registration. Please try again later. (Network or parsing error)');
        } finally {
            setIsLoading(false);
        }
    };


    return (
        <div className="auth-container">
            <h2>Register</h2>
            <form onSubmit={handleSubmit}> {/* Add basic submit handler */}

                {/* Username Input Group */}
                <div className="form-group">
                    <label htmlFor="username">Username</label>
                    <input
                        type="text"
                        id="username"
                        value={username}
                        onChange={(e) => setUsername(e.target.value)}
                        required
                    />
                </div>

                {/* Email Input Group */}
                <div className="form-group">
                    <label htmlFor="email">Email</label>
                    <input
                        type="email"
                        id="email"
                        value={email}
                        onChange={(e) => setEmail(e.target.value)}
                        required
                    />
                </div>

                {/* Password Input Group */}
                <div className="form-group">
                    <label htmlFor="password">Passwordddd</label>
                    <input
                        type="password"
                        id="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                </div>

                {/* Confirm Password Input Group */}
                <div className="form-group">
                    <label htmlFor="confirmPassword">Confirm Password</label>
                    <input
                        type="password"
                        id="confirmPassword"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        required
                    />
                </div>

                {error && <p className="error-message">{error}</p>}

                {/* Submit Button (will add later) */}
                <button type="submit" disabled={isLoading} className="auth-button">
                    {isLoading ? 'Registering...' : 'Register'}
                </button>

            </form>
            <p className="auth-switch">
                Already have an account? <Link to="/login">Login here</Link>
            </p>
        </div>
    )
}

export default RegisterPage