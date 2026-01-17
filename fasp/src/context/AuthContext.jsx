// src/context/AuthContext.jsx

import React, { createContext, useState, useContext, useEffect } from 'react';

const AuthContext = createContext(null);

// Helper function to decode JWT
const decodeJWT = (token) => {
    try {
        const base64Url = token.split('.')[1];
        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(
            atob(base64)
                .split('')
                .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
                .join('')
        );
        return JSON.parse(jsonPayload);
    } catch (error) {
        console.error('Failed to decode JWT:', error);
        return null;
    }
};

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const token = localStorage.getItem('jwtToken');
        if (token) {
            // DECODE THE TOKEN to get actual user data
            const decoded = decodeJWT(token);

            if (decoded) {
                // Extract username from token (adjust field name based on your JWT structure)
                setUser({
                    username: decoded.sub || decoded.username || decoded.email,
                    // Add other fields from token if needed
                    ...decoded
                });
            } else {
                // Token is invalid, remove it
                localStorage.removeItem('jwtToken');
            }
        }
        setIsLoading(false);
    }, []);

    const login = (userData, token) => {
        localStorage.setItem('jwtToken', token);
        setUser(userData);
    };

    const logout = () => {
        localStorage.removeItem('jwtToken');
        setUser(null);
    };

    const value = { currentUser: user, isLoading, login, logout };

    return (
        <AuthContext.Provider value={value}>
            {!isLoading && children}
        </AuthContext.Provider>
    );
};

export const useAuth = () => {
    return useContext(AuthContext);
};