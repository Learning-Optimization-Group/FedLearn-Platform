import React, { createContext, useState, useContext, useEffect, ReactNode } from 'react';

interface User {
    username: string;
    email?: string;
    sub?: string;
    [key: string]: any;
}

interface AuthContextType {
    currentUser: User | null;
    isLoading: boolean;
    login: (userData: User, token: string) => void;
    logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

// Helper function to decode JWT (client-side only, no signature verification)
const decodeJWT = (token: string): any | null => {
    try {
        const base64Url = token.split('.')[1];
        if (!base64Url) return null;

        const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        const jsonPayload = decodeURIComponent(
            atob(base64)
                .split('')
                .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
                .join('')
        );
        const decoded = JSON.parse(jsonPayload);

        // Check token expiration
        if (decoded.exp && decoded.exp * 1000 < Date.now()) {
            return null; // Token expired
        }

        return decoded;
    } catch (error) {
        console.error('Failed to decode JWT:', error);
        return null;
    }
};

interface AuthProviderProps {
    children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
    const [user, setUser] = useState<User | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const token = localStorage.getItem('jwtToken');
        if (token) {
            const decoded = decodeJWT(token);

            if (decoded) {
                setUser({
                    username: decoded.sub || decoded.username || decoded.email || 'User',
                    ...decoded
                });
            } else {
                // Token is invalid or expired, remove it
                localStorage.removeItem('jwtToken');
            }
        }
        setIsLoading(false);
    }, []);

    const login = (userData: User, token: string) => {
        localStorage.setItem('jwtToken', token);
        setUser(userData);
    };

    const logout = () => {
        localStorage.removeItem('jwtToken');
        setUser(null);
    };

    const value: AuthContextType = { currentUser: user, isLoading, login, logout };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
};

export const useAuth = (): AuthContextType => {
    const context = useContext(AuthContext);
    if (!context) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};
