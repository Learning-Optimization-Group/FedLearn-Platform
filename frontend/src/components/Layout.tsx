import React, { useEffect, useState } from 'react';
import { Outlet, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import '../styles/Layout.css';

const Layout: React.FC = () => {
    const [userName, setUserName] = useState('');
    const { currentUser, logout } = useAuth();

    useEffect(() => {
        if (currentUser) {
            setUserName(currentUser.username);
        }
    }, [currentUser]); // Fixed: Added currentUser as dependency

    return (
        <div className="layout-container">
            <header className="app-header">
                <Link to="/" className="logo-link">
                    <div className="logo">
                        <span>FedLearn Platform</span>
                    </div>
                </Link>
                <nav className="app-nav">
                    <Link to="/dashboard">Dashboard</Link>
                    <Link to="/models">Models</Link>
                </nav>
                <div className="user-profile">
                    {userName && (
                        <span className="welcome-text">Welcome, {userName}!</span>
                    )}
                    <button onClick={logout} className="logout-button" aria-label="Logout">
                        Logout
                    </button>
                </div>
            </header>
            <main className="app-content">
                <Outlet />
            </main>
        </div>
    );
};

export default Layout;
