import React, { useEffect, useState } from 'react';
import { Outlet, Link } from 'react-router-dom';
import { useAuth } from '../context/AuthContext'; // Import your auth context
import '../styles/Layout.css'; // We'll add styles for the header
import platformLogo from '../assets/images/logo_black.png';

const Layout = () => {

    const [userName, setUserName] = useState('');
    const { currentUser, logout } = useAuth();

    useEffect(() => {
        console.log('CurrentUser - ', currentUser);
        setUserName(currentUser.username);
    }, []);

    return (
        <div className="layout-container">
            <header className="app-header">
                <Link to="/" className="logo-link">
                    <div className="logo">
                        <img src={platformLogo} alt="FL Platform Logo" className="logo-image" />
                        <span>Your FL Platform</span>
                    </div>
                </Link>
                <nav className="app-nav">
                    <Link to="/dashboard">Dashboard</Link>
                    <Link to="/models">Models</Link>
                    {/* Add other main navigation links here */}
                </nav>
                <div className="user-profile">
                    {/* --- THIS IS THE NEW WELCOME TEXT --- */}
                    {userName && (
                        <span className="welcome-text">Welcome, {userName}!</span>
                    )}
                    <button onClick={logout} className="logout-button">Logout</button>
                </div>
            </header>
            <main className="app-content">
                <Outlet /> {/* This is where your page components (like DashboardPage) will be rendered */}
            </main>
        </div>
    );
};

export default Layout;