import React, { useEffect, useState } from 'react';
import { Outlet, Link, NavLink } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import '../styles/Layout.css';
import { ThemeToggle } from './ThemeToggle';

const Layout: React.FC = () => {
    const [userName, setUserName] = useState('');
    const { currentUser, logout } = useAuth();

    useEffect(() => {
        if (currentUser) {
            setUserName(currentUser.username);
        }
    }, [currentUser]);

    return (
        <div className="layout-container">
            <header className="app-header">
                <Link to="/" className="logo-link">
                    <div className="logo">
                        <span>FedLearn Platform</span>
                    </div>
                </Link>
                <nav className="app-nav">
                    <NavLink to="/dashboard" className={({ isActive }) => (isActive ? 'active' : '')}>
                        Dashboard
                    </NavLink>
                    <NavLink to="/clients" className={({ isActive }) => (isActive ? 'active' : '')}>
                        Clients
                    </NavLink>
                    <NavLink to="/models" className={({ isActive }) => (isActive ? 'active' : '')}>
                        Models
                    </NavLink>
                    <NavLink to="/training" className={({ isActive }) => (isActive ? 'active' : '')}>
                        Training
                    </NavLink>
                    <NavLink to="/settings" className={({ isActive }) => (isActive ? 'active' : '')}>
                        Settings
                    </NavLink>
                </nav>
                <div className="user-profile">
                    <ThemeToggle />
                    <Link to="/v2" className="v2-link" title="Switch to the redesigned UI">
                        Try V2 UI →
                    </Link>
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
