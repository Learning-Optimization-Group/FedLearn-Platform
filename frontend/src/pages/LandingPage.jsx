import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/LandingPage.css'; // We'll update the CSS in the next step
import platformLogo from '../assets/images/logo.png';

const LandingPage = () => {
    return (
        <div className="landing-container">
            <header className="landing-header">
                <Link to="/" className="logo-link">
                    <div className="logo">
                        <img src={platformLogo} alt="FL Platform Logo" className="logo-image" />
                        <span>Your FL Platform</span>
                    </div>
                </Link>
                <nav className="landing-nav">
                    <a href="#services">Core Services</a>
                    <a href="#features">Upcoming Features</a>
                    <a href="#about">About</a>
                </nav>
                <div className="auth-buttons">
                    <Link to="/login" className="btn btn-secondary">Sign In</Link>
                    <Link to="/register" className="btn btn-primary">Sign Up</Link>
                </div>
            </header>

            <main className="hero-section">
                {/* --- NEW, ORIGINAL TEXT --- */}
                <h1 className="hero-title">Unlock Insights from Distributed Data. Securely.</h1>
                <p className="hero-subtitle">
                    Our platform provides the essential toolkit to build, train, and deploy privacy-preserving AI models using Federated Learning.
                </p>

                <div className="services-grid" id="services">
                    <div className="service-card">
                        <h3>Federated Training as a Service</h3>
                        <p>Orchestrate complex training and aggregation across decentralized clients with robust algorithms.</p>
                    </div>
                    <div className="service-card">
                        <h3>On-Demand FL Compute Server</h3>
                        <p>Dynamically launch isolated and secure gRPC server instances for each of your training jobs.</p>
                    </div>
                    <div className="service-card">
                        <h3>Model Initialization & Pre-training</h3>
                        <p>Start with a random baseline or leverage a powerful pre-trained model to accelerate learning.</p>
                    </div>
                    {/* --- MISSING SERVICE ADDED --- */}
                    <div className="service-card">
                        <h3>Secure API Control Plane</h3>
                        <p>Manage the entire platform lifecycle through a secure, authenticated REST API.</p>
                    </div>
                </div>

                <Link to="/login" className="btn-launch">
                    Launch Platform
                </Link>
            </main>

            {/* --- NEW UPCOMING FEATURES SECTION --- */}
            <section className="features-section" id="features">
                <h2 className="section-title">Upcoming Features & Roadmap</h2>
                <div className="features-category">
                    <h3>Platform & MLOps</h3>
                    <div className="features-grid">
                        <div className="feature-card">
                            <h4>Experimental Tracking</h4>
                            <p>Live metrics, charts, and results history for training runs.</p>
                        </div>
                        <div className="feature-card">
                            <h4>Model Hub & Registry</h4>
                            <p>A central place to version, manage, and download trained models.</p>
                        </div>
                        <div className="feature-card">
                            <h4>Model Serving API</h4>
                            <p>Deploy your trained models as live inference endpoints with a single click.</p>
                        </div>
                    </div>
                </div>
                <div className="features-category">
                    <h3>Generative AI & LLM Services</h3>
                    <div className="features-grid">
                        <div className="feature-card">
                            <h4>Federated LLM Fine-Tuning</h4>
                            <p>Adapt foundation models like Llama and OPT on private, distributed data.</p>
                        </div>
                        <div className="feature-card">
                            <h4>Private LLM Inference</h4>
                            <p>Serve fine-tuned models for tasks like Q&A and summarization via a secure API.</p>
                        </div>
                        <div className="feature-card">
                            <h4>Domain-Specific AI Agents</h4>
                            <p>Productize your fine-tuned LLMs as specialized agents (e.g., "Legal Assistant").</p>
                        </div>
                    </div>
                </div>
            </section>

        </div>
    );
};

export default LandingPage;