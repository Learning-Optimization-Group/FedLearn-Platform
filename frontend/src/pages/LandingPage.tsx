import React from 'react';
import { Link } from 'react-router-dom';
import { ThemeToggle } from '../components/ThemeToggle';
import { Network, Server, Fingerprint, Lock } from 'lucide-react';

const LandingPage: React.FC = () => {
    return (
        <div className="min-h-screen font-sans overflow-x-hidden" style={{ backgroundColor: 'var(--background-primary)' }}>
            <div className="absolute inset-0 pointer-events-none" style={{
                background: 'radial-gradient(circle at 50% -10%, var(--glow-accent), transparent 60%)',
                opacity: 0.6
            }} />

            <header className="relative z-10 max-w-[1200px] mx-auto px-6 py-6 flex items-center justify-between">
                <Link to="/" className="flex items-center gap-3 no-underline">
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ backgroundColor: 'var(--accent-primary)' }}>
                        <Network className="w-5 h-5 text-white" />
                    </div>
                    <span className="font-display font-semibold text-[20px] text-(--text-primary)">FedLearn</span>
                </Link>

                <nav className="hidden md:flex items-center gap-8 font-medium text-[14px]">
                    <a href="#services" className="text-(--text-secondary) hover:text-(--text-primary) transition-colors">Core Services</a>
                    <a href="#features" className="text-(--text-secondary) hover:text-(--text-primary) transition-colors">Roadmap</a>
                </nav>

                <div className="flex items-center gap-4">
                    <ThemeToggle />
                    <Link to="/login" className="text-[14px] font-semibold text-(--text-secondary) hover:text-(--text-primary) transition-colors">Sign In</Link>
                    <Link to="/register" className="px-4 py-2 rounded-lg text-[14px] font-semibold text-white transition-all hover:brightness-110" style={{ backgroundColor: 'var(--accent-primary)' }}>
                        Launch Platform
                    </Link>
                </div>
            </header>

            <main className="relative z-10 max-w-[1200px] mx-auto px-6 pt-24 pb-16 text-center">
                <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full border mb-8" style={{ backgroundColor: 'color-mix(in srgb, var(--accent-primary) 10%, transparent)', borderColor: 'color-mix(in srgb, var(--accent-primary) 30%, transparent)' }}>
                    <span className="w-2 h-2 rounded-full animate-pulse" style={{ backgroundColor: 'var(--accent-primary)' }} />
                    <span className="font-mono text-[11px] font-semibold tracking-wider uppercase" style={{ color: 'var(--accent-primary)' }}>V2 Platform Live</span>
                </div>

                <h1 className="font-display text-[48px] md:text-[64px] font-medium tracking-tight text-(--text-primary) leading-[1.1] max-w-[800px] mx-auto mb-6">
                    Unlock insights from distributed data. <span className="italic" style={{ color: 'var(--accent-primary)' }}>Securely.</span>
                </h1>
                
                <p className="text-[18px] text-(--text-secondary) max-w-[600px] mx-auto mb-10 leading-relaxed">
                    Our platform provides the essential toolkit to build, train, and deploy privacy-preserving AI models using federated learning architectures.
                </p>

                <Link to="/register" className="inline-flex items-center justify-center px-8 py-4 rounded-xl text-[16px] font-semibold text-white transition-all hover:scale-105" style={{ backgroundColor: 'var(--accent-primary)', boxShadow: '0 0 20px var(--glow-accent)' }}>
                    Start Building for Free
                </Link>

                <div className="mt-32 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 text-left" id="services">
                    {[
                        { icon: <Network />, title: 'Federated Training', desc: 'Orchestrate complex training and aggregation across decentralized clients.' },
                        { icon: <Server />, title: 'On-Demand Compute', desc: 'Dynamically launch isolated and secure gRPC server instances for jobs.' },
                        { icon: <Fingerprint />, title: 'Model Initialization', desc: 'Start with a random baseline or leverage pre-trained foundation models.' },
                        { icon: <Lock />, title: 'Secure API Control', desc: 'Manage the entire platform lifecycle through a secure REST API.' }
                    ].map((svc, i) => (
                        <div key={i} className="p-6 rounded-2xl border transition-all hover:-translate-y-1" style={{ backgroundColor: 'var(--background-card)', borderColor: 'var(--border-color)', boxShadow: 'var(--shadow-soft)' }}>
                            <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-4" style={{ backgroundColor: 'color-mix(in srgb, var(--accent-primary) 10%, transparent)', color: 'var(--accent-primary)' }}>
                                {React.cloneElement(svc.icon as React.ReactElement<{ className?: string }>, { className: 'w-5 h-5' })}
                            </div>
                            <h3 className="font-display text-[18px] font-medium text-(--text-primary) mb-2">{svc.title}</h3>
                            <p className="text-[14px] text-(--text-secondary) leading-relaxed">{svc.desc}</p>
                        </div>
                    ))}
                </div>
            </main>
        </div>
    );
};

export default LandingPage;
