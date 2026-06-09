import React from 'react';
import { Link } from 'react-router-dom';
import {
    Brain,
    Network,
    Server,
    Cpu,
    ShieldCheck,
    LineChart,
    Boxes,
    Rocket,
    Bot,
    MessageSquareLock,
    Sparkles,
} from 'lucide-react';
import { Button, Card } from '../components/ui';

const services = [
    {
        icon: Network,
        title: 'Federated Training as a Service',
        body: 'Orchestrate complex training and aggregation across decentralized clients with robust algorithms.',
    },
    {
        icon: Server,
        title: 'On-Demand FL Compute Server',
        body: 'Dynamically launch isolated and secure gRPC server instances for each of your training jobs.',
    },
    {
        icon: Cpu,
        title: 'Model Initialization & Pre-training',
        body: 'Start with a random baseline or leverage a powerful pre-trained model to accelerate learning.',
    },
    {
        icon: ShieldCheck,
        title: 'Secure API Control Plane',
        body: 'Manage the entire platform lifecycle through a secure, authenticated REST API.',
    },
];

const platformFeatures = [
    {
        icon: LineChart,
        title: 'Experimental Tracking',
        body: 'Live metrics, charts, and results history for training runs.',
    },
    {
        icon: Boxes,
        title: 'Model Hub & Registry',
        body: 'A central place to version, manage, and download trained models.',
    },
    {
        icon: Rocket,
        title: 'Model Serving API',
        body: 'Deploy your trained models as live inference endpoints with a single click.',
    },
];

const llmFeatures = [
    {
        icon: Sparkles,
        title: 'Federated LLM Fine-Tuning',
        body: 'Adapt foundation models like Llama and OPT on private, distributed data.',
    },
    {
        icon: MessageSquareLock,
        title: 'Private LLM Inference',
        body: 'Serve fine-tuned models for tasks like Q&A and summarization via a secure API.',
    },
    {
        icon: Bot,
        title: 'Domain-Specific AI Agents',
        body: 'Productize your fine-tuned LLMs as specialized agents (e.g., "Legal Assistant").',
    },
];

const FeatureCard: React.FC<{
    icon: React.ComponentType<{ className?: string; strokeWidth?: number }>;
    title: string;
    body: string;
}> = ({ icon: Icon, title, body }) => (
    <Card padding="lg" className="flex flex-col gap-3">
        <div className="w-9 h-9 rounded-md bg-surface-2 border border-hairline flex items-center justify-center">
            <Icon className="w-5 h-5 text-accent" strokeWidth={1.5} />
        </div>
        <h4 className="text-h4 text-fg">{title}</h4>
        <p className="text-body text-fg-muted">{body}</p>
    </Card>
);

const LandingPage: React.FC = () => {
    return (
        <div className="min-h-screen bg-canvas text-fg font-sans">
            <header className="flex items-center justify-between gap-6 px-6 md:px-12 h-20 border-b border-hairline">
                <Link to="/" className="flex items-center gap-3">
                    <div className="w-8 h-8 rounded-md bg-surface-1 border border-hairline flex items-center justify-center">
                        <Brain className="w-5 h-5 text-fg" strokeWidth={1.5} />
                    </div>
                    <span className="text-h4 tracking-tight text-fg">FedLearn Platform</span>
                </Link>

                <nav className="hidden md:flex items-center gap-6">
                    <a href="#services" className="text-label font-medium text-fg-muted hover:text-fg transition-colors">
                        Core Services
                    </a>
                    <a href="#features" className="text-label font-medium text-fg-muted hover:text-fg transition-colors">
                        Upcoming Features
                    </a>
                    <a href="#about" className="text-label font-medium text-fg-muted hover:text-fg transition-colors">
                        About
                    </a>
                </nav>

                <div className="flex items-center gap-3">
                    <Link to="/login">
                        <Button variant="secondary">Sign In</Button>
                    </Link>
                    <Link to="/register">
                        <Button variant="primary">Sign Up</Button>
                    </Link>
                </div>
            </header>

            <main className="px-6 md:px-12 py-16 md:py-24 max-w-6xl mx-auto">
                <section className="text-center max-w-3xl mx-auto">
                    <h1 className="text-h1 text-fg">Unlock Insights from Distributed Data. Securely.</h1>
                    <p className="text-body-lg text-fg-muted mt-5">
                        Our platform provides the essential toolkit to build, train, and deploy
                        privacy-preserving AI models using Federated Learning.
                    </p>
                </section>

                <section id="services" className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-16">
                    {services.map((s) => (
                        <FeatureCard key={s.title} icon={s.icon} title={s.title} body={s.body} />
                    ))}
                </section>

                <div className="flex justify-center mt-12">
                    <Link to="/login">
                        <Button variant="primary" className="h-11 px-6 text-body-lg">
                            <Rocket className="w-5 h-5" strokeWidth={1.5} />
                            Launch Platform
                        </Button>
                    </Link>
                </div>
            </main>

            <section id="features" className="px-6 md:px-12 py-16 border-t border-hairline">
                <div className="max-w-6xl mx-auto">
                    <h2 className="text-h2 text-fg text-center">Upcoming Features & Roadmap</h2>

                    <div className="mt-12">
                        <h3 className="text-h4 text-fg-muted mb-4">Platform & MLOps</h3>
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                            {platformFeatures.map((f) => (
                                <FeatureCard key={f.title} icon={f.icon} title={f.title} body={f.body} />
                            ))}
                        </div>
                    </div>

                    <div className="mt-12">
                        <h3 className="text-h4 text-fg-muted mb-4">Generative AI & LLM Services</h3>
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                            {llmFeatures.map((f) => (
                                <FeatureCard key={f.title} icon={f.icon} title={f.title} body={f.body} />
                            ))}
                        </div>
                    </div>
                </div>
            </section>
        </div>
    );
};

export default LandingPage;
