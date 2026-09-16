import React from 'react';
import { Link } from 'react-router-dom';
import {
    Network,
    Server,
    ShieldCheck,
    LineChart,
    Cpu,
    Boxes,
    ArrowRight,
    SlidersHorizontal,
    MessageSquareLock,
    Bot,
    Rocket,
    Github,
    Lock,
} from 'lucide-react';
import { Button, Card, SectionLabel } from '../components/ui';
import { Wordmark, HeroNetwork } from '../components/brand';

/* ------------------------------ content (plain language) ------------------------------ */

const coreFeatures = [
    {
        icon: Network,
        title: 'Train across many devices',
        body: 'Bring phones, laptops, and servers together to teach one model. FedLearn does all the coordinating for you.',
    },
    {
        icon: Server,
        title: 'Start in one click',
        body: 'Spin up a secure training session instantly. Nothing to install, nothing to configure.',
    },
    {
        icon: ShieldCheck,
        title: 'Private by design',
        body: 'Raw data never leaves a device. Only small, anonymous learning updates are ever shared.',
    },
    {
        icon: LineChart,
        title: 'Watch it learn, live',
        body: 'Follow accuracy and progress in real time as every device pitches in, round after round.',
    },
    {
        icon: Cpu,
        title: 'Runs on your hardware',
        body: 'Mac, Windows, Linux, NVIDIA Jetson, even an Android phone — FedLearn meets your devices where they are.',
    },
    {
        icon: Boxes,
        title: 'Keep every model',
        body: 'Save, compare, and reuse the models you train. Pick up right where you left off, any time.',
    },
];

const steps = [
    {
        n: '01',
        title: 'Create a project',
        body: 'Choose what you want your AI to learn. Sensible defaults are ready, so you can just press go.',
    },
    {
        n: '02',
        title: 'Connect your devices',
        body: 'Each device joins with a tap and trains quietly on its own data — without ever sharing it.',
    },
    {
        n: '03',
        title: 'Get one smart model',
        body: "FedLearn blends everyone's progress into a single model that's smarter than any device alone.",
    },
];

const roadmap = [
    {
        group: 'Platform',
        items: [
            { icon: LineChart, title: 'Experiment tracking', body: 'A clear history of every run, with charts you can revisit.' },
            { icon: Boxes, title: 'Model library', body: 'One place to version, manage, and download trained models.' },
            { icon: Rocket, title: 'One-click serving', body: 'Turn a trained model into a live endpoint with a single click.' },
        ],
    },
    {
        group: 'Language models',
        items: [
            { icon: SlidersHorizontal, title: 'Fine-tune language models', body: 'Teach models like Llama on private data they never have to upload.' },
            { icon: MessageSquareLock, title: 'Private answers & summaries', body: 'Run Q&A and summaries through a secure, private endpoint.' },
            { icon: Bot, title: 'Custom AI assistants', body: 'Package your tuned model as a focused helper for your team.' },
        ],
    },
];

const platforms = ['macOS', 'Windows', 'Linux', 'NVIDIA Jetson', 'Android'];

/* ------------------------------ small pieces ------------------------------ */

type IconType = React.ComponentType<{ className?: string; strokeWidth?: number }>;

const FeatureItem: React.FC<{ icon: IconType; title: string; body: string }> = ({
    icon: Icon,
    title,
    body,
}) => (
    <div className="flex flex-col gap-4">
        <span className="icon-tile">
            <Icon className="h-5 w-5" strokeWidth={1.5} />
        </span>
        <div>
            <h3 className="text-h4 text-fg">{title}</h3>
            <p className="mt-1.5 text-body text-fg-muted">{body}</p>
        </div>
    </div>
);

/* ------------------------------ page ------------------------------ */

const LandingPage: React.FC = () => {
    return (
        <div className="min-h-screen bg-canvas font-sans text-fg">
            {/* header */}
            <header className="sticky top-0 z-40 border-b border-hairline bg-canvas">
                <div className="mx-auto flex h-16 max-w-6xl items-center justify-between gap-6 px-5 md:px-8">
                    <Link to="/" aria-label="FedLearn home">
                        <Wordmark size={28} />
                    </Link>
                    <nav className="hidden items-center gap-8 md:flex">
                        {[
                            ['How it works', '#how'],
                            ['Features', '#features'],
                            ['Roadmap', '#roadmap'],
                        ].map(([label, href]) => (
                            <a
                                key={href}
                                href={href}
                                className="text-label font-medium text-fg-muted transition-colors hover:text-fg"
                            >
                                {label}
                            </a>
                        ))}
                    </nav>
                    <div className="flex items-center gap-2">
                        <Link to="/login" className="hidden sm:block">
                            <Button variant="ghost">Sign in</Button>
                        </Link>
                        <Link to="/register">
                            <Button variant="primary">Get started</Button>
                        </Link>
                    </div>
                </div>
            </header>

            <main>
                {/* hero */}
                <section>
                    <div className="mx-auto max-w-6xl px-5 pb-10 pt-16 text-center md:px-8 md:pt-24">
                        <div className="reveal flex justify-center" style={{ animationDelay: '40ms' }}>
                            <span className="chip">
                                <span className="h-1.5 w-1.5 rounded-full bg-accent" />
                                Open source · Private by design
                            </span>
                        </div>

                        <h1
                            className="reveal display-hero mx-auto mt-7 max-w-4xl text-[42px] leading-[1.02] text-fg sm:text-[60px] lg:text-[74px]"
                            style={{ animationDelay: '100ms' }}
                        >
                            Train AI together.
                            <br />
                            Keep your data <span className="text-accent">home</span>.
                        </h1>

                        <p
                            className="reveal mx-auto mt-6 max-w-2xl text-body-lg text-fg-muted"
                            style={{ animationDelay: '180ms' }}
                        >
                            FedLearn lets many devices improve one shared AI model — without any of
                            them handing over private data. Open, simple, and private from the start.
                        </p>

                        <div
                            className="reveal mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row"
                            style={{ animationDelay: '260ms' }}
                        >
                            <Link to="/register" className="w-full sm:w-auto">
                                <Button variant="primary" size="lg" className="w-full sm:w-auto">
                                    Get started — it's free
                                    <ArrowRight className="h-4 w-4" strokeWidth={2} />
                                </Button>
                            </Link>
                            <a href="#how" className="w-full sm:w-auto">
                                <Button variant="secondary" size="lg" className="w-full sm:w-auto">
                                    See how it works
                                </Button>
                            </a>
                        </div>

                        <div
                            className="reveal mx-auto mt-16 md:mt-20"
                            style={{ animationDelay: '340ms' }}
                        >
                            <HeroNetwork />
                        </div>

                        {/* platform strip */}
                        <div className="mt-12 flex flex-col items-center gap-4">
                            <SectionLabel>Works on the devices you already have</SectionLabel>
                            <div className="flex flex-wrap items-center justify-center gap-x-7 gap-y-3">
                                {platforms.map((p) => (
                                    <span key={p} className="text-label font-medium text-fg-muted">
                                        {p}
                                    </span>
                                ))}
                            </div>
                        </div>
                    </div>
                </section>

                {/* features */}
                <section id="features" className="border-t border-hairline">
                    <div className="mx-auto max-w-6xl px-5 py-20 md:px-8 md:py-28">
                        <div className="max-w-2xl">
                            <SectionLabel>What you get</SectionLabel>
                            <h2 className="mt-4 text-h1 text-fg">
                                Everything you need to train AI, together.
                            </h2>
                            <p className="mt-4 text-body-lg text-fg-muted">
                                One simple platform handles the hard parts — coordination, privacy,
                                and progress — so you can focus on what you want to build.
                            </p>
                        </div>
                        <div className="mt-14 grid grid-cols-1 gap-x-10 gap-y-12 sm:grid-cols-2 lg:grid-cols-3">
                            {coreFeatures.map((f) => (
                                <FeatureItem key={f.title} {...f} />
                            ))}
                        </div>
                    </div>
                </section>

                {/* how it works */}
                <section id="how" className="border-t border-hairline bg-surface-2">
                    <div className="mx-auto max-w-6xl px-5 py-20 md:px-8 md:py-28">
                        <div className="mx-auto max-w-2xl text-center">
                            <SectionLabel>How it works</SectionLabel>
                            <h2 className="mt-4 text-h1 text-fg">
                                Three steps. No expertise required.
                            </h2>
                        </div>
                        <div className="mt-16 grid grid-cols-1 gap-5 md:grid-cols-3">
                            {steps.map((s, i) => (
                                <Card key={s.n} padding="lg" className="relative">
                                    <span className="font-mono text-h2 font-medium text-fg-subtle">
                                        {s.n}
                                    </span>
                                    <h3 className="mt-4 text-h4 text-fg">{s.title}</h3>
                                    <p className="mt-2 text-body text-fg-muted">{s.body}</p>
                                    {i < steps.length - 1 && (
                                        <ArrowRight
                                            className="absolute -right-3.5 top-1/2 hidden h-5 w-5 -translate-y-1/2 text-fg-subtle md:block"
                                            strokeWidth={1.5}
                                        />
                                    )}
                                </Card>
                            ))}
                        </div>
                    </div>
                </section>

                {/* roadmap */}
                <section id="roadmap" className="border-t border-hairline">
                    <div className="mx-auto max-w-6xl px-5 py-20 md:px-8 md:py-28">
                        <div className="max-w-2xl">
                            <SectionLabel>On the way</SectionLabel>
                            <h2 className="mt-4 text-h1 text-fg">
                                Growing into a full AI workshop.
                            </h2>
                            <p className="mt-4 text-body-lg text-fg-muted">
                                FedLearn already trains models together. Here's what's landing next.
                            </p>
                        </div>
                        <div className="mt-14 space-y-14">
                            {roadmap.map((section) => (
                                <div key={section.group}>
                                    <SectionLabel>{section.group}</SectionLabel>
                                    <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                                        {section.items.map((f) => (
                                            <Card key={f.title} padding="lg">
                                                <span className="icon-tile">
                                                    <f.icon className="h-5 w-5" strokeWidth={1.5} />
                                                </span>
                                                <h4 className="mt-4 text-h4 text-fg">{f.title}</h4>
                                                <p className="mt-1.5 text-body text-fg-muted">{f.body}</p>
                                            </Card>
                                        ))}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* CTA */}
                <section className="border-t border-hairline">
                    <div className="mx-auto max-w-6xl px-5 py-24 text-center md:px-8">
                        <h2 className="display-hero mx-auto max-w-2xl text-[32px] leading-tight text-fg sm:text-[46px]">
                            Ready to train your first model?
                        </h2>
                        <p className="mx-auto mt-4 max-w-xl text-body-lg text-fg-muted">
                            Create a free account and start a training session in minutes.
                        </p>
                        <div className="mt-9 flex justify-center">
                            <Link to="/register">
                                <Button variant="primary" size="lg">
                                    Get started — it's free
                                    <ArrowRight className="h-4 w-4" strokeWidth={2} />
                                </Button>
                            </Link>
                        </div>
                    </div>
                </section>
            </main>

            {/* footer */}
            <footer className="border-t border-hairline">
                <div className="mx-auto max-w-6xl px-5 py-16 md:px-8">
                    <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
                        <div className="col-span-2 md:col-span-1">
                            <Wordmark size={26} />
                            <p className="mt-4 max-w-xs text-body text-fg-muted">
                                Open platform for training AI together — privately. Many devices, one
                                shared model.
                            </p>
                        </div>
                        {[
                            { h: 'Product', links: [['How it works', '#how'], ['Features', '#features'], ['Get started', '/register']] },
                            { h: 'Resources', links: [['Documentation', '#'], ['Open source', '#'], ['API', '#']] },
                            { h: 'About', links: [['Privacy', '#'], ['Roadmap', '#roadmap'], ['Contact', '#']] },
                        ].map((col) => (
                            <div key={col.h}>
                                <h4 className="text-label font-medium text-fg">{col.h}</h4>
                                <ul className="mt-4 space-y-2.5">
                                    {col.links.map(([label, href]) => (
                                        <li key={label}>
                                            <a
                                                href={href}
                                                className="text-body text-fg-muted transition-colors hover:text-fg"
                                            >
                                                {label}
                                            </a>
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        ))}
                    </div>
                    <div className="mt-14 flex flex-col items-center justify-between gap-4 border-t border-hairline pt-6 sm:flex-row">
                        <span className="text-caption text-fg-subtle">
                            © {new Date().getFullYear()} FedLearn · Private by design
                        </span>
                        <div className="flex items-center gap-5 text-fg-subtle">
                            <Lock className="h-4 w-4" strokeWidth={1.5} aria-label="Privacy first" />
                            <Github className="h-4 w-4" strokeWidth={1.5} aria-label="Open source" />
                        </div>
                    </div>
                </div>
            </footer>
        </div>
    );
};

export default LandingPage;
