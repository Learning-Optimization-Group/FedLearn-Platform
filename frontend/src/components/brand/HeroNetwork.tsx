import { Laptop, Smartphone, Server, Monitor, Tablet, Watch } from 'lucide-react';
import { BrandMark } from './BrandMark';

interface Device {
    icon: typeof Laptop;
    label: string;
    x: number; // % of container
    y: number;
}

const DEVICES: Device[] = [
    { icon: Laptop, label: 'Laptop', x: 12, y: 24 },
    { icon: Smartphone, label: 'Phone', x: 50, y: 9 },
    { icon: Server, label: 'Server', x: 88, y: 24 },
    { icon: Monitor, label: 'Desktop', x: 89, y: 78 },
    { icon: Tablet, label: 'Tablet', x: 50, y: 93 },
    { icon: Watch, label: 'Wearable', x: 11, y: 78 },
];

/**
 * The federated picture: everyday devices, each keeping its own data, sending
 * only learning updates (the flowing ember dashes) into one shared model at the
 * centre. Purely decorative — labelled aria-hidden.
 */
export function HeroNetwork({ className }: { className?: string }) {
    return (
        <div
            className={`relative mx-auto aspect-[6/5] w-full max-w-[560px] ${className ?? ''}`}
            aria-hidden
        >
            {/* connectors */}
            <svg
                className="absolute inset-0 h-full w-full"
                viewBox="0 0 100 100"
                preserveAspectRatio="none"
                fill="none"
            >
                {DEVICES.map((d) => (
                    <line
                        key={`base-${d.label}`}
                        x1="50"
                        y1="50"
                        x2={d.x}
                        y2={d.y}
                        stroke="var(--line)"
                        strokeWidth="1"
                        vectorEffect="non-scaling-stroke"
                    />
                ))}
                {DEVICES.map((d, i) => (
                    <line
                        key={`flow-${d.label}`}
                        x1={d.x}
                        y1={d.y}
                        x2="50"
                        y2="50"
                        stroke="var(--accent)"
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        vectorEffect="non-scaling-stroke"
                        className="flow-line"
                        style={{ animationDelay: `${i * 0.22}s`, opacity: 0.85 }}
                    />
                ))}
            </svg>

            {/* device nodes */}
            {DEVICES.map((d, i) => (
                <div
                    key={d.label}
                    className="absolute z-10 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center gap-1.5"
                    style={{ left: `${d.x}%`, top: `${d.y}%` }}
                >
                    <div className="icon-tile float-slow" style={{ animationDelay: `${i * 0.4}s` }}>
                        <d.icon className="h-5 w-5" strokeWidth={1.5} />
                    </div>
                    <span className="text-caption text-fg-subtle">{d.label}</span>
                </div>
            ))}

            {/* the shared model */}
            <div className="absolute left-1/2 top-1/2 z-20 flex -translate-x-1/2 -translate-y-1/2 flex-col items-center">
                <div className="relative">
                    <div className="absolute -inset-6 rounded-full bg-accent/25 blur-2xl" />
                    <div className="relative grid h-24 w-24 place-items-center rounded-full border border-accent/30 bg-surface-1/70 backdrop-blur-sm">
                        <BrandMark size={62} />
                    </div>
                </div>
                <span className="chip mt-3 text-fg">Shared model</span>
            </div>
        </div>
    );
}
