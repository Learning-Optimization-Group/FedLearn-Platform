import { type ReactNode } from 'react';

interface PageHeaderProps {
    title: string;
    subtitle?: string;
    /** Right-aligned actions (buttons, search). */
    children?: ReactNode;
}

/** Shared sticky page header for the authed views. */
export function PageHeader({ title, subtitle, children }: PageHeaderProps) {
    return (
        <header className="sticky top-0 z-20 flex h-20 items-center justify-between gap-4 border-b border-hairline bg-canvas/80 px-6 backdrop-blur-xl md:px-10">
            <div className="min-w-0">
                <h1 className="text-h3 font-display font-semibold tracking-tight text-fg">{title}</h1>
                {subtitle && <p className="mt-0.5 truncate text-label text-fg-muted">{subtitle}</p>}
            </div>
            {children && <div className="flex items-center gap-3">{children}</div>}
        </header>
    );
}
