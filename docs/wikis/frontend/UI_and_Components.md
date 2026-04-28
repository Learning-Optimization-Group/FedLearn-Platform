# UI and Components

The FedLearn frontend is transitioning from a traditional custom-CSS layout to a modern, utility-first (Tailwind CSS) approach. 

## Two UI Paradigms

The application currently supports two distinct visual languages side-by-side:

### 1. Legacy UI (Standard Routes)
Located in `src/components/` and `src/pages/`, the legacy UI relies on standard CSS files located in `src/styles/`. Components like `ProjectCard`, `ModelCard`, and standard forms utilize classes defined in files like `Dashboard.css` and `Layout.css`.

### 2. Redesign V2 (Apple-inspired Dark Theme)
Located in `src/components/redesign/`, this is the future direction of the platform. The V2 interface leverages **Tailwind CSS 4.x** to construct a sleek, glassmorphic, and dynamic UI without custom CSS files.
These components are accessed via the `/v2` routes.

## The V2 Design System

The V2 redesign strictly adheres to modern web aesthetics.

- **Layout**: Uses a permanent Sidebar (`Sidebar.tsx`) and a flexible main content area (`LayoutV2.tsx`).
- **Cards**: Elements like `ProjectCard.tsx` in V2 use soft gradients, semi-transparent backgrounds (`bg-gray-800/50`), and subtle borders (`border-white/10`) to simulate glass.
- **Icons**: `lucide-react` provides a consistent, crisp icon set.

### Example: Tailwind Card Component
```tsx
export const Card = ({ children, className }) => (
    <div className={`bg-gray-900/40 border border-white/5 rounded-2xl p-6 backdrop-blur-xl ${className}`}>
        {children}
    </div>
);
```

## Shared Components and Utilities

Regardless of the UI version, some components are universally applied to handle complex logic gracefully:

- **`LogViewer.tsx`**: Renders real-time log outputs. The V2 equivalent is highly stylized to look like an integrated terminal window.
- **`ErrorBoundary.tsx`**: Wraps the entire application to catch rendering errors and prevent white screens of death, providing a fallback UI instead.
- **`DiskLoader.tsx`**: A standardized loading spinner used across the app during asynchronous transitions or bootups.
- **Modals**: E.g., `CreateProjectModal.tsx` / `ResultsModal.tsx` rely on React Portals or absolute positioning to render overlays above the main interface.

## Responsive Design

Both UI versions are designed to be responsive.
- The legacy UI uses media queries inside CSS.
- The V2 UI uses Tailwind's built-in breakpoint prefixes (e.g., `md:flex`, `lg:grid-cols-3`) to reflow layouts fluidly from mobile to desktop sizes.
