import React from 'react';

export type RouteKey = 'projects' | 'discover' | 'requests' | 'models' | 'settings';

interface SidebarProps {
  active: RouteKey;
  username: string;
  onNavigate: (route: RouteKey) => void;
}

const ITEMS: Array<{ key: RouteKey; label: string; icon: string }> = [
  { key: 'projects', label: 'My Projects', icon: '▦' },
  { key: 'discover', label: 'Discover', icon: '◎' },
  { key: 'requests', label: 'My Requests', icon: '↻' },
  { key: 'models', label: 'Models', icon: '◇' },
  { key: 'settings', label: 'Settings', icon: '⚙' },
];

const Sidebar: React.FC<SidebarProps> = ({ active, username, onNavigate }) => {
  const initials = (username || 'U').slice(0, 2).toUpperCase();
  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <div className="sidebar__brand-icon">◆</div>
        <div className="sidebar__brand-name">FedLearn</div>
      </div>
      <nav className="sidebar__nav">
        <div className="sidebar__section-label">Workspace</div>
        {ITEMS.map((item) => (
          <button
            key={item.key}
            type="button"
            className={`sidebar__item${active === item.key ? ' sidebar__item--active' : ''}`}
            onClick={() => onNavigate(item.key)}
          >
            <span className="sidebar__item-icon">{item.icon}</span>
            <span>{item.label}</span>
          </button>
        ))}
      </nav>
      <div className="sidebar__footer">
        <div className="sidebar__avatar">{initials}</div>
        <div className="sidebar__user-name">
          {username || 'User'}
          <span>Signed in</span>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
