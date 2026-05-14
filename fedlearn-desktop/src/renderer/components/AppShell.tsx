import React, { useState } from 'react';
import Sidebar, { RouteKey } from './Sidebar';
import StatusIndicator from './StatusIndicator';
import type { ContainerStatus } from '../App';

interface AppShellProps {
  username: string;
  status: ContainerStatus;
  hardwareLabel: string;
  onLogout: () => void;
  initialRoute?: RouteKey;
  renderView: (route: RouteKey) => React.ReactNode;
  drawer?: React.ReactNode;
}

const ROUTE_TITLES: Record<RouteKey, string> = {
  projects: 'My Projects',
  discover: 'Discover',
  requests: 'My Requests',
  models: 'Models',
  settings: 'Settings',
};

const AppShell: React.FC<AppShellProps> = ({
  username, status, hardwareLabel, onLogout, initialRoute = 'projects', renderView, drawer,
}) => {
  const [route, setRoute] = useState<RouteKey>(initialRoute);
  return (
    <div className="app-shell">
      <div className="app-shell__sidebar">
        <Sidebar active={route} username={username} onNavigate={setRoute} />
      </div>
      <header className="app-shell__header shell-header">
        <div className="shell-header__title">{ROUTE_TITLES[route]}</div>
        <div className="shell-header__actions">
          <span className="shell-header__chip">⚡ {hardwareLabel}</span>
          <StatusIndicator status={status} />
          <button className="shell-header__btn" onClick={onLogout}>Sign Out</button>
        </div>
      </header>
      <main className="app-shell__content">{renderView(route)}</main>
      {drawer && <div className="app-shell__drawer">{drawer}</div>}
    </div>
  );
};

export default AppShell;
