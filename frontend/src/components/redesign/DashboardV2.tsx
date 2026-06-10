// =============================================================================
// FedLearn Frontend — DashboardV2 (Ember design system, real API)
// =============================================================================
// Wired to apiServices for real project data, WebSocket status updates.

import { useState, useEffect, useCallback, useRef } from 'react';
import * as api from '../../services/apiServices';
import { Client as StompClient, StompSubscription } from '@stomp/stompjs';
import { ProjectCard } from './ProjectCard';
import { LogViewerV2 } from './LogViewer';
import { ResultsModalV2 } from './ResultsModal';
import { CreateProjectModalV2 } from './CreateProjectModal';
import { EditProjectModal } from './EditProjectModal';
import { StartProjectModal } from './StartProjectModal';
import { Plus, Search, AlertCircle } from 'lucide-react';
import { Button, Card, Skeleton } from '../ui';
import { BrandMark } from '../brand';
import type { Project, ProjectResult } from '../../services/apiServices';
import { createLogger } from '../../lib/logger';

const log = createLogger('DashboardV2');

const SERVER_ROOT_URL = import.meta.env.VITE_SERVER_ROOT_URL || `http://${window.location.hostname}:8081`;
const WEBSOCKET_URL_BASE = SERVER_ROOT_URL.replace(/^http/, 'ws');

interface StatusUpdate {
  projectId: string;
  newStatus: string;
  serverPort?: number;
}

export function DashboardV2() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [logViewProjectId, setLogViewProjectId] = useState<string | null>(null);
  const [resultsProject, setResultsProject] = useState<{ id: string; name: string } | null>(null);
  const [results, setResults] = useState<ProjectResult[]>([]);
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [isCreating, setIsCreating] = useState(false);

  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [editProject, setEditProject] = useState<Project | null>(null);
  const [isUpdating, setIsUpdating] = useState(false);

  const [isStartModalOpen, setIsStartModalOpen] = useState(false);
  const [startProject, setStartProject] = useState<Project | null>(null);

  // Store results for all projects to power the sparkline charts
  const [resultsMap, setResultsMap] = useState<Record<string, ProjectResult[]>>({});

  const stompClientRef = useRef<StompClient | null>(null);
  const subscriptionStatusRef = useRef<StompSubscription | null>(null);
  const subscriptionResultsRef = useRef<StompSubscription | null>(null);

  const loadProjects = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await api.fetchProjects();
      const loadedProjects = Array.isArray(response.data) ? response.data : [];
      setProjects(loadedProjects);
      setError('');

      // Load historical results for all projects to populate sparklines
      if (loadedProjects.length > 0) {
        Promise.allSettled(
          loadedProjects.map(p => api.fetchProjectResults(p.id).then(res => ({ id: p.id, results: res.data })))
        ).then((resultsData) => {
          const newMap: Record<string, ProjectResult[]> = {};
          resultsData.forEach(r => {
            if (r.status === 'fulfilled') {
              newMap[r.value.id] = r.value.results;
            }
          });
          setResultsMap(prev => ({ ...prev, ...newMap }));
        });
      }
    } catch {
      setError('Failed to fetch projects.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => { loadProjects(); }, [loadProjects]);

  // WebSocket status updates
  useEffect(() => {
    const client = new StompClient({
      brokerURL: `${WEBSOCKET_URL_BASE}/ws-logs`,
      reconnectDelay: 5000,
    });

    client.onConnect = () => {
      // 1. Subscribe to Status updates
      const subStatus = client.subscribe('/topic/status/*', (message) => {
        try {
          const update: StatusUpdate = JSON.parse(message.body);
          setProjects((prev) =>
            prev.map((p) =>
              p.id === update.projectId
                ? { ...p, status: update.newStatus as Project['status'], serverPort: update.serverPort }
                : p
            )
          );
        } catch { /* ignore parse errors */ }
      });
      subscriptionStatusRef.current = subStatus;

      // 2. Subscribe to Telemetry updates
      const subResults = client.subscribe('/topic/results/*', (message) => {
        try {
          const result: ProjectResult = JSON.parse(message.body);
          const destParts = message.headers.destination.split('/');
          const projectId = destParts[destParts.length - 1];
          if (projectId && result) {
            setResultsMap(prev => ({
              ...prev,
              [projectId]: [...(prev[projectId] || []), result]
            }));
          }
        } catch { /* ignore parse errors */ }
      });
      subscriptionResultsRef.current = subResults;
    };

    client.activate();
    stompClientRef.current = client;

    return () => {
      subscriptionStatusRef.current?.unsubscribe();
      subscriptionResultsRef.current?.unsubscribe();
      if (stompClientRef.current?.active) stompClientRef.current.deactivate();
    };
  }, []);

  const handleCreateProject = async (projectData: any) => {
    try {
      setIsCreating(true);
      await api.createProject(projectData);
      setIsCreateModalOpen(false);
      loadProjects();
    } catch (err) {
      setError('Failed to create project.');
      log.error('createProject failed', err);
    } finally {
      setIsCreating(false);
    }
  };

  const handleToggleServer = async (project: Project) => {
    try {
      if (project.status === 'RUNNING') {
        const res = await api.stopProjectServer(project.id);
        setProjects((prev) => prev.map((p) => (p.id === res.data.id ? res.data : p)));
      } else {
        setStartProject(project);
        setIsStartModalOpen(true);
      }
    } catch {
      setError(`Failed to stop server.`);
    }
  };

  const handleStartSubmit = async (projectId: string, config: any) => {
    try {
      const res = await api.startProjectServer(projectId, config);
      setProjects((prev) => prev.map((p) => (p.id === res.data.id ? res.data : p)));
      setIsStartModalOpen(false);
      setStartProject(null);
    } catch {
      setError('Failed to start server.');
    }
  };

  const handleUpdateProject = async (id: string, projectData: Partial<Project>) => {
    try {
      setIsUpdating(true);
      const res = await api.updateProject(id, projectData);
      setProjects((prev) => prev.map((p) => (p.id === id ? res.data : p)));
      setIsEditModalOpen(false);
      setEditProject(null);
    } catch (err) {
      setError('Failed to update project.');
      log.error('updateProject failed', err);
    } finally {
      setIsUpdating(false);
    }
  };

  const handleDeleteProject = async (projectId: string) => {
    try {
      await api.deleteProject(projectId);
      setProjects((prev) => prev.filter((p) => p.id !== projectId));
    } catch (err) {
      setError('Failed to delete project.');
    }
  };

  const handleOpenResults = async (project: Project) => {
    try {
      const res = await api.fetchProjectResults(project.id);
      setResults(res.data);
      setResultsProject({ id: project.id, name: project.name });
    } catch {
      setError('Could not fetch results.');
    }
  };

  const filteredProjects = projects.filter((p) =>
    p.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
      {/* Header */}
      <header className="flex items-center justify-between gap-4 px-6 md:px-10 h-20 border-b border-hairline bg-canvas/80 backdrop-blur-xl sticky top-0 z-20">
        <div>
          <h1 className="text-h3 font-display font-semibold tracking-tight text-fg">Projects</h1>
          <p className="text-label text-fg-muted mt-0.5">Create, run, and watch your training.</p>
        </div>
        <div className="flex items-center gap-3">
          <div className="relative hidden sm:block">
            <Search className="w-[18px] h-[18px] absolute left-3.5 top-1/2 -translate-y-1/2 text-fg-subtle" strokeWidth={1.5} />
            <input
              type="text"
              placeholder="Search projects"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-surface-2 border border-hairline pl-10 pr-4 h-9 rounded-md text-body text-fg placeholder:text-fg-subtle transition-[border-color,box-shadow] duration-[140ms] hover:border-line focus:outline-none focus:border-accent focus:ring-2 focus:ring-accent/20 w-64"
            />
          </div>
          <Button onClick={() => setIsCreateModalOpen(true)}>
            <Plus className="w-[18px] h-[18px]" strokeWidth={2} />
            New project
          </Button>
        </div>
      </header>

      {/* Main Content Grid */}
      <div className="flex-1 overflow-y-auto px-6 md:px-10 py-8 relative z-10 bg-canvas">
        {error && (
          <div className="mb-6 flex items-center gap-2 px-4 py-3 rounded-md border border-danger/30 bg-danger/10 text-danger text-body font-medium max-w-[1600px] mx-auto">
            <AlertCircle className="w-4 h-4 flex-shrink-0" strokeWidth={1.5} />
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 max-w-[1600px] mx-auto">
            {[0, 1, 2].map((i) => (
              <Card key={i} padding="lg" className="flex flex-col gap-5">
                <div className="flex justify-between">
                  <div className="flex flex-col gap-2">
                    <Skeleton className="h-5 w-40" />
                    <Skeleton className="h-4 w-24" />
                  </div>
                  <Skeleton className="h-14 w-14 rounded-full" />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <Skeleton className="h-24" />
                  <Skeleton className="h-24" />
                </div>
                <Skeleton className="h-9 w-full" />
              </Card>
            ))}
          </div>
        ) : filteredProjects.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 max-w-[1600px] mx-auto">
            {filteredProjects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                results={resultsMap[project.id] || []}
                onOpenLogs={() => setLogViewProjectId(project.id)}
                onOpenResults={() => handleOpenResults(project)}
                onToggleServer={() => handleToggleServer(project)}
                onEditProject={() => { setEditProject(project); setIsEditModalOpen(true); }}
                onDeleteProject={() => handleDeleteProject(project.id)}
              />
            ))}
          </div>
        ) : (
          /* Empty state */
          <div className="flex flex-col items-center justify-center text-center gap-5 mt-16 md:mt-24">
            <div className="grid h-20 w-20 place-items-center rounded-card border border-hairline bg-surface-1">
              <BrandMark size={48} />
            </div>
            <div className="max-w-sm">
              <p className="text-h4 font-display text-fg">
                {searchQuery ? 'No projects match your search' : 'No projects yet'}
              </p>
              <p className="text-body text-fg-muted mt-1.5">
                {searchQuery
                  ? 'Try a different name, or create a new project.'
                  : 'Create your first project and start training a model together with your devices.'}
              </p>
            </div>
            <Button size="lg" onClick={() => setIsCreateModalOpen(true)}>
              <Plus className="w-[18px] h-[18px]" strokeWidth={2} />
              Create your first project
            </Button>
          </div>
        )}
      </div>

      {/* Modals */}
      <CreateProjectModalV2
        isOpen={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
        onSubmit={handleCreateProject}
        isLoading={isCreating}
      />

      <EditProjectModal
        isOpen={isEditModalOpen}
        project={editProject}
        onClose={() => { setIsEditModalOpen(false); setEditProject(null); }}
        onSubmit={handleUpdateProject}
        isLoading={isUpdating}
      />

      <StartProjectModal
        isOpen={isStartModalOpen}
        project={startProject}
        onClose={() => { setIsStartModalOpen(false); setStartProject(null); }}
        onSubmit={handleStartSubmit}
      />

      {logViewProjectId && (
        <LogViewerV2
          projectId={logViewProjectId}
          serverUrl={SERVER_ROOT_URL}
          onClose={() => setLogViewProjectId(null)}
        />
      )}

      <ResultsModalV2
        isOpen={!!resultsProject}
        onClose={() => setResultsProject(null)}
        projectName={resultsProject?.name || ''}
        results={results}
      />
    </div>
  );
}
