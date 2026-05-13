import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import * as api from '../../services/apiServices';
import { Client as StompClient, StompSubscription } from '@stomp/stompjs';
import { motion } from 'framer-motion';
import {
  Plus,
  Search,
  Activity,
  ChartLine,
  Server,
  AlertTriangle,
  Layers,
} from 'lucide-react';
import { ProjectCard } from './ProjectCard';
import { LogViewerV2 } from './LogViewer';
import { ResultsModalV2 } from './ResultsModal';
import { CreateProjectModalV2 } from './CreateProjectModal';
import { EditProjectModal } from './EditProjectModal';
import { StartProjectModal } from './StartProjectModal';
import type { Project, ProjectResult } from '../../services/apiServices';
import { createLogger } from '../../lib/logger';
import { cn } from '../../lib/utils';

const log = createLogger('DashboardV2');

const SERVER_ROOT_URL = import.meta.env.VITE_SERVER_ROOT_URL || `http://${window.location.hostname}:8081`;
const WEBSOCKET_URL_BASE = SERVER_ROOT_URL.replace(/^http/, 'ws');

interface StatusUpdate {
  projectId: string;
  newStatus: string;
  serverPort?: number;
}

interface KpiCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  accent?: string;
}

function KpiCard({ icon, label, value, accent = 'var(--accent-primary)' }: KpiCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 14 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.35 }}
      className="rounded-3xl p-5"
      style={{ background: 'var(--background-card)', border: '1px solid var(--border-color)', boxShadow: 'var(--shadow-soft)' }}
    >
      <div className="flex items-center justify-between">
        <div className="text-sm text-(--text-secondary)">{label}</div>
        <div className="h-9 w-9 rounded-xl flex items-center justify-center" style={{ backgroundColor: 'color-mix(in srgb, var(--background-secondary) 88%, transparent)', color: accent }}>
          {icon}
        </div>
      </div>
      <div className="mt-3 text-3xl font-semibold tracking-tight text-(--text-primary)">{value}</div>
    </motion.div>
  );
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

  const [resultsMap, setResultsMap] = useState<Record<string, ProjectResult[]>>({});

  type RelationFilter = 'all' | 'owner' | 'member' | 'client';
  const [filter, setFilter] = useState<RelationFilter>('all');

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

      if (loadedProjects.length > 0) {
        Promise.allSettled(
          loadedProjects.map((p) => api.fetchProjectResults(p.id).then((res) => ({ id: p.id, results: res.data })))
        ).then((resultsData) => {
          const newMap: Record<string, ProjectResult[]> = {};
          resultsData.forEach((r) => {
            if (r.status === 'fulfilled') {
              newMap[r.value.id] = r.value.results;
            }
          });
          setResultsMap((prev) => ({ ...prev, ...newMap }));
        });
      }
    } catch {
      setError('Failed to fetch projects.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    loadProjects();
  }, [loadProjects]);

  useEffect(() => {
    const client = new StompClient({
      brokerURL: `${WEBSOCKET_URL_BASE}/ws-logs`,
      reconnectDelay: 5000,
    });

    client.onConnect = () => {
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
        } catch {
          // Ignore malformed status message.
        }
      });
      subscriptionStatusRef.current = subStatus;

      const subResults = client.subscribe('/topic/results/*', (message) => {
        try {
          const result: ProjectResult = JSON.parse(message.body);
          const destParts = message.headers.destination.split('/');
          const projectId = destParts[destParts.length - 1];
          if (projectId && result) {
            setResultsMap((prev) => ({
              ...prev,
              [projectId]: [...(prev[projectId] || []), result],
            }));
          }
        } catch {
          // Ignore malformed results payload.
        }
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
      setError('Failed to stop server.');
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
    } catch {
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

  const filteredProjects = useMemo(() => {
    let list = projects;
    if (filter === 'owner') list = projects.filter((p) => p.myRelationship === 'OWNER');
    else if (filter === 'member') list = projects.filter((p) => p.myRelationship === 'MEMBER');
    else if (filter === 'client') list = projects.filter((p) => p.myRelationship === 'CLIENT');
    return list.filter((p) => p.name.toLowerCase().includes(searchQuery.toLowerCase()));
  }, [projects, filter, searchQuery]);

  const portfolio = useMemo(() => {
    const running = projects.filter((p) => p.status === 'RUNNING').length;
    const failed = projects.filter((p) => p.status === 'FAILED').length;
    const completed = projects.filter((p) => p.status === 'COMPLETED').length;
    const uniqueModels = new Set(projects.map((p) => `${p.modelType}:${p.modelName}`)).size;

    const allResults = Object.values(resultsMap).flat();
    const latestAccuracy = allResults.length > 0 ? allResults[allResults.length - 1].accuracy : 0;

    return {
      running,
      failed,
      completed,
      uniqueModels,
      latestAccuracy,
    };
  }, [projects, resultsMap]);

  return (
    <div className="flex-1 flex flex-col h-screen overflow-hidden font-sans">
      <div className="border-b px-8 py-6" style={{ borderColor: 'var(--border-color)', backgroundColor: 'var(--surface-glass)', backdropFilter: 'blur(18px) saturate(160%)' }}>
        <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.35 }} className="flex flex-col gap-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <h1 className="font-display text-4xl md:text-5xl font-semibold tracking-tight text-(--text-primary)">Federated Operations</h1>
              <p className="text-sm md:text-base text-(--text-secondary) mt-2 max-w-3xl">
                A unified view of project health, model coverage, and training quality across your distributed fleet.
              </p>
            </div>
            <div className="flex items-center gap-3">
              <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-(--text-secondary)" />
                <input
                  type="text"
                  placeholder="Search projects"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="rounded-full pl-10 pr-4 py-2.5 text-sm w-64"
                  style={{ backgroundColor: 'var(--background-secondary)', color: 'var(--text-primary)', border: '1px solid var(--border-color)' }}
                />
              </div>
              <button
                onClick={() => setIsCreateModalOpen(true)}
                className="inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-semibold text-white"
                style={{ backgroundColor: 'var(--accent-primary)' }}
              >
                <Plus className="w-4 h-4" />
                New Project
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {(['all', 'owner', 'member', 'client'] as RelationFilter[]).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={cn(
                  'px-4 py-1.5 rounded-full text-[13px] font-medium transition-all border',
                  filter === f
                    ? 'bg-(--accent-primary) text-white border-transparent'
                    : 'text-(--text-secondary) border-(--border-color) hover:text-(--text-primary) hover:bg-(--background-card)'
                )}
                style={filter === f ? {} : { backgroundColor: 'var(--background-secondary)' }}
              >
                {f === 'all' ? 'All' : f === 'owner' ? 'Owned by me' : f === 'member' ? 'Member' : 'Client'}
              </button>
            ))}
          </div>

          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <KpiCard icon={<Server className="w-4 h-4" />} label="Active Projects" value={portfolio.running} />
            <KpiCard icon={<ChartLine className="w-4 h-4" />} label="Completed Runs" value={portfolio.completed} accent="#22c55e" />
            <KpiCard icon={<AlertTriangle className="w-4 h-4" />} label="Failures" value={portfolio.failed} accent="#ef4444" />
            <KpiCard icon={<Layers className="w-4 h-4" />} label="Model Families" value={portfolio.uniqueModels} accent="#8b5cf6" />
            <KpiCard icon={<Activity className="w-4 h-4" />} label="Latest Accuracy" value={portfolio.latestAccuracy ? `${(portfolio.latestAccuracy * 100).toFixed(1)}%` : '—'} accent="#0ea5e9" />
          </div>
        </motion.div>
      </div>

      <div className="flex-1 overflow-y-auto px-8 py-8">
        {error && (
          <div className="mb-6 px-5 py-3 rounded-2xl text-sm font-medium" style={{ backgroundColor: 'color-mix(in srgb, #ef4444 12%, transparent)', color: '#ef4444', border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)' }}>
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center h-64 text-(--text-secondary)">Loading projects...</div>
        ) : filteredProjects.length > 0 ? (
          <motion.div
            initial="hidden"
            animate="visible"
            variants={{
              hidden: { opacity: 0 },
              visible: { opacity: 1, transition: { staggerChildren: 0.06 } },
            }}
            className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 max-w-[1700px] mx-auto"
          >
            {filteredProjects.map((project) => (
              <motion.div key={project.id} variants={{ hidden: { opacity: 0, y: 18 }, visible: { opacity: 1, y: 0 } }}>
                <ProjectCard
                  project={project}
                  results={resultsMap[project.id] || []}
                  onOpenLogs={() => setLogViewProjectId(project.id)}
                  onOpenResults={() => handleOpenResults(project)}
                  onToggleServer={() => handleToggleServer(project)}
                  onEditProject={() => {
                    setEditProject(project);
                    setIsEditModalOpen(true);
                  }}
                  onDeleteProject={() => handleDeleteProject(project.id)}
                />
              </motion.div>
            ))}
          </motion.div>
        ) : (
          <div className="flex flex-col items-center justify-center h-64 text-(--text-secondary) gap-2">
            <p className="text-[16px] font-medium text-(--text-primary)">No projects found.</p>
            <p className="text-[14px]">Create one to start federated training.</p>
          </div>
        )}
      </div>

      <CreateProjectModalV2
        isOpen={isCreateModalOpen}
        onClose={() => setIsCreateModalOpen(false)}
        onSubmit={handleCreateProject}
        isLoading={isCreating}
      />

      <EditProjectModal
        isOpen={isEditModalOpen}
        project={editProject}
        onClose={() => {
          setIsEditModalOpen(false);
          setEditProject(null);
        }}
        onSubmit={handleUpdateProject}
        isLoading={isUpdating}
      />

      <StartProjectModal
        isOpen={isStartModalOpen}
        project={startProject}
        onClose={() => {
          setIsStartModalOpen(false);
          setStartProject(null);
        }}
        onSubmit={handleStartSubmit}
      />

      {logViewProjectId && (
        <LogViewerV2 projectId={logViewProjectId} serverUrl={SERVER_ROOT_URL} onClose={() => setLogViewProjectId(null)} />
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
