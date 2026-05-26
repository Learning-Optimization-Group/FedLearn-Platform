import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import * as api from '../../services/apiServices';
import { Client as StompClient, StompSubscription } from '@stomp/stompjs';
import { motion } from 'framer-motion';
import { Plus, Search, Activity, ChartLine, Server, AlertTriangle, Layers } from 'lucide-react';
import { FederationOrrery } from './FederationOrrery';
import { ProjectCard } from './ProjectCard';
import { LogViewerV2 } from './LogViewer';
import { ResultsModalV2 } from './ResultsModal';
import { CreateProjectModalV2 } from './CreateProjectModal';
import { EditProjectModal } from './EditProjectModal';
import { StartProjectModal } from './StartProjectModal';
import type { Project, ProjectResult } from '../../services/apiServices';
import { createLogger } from '../../lib/logger';
import { cn } from '../../lib/utils';
import { useToast } from '../../context/ToastContext';

const log = createLogger('DashboardV2');

const SERVER_ROOT_URL =
  import.meta.env.VITE_SERVER_ROOT_URL || `http://${window.location.hostname}:8081`;
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
      className="flex flex-col gap-1 min-w-[110px]"
    >
      <div className="font-mono text-[10.5px] font-medium tracking-[0.12em] uppercase text-(--text-secondary)">
        {label}
      </div>
      <div className="flex items-center gap-2">
        <span className="font-mono text-[18px] font-medium text-(--text-primary)">{value}</span>
        <div
          className="h-6 w-6 rounded-md flex items-center justify-center ml-auto"
          style={{
            backgroundColor: `color-mix(in srgb, ${accent} 15%, transparent)`,
            color: accent,
          }}
        >
          {icon}
        </div>
      </div>
    </motion.div>
  );
}

export function DashboardV2() {
  const { show: toast } = useToast();
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
          loadedProjects.map((p) =>
            api.fetchProjectResults(p.id).then((res) => ({ id: p.id, results: res.data }))
          )
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
                ? {
                    ...p,
                    status: update.newStatus as Project['status'],
                    serverPort: update.serverPort,
                  }
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
    setIsCreating(true);
    try {
      const res = await api.createProject(projectData);
      toast('success', `Federation "${res.data.name}" created.`);
      loadProjects();
    } catch (err: any) {
      const detail =
        err?.response?.data?.message ||
        err?.response?.data?.error ||
        'Failed to create federation.';
      toast('error', detail);
      log.error('createProject failed', err);
    } finally {
      setIsCreating(false);
      setIsCreateModalOpen(false);
    }
  };

  const handleToggleServer = async (project: Project) => {
    try {
      if (project.status === 'RUNNING') {
        const res = await api.stopProjectServer(project.id);
        setProjects((prev) => prev.map((p) => (p.id === res.data.id ? res.data : p)));
        toast('success', `Server stopped for "${res.data.name}".`);
      } else {
        setStartProject(project);
        setIsStartModalOpen(true);
      }
    } catch (err: any) {
      const detail =
        err?.response?.data?.message || err?.response?.data?.error || 'Failed to stop server.';
      toast('error', detail);
      log.error('stopProjectServer failed', err);
    }
  };

  const handleStartSubmit = async (projectId: string, config: any) => {
    try {
      const res = await api.startProjectServer(projectId, config);
      setProjects((prev) => prev.map((p) => (p.id === res.data.id ? res.data : p)));
      toast('success', `Server started for "${res.data.name}".`);
    } catch (err: any) {
      const detail =
        err?.response?.data?.message || err?.response?.data?.error || 'Failed to start server.';
      toast('error', detail);
      log.error('startProjectServer failed', err);
    } finally {
      setIsStartModalOpen(false);
      setStartProject(null);
    }
  };

  const handleUpdateProject = async (id: string, projectData: Partial<Project>) => {
    setIsUpdating(true);
    try {
      const res = await api.updateProject(id, projectData);
      setProjects((prev) => prev.map((p) => (p.id === id ? res.data : p)));
      toast('success', `Federation "${res.data.name}" updated.`);
    } catch (err: any) {
      const detail =
        err?.response?.data?.message ||
        err?.response?.data?.error ||
        'Failed to update federation.';
      toast('error', detail);
      log.error('updateProject failed', err);
    } finally {
      setIsUpdating(false);
      setIsEditModalOpen(false);
      setEditProject(null);
    }
  };

  const handleDeleteProject = async (projectId: string) => {
    try {
      await api.deleteProject(projectId);
      setProjects((prev) => prev.filter((p) => p.id !== projectId));
      toast('success', 'Federation deleted.');
    } catch (err: any) {
      const detail =
        err?.response?.data?.message ||
        err?.response?.data?.error ||
        'Failed to delete federation.';
      toast('error', detail);
      log.error('deleteProject failed', err);
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
      <div
        className="flex-1 overflow-y-auto px-8 py-8"
        style={{ backgroundColor: 'var(--background-primary)' }}
      >
        <div className="max-w-[1600px] mx-auto">
          {/* Hero band — federation health + stats */}
          <div
            className="rounded-[12px] p-6 mb-6 relative overflow-hidden grid grid-cols-1 md:grid-cols-[1fr_auto] gap-6 items-center border border-(--border-color)"
            style={{
              background:
                'radial-gradient(circle at 95% -20%, var(--glow-accent), transparent 55%), var(--background-card)',
              boxShadow: 'var(--shadow-soft)',
            }}
          >
            <div className="min-w-0">
              <div className="font-mono text-[10.5px] font-medium tracking-[0.12em] uppercase text-(--text-secondary)">
                FEDERATION HEALTH
              </div>
              <div className="flex items-baseline gap-2.5 mt-1.5">
                <span className="text-[38px] font-display italic tracking-[-0.01em] leading-none text-(--text-primary)">
                  Nominal
                </span>
                <span className="font-mono text-[13px] text-(--accent-primary)">● live</span>
              </div>
              <p className="my-2.5 text-[13.5px] text-(--text-secondary) leading-relaxed">
                <span className="font-mono">{portfolio.running}</span> active projects ·{' '}
                <span className="font-mono">Healthy</span> components · last aggregation{' '}
                <span className="font-mono">just now</span>
              </p>
              <div className="flex gap-6 flex-wrap mt-4">
                <KpiCard
                  icon={<Server className="w-3.5 h-3.5" />}
                  label="ACTIVE"
                  value={portfolio.running}
                  accent="var(--accent-primary)"
                />
                <KpiCard
                  icon={<ChartLine className="w-3.5 h-3.5" />}
                  label="COMPLETED"
                  value={portfolio.completed}
                  accent="oklch(0.52 0.16 220)"
                />
                <KpiCard
                  icon={<AlertTriangle className="w-3.5 h-3.5" />}
                  label="FAILURES"
                  value={portfolio.failed}
                  accent="var(--destructive)"
                />
                <KpiCard
                  icon={<Layers className="w-3.5 h-3.5" />}
                  label="MODELS"
                  value={portfolio.uniqueModels}
                  accent="oklch(0.55 0.22 340)"
                />
                <KpiCard
                  icon={<Activity className="w-3.5 h-3.5" />}
                  label="LATEST ACC"
                  value={
                    portfolio.latestAccuracy
                      ? `${(portfolio.latestAccuracy * 100).toFixed(1)}%`
                      : '—'
                  }
                  accent="oklch(0.70 0.18 85)"
                />
              </div>
            </div>

            <div className="flex justify-center items-center">
              <FederationOrrery
                clients={[
                  { name: 'jetson-orin-1', status: 'uploading', contribution: 0.8 },
                  { name: 'lab-mac-m2', status: 'training', contribution: 0.5 },
                  { name: 'gpu-server-x', status: 'training', contribution: 0.9 },
                  { name: 'clinic-node', status: 'offline', contribution: 0.2 },
                  { name: 'research-pc', status: 'idle', contribution: 0.4 },
                ]}
                round={28}
                totalRounds={50}
                size={180}
              />
            </div>
          </div>

          <div className="flex items-center justify-between mb-4 mt-8">
            <div>
              <div className="font-mono text-[10.5px] font-medium tracking-[0.12em] uppercase text-(--text-secondary)">
                PROJECTS
              </div>
              <h2 className="m-0 mt-1 text-[18px] font-medium tracking-tight text-(--text-primary)">
                All federations
              </h2>
            </div>

            <div className="flex items-center gap-4">
              <div className="relative">
                <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-(--text-secondary)" />
                <input
                  type="text"
                  placeholder="Search projects"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="rounded-md pl-9 pr-3 py-2 text-[13px] w-56 transition-colors"
                  style={{
                    backgroundColor: 'var(--input-background)',
                    color: 'var(--text-primary)',
                    border: '1px solid var(--border-color)',
                  }}
                />
              </div>
              <div className="flex border border-(--border-color) rounded-md overflow-hidden bg-(--background-secondary)">
                {(['all', 'owner', 'member', 'client'] as RelationFilter[]).map((f) => (
                  <button
                    key={f}
                    onClick={() => setFilter(f)}
                    className={cn(
                      'px-3 py-1.5 text-[12px] font-medium transition-colors border-r border-(--border-color) last:border-r-0',
                      filter === f
                        ? 'bg-(--background-card) text-(--text-primary)'
                        : 'text-(--text-secondary) hover:text-(--text-primary) hover:bg-(--background-card)'
                    )}
                  >
                    {f === 'all'
                      ? 'All'
                      : f === 'owner'
                        ? 'Owned'
                        : f === 'member'
                          ? 'Member'
                          : 'Client'}
                  </button>
                ))}
              </div>
              <button
                onClick={() => setIsCreateModalOpen(true)}
                className="inline-flex items-center justify-center gap-2 rounded-md px-4 py-2 text-[13px] font-semibold text-(--primary-foreground) transition-all hover:brightness-110"
                style={{ backgroundColor: 'var(--accent-primary)' }}
              >
                <Plus className="w-4 h-4" />
                New federation
              </button>
            </div>
          </div>
          {error && (
            <div
              className="mb-6 px-5 py-3 rounded-2xl text-sm font-medium"
              style={{
                backgroundColor: 'color-mix(in srgb, #ef4444 12%, transparent)',
                color: '#ef4444',
                border: '1px solid color-mix(in srgb, #ef4444 30%, transparent)',
              }}
            >
              {error}
            </div>
          )}

          {isLoading ? (
            <div className="flex items-center justify-center h-64 text-(--text-secondary)">
              Loading projects...
            </div>
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
                <motion.div
                  key={project.id}
                  variants={{ hidden: { opacity: 0, y: 18 }, visible: { opacity: 1, y: 0 } }}
                >
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
