// =============================================================================
// FedLearn Frontend — DashboardV2 (Apple-inspired, real API)
// =============================================================================
// Wired to apiServices for real project data, WebSocket status updates.

import { useState, useEffect, useCallback, useRef } from 'react';
import * as api from '../../services/apiServices';
import { Client as StompClient, StompSubscription } from '@stomp/stompjs';
import { ProjectCard } from './ProjectCard';
import { LogViewerV2 } from './LogViewer';
import { ResultsModalV2 } from './ResultsModal';
import { CreateProjectModalV2 } from './CreateProjectModal';
import { Plus, Search, Filter } from 'lucide-react';
import type { Project, ProjectResult } from '../../services/apiServices';

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

  const stompClientRef = useRef<StompClient | null>(null);
  const subscriptionRef = useRef<StompSubscription | null>(null);

  const loadProjects = useCallback(async () => {
    try {
      setIsLoading(true);
      const response = await api.fetchProjects();
      setProjects(Array.isArray(response.data) ? response.data : []);
      setError('');
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
      const subscription = client.subscribe('/topic/status/*', (message) => {
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
      subscriptionRef.current = subscription;
    };

    client.activate();
    stompClientRef.current = client;

    return () => {
      subscriptionRef.current?.unsubscribe();
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
      console.error(err);
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
        const res = await api.startProjectServer(project.id, { strategy: 'FedAvg', numRounds: 5, minClients: 2 });
        setProjects((prev) => prev.map((p) => (p.id === res.data.id ? res.data : p)));
      }
    } catch {
      setError(`Failed to ${project.status === 'RUNNING' ? 'stop' : 'start'} server.`);
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
    <div className="flex-1 flex flex-col h-screen overflow-hidden bg-black text-[#f5f5f7] font-sans selection:bg-[#0a84ff] selection:text-white">
      {/* Header */}
      <div className="h-24 flex items-center justify-between px-10 border-b border-[#2c2c2e] bg-[rgba(0,0,0,0.65)] backdrop-blur-3xl saturate-[1.8] sticky top-0 z-20">
        <div>
          <h1 className="text-[28px] font-semibold tracking-tight text-[#f5f5f7]">Active Projects</h1>
          <p className="text-[15px] text-[#86868b] mt-0.5 tracking-tight">Manage and monitor federated tasks.</p>
        </div>
        <div className="flex items-center gap-4">
          <div className="relative">
            <Search className="w-[18px] h-[18px] absolute left-4 top-1/2 -translate-y-1/2 text-[#86868b]" />
            <input
              type="text"
              placeholder="Search projects"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-[#1c1c1e] pl-11 pr-4 py-2.5 rounded-full text-[15px] text-[#f5f5f7] placeholder-[#86868b] focus:outline-none focus:ring-[3px] focus:ring-[#0a84ff]/30 transition-all w-72 border border-transparent focus:border-[#0a84ff]/50"
            />
          </div>
          <button className="w-10 h-10 flex items-center justify-center bg-[#1c1c1e] hover:bg-[#2c2c2e] rounded-full text-[#f5f5f7] transition-colors">
            <Filter className="w-4 h-4" />
          </button>
          <div className="w-px h-6 bg-[#2c2c2e] mx-2" />
          <button
            onClick={() => setIsCreateModalOpen(true)}
            className="flex items-center gap-2 bg-[#f5f5f7] text-black hover:bg-white px-5 py-2.5 rounded-full text-[15px] font-medium transition-all duration-200 transform active:scale-95 shadow-[0_2px_10px_rgba(255,255,255,0.1)]"
          >
            <Plus className="w-[18px] h-[18px]" />
            New Project
          </button>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="flex-1 overflow-y-auto px-10 py-10 relative z-10 bg-black">
        {error && (
          <div className="mb-6 px-5 py-3 rounded-2xl bg-[#ff453a]/10 text-[#ff453a] text-[14px] font-medium">
            {error}
          </div>
        )}

        {isLoading ? (
          <div className="flex items-center justify-center h-64 text-[#86868b]">
            Loading projects...
          </div>
        ) : filteredProjects.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 max-w-[1600px] mx-auto">
            {filteredProjects.map((project) => (
              <ProjectCard
                key={project.id}
                project={project}
                onOpenLogs={() => setLogViewProjectId(project.id)}
                onOpenResults={() => handleOpenResults(project)}
                onToggleServer={() => handleToggleServer(project)}
                onDeleteProject={() => handleDeleteProject(project.id)}
              />
            ))}
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center h-64 text-[#86868b] gap-2">
            <p className="text-[17px]">No projects found.</p>
            <p className="text-[14px]">Create one to get started.</p>
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
