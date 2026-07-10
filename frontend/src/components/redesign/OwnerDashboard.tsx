// =============================================================================
// FedLearn Frontend — OwnerDashboard (role: PROJECT_OWNER / PLATFORM_ADMIN)
// =============================================================================
// The project owner's home: their projects + create, plus the owner-only
// controls layered on via ProjectOwnerPanel (visibility, join-request
// approvals, membership management, request-deletion). Reuses ProjectCard and
// the shared project lifecycle (WebSocket status, start/stop, edit, results, logs).

import { useState, useEffect, useCallback } from 'react';
import { Plus, Search, AlertCircle } from 'lucide-react';
import * as api from '../../services/apiServices';
import type { OwnedProject, ProjectResult, Project } from '../../services/apiServices';
import { ProjectCard } from './ProjectCard';
import { LogViewerV2 } from './LogViewer';
import { ResultsModalV2 } from './ResultsModal';
import { CreateProjectModalV2 } from './CreateProjectModal';
import { EditProjectModal } from './EditProjectModal';
import { StartProjectModal } from './StartProjectModal';
import { ProjectOwnerPanel } from './ProjectOwnerPanel';
import { Button, Card, Skeleton, StatusPill } from '../ui';
import { BrandMark } from '../brand';
import { createLogger } from '../../lib/logger';
import { useProjectStatus } from '../../hooks/useProjectStatus';
import { describeStompConnection } from '../../lib/connectionStatus';

const log = createLogger('OwnerDashboard');

/** ProjectCard only reads the base Project fields; OwnedProject is a superset. */
const asProject = (p: OwnedProject): Project => p;

export function OwnerDashboard() {
    const [projects, setProjects] = useState<OwnedProject[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [searchQuery, setSearchQuery] = useState('');

    const [logViewProjectId, setLogViewProjectId] = useState<string | null>(null);
    const [resultsProject, setResultsProject] = useState<{ id: string; name: string } | null>(null);
    const [results, setResults] = useState<ProjectResult[]>([]);
    const [resultsMap, setResultsMap] = useState<Record<string, ProjectResult[]>>({});

    const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
    const [isCreating, setIsCreating] = useState(false);

    const [isEditModalOpen, setIsEditModalOpen] = useState(false);
    const [editProject, setEditProject] = useState<OwnedProject | null>(null);
    const [isUpdating, setIsUpdating] = useState(false);

    const [isStartModalOpen, setIsStartModalOpen] = useState(false);
    const [startProject, setStartProject] = useState<OwnedProject | null>(null);

    const [managedProject, setManagedProject] = useState<OwnedProject | null>(null);

    // Project ids with a deletion request awaiting platform-admin approval.
    const [pendingDeletions, setPendingDeletions] = useState<Record<string, boolean>>({});

    const loadProjects = useCallback(async () => {
        try {
            setIsLoading(true);
            const response = await api.fetchOwnedProjects();
            const loaded = Array.isArray(response.data) ? response.data : [];
            setProjects(loaded);
            setError('');
            if (loaded.length > 0) {
                Promise.allSettled(
                    loaded.map((p) =>
                        api.fetchProjectResults(p.id).then((res) => ({ id: p.id, results: res.data })),
                    ),
                ).then((resultsData) => {
                    const newMap: Record<string, ProjectResult[]> = {};
                    resultsData.forEach((r) => {
                        if (r.status === 'fulfilled') newMap[r.value.id] = r.value.results;
                    });
                    setResultsMap((prev) => ({ ...prev, ...newMap }));
                });

                // Which projects already have a pending deletion request, so the
                // card shows the pending state instead of "Request deletion".
                Promise.allSettled(
                    loaded.map((p) =>
                        api
                            .fetchProjectDeletionRequest(p.id)
                            .then((res) => ({ id: p.id, pending: !api.isEmptyBody(res.data) })),
                    ),
                ).then((delData) => {
                    const nextPending: Record<string, boolean> = {};
                    delData.forEach((d) => {
                        if (d.status === 'fulfilled') nextPending[d.value.id] = d.value.pending;
                    });
                    setPendingDeletions(nextPending);
                });
            } else {
                setPendingDeletions({});
            }
        } catch {
            setError('Failed to fetch your projects.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        loadProjects();
    }, [loadProjects]);

    // WebSocket status + telemetry for every owned project at once — the '*'
    // segment maps to the backend's wildcard destinations `/topic/status/*` +
    // `/topic/results/*`; lifecycle owned by useStompClient (via useProjectStatus).
    const connection = useProjectStatus<ProjectResult>({
        projectId: '*',
        onStatusUpdate: (update) => {
            setProjects((prev) =>
                prev.map((p) =>
                    p.id === update.projectId
                        ? { ...p, status: update.newStatus as OwnedProject['status'], serverPort: update.serverPort }
                        : p,
                ),
            );
        },
        onResult: (projectId, result) => {
            setResultsMap((prev) => ({
                ...prev,
                [projectId]: [...(prev[projectId] || []), result],
            }));
        },
    });

    const handleCreateProject = async (projectData: api.ProjectData): Promise<api.Project> => {
        try {
            setIsCreating(true);
            const res = await api.createProject(projectData);
            // BA-1: the project comes back INITIALIZING; the modal polls it and then closes + refreshes
            // via onClose/onCreated once it's ready or failed.
            return res.data;
        } catch (err) {
            // Let the modal keep itself open and render the detail inline,
            // instead of flashing a banner on the route behind the modal.
            log.error('createProject failed', err);
            throw err;
        } finally {
            setIsCreating(false);
        }
    };

    const handleToggleServer = async (project: OwnedProject) => {
        try {
            if (project.status === 'RUNNING') {
                const res = await api.stopProjectServer(project.id);
                setProjects((prev) => prev.map((p) => (p.id === res.data.id ? { ...p, ...res.data } : p)));
            } else {
                setStartProject(project);
                setIsStartModalOpen(true);
            }
        } catch {
            setError('Failed to stop server.');
        }
    };

    const handleStartSubmit = async (projectId: string, config: api.StartServerData) => {
        // Errors propagate to StartProjectModal, which stays open and renders
        // the backend detail inline; the modal is only closed here on success.
        const res = await api.startProjectServer(projectId, config);
        setProjects((prev) => prev.map((p) => (p.id === res.data.id ? { ...p, ...res.data } : p)));
        setIsStartModalOpen(false);
        setStartProject(null);
    };

    const handleUpdateProject = async (id: string, projectData: Partial<Project>) => {
        try {
            setIsUpdating(true);
            const res = await api.updateProject(id, projectData);
            setProjects((prev) => prev.map((p) => (p.id === id ? { ...p, ...res.data } : p)));
            setIsEditModalOpen(false);
            setEditProject(null);
        } catch (err) {
            // Let EditProjectModal keep itself open and render the detail inline.
            log.error('updateProject failed', err);
            throw err;
        } finally {
            setIsUpdating(false);
        }
    };

    const handleDeleteProject = async (projectId: string) => {
        try {
            await api.deleteProject(projectId);
            setProjects((prev) => prev.filter((p) => p.id !== projectId));
        } catch (err) {
            setError(api.errorMessage(err, 'Failed to delete project.'));
        }
    };

    const handleRequestDeletion = async (projectId: string, reason: string) => {
        try {
            await api.submitDeletionRequest(projectId, reason || undefined);
            setPendingDeletions((prev) => ({ ...prev, [projectId]: true }));
        } catch (err) {
            setError(api.errorMessage(err, 'Could not submit a deletion request.'));
        }
    };

    const handleOpenResults = async (project: OwnedProject) => {
        try {
            const res = await api.fetchProjectResults(project.id);
            setResults(res.data);
            setResultsProject({ id: project.id, name: project.name });
        } catch {
            setError('Could not fetch results.');
        }
    };

    const filtered = projects.filter((p) => p.name.toLowerCase().includes(searchQuery.toLowerCase()));

    const connectionDisplay = describeStompConnection(connection, {
        live: 'Live',
        connecting: 'Connecting…',
        reconnecting: 'Reconnecting…',
        error: 'Connection error',
    });

    return (
        <div className="flex-1 flex flex-col h-screen overflow-hidden bg-canvas text-fg font-sans">
            <header className="flex items-center justify-between gap-4 px-6 md:px-10 h-20 border-b border-hairline bg-canvas/80 backdrop-blur-xl sticky top-0 z-20">
                <div>
                    <div className="flex items-center gap-2.5">
                        <h1 className="text-h3 font-display font-semibold tracking-tight text-fg">My projects</h1>
                        <StatusPill status={connectionDisplay.kind}>{connectionDisplay.label}</StatusPill>
                    </div>
                    <p className="text-label text-fg-muted mt-0.5">Create, run, and manage who can join.</p>
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
                ) : filtered.length > 0 ? (
                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6 max-w-[1600px] mx-auto">
                        {filtered.map((project) => (
                            <ProjectCard
                                key={project.id}
                                project={asProject(project)}
                                results={resultsMap[project.id] || []}
                                onOpenLogs={() => setLogViewProjectId(project.id)}
                                onOpenResults={() => handleOpenResults(project)}
                                onToggleServer={() => handleToggleServer(project)}
                                onEditProject={() => {
                                    setEditProject(project);
                                    setIsEditModalOpen(true);
                                }}
                                onDeleteProject={() => handleDeleteProject(project.id)}
                                onRequestDeletion={(reason) => handleRequestDeletion(project.id, reason)}
                                deletionPending={!!pendingDeletions[project.id]}
                                onManageProject={() => setManagedProject(project)}
                            />
                        ))}
                    </div>
                ) : (
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
                                    : 'Create your first project and invite devices to train it together.'}
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
                onCreated={loadProjects}
                isLoading={isCreating}
            />

            <EditProjectModal
                isOpen={isEditModalOpen}
                project={editProject ? asProject(editProject) : null}
                onClose={() => {
                    setIsEditModalOpen(false);
                    setEditProject(null);
                }}
                onSubmit={handleUpdateProject}
                isLoading={isUpdating}
            />

            <StartProjectModal
                isOpen={isStartModalOpen}
                project={startProject ? asProject(startProject) : null}
                onClose={() => {
                    setIsStartModalOpen(false);
                    setStartProject(null);
                }}
                onSubmit={handleStartSubmit}
            />

            <ProjectOwnerPanel
                open={!!managedProject}
                project={managedProject}
                onClose={() => setManagedProject(null)}
                onChanged={loadProjects}
            />

            {logViewProjectId && (
                <LogViewerV2
                    projectId={logViewProjectId}
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
