import React, { useState, useEffect, useCallback, useRef } from 'react';
import * as api from '../services/apiServices';
import { Client, StompSubscription } from '@stomp/stompjs';
import ProjectCard from '../components/ProjectCard';
import LogViewer from '../components/LogViewer';
import CreateProjectModal from '../components/CreateProjectModal';
import '../styles/Dashboard.css';
import DiskLoader from '../components/DiskLoader';

const SERVER_ROOT_URL = import.meta.env.VITE_SERVER_ROOT_URL || `http://${window.location.hostname}:8081`;
const WEBSOCKET_URL_BASE = SERVER_ROOT_URL.replace(/^http/, 'ws');

interface Project {
    id: string;
    name: string;
    modelType: string;
    modelName: string;
    optimizer: string;
    status: 'RUNNING' | 'STOPPED' | 'COMPLETED' | 'FAILED';
    serverPort?: number;
}

interface StatusUpdate {
    projectId: string;
    newStatus: string;
    serverPort?: number;
}

const DashboardPage: React.FC = () => {
    const [projects, setProjects] = useState<Project[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isLoadingProjectCard, setIsLoadingProjectCard] = useState(false);
    const [error, setError] = useState('');
    const [isModalOpen, setIsModalOpen] = useState(false);
    const [logViewProjectId, setLogViewProjectId] = useState<string | null>(null);

    const stompClientRef = useRef<Client | null>(null);
    const subscriptionRef = useRef<StompSubscription | null>(null);

    const handleShowLogs = (projectId: string) => {
        setLogViewProjectId(projectId);
    };

    const loadProjects = useCallback(async () => {
        try {
            setIsLoading(true);
            const response = await api.fetchProjects();
            const projectsData = Array.isArray(response.data) ? response.data : [];
            setProjects(projectsData);
            setError('');
        } catch (err) {
            setError('Failed to fetch projects. Please try again.');
            console.error(err);
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        loadProjects();
    }, [loadProjects]);

    useEffect(() => {
        const client = new Client({
            brokerURL: `${WEBSOCKET_URL_BASE}/ws-logs`,
            reconnectDelay: 5000,
        });

        client.onConnect = () => {
            const subscription = client.subscribe('/topic/status/*', (message) => {
                try {
                    const statusUpdate: StatusUpdate = JSON.parse(message.body);
                    setProjects(currentProjects =>
                        currentProjects.map(p =>
                            p.id === statusUpdate.projectId
                                ? {
                                    ...p,
                                    status: statusUpdate.newStatus as Project['status'],
                                    serverPort: statusUpdate.serverPort
                                }
                                : p
                        )
                    );
                } catch (err) {
                    console.error('Error parsing status update:', err);
                }
            });

            subscriptionRef.current = subscription;
        };

        client.activate();
        stompClientRef.current = client;

        return () => {
            if (subscriptionRef.current) {
                subscriptionRef.current.unsubscribe();
            }
            if (stompClientRef.current?.active) {
                stompClientRef.current.deactivate();
            }
        };
    }, []);

    const handleCreateProject = async (projectData: any) => {
        try {
            await api.createProject(projectData);
            setIsModalOpen(false);
            loadProjects();
        } catch (err) {
            setError('Failed to create project.');
            console.error(err);
        }
    };

    const handleToggleServer = async (
        project: Project,
        isCurrentlyRunning: boolean,
        startData: any
    ) => {
        try {
            setIsLoadingProjectCard(true);
            let updatedProject: Project;

            if (isCurrentlyRunning) {
                const response = await api.stopProjectServer(project.id);
                updatedProject = response.data;
            } else {
                const response = await api.startProjectServer(project.id, startData);
                updatedProject = response.data;
            }

            setProjects(currentProjects =>
                currentProjects.map(p =>
                    p.id === updatedProject.id ? updatedProject : p
                )
            );
        } catch (err) {
            setError(`Failed to ${isCurrentlyRunning ? 'stop' : 'start'} server.`);
            console.error(err);
        } finally {
            setIsLoadingProjectCard(false);
        }
    };

    const handleUpdateOptimizer = async (projectId: string, newOptimizer: string) => {
        // Implement this when backend endpoint is ready
        console.info(`Updating optimizer for ${projectId} to ${newOptimizer}`);
    };

    const handleDeleteProject = async (projectId: string) => {
        if (!window.confirm("Are you sure you want to delete this project?")) return;
        try {
            await api.deleteProject(projectId);
            setProjects(currentProjects => currentProjects.filter(p => p.id !== projectId));
        } catch (err) {
            setError('Failed to delete project.');
            console.error(err);
        }
    };

    if ((isLoading && projects.length === 0) || isLoadingProjectCard) {
        return <DiskLoader message="Loading Dashboard..." />;
    }

    return (
        <div className="dashboard">
            <header className="dashboard-header">
                <h1>Available Projects</h1>
                <button
                    className="create-project-btn"
                    onClick={() => setIsModalOpen(true)}
                    aria-label="Create new project"
                >
                    + Create New Project
                </button>
            </header>

            {isModalOpen && (
                <CreateProjectModal
                    onSubmit={handleCreateProject}
                    onCancel={() => setIsModalOpen(false)}
                />
            )}

            {error && <div className="error-message" role="alert">{error}</div>}

            <div className="project-grid">
                {projects.length > 0 ? (
                    projects.map(project => (
                        <ProjectCard
                            key={project.id}
                            project={project}
                            onToggleServer={handleToggleServer}
                            onUpdateOptimizer={handleUpdateOptimizer}
                            onShowLogs={handleShowLogs}
                            onDeleteProject={handleDeleteProject}
                        />
                    ))
                ) : (
                    !isLoading && <p>No projects found. Create one to get started!</p>
                )}
            </div>

            {logViewProjectId && (
                <LogViewer
                    projectId={logViewProjectId}
                    serverUrl={SERVER_ROOT_URL}
                    onClose={() => setLogViewProjectId(null)}
                />
            )}
        </div>
    );
};

export default DashboardPage;
