import React, { useState, useEffect, useCallback } from 'react';
import * as api from '../services/apiServices';
import { Client } from '@stomp/stompjs';
import ProjectCard from '../components/ProjectCard';
import LogViewer from '../components/LogViewer';
import CreateProjectModal from '../components/CreateProjectModal';
import '../styles/Dashboard.css';
import ResultsModal from '../components/ResultsModal';
import { SERVER_ROOT_URL, WEBSOCKET_URL_BASE } from '../config';
import DiskLoader from '../components/DiskLoader';

const DashboardPage = () => {
    const [projects, setProjects] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [isLoadingProjectCard, setIsLoadingProjectCard] = useState(false);
    const [stompClient, setStompClient] = useState(null);
    const [error, setError] = useState('');
    const [isModalOpen, setIsModalOpen] = useState(false);

    const [isResultsModalOpen, setIsResultsModalOpen] = useState(false);
    const [selectedProjectResults, setSelectedProjectResults] = useState([]);
    const [selectedProjectName, setSelectedProjectName] = useState('');

    const [logViewProjectId, setLogViewProjectId] = useState(null);
    // const API_SERVER_HOST = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8081/';

    const handleShowLogs = (projectId) => {
        setLogViewProjectId(projectId);
    };

    useEffect(() => {
        // Create a single STOMP client for the whole dashboard
        const client = new Client({
            brokerURL: `${WEBSOCKET_URL_BASE}/ws-logs`, // Or your server URL
            reconnectDelay: 5000,
        });

        client.onConnect = () => {
            console.log("Dashboard WebSocket Connected!");
            // Subscribe to a general status update topic (or you can subscribe individually)
            // Here, we just need to know WHEN to refetch, not the specific data.
            // A more advanced version would use the data from the message.
            // Inside client.subscribe in DashboardPage.jsx
            client.subscribe('/topic/status/*', (message) => {
                const statusUpdate = JSON.parse(message.body); // e.g., { projectId, newStatus, serverPort }
                console.log("Received status update:", statusUpdate);

                setProjects(currentProjects =>
                    currentProjects.map(p =>
                        p.id === statusUpdate.projectId
                            ? { ...p, status: statusUpdate.newStatus, serverPort: statusUpdate.serverPort }
                            : p
                    )
                );
            });
        };

        client.activate();
        setStompClient(client);

        // Disconnect on cleanup
        return () => {
            if (client.active) {
                client.deactivate();
            }
        };
    }, []);

    const loadProjects = useCallback(async () => {
        try {
            setIsLoading(true);
            console.log("Calling fetchProjects...");
            const response = await api.fetchProjects();
            console.log("Data received from /api/projects:", response.data);
            const projectsData = Array.isArray(response.data) ? response.data : [];
            //ssds
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
        console.log('load projects')
        loadProjects();
    }, [loadProjects]);


    const handleFetchResults = async (projectId) => {
        try {
            const project = projects.find(p => p.id === projectId);
            if (!project) return;

            console.log(`Fetching results for project ${projectId}`);
            const response = await api.fetchProjectResults(projectId);

            setSelectedProjectResults(response.data);
            setSelectedProjectName(project.name);
            setIsResultsModalOpen(true); // Open the modal with the fetched data
        } catch (err) {
            setError(`Failed to fetch results for project ${projectId}.`);
            console.error(err);
        }
    };

    const handleCreateProject = async (projectData) => {
        try {
            await api.createProject(projectData);
            setIsModalOpen(false);
            loadProjects(); // Refresh the list with the new project
        } catch (err) {
            setError('Failed to create project.');
            console.error(err);
        }
    };

    const handleToggleServer = async (project, isCurrentlyRunning, startData) => {
        try {
            setIsLoadingProjectCard(true);
            console.log('isCurrentlyRunning - ', isCurrentlyRunning)
            let updatedProject;
            if (isCurrentlyRunning) {
                const response = await api.stopProjectServer(project.id);
                updatedProject = response.data; // Use the response directly
            } else {
                const response = await api.startProjectServer(project.id, startData);
                updatedProject = response.data; // Use the response directly
            }

            // Update the state without a full refresh
            setProjects(currentProjects =>
                currentProjects.map(p =>
                    p.id === updatedProject.id ? updatedProject : p
                )
            );
            setIsLoadingProjectCard(false);
        } catch (err) {
            setError(`Failed to ${isCurrentlyRunning ? 'stop' : 'start'} server.`);
            console.error(err);
        }
    };

    // You can implement the optimizer update later
    const handleUpdateOptimizer = async (projectId, newOptimizer) => {
        console.log(`Updating optimizer for ${projectId} to ${newOptimizer}...`);
        // await api.updateProject(projectId, { optimizer: newOptimizer });
        // loadProjects();
    };


    if ((isLoading && projects.length === 0) || isLoadingProjectCard) {
        return <div className="loading-fullscreen">Loading Dashboard...</div>;
    }

    return (
        <div className="dashboard">
            <header className="dashboard-header">
                <h1>Available Projects</h1>
                <button className="create-project-btn" onClick={() => setIsModalOpen(true)}>
                    + Create New Project
                </button>
            </header>

            {isModalOpen &&
                <CreateProjectModal
                    onSubmit={handleCreateProject}
                    onCancel={() => setIsModalOpen(false)}
                />
            }

            {/* {isResultsModalOpen &&
                <ResultsModal
                    results={selectedProjectResults}
                    projectName={selectedProjectName}
                    onCancel={() => setIsResultsModalOpen(false)} // Pass the close function
                />
            } */}

            {isLoading && projects.length === 0 && <div>Loading projects...</div>}
            {error && <div className="error-message">{error}</div>}

            <div className="project-grid">
                {projects?.length > 0 ? (
                    projects.map(project => (
                        <ProjectCard
                            key={project?.id}
                            project={project}
                            onToggleServer={handleToggleServer}
                            onUpdateOptimizer={handleUpdateOptimizer}
                            onShowLogs={handleShowLogs}
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