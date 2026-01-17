// Import your single, configured axios instance instead of the generic one.
import api from '../api/axiosConfig';

// =======================================================
// --- Authentication Endpoints ---
// =======================================================
export const loginUser = (credentials) => {
    // Note: The interceptor will NOT add a token to this call, which is correct.
    return api.post('/auth/login', credentials);
};

export const registerUser = (userData) => {
    return api.post('/auth/register', userData);
};

// =======================================================
// --- Project Management Endpoints ---
// All calls below will automatically have the Authorization header added by the interceptor.
// =======================================================

/**
 * Fetches all projects for the currently authenticated user.
 */


export const fetchProjects = async () => {
    console.log('Calling fetchProjects...');
    try {
        const res = await api.get('/projects');
        console.log('Projects fetched:', res);
        return res;
    } catch (err) {
        console.error('fetchProjects error:', err);
        throw err;
    }
};


/**
 * Creates a new project.
 * @param {object} projectData - The data for the new project.
 */
export const createProject = (projectData) => {
    return api.post('/projects', projectData);
};

/**
 * Starts the Flower server for a specific project.
 * @param {string} projectId - The UUID of the project.
 * @param {object} startData - Configuration for the server (e.g., strategy, numRounds).
 */
export const startProjectServer = (projectId, startData) => {
    console.log('startData - ', startData)
    const params = {};
    if (startData && startData.strategy) {
        params.strategy = startData.strategy;
    }
    if (startData && startData.numRounds) {
        params.numRounds = startData.numRounds;
    }
    // Send params as query parameters
    return api.post(`/projects/${projectId}/start`, null, { params });
};


/**
 * Stops the Flower server for a specific project.
 * @param {string} projectId - The UUID of the project.
 */
export const stopProjectServer = (projectId) => {
    return api.post(`/projects/${projectId}/stop`, {});
};

/**
 * Updates a project's details (e.g., the optimizer).
 * @param {string} projectId - The UUID of the project.
 * @param {object} updateData - The fields to update.
 */
export const updateProject = (projectId, updateData) => {
    return api.put(`/projects/${projectId}`, updateData);
};

/**
 * Fetches the round-by-round results for a completed project.
 * @param {string} projectId - The UUID of the project.
 */
export const fetchProjectResults = (projectId) => {
    return api.get(`/projects/${projectId}/results`);
};