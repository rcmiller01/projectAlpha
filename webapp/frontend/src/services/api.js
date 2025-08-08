import axios from 'axios';

// Create axios instance with base configuration
const apiClient = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:3001/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
apiClient.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('authToken');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle authentication errors
      localStorage.removeItem('authToken');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Agent API functions
export const agentApi = {
  getAgents: () => apiClient.get('/agents'),

  invokeAgent: (agentType, data) =>
    apiClient.post(`/agents/${agentType}/invoke`, data),

  streamAgent: (agentType, data) =>
    apiClient.post(`/agents/${agentType}/stream`, data),

  batchInvoke: (requests) =>
    apiClient.post('/agents/batch', { requests }),
};

// Thread API functions
export const threadApi = {
  getThreads: (params) =>
    apiClient.get('/threads', { params }),

  getThread: (threadId) =>
    apiClient.get(`/threads/${threadId}`),

  createThread: (data) =>
    apiClient.post('/threads', data),

  updateThread: (threadId, data) =>
    apiClient.put(`/threads/${threadId}`, data),

  deleteThread: (threadId) =>
    apiClient.delete(`/threads/${threadId}`),

  addMessage: (threadId, message) =>
    apiClient.post(`/threads/${threadId}/messages`, message),

  archiveThread: (threadId) =>
    apiClient.post(`/threads/${threadId}/archive`),

  searchThreads: (query) =>
    apiClient.get('/threads/search', { params: { q: query } }),
};

// Project API functions
export const projectApi = {
  getProjects: (params) =>
    apiClient.get('/projects', { params }),

  getProject: (projectId) =>
    apiClient.get(`/projects/${projectId}`),

  createProject: (data) =>
    apiClient.post('/projects', data),

  updateProject: (projectId, data) =>
    apiClient.put(`/projects/${projectId}`, data),

  deleteProject: (projectId) =>
    apiClient.delete(`/projects/${projectId}`),

  updateProgress: (projectId, progress) =>
    apiClient.put(`/projects/${projectId}/progress`, { progress }),

  addThread: (projectId, threadId) =>
    apiClient.post(`/projects/${projectId}/threads`, { threadId }),

  removeThread: (projectId, threadId) =>
    apiClient.delete(`/projects/${projectId}/threads/${threadId}`),

  archiveProject: (projectId) =>
    apiClient.post(`/projects/${projectId}/archive`),

  getStats: () =>
    apiClient.get('/projects/stats'),
};

// Health check
export const healthApi = {
  check: () => apiClient.get('/health'),
};

export default apiClient;
