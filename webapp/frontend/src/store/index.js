import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';

// Main application store
export const useAppStore = create(
  subscribeWithSelector((set, get) => ({
    // UI State
    sidebarOpen: true,
    currentView: 'chat',
    darkMode: true,

    // Connection status
    connectionStatus: {
      api: 'connected',
      socket: 'connected',
      agents: 'idle'
    },

    // User preferences
    preferences: {
      autoSave: true,
      notifications: true,
      streamingResponses: true,
      defaultAgent: 'general',
      voiceSettings: {
        enabled: true,
        mood: 'balanced',
        energy: 0.6,
        intimacy: 0.6,
        agentSpecific: true
      }
    },

    // Actions
    setSidebarOpen: (open) => set({ sidebarOpen: open }),
    setCurrentView: (view) => set({ currentView: view }),
    setDarkMode: (mode) => set({ darkMode: mode }),
    setConnectionStatus: (status) => set(state => ({
      connectionStatus: { ...state.connectionStatus, ...status }
    })),
    setPreferences: (prefs) => set(state => ({
      preferences: { ...state.preferences, ...prefs }
    }))
  }))
);

// Thread store
export const useThreadStore = create(
  subscribeWithSelector((set, get) => ({
    threads: [],
    currentThread: null,
    loading: false,
    error: null,
    searchQuery: '',
    filter: 'all', // all, active, archived

    // Actions
    setThreads: (threads) => set({ threads }),
    setCurrentThread: (thread) => set({ currentThread: thread }),
    setLoading: (loading) => set({ loading }),
    setError: (error) => set({ error }),
    setSearchQuery: (query) => set({ searchQuery: query }),
    setFilter: (filter) => set({ filter }),

    addThread: (thread) => set(state => ({
      threads: [thread, ...state.threads]
    })),

    updateThread: (threadId, updates) => set(state => ({
      threads: state.threads.map(thread =>
        thread.id === threadId ? { ...thread, ...updates } : thread
      ),
      currentThread: state.currentThread?.id === threadId
        ? { ...state.currentThread, ...updates }
        : state.currentThread
    })),

    removeThread: (threadId) => set(state => ({
      threads: state.threads.filter(thread => thread.id !== threadId),
      currentThread: state.currentThread?.id === threadId ? null : state.currentThread
    })),

    addMessage: (threadId, message) => set(state => {
      const updatedThreads = state.threads.map(thread => {
        if (thread.id === threadId) {
          return {
            ...thread,
            messages: [...(thread.messages || []), message],
            messageCount: (thread.messageCount || 0) + 1,
            lastMessage: message.content.substring(0, 100),
            updatedAt: new Date().toISOString()
          };
        }
        return thread;
      });

      const updatedCurrentThread = state.currentThread?.id === threadId
        ? {
            ...state.currentThread,
            messages: [...(state.currentThread.messages || []), message],
            messageCount: (state.currentThread.messageCount || 0) + 1,
            lastMessage: message.content.substring(0, 100),
            updatedAt: new Date().toISOString()
          }
        : state.currentThread;

      return {
        threads: updatedThreads,
        currentThread: updatedCurrentThread
      };
    }),

    // Filtered threads getter
    getFilteredThreads: () => {
      const { threads, searchQuery, filter } = get();
      let filtered = threads;

      // Apply status filter
      if (filter === 'archived') {
        filtered = filtered.filter(thread => thread.isArchived);
      } else if (filter === 'active') {
        filtered = filtered.filter(thread => !thread.isArchived);
      }

      // Apply search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase();
        filtered = filtered.filter(thread =>
          thread.title.toLowerCase().includes(query) ||
          thread.lastMessage?.toLowerCase().includes(query) ||
          thread.tags?.some(tag => tag.toLowerCase().includes(query))
        );
      }

      return filtered.sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));
    }
  }))
);

// Project store
export const useProjectStore = create(
  subscribeWithSelector((set, get) => ({
    projects: [],
    currentProject: null,
    loading: false,
    error: null,
    stats: null,
    filter: 'active', // active, completed, all

    // Actions
    setProjects: (projects) => set({ projects }),
    setCurrentProject: (project) => set({ currentProject: project }),
    setLoading: (loading) => set({ loading }),
    setError: (error) => set({ error }),
    setStats: (stats) => set({ stats }),
    setFilter: (filter) => set({ filter }),

    addProject: (project) => set(state => ({
      projects: [project, ...state.projects]
    })),

    updateProject: (projectId, updates) => set(state => ({
      projects: state.projects.map(project =>
        project.id === projectId ? { ...project, ...updates } : project
      ),
      currentProject: state.currentProject?.id === projectId
        ? { ...state.currentProject, ...updates }
        : state.currentProject
    })),

    removeProject: (projectId) => set(state => ({
      projects: state.projects.filter(project => project.id !== projectId),
      currentProject: state.currentProject?.id === projectId ? null : state.currentProject
    })),

    // Filtered projects getter
    getFilteredProjects: () => {
      const { projects, filter } = get();
      let filtered = projects;

      if (filter === 'active') {
        filtered = filtered.filter(project => project.status === 'active');
      } else if (filter === 'completed') {
        filtered = filtered.filter(project => project.status === 'completed');
      }

      return filtered.sort((a, b) => {
        // Sort by priority first, then by update time
        const priorityOrder = { urgent: 3, high: 2, medium: 1, low: 0 };
        const aPriority = priorityOrder[a.priority] || 0;
        const bPriority = priorityOrder[b.priority] || 0;

        if (aPriority !== bPriority) {
          return bPriority - aPriority;
        }

        return new Date(b.updatedAt) - new Date(a.updatedAt);
      });
    }
  }))
);

// Agent store
export const useAgentStore = create(
  subscribeWithSelector((set, get) => ({
    agents: [],
    currentAgent: 'general',
    agentStatuses: {},
    activeRequests: new Map(),
    loading: false,
    error: null,

    // Actions
    setAgents: (agents) => set({ agents }),
    setCurrentAgent: (agent) => set({ currentAgent: agent }),
    setAgentStatuses: (statuses) => set({ agentStatuses: statuses }),
    setLoading: (loading) => set({ loading }),
    setError: (error) => set({ error }),

    updateAgentStatus: (agentType, status) => set(state => ({
      agentStatuses: { ...state.agentStatuses, [agentType]: status }
    })),

    addActiveRequest: (requestId, request) => set(state => {
      const newRequests = new Map(state.activeRequests);
      newRequests.set(requestId, request);
      return { activeRequests: newRequests };
    }),

    updateActiveRequest: (requestId, updates) => set(state => {
      const newRequests = new Map(state.activeRequests);
      const existing = newRequests.get(requestId);
      if (existing) {
        newRequests.set(requestId, { ...existing, ...updates });
      }
      return { activeRequests: newRequests };
    }),

    removeActiveRequest: (requestId) => set(state => {
      const newRequests = new Map(state.activeRequests);
      newRequests.delete(requestId);
      return { activeRequests: newRequests };
    }),

    // Get agent by type
    getAgent: (agentType) => {
      const { agents } = get();
      return agents.find(agent => agent.type === agentType);
    },

    // Get active requests as array
    getActiveRequests: () => {
      const { activeRequests } = get();
      return Array.from(activeRequests.values());
    }
  }))
);

// Subscribe to store changes for persistence
useAppStore.subscribe(
  (state) => state.preferences,
  (preferences) => {
    localStorage.setItem('app_preferences', JSON.stringify(preferences));
  }
);

// Initialize preferences from localStorage
const savedPreferences = localStorage.getItem('app_preferences');
if (savedPreferences) {
  try {
    const preferences = JSON.parse(savedPreferences);
    useAppStore.getState().setPreferences(preferences);
  } catch (error) {
    console.error('Failed to load saved preferences:', error);
  }
}
