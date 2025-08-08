import { io } from 'socket.io-client';

class SocketService {
  constructor() {
    this.socket = null;
    this.isConnected = false;
    this.listeners = new Map();
  }

  connect() {
    if (this.socket) {
      return this.socket;
    }

    const serverUrl = process.env.REACT_APP_SOCKET_URL || 'http://localhost:3001';

    this.socket = io(serverUrl, {
      transports: ['websocket'],
      upgrade: true,
      rememberUpgrade: true,
      timeout: 30000,
    });

    this.socket.on('connect', () => {
      console.log('Connected to server');
      this.isConnected = true;
    });

    this.socket.on('disconnect', () => {
      console.log('Disconnected from server');
      this.isConnected = false;
    });

    this.socket.on('connect_error', (error) => {
      console.error('Connection error:', error);
      this.isConnected = false;
    });

    // Handle agent streaming responses
    this.socket.on('agent_response', (data) => {
      this.emit('agent_response', data);
    });

    this.socket.on('agent_error', (data) => {
      this.emit('agent_error', data);
    });

    this.socket.on('agent_complete', (data) => {
      this.emit('agent_complete', data);
    });

    return this.socket;
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
    }
  }

  // Event listener management
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, []);
    }
    this.listeners.get(event).push(callback);

    // Also listen on socket if connected
    if (this.socket) {
      this.socket.on(event, callback);
    }
  }

  off(event, callback) {
    if (this.listeners.has(event)) {
      const callbacks = this.listeners.get(event);
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }

    // Also remove from socket if connected
    if (this.socket) {
      this.socket.off(event, callback);
    }
  }

  emit(event, data) {
    // Emit to local listeners
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error(`Error in ${event} listener:`, error);
        }
      });
    }

    // Emit to server if connected
    if (this.socket && this.isConnected) {
      this.socket.emit(event, data);
    }
  }

  // Agent interaction methods
  invokeAgent(agentType, message, options = {}) {
    return new Promise((resolve, reject) => {
      if (!this.isConnected) {
        reject(new Error('Socket not connected'));
        return;
      }

      const requestId = Date.now().toString();
      const timeout = options.timeout || 30000;

      // Set up response handlers
      const handleResponse = (data) => {
        if (data.requestId === requestId) {
          this.socket.off('agent_complete', handleComplete);
          this.socket.off('agent_error', handleError);
          if (options.onProgress) {
            options.onProgress(data);
          }
        }
      };

      const handleComplete = (data) => {
        if (data.requestId === requestId) {
          this.socket.off('agent_response', handleResponse);
          this.socket.off('agent_error', handleError);
          clearTimeout(timeoutId);
          resolve(data);
        }
      };

      const handleError = (data) => {
        if (data.requestId === requestId) {
          this.socket.off('agent_response', handleResponse);
          this.socket.off('agent_complete', handleComplete);
          clearTimeout(timeoutId);
          reject(new Error(data.error || 'Agent invocation failed'));
        }
      };

      const timeoutId = setTimeout(() => {
        this.socket.off('agent_response', handleResponse);
        this.socket.off('agent_complete', handleComplete);
        this.socket.off('agent_error', handleError);
        reject(new Error('Agent invocation timeout'));
      }, timeout);

      // Set up listeners
      this.socket.on('agent_response', handleResponse);
      this.socket.on('agent_complete', handleComplete);
      this.socket.on('agent_error', handleError);

      // Send request
      this.socket.emit('invoke_agent', {
        requestId,
        agentType,
        message,
        ...options
      });
    });
  }

  // Join/leave thread rooms for real-time updates
  joinThread(threadId) {
    if (this.isConnected) {
      this.socket.emit('join_thread', { threadId });
    }
  }

  leaveThread(threadId) {
    if (this.isConnected) {
      this.socket.emit('leave_thread', { threadId });
    }
  }

  // Join/leave project rooms
  joinProject(projectId) {
    if (this.isConnected) {
      this.socket.emit('join_project', { projectId });
    }
  }

  leaveProject(projectId) {
    if (this.isConnected) {
      this.socket.emit('leave_project', { projectId });
    }
  }

  getConnectionStatus() {
    return {
      connected: this.isConnected,
      socketId: this.socket?.id || null,
    };
  }
}

// Create singleton instance
const socketService = new SocketService();

export default socketService;
