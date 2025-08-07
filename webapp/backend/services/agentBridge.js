const { spawn } = require('child_process');
const path = require('path');
const EventEmitter = require('events');

/**
 * AgentBridge - Bridge between Node.js backend and Python SLiM agents
 * 
 * This service manages communication with the ProjectAlpha Python backend,
 * specifically the HRM router and SLiM agent system.
 */
class AgentBridge extends EventEmitter {
  constructor() {
    super();
    this.pythonProcess = null;
    this.isConnected = false;
    this.pendingRequests = new Map();
    this.requestCounter = 0;
    
    // Path to the ProjectAlpha Python backend
    this.projectRoot = path.resolve(__dirname, '../../../');
    this.pythonScript = path.join(this.projectRoot, 'webapp/backend/services/agent_bridge.py');
    
    this.initializePythonBridge();
  }

  /**
   * Initialize the Python bridge process
   */
  initializePythonBridge() {
    try {
      console.log('🐍 Starting Python Agent Bridge...');
      console.log(`📁 Project root: ${this.projectRoot}`);
      console.log(`🔧 Python script: ${this.pythonScript}`);
      
      this.pythonProcess = spawn('python', [this.pythonScript], {
        cwd: this.projectRoot,
        stdio: ['pipe', 'pipe', 'pipe']
      });

      this.pythonProcess.stdout.on('data', (data) => {
        this.handlePythonOutput(data.toString());
      });

      this.pythonProcess.stderr.on('data', (data) => {
        console.error(`🚨 Python Agent Error: ${data.toString()}`);
      });

      this.pythonProcess.on('close', (code) => {
        console.log(`🐍 Python process exited with code ${code}`);
        this.isConnected = false;
        this.emit('disconnected');
        
        // Auto-restart after 5 seconds
        setTimeout(() => {
          console.log('🔄 Restarting Python Agent Bridge...');
          this.initializePythonBridge();
        }, 5000);
      });

      this.pythonProcess.on('error', (error) => {
        console.error(`🚨 Failed to start Python process: ${error.message}`);
        this.isConnected = false;
      });

      // Send initialization message
      this.sendToPython({
        type: 'init',
        payload: {
          project_root: this.projectRoot
        }
      });

    } catch (error) {
      console.error(`🚨 AgentBridge initialization error: ${error.message}`);
    }
  }

  /**
   * Handle output from Python process
   */
  handlePythonOutput(data) {
    try {
      const lines = data.trim().split('\n');
      
      for (const line of lines) {
        if (line.startsWith('{')) {
          const message = JSON.parse(line);
          this.handlePythonMessage(message);
        } else {
          console.log(`🐍 Python: ${line}`);
        }
      }
    } catch (error) {
      console.error(`🚨 Error parsing Python output: ${error.message}`);
      console.log(`🐍 Raw output: ${data}`);
    }
  }

  /**
   * Handle structured messages from Python
   */
  handlePythonMessage(message) {
    const { type, request_id, payload, error } = message;

    switch (type) {
      case 'init_response':
        this.isConnected = payload.success;
        if (this.isConnected) {
          console.log('✅ Python Agent Bridge connected successfully');
          this.emit('connected', payload);
        } else {
          console.error(`🚨 Python Agent Bridge failed to initialize: ${payload.error}`);
        }
        break;

      case 'agent_response':
        if (this.pendingRequests.has(request_id)) {
          const { resolve } = this.pendingRequests.get(request_id);
          this.pendingRequests.delete(request_id);
          resolve(payload);
        }
        break;

      case 'error':
        if (request_id && this.pendingRequests.has(request_id)) {
          const { reject } = this.pendingRequests.get(request_id);
          this.pendingRequests.delete(request_id);
          reject(new Error(error));
        } else {
          console.error(`🚨 Python Agent Error: ${error}`);
        }
        break;

      case 'agent_list':
        this.emit('agents_updated', payload);
        break;

      default:
        console.log(`🐍 Unknown message type: ${type}`, payload);
    }
  }

  /**
   * Send message to Python process
   */
  sendToPython(message) {
    if (this.pythonProcess && this.pythonProcess.stdin.writable) {
      this.pythonProcess.stdin.write(JSON.stringify(message) + '\n');
    } else {
      throw new Error('Python process not available');
    }
  }

  /**
   * Invoke a SLiM agent with a prompt
   */
  async invokeAgent(agentType, prompt, options = {}) {
    return new Promise((resolve, reject) => {
      const requestId = `req_${++this.requestCounter}_${Date.now()}`;
      
      this.pendingRequests.set(requestId, { resolve, reject });
      
      // Set timeout for request
      setTimeout(() => {
        if (this.pendingRequests.has(requestId)) {
          this.pendingRequests.delete(requestId);
          reject(new Error('Request timeout'));
        }
      }, options.timeout || 30000);

      this.sendToPython({
        type: 'invoke_agent',
        request_id: requestId,
        payload: {
          agent_type: agentType,
          prompt: prompt,
          depth: options.depth || 1,
          context: options.context || {}
        }
      });
    });
  }

  /**
   * Get list of available agents
   */
  async getAgents() {
    return new Promise((resolve, reject) => {
      const requestId = `agents_${++this.requestCounter}_${Date.now()}`;
      
      this.pendingRequests.set(requestId, { resolve, reject });
      
      setTimeout(() => {
        if (this.pendingRequests.has(requestId)) {
          this.pendingRequests.delete(requestId);
          reject(new Error('Request timeout'));
        }
      }, 10000);

      this.sendToPython({
        type: 'get_agents',
        request_id: requestId,
        payload: {}
      });
    });
  }

  /**
   * Get agent bridge status
   */
  async getStatus() {
    return {
      connected: this.isConnected,
      python_process_running: this.pythonProcess && !this.pythonProcess.killed,
      pending_requests: this.pendingRequests.size
    };
  }

  /**
   * Shutdown the agent bridge
   */
  shutdown() {
    if (this.pythonProcess) {
      this.pythonProcess.kill();
      this.pythonProcess = null;
    }
    this.isConnected = false;
    this.pendingRequests.clear();
  }
}

module.exports = AgentBridge;
