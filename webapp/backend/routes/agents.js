const express = require('express');
const router = express.Router();

/**
 * Agents API Routes
 *
 * Provides REST endpoints for interacting with ProjectAlpha SLiM agents
 */

// Get list of available agents
router.get('/', async (req, res) => {
  try {
    const agents = await req.agentBridge.getAgents();
    res.json({
      success: true,
      agents: agents.agents || {},
      count: agents.count || 0,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    req.logger.error(`Error getting agents: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get agent status and capabilities
router.get('/:agentType', async (req, res) => {
  try {
    const { agentType } = req.params;
    const agents = await req.agentBridge.getAgents();

    if (agents.success && agents.agents[agentType]) {
      res.json({
        success: true,
        agent: agents.agents[agentType],
        available: true
      });
    } else {
      res.status(404).json({
        success: false,
        error: `Agent '${agentType}' not found`,
        available_agents: Object.keys(agents.agents || {})
      });
    }
  } catch (error) {
    req.logger.error(`Error getting agent ${req.params.agentType}: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Invoke a specific agent
router.post('/:agentType/invoke', async (req, res) => {
  try {
    const { agentType } = req.params;
    const { prompt, depth = 1, context = {} } = req.body;

    if (!prompt) {
      return res.status(400).json({
        success: false,
        error: 'Prompt is required'
      });
    }

    req.logger.info(`Invoking ${agentType} agent with prompt: ${prompt.substring(0, 100)}...`);

    const result = await req.agentBridge.invokeAgent(agentType, prompt, {
      depth,
      context,
      timeout: 30000
    });

    if (result.success) {
      res.json({
        success: true,
        agent_type: agentType,
        prompt: prompt,
        response: result.response,
        metadata: {
          depth: depth,
          timestamp: result.timestamp,
          processing_time: result.processing_time || 0
        }
      });
    } else {
      res.status(500).json({
        success: false,
        error: result.error,
        agent_type: agentType
      });
    }

  } catch (error) {
    req.logger.error(`Error invoking agent ${req.params.agentType}: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message,
      agent_type: req.params.agentType
    });
  }
});

// Batch invoke multiple agents
router.post('/batch', async (req, res) => {
  try {
    const { requests } = req.body;

    if (!Array.isArray(requests) || requests.length === 0) {
      return res.status(400).json({
        success: false,
        error: 'Requests array is required and must not be empty'
      });
    }

    req.logger.info(`Processing batch request with ${requests.length} agent invocations`);

    const results = [];

    // Process requests sequentially to avoid overwhelming the system
    for (let i = 0; i < requests.length; i++) {
      const request = requests[i];
      const { agentType, prompt, depth = 1, context = {} } = request;

      try {
        const result = await req.agentBridge.invokeAgent(agentType, prompt, {
          depth,
          context,
          timeout: 30000
        });

        results.push({
          index: i,
          agent_type: agentType,
          success: result.success,
          response: result.success ? result.response : null,
          error: result.success ? null : result.error
        });

      } catch (error) {
        results.push({
          index: i,
          agent_type: agentType,
          success: false,
          response: null,
          error: error.message
        });
      }
    }

    const successCount = results.filter(r => r.success).length;

    res.json({
      success: true,
      total_requests: requests.length,
      successful: successCount,
      failed: requests.length - successCount,
      results: results,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    req.logger.error(`Error processing batch request: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Stream agent response (for real-time communication)
router.post('/:agentType/stream', async (req, res) => {
  try {
    const { agentType } = req.params;
    const { prompt, depth = 1, context = {} } = req.body;

    if (!prompt) {
      return res.status(400).json({
        success: false,
        error: 'Prompt is required'
      });
    }

    // Set up Server-Sent Events
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Headers': 'Cache-Control'
    });

    // Send initial status
    res.write(`data: ${JSON.stringify({
      type: 'status',
      message: `Starting ${agentType} agent...`,
      timestamp: new Date().toISOString()
    })}\n\n`);

    try {
      const result = await req.agentBridge.invokeAgent(agentType, prompt, {
        depth,
        context,
        timeout: 30000
      });

      // Send response
      res.write(`data: ${JSON.stringify({
        type: 'response',
        agent_type: agentType,
        success: result.success,
        response: result.success ? result.response : null,
        error: result.success ? null : result.error,
        timestamp: new Date().toISOString()
      })}\n\n`);

      // Send completion
      res.write(`data: ${JSON.stringify({
        type: 'complete',
        timestamp: new Date().toISOString()
      })}\n\n`);

    } catch (error) {
      res.write(`data: ${JSON.stringify({
        type: 'error',
        error: error.message,
        timestamp: new Date().toISOString()
      })}\n\n`);
    }

    res.end();

  } catch (error) {
    req.logger.error(`Error streaming agent ${req.params.agentType}: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

module.exports = router;
