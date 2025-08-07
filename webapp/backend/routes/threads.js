const express = require('express');
const router = express.Router();
const Thread = require('../models/Thread');

/**
 * Threads API Routes
 * 
 * Manages conversation threads for the ProjectAlpha web UI
 */

// Get all threads for a user
router.get('/', async (req, res) => {
  try {
    const { userId = 'default' } = req.query;
    
    const threads = await Thread.find({ userId })
      .sort({ updatedAt: -1 })
      .select('id title lastMessage messageCount createdAt updatedAt tags isArchived');

    res.json({
      success: true,
      threads: threads,
      count: threads.length,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    req.logger.error(`Error getting threads: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Create a new thread
router.post('/', async (req, res) => {
  try {
    const { title, userId = 'default', tags = [], agentType = 'general' } = req.body;

    if (!title) {
      return res.status(400).json({
        success: false,
        error: 'Thread title is required'
      });
    }

    const thread = new Thread({
      title,
      userId,
      tags,
      agentType,
      messages: [],
      messageCount: 0,
      lastMessage: null
    });

    await thread.save();

    req.logger.info(`Created new thread: ${thread.id} for user: ${userId}`);

    res.status(201).json({
      success: true,
      thread: thread,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    req.logger.error(`Error creating thread: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get a specific thread with messages
router.get('/:threadId', async (req, res) => {
  try {
    const { threadId } = req.params;
    const { includeMessages = true } = req.query;

    const selectFields = includeMessages 
      ? '' // Include all fields
      : '-messages'; // Exclude messages

    const thread = await Thread.findById(threadId).select(selectFields);

    if (!thread) {
      return res.status(404).json({
        success: false,
        error: 'Thread not found'
      });
    }

    res.json({
      success: true,
      thread: thread,
      messageCount: thread.messageCount,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    req.logger.error(`Error getting thread ${req.params.threadId}: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Update thread metadata
router.put('/:threadId', async (req, res) => {
  try {
    const { threadId } = req.params;
    const updateData = req.body;

    // Don't allow updating messages directly through this endpoint
    delete updateData.messages;
    delete updateData.messageCount;

    const thread = await Thread.findByIdAndUpdate(
      threadId,
      { ...updateData, updatedAt: new Date() },
      { new: true, runValidators: true }
    );

    if (!thread) {
      return res.status(404).json({
        success: false,
        error: 'Thread not found'
      });
    }

    req.logger.info(`Updated thread: ${threadId}`);

    res.json({
      success: true,
      thread: thread,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    req.logger.error(`Error updating thread ${req.params.threadId}: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Add message to thread
router.post('/:threadId/messages', async (req, res) => {
  try {
    const { threadId } = req.params;
    const { content, sender, agentType, metadata = {} } = req.body;

    if (!content || !sender) {
      return res.status(400).json({
        success: false,
        error: 'Message content and sender are required'
      });
    }

    const message = {
      content,
      sender,
      agentType,
      metadata,
      timestamp: new Date()
    };

    const thread = await Thread.findByIdAndUpdate(
      threadId,
      {
        $push: { messages: message },
        $inc: { messageCount: 1 },
        $set: { 
          lastMessage: content.substring(0, 100),
          updatedAt: new Date()
        }
      },
      { new: true }
    );

    if (!thread) {
      return res.status(404).json({
        success: false,
        error: 'Thread not found'
      });
    }

    const addedMessage = thread.messages[thread.messages.length - 1];

    req.logger.info(`Added message to thread ${threadId} from ${sender}`);

    res.status(201).json({
      success: true,
      message: addedMessage,
      thread_id: threadId,
      total_messages: thread.messageCount,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    req.logger.error(`Error adding message to thread ${req.params.threadId}: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get messages from a thread with pagination
router.get('/:threadId/messages', async (req, res) => {
  try {
    const { threadId } = req.params;
    const { limit = 50, offset = 0, since } = req.query;

    const thread = await Thread.findById(threadId);

    if (!thread) {
      return res.status(404).json({
        success: false,
        error: 'Thread not found'
      });
    }

    let messages = thread.messages;

    // Filter by timestamp if 'since' parameter is provided
    if (since) {
      const sinceDate = new Date(since);
      messages = messages.filter(msg => new Date(msg.timestamp) > sinceDate);
    }

    // Apply pagination
    const startIndex = parseInt(offset);
    const endIndex = startIndex + parseInt(limit);
    const paginatedMessages = messages.slice(startIndex, endIndex);

    res.json({
      success: true,
      messages: paginatedMessages,
      pagination: {
        total: messages.length,
        limit: parseInt(limit),
        offset: parseInt(offset),
        hasMore: endIndex < messages.length
      },
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    req.logger.error(`Error getting messages for thread ${req.params.threadId}: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Delete a thread
router.delete('/:threadId', async (req, res) => {
  try {
    const { threadId } = req.params;

    const thread = await Thread.findByIdAndDelete(threadId);

    if (!thread) {
      return res.status(404).json({
        success: false,
        error: 'Thread not found'
      });
    }

    req.logger.info(`Deleted thread: ${threadId}`);

    res.json({
      success: true,
      message: 'Thread deleted successfully',
      deletedThread: {
        id: thread.id,
        title: thread.title,
        messageCount: thread.messageCount
      },
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    req.logger.error(`Error deleting thread ${req.params.threadId}: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Archive/unarchive a thread
router.patch('/:threadId/archive', async (req, res) => {
  try {
    const { threadId } = req.params;
    const { archived = true } = req.body;

    const thread = await Thread.findByIdAndUpdate(
      threadId,
      { isArchived: archived, updatedAt: new Date() },
      { new: true }
    );

    if (!thread) {
      return res.status(404).json({
        success: false,
        error: 'Thread not found'
      });
    }

    req.logger.info(`${archived ? 'Archived' : 'Unarchived'} thread: ${threadId}`);

    res.json({
      success: true,
      thread: thread,
      action: archived ? 'archived' : 'unarchived',
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    req.logger.error(`Error archiving thread ${req.params.threadId}: ${error.message}`);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

module.exports = router;
