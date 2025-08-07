const express = require('express');
const router = express.Router();
const { Project } = require('../models');

// GET /api/projects - Get all projects for user
router.get('/', async (req, res) => {
  try {
    const { 
      status, 
      type, 
      priority, 
      search, 
      page = 1, 
      limit = 20,
      userId = 'default' 
    } = req.query;

    let query = { userId };

    // Apply filters
    if (status) query.status = status;
    if (type) query.type = type;
    if (priority) query.priority = priority;

    // Handle search
    if (search) {
      const searchRegex = new RegExp(search, 'i');
      query.$or = [
        { name: searchRegex },
        { description: searchRegex },
        { tags: { $in: [searchRegex] } }
      ];
    }

    const skip = (parseInt(page) - 1) * parseInt(limit);
    
    const [projects, total] = await Promise.all([
      Project.find(query)
        .sort({ priority: -1, updatedAt: -1 })
        .skip(skip)
        .limit(parseInt(limit))
        .populate('activeThreads', 'title agentType messageCount updatedAt'),
      Project.countDocuments(query)
    ]);

    res.json({
      projects: projects.map(p => p.getSummary()),
      pagination: {
        page: parseInt(page),
        limit: parseInt(limit),
        total,
        pages: Math.ceil(total / parseInt(limit))
      }
    });
  } catch (error) {
    console.error('Error fetching projects:', error);
    res.status(500).json({ error: 'Failed to fetch projects' });
  }
});

// GET /api/projects/stats - Get dashboard statistics
router.get('/stats', async (req, res) => {
  try {
    const { userId = 'default' } = req.query;
    const stats = await Project.getDashboardStats(userId);
    res.json(stats);
  } catch (error) {
    console.error('Error fetching project stats:', error);
    res.status(500).json({ error: 'Failed to fetch statistics' });
  }
});

// GET /api/projects/:id - Get specific project with threads
router.get('/:id', async (req, res) => {
  try {
    const { userId = 'default' } = req.query;
    
    const project = await Project.findOne({ 
      _id: req.params.id, 
      userId 
    }).populate('activeThreads');

    if (!project) {
      return res.status(404).json({ error: 'Project not found' });
    }

    res.json(project);
  } catch (error) {
    console.error('Error fetching project:', error);
    res.status(500).json({ error: 'Failed to fetch project' });
  }
});

// POST /api/projects - Create new project
router.post('/', async (req, res) => {
  try {
    const { userId = 'default' } = req.body;
    
    const projectData = {
      ...req.body,
      userId
    };

    const project = new Project(projectData);
    await project.save();

    res.status(201).json(project.getSummary());
  } catch (error) {
    if (error.code === 11000) {
      return res.status(400).json({ error: 'Project name already exists' });
    }
    console.error('Error creating project:', error);
    res.status(500).json({ error: 'Failed to create project' });
  }
});

// PUT /api/projects/:id - Update project
router.put('/:id', async (req, res) => {
  try {
    const { userId = 'default' } = req.body;
    
    const project = await Project.findOneAndUpdate(
      { _id: req.params.id, userId },
      { 
        $set: {
          ...req.body,
          userId,
          'metadata.lastActivity': new Date()
        }
      },
      { new: true, runValidators: true }
    );

    if (!project) {
      return res.status(404).json({ error: 'Project not found' });
    }

    res.json(project.getSummary());
  } catch (error) {
    console.error('Error updating project:', error);
    res.status(500).json({ error: 'Failed to update project' });
  }
});

// PUT /api/projects/:id/progress - Update project progress
router.put('/:id/progress', async (req, res) => {
  try {
    const { progress, userId = 'default' } = req.body;
    
    const project = await Project.findOne({ 
      _id: req.params.id, 
      userId 
    });

    if (!project) {
      return res.status(404).json({ error: 'Project not found' });
    }

    await project.updateProgress(progress);
    res.json(project.getSummary());
  } catch (error) {
    console.error('Error updating project progress:', error);
    res.status(500).json({ error: 'Failed to update progress' });
  }
});

// POST /api/projects/:id/threads - Add thread to project
router.post('/:id/threads', async (req, res) => {
  try {
    const { threadId, userId = 'default' } = req.body;
    
    const project = await Project.findOne({ 
      _id: req.params.id, 
      userId 
    });

    if (!project) {
      return res.status(404).json({ error: 'Project not found' });
    }

    await project.addThread(threadId);
    res.json(project.getSummary());
  } catch (error) {
    console.error('Error adding thread to project:', error);
    res.status(500).json({ error: 'Failed to add thread' });
  }
});

// DELETE /api/projects/:id/threads/:threadId - Remove thread from project
router.delete('/:id/threads/:threadId', async (req, res) => {
  try {
    const { userId = 'default' } = req.query;
    
    const project = await Project.findOne({ 
      _id: req.params.id, 
      userId 
    });

    if (!project) {
      return res.status(404).json({ error: 'Project not found' });
    }

    await project.removeThread(req.params.threadId);
    res.json(project.getSummary());
  } catch (error) {
    console.error('Error removing thread from project:', error);
    res.status(500).json({ error: 'Failed to remove thread' });
  }
});

// DELETE /api/projects/:id - Delete project
router.delete('/:id', async (req, res) => {
  try {
    const { userId = 'default' } = req.query;
    
    const project = await Project.findOneAndDelete({ 
      _id: req.params.id, 
      userId 
    });

    if (!project) {
      return res.status(404).json({ error: 'Project not found' });
    }

    res.json({ message: 'Project deleted successfully' });
  } catch (error) {
    console.error('Error deleting project:', error);
    res.status(500).json({ error: 'Failed to delete project' });
  }
});

// POST /api/projects/:id/archive - Archive project
router.post('/:id/archive', async (req, res) => {
  try {
    const { userId = 'default' } = req.body;
    
    const project = await Project.findOneAndUpdate(
      { _id: req.params.id, userId },
      { 
        $set: { 
          status: 'archived',
          'metadata.lastActivity': new Date()
        }
      },
      { new: true }
    );

    if (!project) {
      return res.status(404).json({ error: 'Project not found' });
    }

    res.json(project.getSummary());
  } catch (error) {
    console.error('Error archiving project:', error);
    res.status(500).json({ error: 'Failed to archive project' });
  }
});

module.exports = router;
