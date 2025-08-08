import React from 'react';
import {
  Container,
  Typography,
  Grid,
  Card,
  CardContent,
  Box,
  Button,
  Chip,
  LinearProgress,
  IconButton
} from '@mui/material';
import {
  Add,
  Folder,
  Edit,
  Archive,
  TrendingUp
} from '@mui/icons-material';
import { motion } from 'framer-motion';

import { useProjectStore } from '../store';

const Projects = () => {
  const { projects, getFilteredProjects } = useProjectStore();

  // Mock projects for demonstration
  const mockProjects = [
    {
      id: 1,
      name: 'AI Agent Research',
      description: 'Comprehensive research on SLiM agent architecture and implementation patterns',
      type: 'research',
      status: 'active',
      priority: 'high',
      progress: 75,
      threadCount: 8,
      tags: ['ai', 'research', 'slim'],
      lastActivity: new Date()
    },
    {
      id: 2,
      name: 'Voice Integration System',
      description: 'Integration of voice cadence modulation for enhanced agent interaction',
      type: 'development',
      status: 'active',
      priority: 'medium',
      progress: 45,
      threadCount: 3,
      tags: ['voice', 'integration', 'tts'],
      lastActivity: new Date()
    },
    {
      id: 3,
      name: 'System Documentation',
      description: 'Complete documentation of ProjectAlpha system architecture',
      type: 'documentation',
      status: 'active',
      priority: 'low',
      progress: 90,
      threadCount: 12,
      tags: ['docs', 'architecture'],
      lastActivity: new Date()
    }
  ];

  const displayProjects = projects.length > 0 ? getFilteredProjects() : mockProjects;

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'urgent': return 'error';
      case 'high': return 'warning';
      case 'medium': return 'primary';
      case 'low': return 'info';
      default: return 'default';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'active': return 'success';
      case 'completed': return 'primary';
      case 'paused': return 'warning';
      case 'archived': return 'default';
      default: return 'default';
    }
  };

  return (
    <Container maxWidth="lg" sx={{ py: 3 }}>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
          <Box>
            <Typography variant="h4" component="h1" sx={{ mb: 1 }}>
              Projects
            </Typography>
            <Typography variant="body1" color="text.secondary">
              Organize your conversations and research into structured projects
            </Typography>
          </Box>
          <Button
            variant="contained"
            startIcon={<Add />}
            onClick={() => {
              // TODO: Implement create project dialog
              alert('Create project functionality will be implemented');
            }}
          >
            New Project
          </Button>
        </Box>
      </motion.div>

      {/* Projects Grid */}
      <Grid container spacing={3}>
        {displayProjects.map((project, index) => (
          <Grid item xs={12} md={6} lg={4} key={project.id}>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: index * 0.1 }}
            >
              <Card sx={{
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                '&:hover': {
                  boxShadow: 4,
                  transform: 'translateY(-2px)',
                  transition: 'all 0.2s ease-in-out'
                }
              }}>
                <CardContent sx={{ flexGrow: 1 }}>
                  {/* Project Header */}
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Folder sx={{ mr: 1, color: 'primary.main' }} />
                      <Typography variant="h6" component="h2" sx={{ flexGrow: 1 }}>
                        {project.name}
                      </Typography>
                    </Box>
                    <IconButton size="small">
                      <Edit fontSize="small" />
                    </IconButton>
                  </Box>

                  {/* Description */}
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {project.description}
                  </Typography>

                  {/* Tags */}
                  <Box sx={{ mb: 2 }}>
                    {project.tags?.map((tag) => (
                      <Chip
                        key={tag}
                        label={tag}
                        size="small"
                        variant="outlined"
                        sx={{ mr: 0.5, mb: 0.5 }}
                      />
                    ))}
                  </Box>

                  {/* Progress */}
                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        Progress
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {project.progress}%
                      </Typography>
                    </Box>
                    <LinearProgress
                      variant="determinate"
                      value={project.progress}
                      sx={{ height: 6, borderRadius: 3 }}
                    />
                  </Box>

                  {/* Project Stats */}
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <TrendingUp sx={{ fontSize: 16, mr: 0.5, color: 'text.secondary' }} />
                      <Typography variant="body2" color="text.secondary">
                        {project.threadCount} threads
                      </Typography>
                    </Box>
                    <Typography variant="caption" color="text.secondary">
                      {project.lastActivity.toLocaleDateString()}
                    </Typography>
                  </Box>

                  {/* Status and Priority */}
                  <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                    <Chip
                      label={project.status}
                      size="small"
                      color={getStatusColor(project.status)}
                    />
                    <Chip
                      label={`${project.priority} priority`}
                      size="small"
                      color={getPriorityColor(project.priority)}
                      variant="outlined"
                    />
                    <Chip
                      label={project.type}
                      size="small"
                      variant="outlined"
                    />
                  </Box>
                </CardContent>

                {/* Actions */}
                <Box sx={{ p: 2, pt: 0 }}>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      variant="outlined"
                      size="small"
                      fullWidth
                      onClick={() => {
                        // TODO: Navigate to project detail view
                        alert(`Open project: ${project.name}`);
                      }}
                    >
                      Open
                    </Button>
                    <IconButton
                      size="small"
                      onClick={() => {
                        // TODO: Archive project
                        alert(`Archive project: ${project.name}`);
                      }}
                    >
                      <Archive fontSize="small" />
                    </IconButton>
                  </Box>
                </Box>
              </Card>
            </motion.div>
          </Grid>
        ))}
      </Grid>

      {/* Empty State */}
      {displayProjects.length === 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Box sx={{ textAlign: 'center', py: 8 }}>
            <Folder sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" sx={{ mb: 1 }}>
              No projects yet
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
              Create your first project to organize conversations and research
            </Typography>
            <Button variant="contained" startIcon={<Add />}>
              Create Project
            </Button>
          </Box>
        </motion.div>
      )}
    </Container>
  );
};

export default Projects;
