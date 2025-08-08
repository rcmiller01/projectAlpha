import React from 'react';
import {
  Container,
  Grid,
  Card,
  CardContent,
  Typography,
  Box,
  Button,
  Chip,
  LinearProgress
} from '@mui/material';
import {
  Chat,
  Psychology,
  Assignment,
  TrendingUp,
  SmartToy
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';

import { useThreadStore, useProjectStore, useAgentStore } from '../store';

const Dashboard = () => {
  const navigate = useNavigate();
  const { threads } = useThreadStore();
  const { projects } = useProjectStore();
  const { agents, agentStatuses } = useAgentStore();

  // Mock data for demonstration
  const stats = {
    totalThreads: threads.length || 12,
    activeProjects: projects.filter(p => p.status === 'active').length || 3,
    totalMessages: 247,
    agentsAvailable: agents.length || 4
  };

  const recentThreads = threads.slice(0, 5) || [
    { id: 1, title: 'SLiM Agent Architecture Discussion', agentType: 'deduction', updatedAt: new Date() },
    { id: 2, title: 'Creative Writing Project', agentType: 'metaphor', updatedAt: new Date() },
    { id: 3, title: 'Task Planning Session', agentType: 'planner', updatedAt: new Date() }
  ];

  const activeProjects = projects.slice(0, 3) || [
    { id: 1, name: 'AI Agent Research', progress: 75, type: 'research' },
    { id: 2, name: 'Voice Integration', progress: 45, type: 'development' },
    { id: 3, name: 'System Documentation', progress: 90, type: 'documentation' }
  ];

  const agentStatusList = [
    { type: 'deduction', status: 'active', name: 'Deduction Agent' },
    { type: 'metaphor', status: 'active', name: 'Metaphor Agent' },
    { type: 'planner', status: 'idle', name: 'Planner Agent' },
    { type: 'ritual', status: 'idle', name: 'Ritual Agent' }
  ];

  const StatCard = ({ icon, title, value, description, color = 'primary' }) => (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <Card sx={{ height: '100%' }}>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
            <Box sx={{
              p: 1,
              borderRadius: 1,
              bgcolor: `${color}.main`,
              color: `${color}.contrastText`,
              mr: 2
            }}>
              {icon}
            </Box>
            <Typography variant="h6" component="h2">
              {title}
            </Typography>
          </Box>
          <Typography variant="h3" component="div" sx={{ mb: 1 }}>
            {value}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {description}
          </Typography>
        </CardContent>
      </Card>
    </motion.div>
  );

  return (
    <Container maxWidth="lg" sx={{ py: 3 }}>
      {/* Welcome Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" component="h1" sx={{ mb: 1 }}>
            Welcome to ProjectAlpha
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Your SLiM agent interaction dashboard
          </Typography>
        </Box>
      </motion.div>

      {/* Stats Cards */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<Chat />}
            title="Total Threads"
            value={stats.totalThreads}
            description="Active conversations"
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<Assignment />}
            title="Active Projects"
            value={stats.activeProjects}
            description="In progress"
            color="success"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<TrendingUp />}
            title="Total Messages"
            value={stats.totalMessages}
            description="All time"
            color="info"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<SmartToy />}
            title="Agents Available"
            value={stats.agentsAvailable}
            description="Ready to assist"
            color="secondary"
          />
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        {/* Recent Threads */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" component="h2">
                    Recent Conversations
                  </Typography>
                  <Button
                    size="small"
                    onClick={() => navigate('/chat')}
                  >
                    View All
                  </Button>
                </Box>

                {recentThreads.map((thread, index) => (
                  <Box
                    key={thread.id}
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      py: 1,
                      borderBottom: index < recentThreads.length - 1 ? 1 : 0,
                      borderColor: 'divider',
                      cursor: 'pointer',
                      '&:hover': { bgcolor: 'action.hover' }
                    }}
                    onClick={() => navigate(`/chat/${thread.id}`)}
                  >
                    <Box>
                      <Typography variant="body1">{thread.title}</Typography>
                      <Typography variant="caption" color="text.secondary">
                        {thread.updatedAt.toLocaleDateString()}
                      </Typography>
                    </Box>
                    <Chip
                      label={thread.agentType}
                      size="small"
                      color="primary"
                      variant="outlined"
                    />
                  </Box>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Active Projects */}
        <Grid item xs={12} md={6}>
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.3 }}
          >
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6" component="h2">
                    Active Projects
                  </Typography>
                  <Button
                    size="small"
                    onClick={() => navigate('/projects')}
                  >
                    View All
                  </Button>
                </Box>

                {activeProjects.map((project, index) => (
                  <Box
                    key={project.id}
                    sx={{
                      py: 2,
                      borderBottom: index < activeProjects.length - 1 ? 1 : 0,
                      borderColor: 'divider'
                    }}
                  >
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                      <Typography variant="body1">{project.name}</Typography>
                      <Chip
                        label={project.type}
                        size="small"
                        color="secondary"
                        variant="outlined"
                      />
                    </Box>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <LinearProgress
                        variant="determinate"
                        value={project.progress}
                        sx={{ flexGrow: 1, height: 6, borderRadius: 3 }}
                      />
                      <Typography variant="caption" color="text.secondary">
                        {project.progress}%
                      </Typography>
                    </Box>
                  </Box>
                ))}
              </CardContent>
            </Card>
          </motion.div>
        </Grid>

        {/* Agent Status */}
        <Grid item xs={12}>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          >
            <Card>
              <CardContent>
                <Typography variant="h6" component="h2" sx={{ mb: 2 }}>
                  Agent Status
                </Typography>

                <Grid container spacing={2}>
                  {agentStatusList.map((agent) => (
                    <Grid item xs={12} sm={6} md={3} key={agent.type}>
                      <Box sx={{
                        p: 2,
                        border: 1,
                        borderColor: 'divider',
                        borderRadius: 1,
                        textAlign: 'center'
                      }}>
                        <SmartToy sx={{ fontSize: 40, color: 'primary.main', mb: 1 }} />
                        <Typography variant="subtitle1">{agent.name}</Typography>
                        <Chip
                          label={agent.status}
                          size="small"
                          color={agent.status === 'active' ? 'success' : 'default'}
                        />
                      </Box>
                    </Grid>
                  ))}
                </Grid>
              </CardContent>
            </Card>
          </motion.div>
        </Grid>
      </Grid>
    </Container>
  );
};

export default Dashboard;
