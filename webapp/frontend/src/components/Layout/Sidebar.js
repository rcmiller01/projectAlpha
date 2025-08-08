import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Box,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Typography,
  Divider,
  IconButton,
  TextField,
  InputAdornment,
  Chip,
  Avatar
} from '@mui/material';
import {
  Dashboard,
  Chat,
  Folder,
  Settings,
  Search,
  Add,
  SmartToy,
  Close
} from '@mui/icons-material';
import { motion } from 'framer-motion';

import { useThreadStore, useProjectStore } from '../../store';

const Sidebar = ({ onClose }) => {
  const location = useLocation();
  const navigate = useNavigate();
  const [searchQuery, setSearchQuery] = useState('');

  const { threads, getFilteredThreads } = useThreadStore();
  const { projects, getFilteredProjects } = useProjectStore();

  // Mock data for demonstration
  const mockThreads = [
    { id: 1, title: 'SLiM Agent Discussion', agentType: 'deduction', updatedAt: new Date() },
    { id: 2, title: 'Creative Writing', agentType: 'metaphor', updatedAt: new Date() },
    { id: 3, title: 'Project Planning', agentType: 'planner', updatedAt: new Date() }
  ];

  const mockProjects = [
    { id: 1, name: 'AI Research', threadCount: 8, status: 'active' },
    { id: 2, name: 'Voice Integration', threadCount: 3, status: 'active' },
    { id: 3, name: 'Documentation', threadCount: 12, status: 'active' }
  ];

  const displayThreads = threads.length > 0 ? getFilteredThreads() : mockThreads;
  const displayProjects = projects.length > 0 ? getFilteredProjects() : mockProjects;

  const navigation = [
    { path: '/', icon: <Dashboard />, label: 'Dashboard' },
    { path: '/chat', icon: <Chat />, label: 'Chat' },
    { path: '/projects', icon: <Folder />, label: 'Projects' },
    { path: '/settings', icon: <Settings />, label: 'Settings' }
  ];

  const isActive = (path) => {
    if (path === '/') {
      return location.pathname === '/';
    }
    return location.pathname.startsWith(path);
  };

  const filteredThreads = displayThreads.filter(thread =>
    thread.title.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const filteredProjects = displayProjects.filter(project =>
    project.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  const getAgentColor = (agentType) => {
    const colors = {
      deduction: 'primary',
      metaphor: 'secondary',
      planner: 'success',
      ritual: 'warning',
      general: 'info'
    };
    return colors[agentType] || 'default';
  };

  return (
    <Box sx={{
      height: '100vh',
      display: 'flex',
      flexDirection: 'column',
      borderRight: 1,
      borderColor: 'divider',
      bgcolor: 'background.paper'
    }}>
      {/* Header */}
      <Box sx={{ p: 2, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
          <Typography variant="h6" component="h1" sx={{ fontWeight: 600 }}>
            ProjectAlpha
          </Typography>
          <IconButton onClick={onClose} size="small" sx={{ display: { md: 'none' } }}>
            <Close />
          </IconButton>
        </Box>

        {/* Search */}
        <TextField
          fullWidth
          size="small"
          placeholder="Search threads & projects..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          InputProps={{
            startAdornment: (
              <InputAdornment position="start">
                <Search fontSize="small" />
              </InputAdornment>
            ),
          }}
        />
      </Box>

      {/* Navigation */}
      <Box sx={{ px: 1, py: 2 }}>
        <List dense>
          {navigation.map((item) => (
            <ListItem key={item.path} disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton
                onClick={() => {
                  navigate(item.path);
                  onClose?.();
                }}
                selected={isActive(item.path)}
                sx={{
                  borderRadius: 1,
                  '&.Mui-selected': {
                    bgcolor: 'primary.main',
                    color: 'primary.contrastText',
                    '&:hover': { bgcolor: 'primary.dark' }
                  }
                }}
              >
                <ListItemIcon sx={{
                  minWidth: 36,
                  color: isActive(item.path) ? 'inherit' : 'text.secondary'
                }}>
                  {item.icon}
                </ListItemIcon>
                <ListItemText primary={item.label} />
              </ListItemButton>
            </ListItem>
          ))}
        </List>
      </Box>

      <Divider />

      {/* Recent Threads */}
      <Box sx={{ flexGrow: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <Box sx={{ p: 2, pb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Recent Threads
            </Typography>
            <IconButton
              size="small"
              onClick={() => {
                navigate('/chat');
                onClose?.();
              }}
            >
              <Add fontSize="small" />
            </IconButton>
          </Box>
        </Box>

        <Box sx={{ flexGrow: 1, overflow: 'auto', px: 1 }}>
          <List dense>
            {filteredThreads.slice(0, 8).map((thread, index) => (
              <motion.div
                key={thread.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <ListItem disablePadding sx={{ mb: 0.5 }}>
                  <ListItemButton
                    onClick={() => {
                      navigate(`/chat/${thread.id}`);
                      onClose?.();
                    }}
                    sx={{
                      borderRadius: 1,
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                      py: 1,
                      '&:hover': { bgcolor: 'action.hover' }
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', mb: 0.5 }}>
                      <Avatar sx={{
                        width: 20,
                        height: 20,
                        mr: 1,
                        bgcolor: `${getAgentColor(thread.agentType)}.main`
                      }}>
                        <SmartToy sx={{ fontSize: 12 }} />
                      </Avatar>
                      <Typography
                        variant="body2"
                        sx={{
                          flexGrow: 1,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}
                      >
                        {thread.title}
                      </Typography>
                    </Box>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', width: '100%' }}>
                      <Chip
                        label={thread.agentType}
                        size="small"
                        color={getAgentColor(thread.agentType)}
                        variant="outlined"
                        sx={{ height: 16, fontSize: '0.65rem' }}
                      />
                      <Typography variant="caption" color="text.secondary">
                        {thread.updatedAt.toLocaleDateString()}
                      </Typography>
                    </Box>
                  </ListItemButton>
                </ListItem>
              </motion.div>
            ))}
          </List>
        </Box>

        <Divider sx={{ mx: 2 }} />

        {/* Projects */}
        <Box sx={{ p: 2, pb: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 1 }}>
            <Typography variant="subtitle2" color="text.secondary">
              Active Projects
            </Typography>
            <IconButton
              size="small"
              onClick={() => {
                navigate('/projects');
                onClose?.();
              }}
            >
              <Add fontSize="small" />
            </IconButton>
          </Box>
        </Box>

        <Box sx={{ overflow: 'auto', px: 1, pb: 2 }}>
          <List dense>
            {filteredProjects.slice(0, 5).map((project, index) => (
              <motion.div
                key={project.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <ListItem disablePadding sx={{ mb: 0.5 }}>
                  <ListItemButton
                    onClick={() => {
                      navigate(`/projects/${project.id}`);
                      onClose?.();
                    }}
                    sx={{
                      borderRadius: 1,
                      flexDirection: 'column',
                      alignItems: 'flex-start',
                      py: 1,
                      '&:hover': { bgcolor: 'action.hover' }
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'center', width: '100%', mb: 0.5 }}>
                      <Folder sx={{ fontSize: 16, mr: 1, color: 'primary.main' }} />
                      <Typography
                        variant="body2"
                        sx={{
                          flexGrow: 1,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap'
                        }}
                      >
                        {project.name}
                      </Typography>
                    </Box>
                    <Typography variant="caption" color="text.secondary">
                      {project.threadCount} threads
                    </Typography>
                  </ListItemButton>
                </ListItem>
              </motion.div>
            ))}
          </List>
        </Box>
      </Box>
    </Box>
  );
};

export default Sidebar;
