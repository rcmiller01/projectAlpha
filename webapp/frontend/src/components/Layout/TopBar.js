import React from 'react';
import {
  AppBar,
  Toolbar,
  Typography,
  IconButton,
  Box,
  Chip,
  Avatar,
  Button
} from '@mui/material';
import {
  Menu,
  Notifications,
  AccountCircle,
  Circle
} from '@mui/icons-material';

import { useAppStore, useAgentStore } from '../../store';

const TopBar = ({ onSidebarToggle, sidebarOpen, height }) => {
  const { connectionStatus } = useAppStore();
  const { currentAgent } = useAgentStore();

  const getConnectionColor = (status) => {
    switch (status) {
      case 'connected': return 'success';
      case 'connecting': return 'warning';
      case 'disconnected': return 'error';
      default: return 'default';
    }
  };

  const getConnectionText = (status) => {
    switch (status) {
      case 'connected': return 'Connected';
      case 'connecting': return 'Connecting...';
      case 'disconnected': return 'Disconnected';
      default: return 'Unknown';
    }
  };

  return (
    <AppBar 
      position="static" 
      elevation={0}
      sx={{ 
        bgcolor: 'background.paper',
        borderBottom: 1,
        borderColor: 'divider',
        color: 'text.primary',
        height: height
      }}
    >
      <Toolbar sx={{ minHeight: `${height}px !important` }}>
        {/* Menu Button */}
        <IconButton
          edge="start"
          color="inherit"
          aria-label="menu"
          onClick={onSidebarToggle}
          sx={{ mr: 2 }}
        >
          <Menu />
        </IconButton>

        {/* Page Title - Dynamic based on route */}
        <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
          {/* This could be enhanced to show current page context */}
          SLiM Agent Interface
        </Typography>

        {/* Current Agent Indicator */}
        <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
          <Typography variant="body2" color="text.secondary" sx={{ mr: 1 }}>
            Agent:
          </Typography>
          <Chip
            label={currentAgent || 'general'}
            size="small"
            color="primary"
            sx={{ textTransform: 'capitalize' }}
          />
        </Box>

        {/* Connection Status */}
        <Box sx={{ display: 'flex', alignItems: 'center', mr: 2 }}>
          <Circle 
            sx={{ 
              fontSize: 12, 
              mr: 1,
              color: `${getConnectionColor(connectionStatus.api)}.main`
            }} 
          />
          <Typography variant="body2" color="text.secondary">
            {getConnectionText(connectionStatus.api)}
          </Typography>
        </Box>

        {/* Notifications */}
        <IconButton
          color="inherit"
          aria-label="notifications"
          sx={{ mr: 1 }}
        >
          <Notifications />
        </IconButton>

        {/* User Avatar */}
        <IconButton
          color="inherit"
          aria-label="account"
        >
          <AccountCircle />
        </IconButton>
      </Toolbar>
    </AppBar>
  );
};

export default TopBar;
