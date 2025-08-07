import React from 'react';
import {
  Container,
  Typography,
  Box,
  Card,
  CardContent,
  Switch,
  FormControlLabel,
  Divider,
  Grid,
  Button,
  Alert
} from '@mui/material';
import {
  Settings as SettingsIcon,
  Notifications,
  Save,
  RestartAlt
} from '@mui/icons-material';
import { motion } from 'framer-motion';

import VoiceSettingsPanel from '../components/Settings/VoiceSettingsPanel';
import { useAppStore } from '../store';

const Settings = () => {
  const { preferences, setPreferences } = useAppStore();

  const handlePreferenceChange = (key, value) => {
    setPreferences({
      [key]: value
    });
  };

  const handleReset = () => {
    const defaultPreferences = {
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
    };
    setPreferences(defaultPreferences);
  };

  return (
    <Container maxWidth="md" sx={{ py: 3 }}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="h4" component="h1" sx={{ mb: 1, display: 'flex', alignItems: 'center' }}>
            <SettingsIcon sx={{ mr: 2, color: 'primary.main' }} />
            Settings
          </Typography>
          <Typography variant="body1" color="text.secondary">
            Customize your ProjectAlpha experience and agent interactions
          </Typography>
        </Box>

        {/* General Settings */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" component="h2" sx={{ mb: 3 }}>
              General Settings
            </Typography>

            <Grid container spacing={3}>
              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={preferences.autoSave || true}
                      onChange={(e) => handlePreferenceChange('autoSave', e.target.checked)}
                      color="primary"
                    />
                  }
                  label="Auto-save conversations"
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={preferences.notifications || true}
                      onChange={(e) => handlePreferenceChange('notifications', e.target.checked)}
                      color="primary"
                    />
                  }
                  label={
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      <Notifications sx={{ mr: 1, fontSize: '1.1rem' }} />
                      Enable notifications
                    </Box>
                  }
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <FormControlLabel
                  control={
                    <Switch
                      checked={preferences.streamingResponses !== false}
                      onChange={(e) => handlePreferenceChange('streamingResponses', e.target.checked)}
                      color="primary"
                    />
                  }
                  label="Streaming responses"
                />
              </Grid>
            </Grid>

            <Divider sx={{ my: 3 }} />

            <Typography variant="subtitle1" sx={{ mb: 2 }}>
              Default Agent
            </Typography>
            <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
              {['general', 'deduction', 'metaphor', 'planner', 'ritual'].map((agent) => (
                <Button
                  key={agent}
                  variant={preferences.defaultAgent === agent ? 'contained' : 'outlined'}
                  size="small"
                  onClick={() => handlePreferenceChange('defaultAgent', agent)}
                  sx={{ textTransform: 'capitalize' }}
                >
                  {agent}
                </Button>
              ))}
            </Box>
          </CardContent>
        </Card>

        {/* Voice Settings Panel */}
        <VoiceSettingsPanel />

        {/* Agent Settings */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" component="h2" sx={{ mb: 3 }}>
              Agent Behavior
            </Typography>

            <Alert severity="info" sx={{ mb: 2 }}>
              Additional agent-specific settings will be available as more agents are integrated.
            </Alert>

            <Typography variant="body2" color="text.secondary">
              Configure how agents respond to different types of queries and conversations.
              These settings will apply across all agent interactions.
            </Typography>
          </CardContent>
        </Card>

        {/* System Settings */}
        <Card sx={{ mb: 3 }}>
          <CardContent>
            <Typography variant="h6" component="h2" sx={{ mb: 3 }}>
              System
            </Typography>

            <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Button
                variant="outlined"
                startIcon={<Save />}
                onClick={() => {
                  // Settings are auto-saved via Zustand store
                  alert('Settings saved successfully!');
                }}
              >
                Save Settings
              </Button>

              <Button
                variant="outlined"
                color="warning"
                startIcon={<RestartAlt />}
                onClick={handleReset}
              >
                Reset to Defaults
              </Button>
            </Box>

            <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>
              Settings are automatically saved to your browser's local storage.
            </Typography>
          </CardContent>
        </Card>

        {/* Version Info */}
        <Card>
          <CardContent>
            <Typography variant="h6" component="h2" sx={{ mb: 2 }}>
              About
            </Typography>
            <Typography variant="body2" color="text.secondary">
              ProjectAlpha Web Interface v1.0.0
              <br />
              SLiM Agent Integration System
              <br />
              Built with React, Material-UI, and Node.js
            </Typography>
          </CardContent>
        </Card>
      </motion.div>
    </Container>
  );
};

export default Settings;
