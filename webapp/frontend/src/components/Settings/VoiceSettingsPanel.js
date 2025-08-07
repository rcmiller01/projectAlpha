import React from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Switch,
  Slider,
  FormControlLabel,
  Grid,
  Chip,
  Divider,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
import {
  VolumeUp,
  Speed,
  Favorite,
  Psychology
} from '@mui/icons-material';
import { useVoiceSettings } from '../../hooks/useVoiceCadence';

const VoiceSettingsPanel = () => {
  const { voiceSettings, updateVoiceSettings } = useVoiceSettings();

  const handleSettingChange = (setting, value) => {
    updateVoiceSettings({ [setting]: value });
  };

  const moodOptions = [
    { value: 'contemplative', label: 'Contemplative', desc: 'Thoughtful and measured' },
    { value: 'creative', label: 'Creative', desc: 'Flowing and imaginative' },
    { value: 'analytical', label: 'Analytical', desc: 'Clear and precise' },
    { value: 'warm', label: 'Warm', desc: 'Friendly and inviting' },
    { value: 'balanced', label: 'Balanced', desc: 'Adaptive to context' },
    { value: 'mystical', label: 'Mystical', desc: 'Soft and ethereal' }
  ];

  const energyLevels = [
    { value: 0.2, label: 'Very Calm' },
    { value: 0.4, label: 'Calm' },
    { value: 0.6, label: 'Moderate' },
    { value: 0.8, label: 'Energetic' },
    { value: 1.0, label: 'Very Energetic' }
  ];

  const intimacyLevels = [
    { value: 0.2, label: 'Formal' },
    { value: 0.4, label: 'Professional' },
    { value: 0.6, label: 'Friendly' },
    { value: 0.8, label: 'Personal' },
    { value: 1.0, label: 'Intimate' }
  ];

  return (
    <Card sx={{ mb: 3 }}>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <VolumeUp sx={{ mr: 1, color: 'primary.main' }} />
          <Typography variant="h6" component="h2">
            Voice & Speech Settings
          </Typography>
        </Box>

        {/* Enable/Disable Voice Modulation */}
        <Box sx={{ mb: 3 }}>
          <FormControlLabel
            control={
              <Switch
                checked={voiceSettings.enabled}
                onChange={(e) => handleSettingChange('enabled', e.target.checked)}
                color="primary"
              />
            }
            label="Enable Voice Modulation"
          />
          <Typography variant="body2" color="text.secondary" sx={{ ml: 4 }}>
            Apply emotional and contextual voice characteristics to agent responses
          </Typography>
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Voice Mood Selection */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="subtitle1" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
            <Psychology sx={{ mr: 1, fontSize: '1.2rem' }} />
            Voice Mood
          </Typography>
          <FormControl fullWidth>
            <InputLabel>Mood</InputLabel>
            <Select
              value={voiceSettings.mood || 'balanced'}
              label="Mood"
              onChange={(e) => handleSettingChange('mood', e.target.value)}
              disabled={!voiceSettings.enabled}
            >
              {moodOptions.map((mood) => (
                <MenuItem key={mood.value} value={mood.value}>
                  <Box>
                    <Typography variant="body1">{mood.label}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {mood.desc}
                    </Typography>
                  </Box>
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        </Box>

        {/* Energy Level Slider */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="subtitle1" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
            <Speed sx={{ mr: 1, fontSize: '1.2rem' }} />
            Energy Level
          </Typography>
          <Slider
            value={voiceSettings.energy || 0.6}
            onChange={(e, value) => handleSettingChange('energy', value)}
            disabled={!voiceSettings.enabled}
            min={0.2}
            max={1.0}
            step={0.1}
            marks={energyLevels}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => 
              energyLevels.find(level => level.value === value)?.label || value
            }
            sx={{ mt: 2 }}
          />
        </Box>

        {/* Intimacy Level Slider */}
        <Box sx={{ mb: 4 }}>
          <Typography variant="subtitle1" sx={{ mb: 2, display: 'flex', alignItems: 'center' }}>
            <Favorite sx={{ mr: 1, fontSize: '1.2rem' }} />
            Intimacy Level
          </Typography>
          <Slider
            value={voiceSettings.intimacy || 0.6}
            onChange={(e, value) => handleSettingChange('intimacy', value)}
            disabled={!voiceSettings.enabled}
            min={0.2}
            max={1.0}
            step={0.1}
            marks={intimacyLevels}
            valueLabelDisplay="auto"
            valueLabelFormat={(value) => 
              intimacyLevels.find(level => level.value === value)?.label || value
            }
            sx={{ mt: 2 }}
          />
        </Box>

        <Divider sx={{ my: 3 }} />

        {/* Agent-Specific Settings */}
        <Box sx={{ mb: 3 }}>
          <FormControlLabel
            control={
              <Switch
                checked={voiceSettings.agentSpecific || true}
                onChange={(e) => handleSettingChange('agentSpecific', e.target.checked)}
                color="primary"
                disabled={!voiceSettings.enabled}
              />
            }
            label="Agent-Specific Voice Characteristics"
          />
          <Typography variant="body2" color="text.secondary" sx={{ ml: 4, mb: 2 }}>
            Allow each agent type to have unique voice characteristics
          </Typography>

          {/* Agent Voice Previews */}
          {voiceSettings.agentSpecific && voiceSettings.enabled && (
            <Grid container spacing={1} sx={{ ml: 4 }}>
              <Grid item>
                <Chip 
                  label="Deduction: Analytical" 
                  size="small" 
                  color="primary" 
                  variant="outlined" 
                />
              </Grid>
              <Grid item>
                <Chip 
                  label="Metaphor: Melodic" 
                  size="small" 
                  color="secondary" 
                  variant="outlined" 
                />
              </Grid>
              <Grid item>
                <Chip 
                  label="Planner: Confident" 
                  size="small" 
                  color="success" 
                  variant="outlined" 
                />
              </Grid>
              <Grid item>
                <Chip 
                  label="Ritual: Whispery" 
                  size="small" 
                  color="warning" 
                  variant="outlined" 
                />
              </Grid>
            </Grid>
          )}
        </Box>

        {/* Voice Preview */}
        {voiceSettings.enabled && (
          <Box sx={{ p: 2, backgroundColor: 'action.hover', borderRadius: 1 }}>
            <Typography variant="body2" color="text.secondary">
              <strong>Current Voice Profile:</strong> {' '}
              {moodOptions.find(m => m.value === voiceSettings.mood)?.label || 'Balanced'} mood, {' '}
              {energyLevels.find(e => e.value === voiceSettings.energy)?.label || 'Moderate'} energy, {' '}
              {intimacyLevels.find(i => i.value === voiceSettings.intimacy)?.label || 'Friendly'} intimacy
            </Typography>
          </Box>
        )}
      </CardContent>
    </Card>
  );
};

export default VoiceSettingsPanel;
