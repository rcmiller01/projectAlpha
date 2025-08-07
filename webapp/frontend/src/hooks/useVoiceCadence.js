import { useEffect, useRef, useState, useCallback } from 'react';
import { useAppStore } from '../store';
import VoiceCadenceModulator from '../services/VoiceCadenceModulator';

/**
 * React hook for integrating VoiceCadenceModulator with the webapp
 * Provides voice modulation capabilities for agent responses
 */
export const useVoiceCadence = () => {
  const modulatorRef = useRef(null);
  const [isInitialized, setIsInitialized] = useState(false);
  const [currentVoiceProfile, setCurrentVoiceProfile] = useState('balanced');
  const { preferences } = useAppStore();

  // Initialize the VoiceCadenceModulator
  useEffect(() => {
    if (!modulatorRef.current) {
      modulatorRef.current = new VoiceCadenceModulator();
      setIsInitialized(true);
    }
  }, []);

  // Update voice settings based on app preferences
  useEffect(() => {
    if (modulatorRef.current && preferences.voiceSettings) {
      const { mood, energy, intimacy } = preferences.voiceSettings;
      updateVoiceProfile({ mood, energy, intimacy });
    }
  }, [preferences.voiceSettings]);

  /**
   * Generate voice parameters for agent response
   * @param {Object} context - Response context (agentType, emotion, urgency)
   * @param {string} text - Text to be spoken
   * @returns {Object} Voice modulation parameters
   */
  const generateVoiceParams = useCallback((context = {}, text = '') => {
    if (!modulatorRef.current) {
      return getDefaultVoiceParams();
    }

    const {
      agentType = 'general',
      emotion = 'neutral',
      urgency = 'medium',
      intimacy = 'medium'
    } = context;

    try {
      // Map agent types to voice characteristics
      const agentVoiceMap = {
        deduction: {
          mood: 'contemplative',
          energy: 'steady',
          tone: 'analytical'
        },
        metaphor: {
          mood: 'creative',
          energy: 'flowing',
          tone: 'melodic'
        },
        planner: {
          mood: 'organized',
          energy: 'controlled',
          tone: 'confident'
        },
        ritual: {
          mood: 'mystical',
          energy: 'ceremonial',
          tone: 'whispery'
        },
        general: {
          mood: 'balanced',
          energy: 'adaptive',
          tone: 'friendly'
        }
      };

      const voiceProfile = agentVoiceMap[agentType] || agentVoiceMap.general;
      
      // Generate modulation parameters
      const params = modulatorRef.current.modulateForContext({
        ...voiceProfile,
        emotion,
        urgency,
        intimacy,
        textLength: text.length,
        wordCount: text.split(' ').length
      });

      return {
        ...params,
        agentType,
        profile: voiceProfile
      };
    } catch (error) {
      console.error('Error generating voice parameters:', error);
      return getDefaultVoiceParams();
    }
  }, []);

  /**
   * Update voice profile for current session
   * @param {Object} profile - Voice profile settings
   */
  const updateVoiceProfile = useCallback((profile) => {
    if (modulatorRef.current) {
      try {
        modulatorRef.current.updateProfile(profile);
        setCurrentVoiceProfile(profile.mood || 'balanced');
      } catch (error) {
        console.error('Error updating voice profile:', error);
      }
    }
  }, []);

  /**
   * Get voice parameters for streaming text
   * @param {string} chunk - Text chunk being streamed
   * @param {Object} context - Streaming context
   * @returns {Object} Streaming voice parameters
   */
  const getStreamingParams = useCallback((chunk, context = {}) => {
    if (!modulatorRef.current) {
      return { tempo: 'steady', emphasis: 'medium' };
    }

    try {
      return modulatorRef.current.getStreamingModulation(chunk, context);
    } catch (error) {
      console.error('Error getting streaming parameters:', error);
      return { tempo: 'steady', emphasis: 'medium' };
    }
  }, []);

  /**
   * Apply emotion-based voice modulation
   * @param {string} emotion - Current emotion state
   * @param {number} intensity - Emotion intensity (0-1)
   * @returns {Object} Emotion-based voice parameters
   */
  const applyEmotionalModulation = useCallback((emotion, intensity = 0.5) => {
    if (!modulatorRef.current) {
      return getDefaultVoiceParams();
    }

    try {
      return modulatorRef.current.emotionalModulation(emotion, intensity);
    } catch (error) {
      console.error('Error applying emotional modulation:', error);
      return getDefaultVoiceParams();
    }
  }, []);

  /**
   * Get voice parameters for different message types
   * @param {string} messageType - Type of message (greeting, response, error, etc.)
   * @param {Object} context - Message context
   * @returns {Object} Message-specific voice parameters
   */
  const getMessageTypeParams = useCallback((messageType, context = {}) => {
    const messageVoiceMap = {
      greeting: {
        tempo: 'steady',
        warmth: 'high',
        emphasis: 'early',
        tone: 'welcoming'
      },
      response: {
        tempo: 'adaptive',
        clarity: 'high',
        emphasis: 'centered',
        tone: 'conversational'
      },
      error: {
        tempo: 'slow',
        clarity: 'high',
        emphasis: 'early',
        tone: 'apologetic'
      },
      thinking: {
        tempo: 'variable',
        pauses: 'thoughtful',
        emphasis: 'wave',
        tone: 'contemplative'
      },
      conclusion: {
        tempo: 'steady',
        finality: 'high',
        emphasis: 'late',
        tone: 'conclusive'
      }
    };

    return messageVoiceMap[messageType] || messageVoiceMap.response;
  }, []);

  // Default voice parameters fallback
  const getDefaultVoiceParams = () => ({
    tempo: 'steady',
    pauseFrequency: 'medium',
    emphasis: 'centered',
    tone: 'balanced',
    clarity: 'high'
  });

  return {
    isInitialized,
    currentVoiceProfile,
    generateVoiceParams,
    updateVoiceProfile,
    getStreamingParams,
    applyEmotionalModulation,
    getMessageTypeParams,
    modulator: modulatorRef.current
  };
};

/**
 * Voice settings context for the app
 */
export const useVoiceSettings = () => {
  const { preferences, setPreferences } = useAppStore();
  const voiceSettings = preferences.voiceSettings || {
    enabled: true,
    mood: 'balanced',
    energy: 'medium',
    intimacy: 'medium',
    agentSpecific: true
  };

  const updateVoiceSettings = useCallback((newSettings) => {
    setPreferences({
      voiceSettings: {
        ...voiceSettings,
        ...newSettings
      }
    });
  }, [voiceSettings, setPreferences]);

  const toggleVoiceEnabled = useCallback(() => {
    updateVoiceSettings({ enabled: !voiceSettings.enabled });
  }, [voiceSettings.enabled, updateVoiceSettings]);

  return {
    voiceSettings,
    updateVoiceSettings,
    toggleVoiceEnabled
  };
};
