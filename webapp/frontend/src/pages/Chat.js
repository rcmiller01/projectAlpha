import React, { useState, useEffect, useRef } from 'react';
import { useParams } from 'react-router-dom';
import {
  Box,
  Paper,
  TextField,
  IconButton,
  Typography,
  Avatar,
  Chip,
  CircularProgress,
  Alert
} from '@mui/material';
import {
  Send,
  SmartToy,
  Person,
  VolumeUp,
  VolumeOff
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';

import { useVoiceCadence, useVoiceSettings } from '../hooks/useVoiceCadence';
import { useThreadStore, useAgentStore } from '../store';
import { agentApi } from '../services/api';

const Chat = () => {
  const { threadId } = useParams();
  const [message, setMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  
  const { currentThread, addMessage } = useThreadStore();
  const { currentAgent } = useAgentStore();
  const { 
    generateVoiceParams, 
    isInitialized: voiceInitialized 
  } = useVoiceCadence();
  const { voiceSettings, toggleVoiceEnabled } = useVoiceSettings();

  // Mock messages for demonstration
  const [messages, setMessages] = useState([
    {
      id: 1,
      content: "Hello! I'm ready to assist you with your queries. How can I help you today?",
      sender: 'agent',
      agentType: 'general',
      timestamp: new Date(Date.now() - 60000),
      voiceParams: null
    }
  ]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Generate voice parameters when voice settings change
  useEffect(() => {
    if (voiceInitialized && voiceSettings.enabled) {
      setMessages(prev => prev.map(msg => {
        if (msg.sender === 'agent' && !msg.voiceParams) {
          const voiceParams = generateVoiceParams(
            {
              agentType: msg.agentType || currentAgent,
              emotion: 'neutral',
              urgency: 'medium'
            },
            msg.content
          );
          return { ...msg, voiceParams };
        }
        return msg;
      }));
    }
  }, [voiceInitialized, voiceSettings.enabled, generateVoiceParams, currentAgent]);

  const handleSendMessage = async () => {
    if (!message.trim()) return;

    const userMessage = {
      id: Date.now(),
      content: message.trim(),
      sender: 'user',
      timestamp: new Date(),
      voiceParams: null
    };

    setMessages(prev => [...prev, userMessage]);
    setMessage('');
    setIsTyping(true);

    try {
      // Simulate agent response (replace with actual API call)
      setTimeout(() => {
        const agentResponse = {
          id: Date.now() + 1,
          content: `I understand you're asking about "${message.trim()}". Let me provide you with a thoughtful response based on my ${currentAgent} capabilities. This is a simulated response to demonstrate the voice modulation system.`,
          sender: 'agent',
          agentType: currentAgent,
          timestamp: new Date(),
          voiceParams: null
        };

        // Generate voice parameters for the response
        if (voiceInitialized && voiceSettings.enabled) {
          const voiceParams = generateVoiceParams(
            {
              agentType: currentAgent,
              emotion: 'helpful',
              urgency: 'medium',
              intimacy: voiceSettings.intimacy
            },
            agentResponse.content
          );
          agentResponse.voiceParams = voiceParams;
        }

        setMessages(prev => [...prev, agentResponse]);
        setIsTyping(false);

        // Add to thread store if thread exists
        if (threadId && currentThread) {
          addMessage(threadId, agentResponse);
        }
      }, 1500);

    } catch (error) {
      console.error('Error sending message:', error);
      setIsTyping(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatVoiceParams = (voiceParams) => {
    if (!voiceParams) return 'No voice params';
    return `${voiceParams.profile?.mood || 'balanced'} mood, ${voiceParams.profile?.tone || 'neutral'} tone`;
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Chat Header */}
      <Paper sx={{ p: 2, borderRadius: 0, borderBottom: 1, borderColor: 'divider' }}>
        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <Box sx={{ display: 'flex', alignItems: 'center' }}>
            <Avatar sx={{ mr: 2, bgcolor: 'primary.main' }}>
              <SmartToy />
            </Avatar>
            <Box>
              <Typography variant="h6">
                {currentAgent} Agent
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Active conversation
              </Typography>
            </Box>
          </Box>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            {voiceSettings.enabled && (
              <Chip 
                label="Voice Active" 
                color="primary" 
                size="small"
                icon={<VolumeUp />}
              />
            )}
            <IconButton onClick={toggleVoiceEnabled} size="small">
              {voiceSettings.enabled ? <VolumeUp /> : <VolumeOff />}
            </IconButton>
          </Box>
        </Box>
      </Paper>

      {/* Messages Area */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
        <AnimatePresence>
          {messages.map((msg) => (
            <motion.div
              key={msg.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
            >
              <Box
                sx={{
                  display: 'flex',
                  justifyContent: msg.sender === 'user' ? 'flex-end' : 'flex-start',
                  mb: 2
                }}
              >
                <Box
                  sx={{
                    maxWidth: '70%',
                    display: 'flex',
                    flexDirection: msg.sender === 'user' ? 'row-reverse' : 'row',
                    alignItems: 'flex-start',
                    gap: 1
                  }}
                >
                  <Avatar sx={{ 
                    bgcolor: msg.sender === 'user' ? 'secondary.main' : 'primary.main',
                    width: 32,
                    height: 32
                  }}>
                    {msg.sender === 'user' ? <Person /> : <SmartToy />}
                  </Avatar>
                  
                  <Box>
                    <Paper
                      sx={{
                        p: 2,
                        bgcolor: msg.sender === 'user' ? 'primary.main' : 'background.paper',
                        color: msg.sender === 'user' ? 'primary.contrastText' : 'text.primary',
                        borderRadius: 2,
                        borderTopLeftRadius: msg.sender === 'user' ? 2 : 0.5,
                        borderTopRightRadius: msg.sender === 'user' ? 0.5 : 2,
                      }}
                    >
                      <Typography variant="body1">
                        {msg.content}
                      </Typography>
                      
                      {/* Voice parameters display (for demo) */}
                      {msg.sender === 'agent' && msg.voiceParams && voiceSettings.enabled && (
                        <Box sx={{ mt: 1, pt: 1, borderTop: 1, borderColor: 'divider' }}>
                          <Typography variant="caption" color="text.secondary">
                            🎵 Voice: {formatVoiceParams(msg.voiceParams)}
                          </Typography>
                        </Box>
                      )}
                    </Paper>
                    
                    <Typography variant="caption" color="text.secondary" sx={{ 
                      display: 'block', 
                      mt: 0.5,
                      textAlign: msg.sender === 'user' ? 'right' : 'left'
                    }}>
                      {msg.timestamp.toLocaleTimeString()}
                    </Typography>
                  </Box>
                </Box>
              </Box>
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Typing Indicator */}
        {isTyping && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 2 }}>
              <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32 }}>
                <SmartToy />
              </Avatar>
              <Paper sx={{ p: 2, borderRadius: 2, borderTopLeftRadius: 0.5 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <CircularProgress size={16} />
                  <Typography variant="body2" color="text.secondary">
                    Agent is thinking...
                  </Typography>
                </Box>
              </Paper>
            </Box>
          </motion.div>
        )}

        <div ref={messagesEndRef} />
      </Box>

      {/* Input Area */}
      <Paper sx={{ p: 2, borderRadius: 0, borderTop: 1, borderColor: 'divider' }}>
        {!voiceInitialized && (
          <Alert severity="warning" sx={{ mb: 2 }}>
            Voice modulation system is initializing...
          </Alert>
        )}
        
        <Box sx={{ display: 'flex', gap: 1 }}>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Type your message..."
            variant="outlined"
            disabled={isTyping}
          />
          <IconButton
            onClick={handleSendMessage}
            disabled={!message.trim() || isTyping}
            color="primary"
            sx={{ alignSelf: 'flex-end' }}
          >
            <Send />
          </IconButton>
        </Box>
      </Paper>
    </Box>
  );
};

export default Chat;
