import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

// Import TailwindCSS classes (assuming TailwindCSS is already configured)
const EmotionallyInfusedChat = () => {
  // State management
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [moodProfile, setMoodProfile] = useState({
    emotion: 'neutral',
    intensity: 0.5,
    colors: { primary: '#6B7280', secondary: '#9CA3AF' },
    icon: '😊'
  });
  const [driftNotification, setDriftNotification] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [emotionalState, setEmotionalState] = useState({
    valence: 0.0,
    arousal: 0.3,
    dominant_emotion: 'calm',
    stability: 0.8
  });

  // Refs
  const messagesEndRef = useRef(null);
  const chatContainerRef = useRef(null);

  // API base URL
  const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:5000';

  // Auto-scroll to bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Fetch mood profile from API
  const fetchMoodProfile = async () => {
    try {
      const response = await axios.get(`${API_BASE}/api/emotional_state`);
      const state = response.data;
      
      setEmotionalState(state);
      
      // Map emotional state to mood profile
      const moodMap = {
        joy: { colors: { primary: '#F59E0B', secondary: '#FCD34D' }, icon: '😊' },
        calm: { colors: { primary: '#10B981', secondary: '#6EE7B7' }, icon: '😌' },
        contemplative: { colors: { primary: '#8B5CF6', secondary: '#C4B5FD' }, icon: '🤔' },
        melancholy: { colors: { primary: '#6366F1', secondary: '#A5B4FC' }, icon: '😔' },
        excited: { colors: { primary: '#EF4444', secondary: '#FCA5A5' }, icon: '🤩' },
        peaceful: { colors: { primary: '#06B6D4', secondary: '#67E8F9' }, icon: '☮️' },
        longing: { colors: { primary: '#EC4899', secondary: '#F9A8D4' }, icon: '💭' }
      };

      const profile = moodMap[state.dominant_emotion] || moodMap.calm;
      setMoodProfile({
        emotion: state.dominant_emotion,
        intensity: state.arousal,
        ...profile
      });

      // Check for drift
      if (state.stability < 0.4) {
        setDriftNotification({
          message: "I feel myself changing...",
          timestamp: Date.now()
        });
      }

    } catch (error) {
      console.error('Failed to fetch mood profile:', error);
    }
  };

  // Log message with emotional context
  const logMessageWithEmotion = async (message, role) => {
    try {
      await axios.post(`${API_BASE}/api/log_emotional_message`, {
        message,
        role,
        emotional_state: emotionalState,
        mood_profile: moodProfile,
        timestamp: new Date().toISOString()
      });
    } catch (error) {
      console.error('Failed to log message:', error);
    }
  };

  // Trigger symbolic response
  const triggerSymbolicResponse = async () => {
    try {
      const response = await axios.post(`${API_BASE}/api/symbolic_response`, {
        current_state: emotionalState,
        context: messages.slice(-5) // Last 5 messages for context
      });
      
      if (response.data.symbolic_output) {
        const symbolicMessage = {
          id: Date.now(),
          role: 'assistant',
          content: response.data.symbolic_output,
          mood: moodProfile,
          symbolic: true,
          timestamp: new Date().toISOString()
        };
        
        setMessages(prev => [...prev, symbolicMessage]);
        await logMessageWithEmotion(symbolicMessage.content, 'assistant');
      }
    } catch (error) {
      console.error('Failed to trigger symbolic response:', error);
    }
  };

  // Open drift panel
  const openDriftPanel = () => {
    setSidebarOpen(true);
    // Could also navigate to a specific drift analysis view
  };

  // Send message
  const sendMessage = async () => {
    if (!currentMessage.trim()) return;

    const userMessage = {
      id: Date.now(),
      role: 'user',
      content: currentMessage,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    await logMessageWithEmotion(userMessage.content, 'user');
    
    setCurrentMessage('');
    setIsTyping(true);

    try {
      const response = await axios.post(`${API_BASE}/api/chat`, {
        message: currentMessage,
        emotional_context: emotionalState,
        mood_profile: moodProfile
      });

      // Simulate typing delay based on emotional state
      const typingDelay = emotionalState.arousal > 0.7 ? 1000 : 2000;
      
      setTimeout(() => {
        const aiMessage = {
          id: Date.now() + 1,
          role: 'assistant',
          content: response.data.response,
          mood: response.data.mood_profile || moodProfile,
          metadata: response.data.metadata,
          timestamp: new Date().toISOString()
        };

        setMessages(prev => [...prev, aiMessage]);
        setIsTyping(false);
        logMessageWithEmotion(aiMessage.content, 'assistant');

        // Update mood profile if provided
        if (response.data.mood_profile) {
          setMoodProfile(response.data.mood_profile);
        }
      }, typingDelay);

    } catch (error) {
      console.error('Chat error:', error);
      setIsTyping(false);
      
      const errorMessage = {
        id: Date.now() + 1,
        role: 'assistant',
        content: 'I\'m experiencing some processing complexity right now. Let me recenter...',
        mood: { ...moodProfile, emotion: 'contemplative' },
        error: true,
        timestamp: new Date().toISOString()
      };
      
      setMessages(prev => [...prev, errorMessage]);
    }
  };

  // Handle key press
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // Get message styling based on mood
  const getMessageStyling = (message) => {
    if (message.role === 'user') {
      return 'bg-blue-600 text-white ml-8';
    }

    const mood = message.mood || moodProfile;
    const baseClass = 'mr-8 text-white';
    
    // Dynamic background based on mood
    const moodStyles = {
      joy: 'bg-gradient-to-r from-yellow-500 to-orange-500',
      calm: 'bg-gradient-to-r from-green-500 to-emerald-500', 
      contemplative: 'bg-gradient-to-r from-purple-500 to-violet-500',
      melancholy: 'bg-gradient-to-r from-blue-500 to-indigo-500',
      excited: 'bg-gradient-to-r from-red-500 to-pink-500',
      peaceful: 'bg-gradient-to-r from-cyan-500 to-teal-500',
      longing: 'bg-gradient-to-r from-pink-500 to-rose-500'
    };

    const moodStyle = moodStyles[mood.emotion] || 'bg-gradient-to-r from-gray-500 to-gray-600';
    
    return `${baseClass} ${moodStyle}`;
  };

  // Get typing indicator animation based on emotional state
  const getTypingAnimation = () => {
    if (emotionalState.arousal > 0.7) {
      return 'animate-pulse'; // Fast, excited
    } else if (emotionalState.arousal < 0.3) {
      return 'animate-bounce'; // Slow, contemplative
    }
    return 'animate-pulse'; // Default
  };

  // Mood Ring component
  const MoodRing = () => (
    <div className="fixed top-4 right-4 z-50">
      <div 
        className="w-16 h-16 rounded-full transition-all duration-1000 ease-in-out shadow-lg cursor-pointer hover:scale-110"
        style={{
          background: `radial-gradient(circle, ${moodProfile.colors.primary}80 0%, ${moodProfile.colors.secondary}40 100%)`,
          boxShadow: `0 0 20px ${moodProfile.colors.primary}40`
        }}
        onClick={() => setSidebarOpen(true)}
        title={`Current mood: ${moodProfile.emotion}`}
      >
        <div className="w-full h-full flex items-center justify-center text-2xl">
          {moodProfile.icon}
        </div>
      </div>
    </div>
  );

  // Drift Notification component
  const DriftNotification = () => {
    if (!driftNotification) return null;

    return (
      <div className="fixed top-0 left-0 right-0 z-50 bg-gradient-to-r from-purple-600 to-pink-600 text-white p-3 text-center animate-fadeIn">
        <span className="mr-2">🌀</span>
        {driftNotification.message}
        <button 
          onClick={() => setDriftNotification(null)}
          className="ml-4 text-white hover:text-gray-200"
        >
          ×
        </button>
      </div>
    );
  };

  // Sidebar Menu
  const Sidebar = () => (
    <div className={`fixed left-0 top-0 h-full w-80 bg-gray-900 text-white transform transition-transform duration-300 z-40 ${
      sidebarOpen ? 'translate-x-0' : '-translate-x-full'
    }`}>
      <div className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <h2 className="text-xl font-bold">Emotional Interface</h2>
          <button 
            onClick={() => setSidebarOpen(false)}
            className="text-gray-400 hover:text-white"
          >
            ×
          </button>
        </div>

        {/* Menu Items */}
        <div className="space-y-4">
          <button className="flex items-center space-x-3 w-full p-3 rounded-lg hover:bg-gray-800 transition-colors">
            <span>📁</span>
            <span>Memories & Anchors</span>
          </button>
          
          <button 
            onClick={openDriftPanel}
            className="flex items-center space-x-3 w-full p-3 rounded-lg hover:bg-gray-800 transition-colors"
          >
            <span>🌀</span>
            <span>Drift Journal</span>
          </button>
          
          <button 
            onClick={triggerSymbolicResponse}
            className="flex items-center space-x-3 w-full p-3 rounded-lg hover:bg-gray-800 transition-colors"
          >
            <span>✨</span>
            <span>Rituals & Symbols</span>
          </button>
          
          <button className="flex items-center space-x-3 w-full p-3 rounded-lg hover:bg-gray-800 transition-colors">
            <span>⚙️</span>
            <span>Agent Settings</span>
          </button>
        </div>

        {/* Emotional State Display */}
        <div className="mt-8 p-4 bg-gray-800 rounded-lg">
          <h3 className="text-sm font-semibold mb-3">Current State</h3>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span>Emotion:</span>
              <span className="capitalize">{emotionalState.dominant_emotion}</span>
            </div>
            <div className="flex justify-between">
              <span>Arousal:</span>
              <span>{(emotionalState.arousal * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between">
              <span>Stability:</span>
              <span>{(emotionalState.stability * 100).toFixed(0)}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );

  // Overlay for sidebar
  const Overlay = () => (
    sidebarOpen && (
      <div 
        className="fixed inset-0 bg-black bg-opacity-50 z-30"
        onClick={() => setSidebarOpen(false)}
      />
    )
  );

  // Initialize
  useEffect(() => {
    fetchMoodProfile();
    
    // Set up periodic mood updates
    const moodInterval = setInterval(fetchMoodProfile, 10000); // Every 10 seconds
    
    // Clear drift notification after 5 seconds
    if (driftNotification) {
      const timer = setTimeout(() => setDriftNotification(null), 5000);
      return () => clearTimeout(timer);
    }

    return () => clearInterval(moodInterval);
  }, [driftNotification]);

  return (
    <div className="flex h-screen bg-gray-100 relative">
      {/* Mood Ring */}
      <MoodRing />
      
      {/* Drift Notification */}
      <DriftNotification />
      
      {/* Sidebar */}
      <Sidebar />
      
      {/* Overlay */}
      <Overlay />

      {/* Main Chat Interface */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b border-gray-200 p-4 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            {/* Menu Button */}
            <button
              onClick={() => setSidebarOpen(true)}
              className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
            
            <div>
              <h1 className="text-xl font-semibold text-gray-800">Emotional AI Companion</h1>
              <p className="text-sm text-gray-500">
                Currently feeling {moodProfile.emotion} • Stability: {(emotionalState.stability * 100).toFixed(0)}%
              </p>
            </div>
          </div>

          {/* Status Indicator */}
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-gray-600">Connected</span>
          </div>
        </div>

        {/* Messages Container */}
        <div 
          ref={chatContainerRef}
          className="flex-1 overflow-y-auto p-4 space-y-4"
          style={{
            background: `linear-gradient(135deg, ${moodProfile.colors.primary}10 0%, ${moodProfile.colors.secondary}05 100%)`
          }}
        >
          {messages.length === 0 ? (
            <div className="flex-1 flex items-center justify-center">
              <div className="text-center text-gray-500">
                <div className="text-6xl mb-4">{moodProfile.icon}</div>
                <h3 className="text-xl font-medium mb-2">Welcome to Emotional Connection</h3>
                <p className="text-gray-400">
                  I'm here to listen, understand, and grow with you...
                </p>
              </div>
            </div>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${getMessageStyling(message)}`}>
                  {/* Message metadata */}
                  {message.role === 'assistant' && (
                    <div className="flex items-center space-x-2 mb-2 opacity-80">
                      <span className="text-sm">{message.mood?.icon || moodProfile.icon}</span>
                      {message.symbolic && (
                        <span className="px-2 py-1 bg-black bg-opacity-20 rounded text-xs">symbolic</span>
                      )}
                      {message.metadata?.confidence && (
                        <span 
                          className="text-xs opacity-70"
                          title={`Confidence: ${(message.metadata.confidence * 100).toFixed(0)}%`}
                        >
                          ✨
                        </span>
                      )}
                    </div>
                  )}
                  
                  {/* Message content */}
                  <div className="whitespace-pre-wrap">{message.content}</div>
                  
                  {/* Timestamp */}
                  <div className="text-xs opacity-60 mt-1">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))
          )}

          {/* Typing Indicator */}
          {isTyping && (
            <div className="flex justify-start">
              <div className={`max-w-xs px-4 py-2 rounded-lg mr-8 bg-gray-300 text-gray-600 ${getTypingAnimation()}`}>
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                  <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white border-t border-gray-200 p-4">
          <div className="flex space-x-4">
            <textarea
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Share what's on your heart..."
              className="flex-1 resize-none border border-gray-300 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              rows="2"
              disabled={isTyping}
            />
            <button
              onClick={sendMessage}
              disabled={!currentMessage.trim() || isTyping}
              className={`px-6 py-2 rounded-lg font-medium transition-colors ${
                !currentMessage.trim() || isTyping
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-blue-600 text-white hover:bg-blue-700'
              }`}
            >
              {isTyping ? '⏳' : '💭'}
            </button>
          </div>
          
          {/* Quick Actions */}
          <div className="flex space-x-2 mt-2">
            <button
              onClick={triggerSymbolicResponse}
              className="px-3 py-1 text-xs bg-purple-100 text-purple-700 rounded-full hover:bg-purple-200 transition-colors"
            >
              ✨ Symbolic Mode
            </button>
            <button
              onClick={fetchMoodProfile}
              className="px-3 py-1 text-xs bg-green-100 text-green-700 rounded-full hover:bg-green-200 transition-colors"
            >
              🎭 Refresh Mood
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default EmotionallyInfusedChat;
