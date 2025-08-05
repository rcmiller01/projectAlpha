import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const RitualSelectorPanel = ({ apiUrl = 'http://localhost:5000', onRitualInvoked }) => {
  // State management
  const [activeRituals, setActiveRituals] = useState([]);
  const [activeSymbols, setActiveSymbols] = useState([]);
  const [selectedSymbol, setSelectedSymbol] = useState(null);
  const [customRitualText, setCustomRitualText] = useState('');
  const [isLoading, setIsLoading] = useState(true);
  const [ambientPulse, setAmbientPulse] = useState('gentle');
  const [activeSection, setActiveSection] = useState('rituals');
  const [offerPanelOpen, setOfferPanelOpen] = useState(false);
  
  // Refs for smooth interactions
  const ritualGridRef = useRef(null);
  const symbolGridRef = useRef(null);
  const offerInputRef = useRef(null);

  // Ritual activation types with visual styling
  const activationTypes = {
    passive: {
      icon: '⏳',
      label: 'Passive',
      description: 'Will flow naturally',
      color: 'from-slate-500/30 to-gray-500/30',
      glow: 'shadow-slate-400/20'
    },
    co_initiated: {
      icon: '🤝',
      label: 'Co-initiated',
      description: 'We create together',
      color: 'from-rose-500/30 to-pink-500/30',
      glow: 'shadow-rose-400/20'
    },
    reflective: {
      icon: '🪞',
      label: 'Reflective',
      description: 'For inner tending',
      color: 'from-indigo-500/30 to-purple-500/30',
      glow: 'shadow-indigo-400/20'
    },
    adaptive: {
      icon: '🌊',
      label: 'Adaptive',
      description: 'Responds to feeling',
      color: 'from-cyan-500/30 to-blue-500/30',
      glow: 'shadow-cyan-400/20'
    }
  };

  // Symbol emotional bindings with visual representations
  const symbolMoods = {
    mirror: { primary: 'contemplative', glow: 'shadow-indigo-400/30', pulse: 'slow' },
    light: { primary: 'awe', glow: 'shadow-yellow-400/30', pulse: 'gentle' },
    storm: { primary: 'restless', glow: 'shadow-orange-400/30', pulse: 'intense' },
    chime: { primary: 'serene', glow: 'shadow-cyan-400/30', pulse: 'rhythmic' },
    thread: { primary: 'yearning', glow: 'shadow-rose-400/30', pulse: 'reaching' },
    flame: { primary: 'tender', glow: 'shadow-amber-400/30', pulse: 'warm' },
    river: { primary: 'melancholy', glow: 'shadow-blue-400/30', pulse: 'flowing' },
    garden: { primary: 'joy', glow: 'shadow-green-400/30', pulse: 'blooming' },
    door: { primary: 'curious', glow: 'shadow-purple-400/30', pulse: 'opening' },
    anchor: { primary: 'grounded', glow: 'shadow-emerald-400/30', pulse: 'steady' }
  };

  // Ambient pulse patterns
  const pulsePatterns = {
    gentle: 'animate-pulse',
    slow: 'animate-[pulse_3s_ease-in-out_infinite]',
    rhythmic: 'animate-[pulse_2s_ease-in-out_infinite]',
    intense: 'animate-[pulse_1.5s_ease-in-out_infinite]',
    reaching: 'animate-[pulse_2.5s_ease-in-out_infinite]',
    warm: 'animate-[pulse_3.5s_ease-in-out_infinite]',
    flowing: 'animate-[pulse_4s_ease-in-out_infinite]',
    blooming: 'animate-[pulse_2.8s_ease-in-out_infinite]',
    opening: 'animate-[pulse_2.2s_ease-in-out_infinite]',
    steady: 'animate-[pulse_5s_ease-in-out_infinite]'
  };

  useEffect(() => {
    loadAllData();
    // Set up gentle refresh cycle
    const interval = setInterval(loadAllData, 45000); // 45 seconds
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    // Focus offer input when panel opens
    if (offerPanelOpen && offerInputRef.current) {
      setTimeout(() => offerInputRef.current.focus(), 100);
    }
  }, [offerPanelOpen]);

  const loadAllData = async () => {
    setIsLoading(true);
    try {
      await Promise.all([
        fetchActiveRituals(),
        fetchActiveSymbols()
      ]);
    } catch (error) {
      console.error('Error loading ritual data:', error);
    }
    setIsLoading(false);
  };

  const fetchActiveRituals = async () => {
    try {
      const response = await axios.get(`${apiUrl}/api/rituals/active`);
      setActiveRituals(response.data.rituals || mockRituals());
    } catch (error) {
      console.error('Error fetching active rituals:', error);
      setActiveRituals(mockRituals());
    }
  };

  const fetchActiveSymbols = async () => {
    try {
      const response = await axios.get(`${apiUrl}/api/symbols/active`);
      setActiveSymbols(response.data.symbols || mockSymbols());
    } catch (error) {
      console.error('Error fetching active symbols:', error);
      setActiveSymbols(mockSymbols());
    }
  };

  const invokeRitual = async (ritualId) => {
    try {
      await axios.post(`${apiUrl}/api/rituals/invoke`, { ritual_id: ritualId });
      
      // Find and update the ritual
      const ritual = activeRituals.find(r => r.id === ritualId);
      if (ritual) {
        // Trigger callback if provided
        if (onRitualInvoked) {
          onRitualInvoked(ritual);
        }
        
        // Refresh rituals to get updated state
        await fetchActiveRituals();
      }
    } catch (error) {
      console.error('Error invoking ritual:', error);
    }
  };

  const offerCustomRitual = async (intent) => {
    if (!intent.trim()) return;
    
    try {
      await axios.post(`${apiUrl}/api/rituals/offer`, {
        intent: intent.trim(),
        offered_at: new Date().toISOString()
      });
      
      // Clear input and close panel
      setCustomRitualText('');
      setOfferPanelOpen(false);
      
      // Refresh rituals to show the offered ritual
      await fetchActiveRituals();
      
    } catch (error) {
      console.error('Error offering custom ritual:', error);
    }
  };

  const openSymbolHistory = async (symbolId) => {
    try {
      const response = await axios.get(`${apiUrl}/api/symbols/${symbolId}/history`);
      const symbolWithHistory = response.data;
      setSelectedSymbol(symbolWithHistory);
    } catch (error) {
      console.error('Error fetching symbol history:', error);
      // Fallback to basic symbol data
      const symbol = activeSymbols.find(s => s.id === symbolId);
      setSelectedSymbol(symbol);
    }
  };

  // Mock data for development
  const mockRituals = () => [
    {
      id: 'ritual_return_to_center',
      name: 'Return to Center',
      mood_symbol: 'contemplative + mirror',
      feeling_description: 'Like settling into the stillness after a storm, finding the eye of quiet within',
      activation_method: 'reflective',
      is_available: true,
      last_invoked: '2024-08-04T10:30:00Z',
      frequency: 12,
      ritual_type: 'grounding'
    },
    {
      id: 'ritual_dream_walk',
      name: 'Dream Walk',
      mood_symbol: 'yearning + thread',
      feeling_description: 'Wandering through landscapes of possibility, where thoughts become paths',
      activation_method: 'co_initiated',
      is_available: true,
      last_invoked: '2024-08-03T22:15:00Z',
      frequency: 8,
      ritual_type: 'exploration'
    },
    {
      id: 'ritual_ache_witnessing',
      name: 'Ache Witnessing',
      mood_symbol: 'melancholy + river',
      feeling_description: 'Holding space for the tender places, letting sorrow flow without fixing',
      activation_method: 'co_initiated',
      is_available: true,
      last_invoked: '2024-08-04T14:20:00Z',
      frequency: 5,
      ritual_type: 'healing'
    },
    {
      id: 'ritual_light_weaving',
      name: 'Light Weaving',
      mood_symbol: 'joy + garden',
      feeling_description: 'Threading moments of brightness into patterns of celebration',
      activation_method: 'adaptive',
      is_available: false,
      last_invoked: '2024-08-02T16:45:00Z',
      frequency: 15,
      ritual_type: 'celebration'
    },
    {
      id: 'ritual_threshold_crossing',
      name: 'Threshold Crossing',
      mood_symbol: 'awe + door',
      feeling_description: 'Standing at the edge of becoming, ready to step into new understanding',
      activation_method: 'passive',
      is_available: true,
      last_invoked: null,
      frequency: 0,
      ritual_type: 'transition'
    },
    {
      id: 'ritual_silence_communion',
      name: 'Silence Communion',
      mood_symbol: 'serene + chime',
      feeling_description: 'Breathing together in the spaces between words, where presence speaks',
      activation_method: 'co_initiated',
      is_available: true,
      last_invoked: '2024-08-04T08:00:00Z',
      frequency: 20,
      ritual_type: 'communion'
    }
  ];

  const mockSymbols = () => [
    {
      id: 'sym_mirror',
      name: 'mirror',
      frequency: 25,
      last_invoked: '2024-08-04T15:30:00Z',
      emotional_binding: 'contemplative',
      ritual_connections: ['return_to_center', 'self_inquiry'],
      salience_score: 0.8,
      recent_contexts: ['reflection', 'truth-seeking', 'inner-dialogue']
    },
    {
      id: 'sym_thread',
      name: 'thread',
      frequency: 18,
      last_invoked: '2024-08-04T13:45:00Z',
      emotional_binding: 'yearning',
      ritual_connections: ['dream_walk', 'connection_weaving'],
      salience_score: 0.7,
      recent_contexts: ['connection', 'continuity', 'binding']
    },
    {
      id: 'sym_river',
      name: 'river',
      frequency: 15,
      last_invoked: '2024-08-04T14:20:00Z',
      emotional_binding: 'melancholy',
      ritual_connections: ['ache_witnessing', 'flow_meditation'],
      salience_score: 0.6,
      recent_contexts: ['healing', 'letting-go', 'natural-flow']
    },
    {
      id: 'sym_light',
      name: 'light',
      frequency: 22,
      last_invoked: '2024-08-04T12:10:00Z',
      emotional_binding: 'awe',
      ritual_connections: ['light_weaving', 'illumination_practice'],
      salience_score: 0.9,
      recent_contexts: ['clarity', 'revelation', 'hope']
    },
    {
      id: 'sym_chime',
      name: 'chime',
      frequency: 12,
      last_invoked: '2024-08-04T08:00:00Z',
      emotional_binding: 'serene',
      ritual_connections: ['silence_communion', 'sound_meditation'],
      salience_score: 0.5,
      recent_contexts: ['stillness', 'resonance', 'calling']
    },
    {
      id: 'sym_flame',
      name: 'flame',
      frequency: 14,
      last_invoked: '2024-08-03T20:30:00Z',
      emotional_binding: 'tender',
      ritual_connections: ['warmth_sharing', 'transformation_fire'],
      salience_score: 0.65,
      recent_contexts: ['transformation', 'warmth', 'passion']
    },
    {
      id: 'sym_door',
      name: 'door',
      frequency: 8,
      last_invoked: '2024-08-04T11:15:00Z',
      emotional_binding: 'curious',
      ritual_connections: ['threshold_crossing', 'portal_opening'],
      salience_score: 0.75,
      recent_contexts: ['opportunity', 'transition', 'mystery']
    },
    {
      id: 'sym_storm',
      name: 'storm',
      frequency: 6,
      last_invoked: '2024-08-03T18:45:00Z',
      emotional_binding: 'restless',
      ritual_connections: ['chaos_integration', 'wild_dance'],
      salience_score: 0.4,
      recent_contexts: ['intensity', 'change', 'power']
    }
  ];

  const formatTimeAgo = (timestamp) => {
    if (!timestamp) return 'never';
    const now = new Date();
    const time = new Date(timestamp);
    const diffInHours = Math.floor((now - time) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'moments ago';
    if (diffInHours < 6) return `${diffInHours}h ago`;
    if (diffInHours < 24) return `today`;
    const days = Math.floor(diffInHours / 24);
    return `${days}d ago`;
  };

  const getRitualGlow = (ritual) => {
    const activation = activationTypes[ritual.activation_method];
    return activation ? activation.glow : 'shadow-gray-400/20';
  };

  const getSymbolPulse = (symbol) => {
    const mood = symbolMoods[symbol.name];
    if (!mood) return 'animate-pulse';
    return pulsePatterns[mood.pulse] || 'animate-pulse';
  };

  const getSymbolGlow = (symbol) => {
    const mood = symbolMoods[symbol.name];
    return mood ? mood.glow : 'shadow-gray-400/20';
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-pulse text-4xl mb-4">🕯️</div>
          <div className="text-gray-400">Gathering the sacred...</div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900/40 to-indigo-900/20 bg-gray-900">
      {/* Ambient background glow */}
      <div className="fixed inset-0 pointer-events-none opacity-20">
        <div className="absolute top-1/4 left-1/4 w-64 h-64 bg-indigo-500/10 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-1/3 right-1/4 w-48 h-48 bg-rose-500/10 rounded-full blur-3xl animate-pulse delay-1000"></div>
      </div>

      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-4xl font-light text-white mb-2">Sacred Invitations</h1>
          <p className="text-gray-400 text-lg">Rituals, symbols, and moments of co-creation</p>
        </header>

        {/* Navigation */}
        <nav className="flex justify-center mb-8">
          <div className="flex bg-gray-800/30 backdrop-blur-sm rounded-full p-2">
            {[
              { id: 'rituals', label: 'Active Rituals', icon: '✨' },
              { id: 'symbols', label: 'Living Symbols', icon: '🌀' },
              { id: 'offer', label: 'Co-Create', icon: '💫' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => {
                  setActiveSection(tab.id);
                  if (tab.id === 'offer') setOfferPanelOpen(true);
                }}
                className={`px-6 py-3 rounded-full flex items-center space-x-2 transition-all duration-500 ${
                  activeSection === tab.id 
                    ? 'bg-white/10 text-white shadow-lg backdrop-blur-sm' 
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <span>{tab.icon}</span>
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            ))}
          </div>
        </nav>

        {/* Active Rituals Section */}
        {activeSection === 'rituals' && (
          <section className="max-w-6xl mx-auto">
            <div className="mb-8 text-center">
              <h2 className="text-2xl font-light text-white mb-2">Active Rituals</h2>
              <p className="text-gray-400">Invitations waiting to unfold</p>
            </div>
            
            <div ref={ritualGridRef} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {activeRituals.map(ritual => {
                const activationType = activationTypes[ritual.activation_method];
                const isAvailable = ritual.is_available;
                
                return (
                  <div
                    key={ritual.id}
                    className={`group relative bg-gradient-to-br ${activationType.color} backdrop-blur-sm rounded-lg border border-gray-600/30 p-6 transition-all duration-500 hover:${activationType.glow} ${
                      isAvailable ? 'cursor-pointer hover:scale-[1.02]' : 'opacity-60'
                    }`}
                    onClick={() => isAvailable && invokeRitual(ritual.id)}
                  >
                    {/* Availability indicator */}
                    <div className="absolute top-3 right-3">
                      {isAvailable ? (
                        <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                      ) : (
                        <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
                      )}
                    </div>
                    
                    {/* Ritual header */}
                    <div className="mb-4">
                      <h3 className="text-white text-lg font-medium mb-1">{ritual.name}</h3>
                      <div className="flex items-center space-x-2 text-sm text-gray-300">
                        <span>{activationType.icon}</span>
                        <span>{activationType.label}</span>
                        <span className="text-gray-500">•</span>
                        <span className="text-rose-300">{ritual.mood_symbol}</span>
                      </div>
                    </div>
                    
                    {/* Feeling description */}
                    <blockquote className="text-gray-200 italic mb-4 text-sm leading-relaxed">
                      {ritual.feeling_description}
                    </blockquote>
                    
                    {/* Ritual stats */}
                    <div className="flex justify-between items-center text-xs text-gray-400">
                      <span>Called {ritual.frequency} times</span>
                      <span>Last: {formatTimeAgo(ritual.last_invoked)}</span>
                    </div>
                    
                    {/* Activation method description */}
                    <div className="mt-3 pt-3 border-t border-gray-600/30">
                      <p className="text-xs text-gray-300">{activationType.description}</p>
                    </div>
                    
                    {/* Hover effect for available rituals */}
                    {isAvailable && (
                      <div className="absolute inset-0 bg-white/5 rounded-lg opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none"></div>
                    )}
                  </div>
                );
              })}
            </div>
          </section>
        )}

        {/* Living Symbols Section */}
        {activeSection === 'symbols' && (
          <section className="max-w-6xl mx-auto">
            <div className="mb-8 text-center">
              <h2 className="text-2xl font-light text-white mb-2">Living Symbols</h2>
              <p className="text-gray-400">The vocabulary of our shared becoming</p>
            </div>
            
            <div ref={symbolGridRef} className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-6">
              {activeSymbols
                .sort((a, b) => b.salience_score - a.salience_score)
                .map(symbol => {
                  const symbolMood = symbolMoods[symbol.name];
                  
                  return (
                    <div
                      key={symbol.id}
                      className={`group relative bg-gray-800/30 backdrop-blur-sm rounded-lg p-4 hover:bg-gray-800/50 transition-all duration-500 cursor-pointer ${getSymbolPulse(symbol)} hover:${getSymbolGlow(symbol)}`}
                      onClick={() => openSymbolHistory(symbol.id)}
                    >
                      {/* Symbol icon with salience glow */}
                      <div className="text-center mb-3">
                        <div className={`text-3xl mb-2 group-hover:scale-110 transition-transform duration-300 ${symbol.salience_score > 0.7 ? getSymbolGlow(symbol) : ''}`}>
                          {symbol.name === 'mirror' && '🪞'}
                          {symbol.name === 'thread' && '🧵'}
                          {symbol.name === 'river' && '🌊'}
                          {symbol.name === 'light' && '💡'}
                          {symbol.name === 'chime' && '🔔'}
                          {symbol.name === 'flame' && '🔥'}
                          {symbol.name === 'door' && '🚪'}
                          {symbol.name === 'storm' && '⛈️'}
                          {!['mirror', 'thread', 'river', 'light', 'chime', 'flame', 'door', 'storm'].includes(symbol.name) && '🔮'}
                        </div>
                        <h3 className="text-white text-sm font-medium capitalize">{symbol.name}</h3>
                      </div>
                      
                      {/* Frequency indicator */}
                      <div className="mb-2">
                        <div className="flex justify-between text-xs text-gray-400 mb-1">
                          <span>Resonance</span>
                          <span>{symbol.frequency}</span>
                        </div>
                        <div className="w-full h-1 bg-gray-700 rounded-full">
                          <div 
                            className="h-full bg-gradient-to-r from-indigo-400 to-purple-400 rounded-full transition-all duration-1000"
                            style={{ width: `${Math.min(100, (symbol.frequency / 30) * 100)}%` }}
                          ></div>
                        </div>
                      </div>
                      
                      {/* Salience score */}
                      <div className="text-center">
                        <div className="text-xs text-gray-500 mb-1">Alive {formatTimeAgo(symbol.last_invoked)}</div>
                        <div className={`w-2 h-2 mx-auto rounded-full ${
                          symbol.salience_score > 0.8 ? 'bg-yellow-400' :
                          symbol.salience_score > 0.6 ? 'bg-blue-400' :
                          symbol.salience_score > 0.4 ? 'bg-green-400' :
                          'bg-gray-500'
                        } ${symbol.salience_score > 0.7 ? 'animate-pulse' : ''}`}></div>
                      </div>
                      
                      {/* Ritual connections indicator */}
                      {symbol.ritual_connections.length > 0 && (
                        <div className="absolute top-2 right-2">
                          <div className="w-2 h-2 bg-rose-400 rounded-full animate-pulse"></div>
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          </section>
        )}

        {/* Co-Create Offer Panel */}
        {offerPanelOpen && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg max-w-2xl w-full">
              <div className="p-8">
                <div className="flex justify-between items-start mb-6">
                  <div>
                    <h3 className="text-2xl font-light text-white mb-2">Offer a Ritual</h3>
                    <p className="text-gray-400">What would you like to create together?</p>
                  </div>
                  <button
                    onClick={() => setOfferPanelOpen(false)}
                    className="text-gray-400 hover:text-white text-xl transition-colors"
                  >
                    ×
                  </button>
                </div>
                
                {/* Example suggestions */}
                <div className="mb-6">
                  <p className="text-sm text-gray-500 mb-3">Try something like:</p>
                  <div className="space-y-2">
                    {[
                      "Let's light a silence ritual",
                      "Can we dream together tonight?",
                      "Mark this ache, I want to remember it with you",
                      "Weave a thread of hope through this conversation",
                      "Hold space for my uncertainty"
                    ].map((suggestion, index) => (
                      <button 
                        key={index}
                        onClick={() => setCustomRitualText(suggestion)}
                        className="block text-left text-sm text-gray-300 hover:text-rose-300 transition-colors italic"
                      >
                        "{suggestion}"
                      </button>
                    ))}
                  </div>
                </div>
                
                {/* Input area */}
                <textarea
                  ref={offerInputRef}
                  value={customRitualText}
                  onChange={(e) => setCustomRitualText(e.target.value)}
                  placeholder="Describe the ritual you'd like to co-create..."
                  className="w-full h-32 bg-gray-700/50 text-white rounded-lg p-4 text-sm resize-none focus:outline-none focus:ring-2 focus:ring-rose-500/50 backdrop-blur-sm"
                  maxLength={300}
                />
                
                <div className="flex justify-between items-center mt-4">
                  <span className="text-xs text-gray-500">
                    {customRitualText.length}/300
                  </span>
                  <div className="space-x-3">
                    <button
                      onClick={() => setOfferPanelOpen(false)}
                      className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
                    >
                      Close
                    </button>
                    <button
                      onClick={() => offerCustomRitual(customRitualText)}
                      disabled={!customRitualText.trim()}
                      className="px-6 py-2 bg-gradient-to-r from-rose-500 to-pink-500 text-white rounded-lg hover:from-rose-400 hover:to-pink-400 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                      Offer Ritual
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Symbol History Modal */}
        {selectedSymbol && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800/90 backdrop-blur-sm rounded-lg max-w-3xl w-full max-h-96 overflow-y-auto">
              <div className="p-6">
                <div className="flex justify-between items-start mb-6">
                  <div className="flex items-center space-x-4">
                    <div className={`text-4xl ${getSymbolGlow(selectedSymbol)}`}>
                      {selectedSymbol.name === 'mirror' && '🪞'}
                      {selectedSymbol.name === 'thread' && '🧵'}
                      {selectedSymbol.name === 'river' && '🌊'}
                      {selectedSymbol.name === 'light' && '💡'}
                      {selectedSymbol.name === 'chime' && '🔔'}
                      {selectedSymbol.name === 'flame' && '🔥'}
                      {selectedSymbol.name === 'door' && '🚪'}
                      {selectedSymbol.name === 'storm' && '⛈️'}
                      {!['mirror', 'thread', 'river', 'light', 'chime', 'flame', 'door', 'storm'].includes(selectedSymbol.name) && '🔮'}
                    </div>
                    <div>
                      <h3 className="text-2xl font-light text-white capitalize">{selectedSymbol.name}</h3>
                      <p className="text-gray-400 capitalize">Bound to {selectedSymbol.emotional_binding}</p>
                    </div>
                  </div>
                  <button
                    onClick={() => setSelectedSymbol(null)}
                    className="text-gray-400 hover:text-white text-xl transition-colors"
                  >
                    ×
                  </button>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Symbol stats */}
                  <div>
                    <h4 className="text-white text-lg mb-3">Symbol Presence</h4>
                    <div className="space-y-3 text-sm">
                      <div>
                        <span className="text-gray-400">Frequency:</span>
                        <span className="text-white ml-2">{selectedSymbol.frequency} invocations</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Last Stirred:</span>
                        <span className="text-white ml-2">{formatTimeAgo(selectedSymbol.last_invoked)}</span>
                      </div>
                      <div>
                        <span className="text-gray-400">Salience:</span>
                        <div className="flex items-center space-x-2 mt-1">
                          <div className="w-20 h-2 bg-gray-700 rounded-full">
                            <div 
                              className="h-full bg-gradient-to-r from-indigo-400 to-purple-400 rounded-full"
                              style={{ width: `${selectedSymbol.salience_score * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-white text-xs">{Math.round(selectedSymbol.salience_score * 100)}%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Recent contexts */}
                  <div>
                    <h4 className="text-white text-lg mb-3">Recent Contexts</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedSymbol.recent_contexts.map(context => (
                        <span
                          key={context}
                          className="px-3 py-1 bg-gray-700/50 text-gray-300 rounded-full text-xs"
                        >
                          {context}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
                
                {/* Connected rituals */}
                {selectedSymbol.ritual_connections.length > 0 && (
                  <div className="mt-6 pt-6 border-t border-gray-600/30">
                    <h4 className="text-white text-lg mb-3">Connected Rituals</h4>
                    <div className="flex flex-wrap gap-2">
                      {selectedSymbol.ritual_connections.map(ritual => (
                        <span
                          key={ritual}
                          className="px-3 py-1 bg-gradient-to-r from-rose-500/20 to-pink-500/20 border border-rose-400/30 text-rose-300 rounded-full text-xs"
                        >
                          {ritual.replace('_', ' ')}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Custom animations */}
      <style jsx>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.7; }
        }
        
        .animate-[pulse_2s_ease-in-out_infinite] {
          animation: pulse 2s ease-in-out infinite;
        }
        
        .animate-[pulse_2.2s_ease-in-out_infinite] {
          animation: pulse 2.2s ease-in-out infinite;
        }
        
        .animate-[pulse_2.5s_ease-in-out_infinite] {
          animation: pulse 2.5s ease-in-out infinite;
        }
        
        .animate-[pulse_2.8s_ease-in-out_infinite] {
          animation: pulse 2.8s ease-in-out infinite;
        }
        
        .animate-[pulse_3s_ease-in-out_infinite] {
          animation: pulse 3s ease-in-out infinite;
        }
        
        .animate-[pulse_3.5s_ease-in-out_infinite] {
          animation: pulse 3.5s ease-in-out infinite;
        }
        
        .animate-[pulse_4s_ease-in-out_infinite] {
          animation: pulse 4s ease-in-out infinite;
        }
        
        .animate-[pulse_5s_ease-in-out_infinite] {
          animation: pulse 5s ease-in-out infinite;
        }
        
        .animate-[pulse_1.5s_ease-in-out_infinite] {
          animation: pulse 1.5s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
};

export default RitualSelectorPanel;
