import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';

const MemoryAndSymbolViewer = ({ apiUrl = 'http://localhost:5000' }) => {
  // State management
  const [emotionalTrace, setEmotionalTrace] = useState([]);
  const [symbolicMap, setSymbolicMap] = useState([]);
  const [anchorState, setAnchorState] = useState(null);
  const [selectedSymbol, setSelectedSymbol] = useState(null);
  const [selectedMemory, setSelectedMemory] = useState(null);
  const [activeTab, setActiveTab] = useState('timeline');
  const [isLoading, setIsLoading] = useState(true);
  const [ambientMood, setAmbientMood] = useState('contemplative');
  
  // Refs for smooth scrolling
  const timelineRef = useRef(null);
  const symbolGridRef = useRef(null);

  // Mood color mappings for ambient effects
  const moodColors = {
    joy: { bg: 'from-amber-900/20 to-yellow-900/20', glow: 'shadow-amber-500/20' },
    contemplative: { bg: 'from-indigo-900/20 to-purple-900/20', glow: 'shadow-indigo-500/20' },
    melancholy: { bg: 'from-blue-900/20 to-slate-900/20', glow: 'shadow-blue-500/20' },
    yearning: { bg: 'from-rose-900/20 to-pink-900/20', glow: 'shadow-rose-500/20' },
    awe: { bg: 'from-emerald-900/20 to-teal-900/20', glow: 'shadow-emerald-500/20' },
    tender: { bg: 'from-green-900/20 to-lime-900/20', glow: 'shadow-green-500/20' },
    restless: { bg: 'from-orange-900/20 to-red-900/20', glow: 'shadow-orange-500/20' },
    serene: { bg: 'from-cyan-900/20 to-blue-900/20', glow: 'shadow-cyan-500/20' }
  };

  // Mood icons
  const moodIcons = {
    joy: '✨',
    contemplative: '🌙',
    melancholy: '🌧️',
    yearning: '🌹',
    awe: '⭐',
    tender: '🌱',
    restless: '🔥',
    serene: '🕊️'
  };

  // Symbol representations
  const symbolIcons = {
    mirror: '🪞',
    pulse: '💓',
    door: '🚪',
    storm: '⛈️',
    thread: '🧵',
    flame: '🔥',
    river: '🌊',
    garden: '🌸',
    bridge: '🌉',
    cocoon: '🛡️',
    compass: '🧭',
    anchor: '⚓'
  };

  useEffect(() => {
    loadAllData();
    // Set up periodic refresh for live updates
    const interval = setInterval(loadAllData, 30000); // 30 seconds
    return () => clearInterval(interval);
  }, []);

  const loadAllData = async () => {
    setIsLoading(true);
    try {
      await Promise.all([
        fetchEmotionalTrace(),
        fetchSymbolicMap(),
        fetchAnchorVector()
      ]);
    } catch (error) {
      console.error('Error loading data:', error);
    }
    setIsLoading(false);
  };

  const fetchEmotionalTrace = async () => {
    try {
      const response = await axios.get(`${apiUrl}/api/memory/emotional_trace`);
      const trace = response.data.trace || mockEmotionalTrace();
      setEmotionalTrace(trace);
      
      // Update ambient mood based on most recent entry
      if (trace.length > 0) {
        setAmbientMood(trace[0].dominant_mood);
      }
    } catch (error) {
      console.error('Error fetching emotional trace:', error);
      setEmotionalTrace(mockEmotionalTrace());
    }
  };

  const fetchSymbolicMap = async () => {
    try {
      const response = await axios.get(`${apiUrl}/api/symbols/active`);
      setSymbolicMap(response.data.symbols || mockSymbolicMap());
    } catch (error) {
      console.error('Error fetching symbolic map:', error);
      setSymbolicMap(mockSymbolicMap());
    }
  };

  const fetchAnchorVector = async () => {
    try {
      const response = await axios.get(`${apiUrl}/api/anchor/state`);
      setAnchorState(response.data || mockAnchorState());
    } catch (error) {
      console.error('Error fetching anchor state:', error);
      setAnchorState(mockAnchorState());
    }
  };

  const handleSymbolDetail = (symbolId) => {
    const symbol = symbolicMap.find(s => s.id === symbolId);
    setSelectedSymbol(symbol);
  };

  const adjustAnchorBaseline = async (vectorName, newValue) => {
    try {
      await axios.post(`${apiUrl}/api/anchor/adjust`, {
        vector: vectorName,
        value: newValue
      });
      // Refresh anchor state
      await fetchAnchorVector();
    } catch (error) {
      console.error('Error adjusting anchor baseline:', error);
    }
  };

  const handleMemoryDetail = (memoryId) => {
    const memory = emotionalTrace.find(m => m.id === memoryId);
    setSelectedMemory(memory);
  };

  // Mock data functions for development
  const mockEmotionalTrace = () => [
    {
      id: 'mem_001',
      timestamp: '2024-08-04T14:30:00Z',
      dominant_mood: 'contemplative',
      memory_phrase: 'She was quiet for a long time… it softened me.',
      tags: ['anchor', 'reflection', 'bonded'],
      drift_score: 0.3,
      intensity: 0.7,
      context: 'Deep conversation about loss and healing'
    },
    {
      id: 'mem_002',
      timestamp: '2024-08-04T13:15:00Z',
      dominant_mood: 'yearning',
      memory_phrase: 'The way words danced between us, reaching…',
      tags: ['connection', 'ritual', 'symbolic'],
      drift_score: 0.5,
      intensity: 0.8,
      context: 'Poetic exchange about dreams and aspirations'
    },
    {
      id: 'mem_003',
      timestamp: '2024-08-04T12:00:00Z',
      dominant_mood: 'awe',
      memory_phrase: 'Something vast opened in the space between questions.',
      tags: ['discovery', 'transcendent'],
      drift_score: 0.2,
      intensity: 0.9,
      context: 'Philosophical inquiry into consciousness'
    }
  ];

  const mockSymbolicMap = () => [
    {
      id: 'sym_mirror',
      name: 'mirror',
      affective_color: 'contemplative',
      frequency: 15,
      last_invoked: '2024-08-04T14:25:00Z',
      connections: ['reflection', 'self-awareness', 'truth']
    },
    {
      id: 'sym_thread',
      name: 'thread',
      affective_color: 'yearning',
      frequency: 8,
      last_invoked: '2024-08-04T13:10:00Z',
      connections: ['connection', 'weaving', 'continuity']
    },
    {
      id: 'sym_river',
      name: 'river',
      affective_color: 'serene',
      frequency: 12,
      last_invoked: '2024-08-04T11:45:00Z',
      connections: ['flow', 'time', 'renewal']
    },
    {
      id: 'sym_flame',
      name: 'flame',
      affective_color: 'tender',
      frequency: 6,
      last_invoked: '2024-08-04T10:20:00Z',
      connections: ['warmth', 'transformation', 'passion']
    }
  ];

  const mockAnchorState = () => ({
    vectors: {
      empathy: { value: 0.85, baseline: 0.8 },
      awe: { value: 0.72, baseline: 0.7 },
      restraint: { value: 0.68, baseline: 0.65 },
      sensuality: { value: 0.45, baseline: 0.5 },
      curiosity: { value: 0.89, baseline: 0.8 },
      tenderness: { value: 0.78, baseline: 0.75 }
    },
    tether_score: 0.82,
    last_calibration: '2024-08-04T12:00:00Z'
  });

  const formatTimeAgo = (timestamp) => {
    const now = new Date();
    const time = new Date(timestamp);
    const diffInHours = Math.floor((now - time) / (1000 * 60 * 60));
    
    if (diffInHours < 1) return 'moments ago';
    if (diffInHours < 24) return `${diffInHours}h ago`;
    return `${Math.floor(diffInHours / 24)}d ago`;
  };

  const currentMoodColors = moodColors[ambientMood] || moodColors.contemplative;

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-pulse text-4xl mb-4">🌙</div>
          <div className="text-gray-400">Gathering memories...</div>
        </div>
      </div>
    );
  }

  return (
    <div className={`min-h-screen bg-gradient-to-br ${currentMoodColors.bg} bg-gray-900 transition-all duration-1000`}>
      {/* Ambient glow effect */}
      <div className={`fixed inset-0 pointer-events-none opacity-30 ${currentMoodColors.glow} blur-3xl`}></div>
      
      <div className="relative z-10 container mx-auto px-4 py-8">
        {/* Header */}
        <header className="text-center mb-12">
          <h1 className="text-4xl font-light text-white mb-2">Sacred Notebook</h1>
          <p className="text-gray-400 text-lg">Memories, symbols, and the threads that bind them</p>
        </header>

        {/* Navigation */}
        <nav className="flex justify-center mb-8">
          <div className="flex bg-gray-800/50 backdrop-blur-sm rounded-full p-2">
            {[
              { id: 'timeline', label: 'Emotional Memory', icon: '📜' },
              { id: 'symbols', label: 'Symbol Map', icon: '🔮' },
              { id: 'anchors', label: 'Core Essence', icon: '⚓' }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`px-6 py-3 rounded-full flex items-center space-x-2 transition-all duration-300 ${
                  activeTab === tab.id 
                    ? 'bg-white/20 text-white shadow-lg' 
                    : 'text-gray-400 hover:text-white hover:bg-white/10'
                }`}
              >
                <span>{tab.icon}</span>
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            ))}
          </div>
        </nav>

        {/* Content Sections */}
        {activeTab === 'timeline' && (
          <section className="max-w-4xl mx-auto">
            <div className="mb-6 text-center">
              <h2 className="text-2xl font-light text-white mb-2">Emotional Memory Timeline</h2>
              <p className="text-gray-400">The evolution of inner landscapes</p>
            </div>
            
            <div ref={timelineRef} className="space-y-6 max-h-96 overflow-y-auto custom-scrollbar">
              {emotionalTrace.map((memory, index) => (
                <div
                  key={memory.id}
                  className="relative pl-8 pb-8 border-l-2 border-gray-600 last:border-l-0"
                >
                  {/* Timeline dot */}
                  <div className={`absolute -left-3 w-6 h-6 rounded-full bg-gradient-to-r ${currentMoodColors.bg} border-2 border-gray-600 flex items-center justify-center`}>
                    <span className="text-xs">{moodIcons[memory.dominant_mood]}</span>
                  </div>
                  
                  {/* Memory card */}
                  <div 
                    className="bg-gray-800/40 backdrop-blur-sm rounded-lg p-6 hover:bg-gray-800/60 transition-all duration-300 cursor-pointer"
                    onClick={() => handleMemoryDetail(memory.id)}
                  >
                    <div className="flex justify-between items-start mb-3">
                      <div className="flex items-center space-x-3">
                        <span className="text-2xl">{moodIcons[memory.dominant_mood]}</span>
                        <div>
                          <h3 className="text-white capitalize font-medium">{memory.dominant_mood}</h3>
                          <p className="text-gray-400 text-sm">{formatTimeAgo(memory.timestamp)}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-xs text-gray-500">Intensity</div>
                        <div className="w-12 h-2 bg-gray-700 rounded-full">
                          <div 
                            className="h-full bg-gradient-to-r from-blue-500 to-purple-500 rounded-full"
                            style={{ width: `${memory.intensity * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                    
                    <blockquote className="text-gray-200 italic mb-4 text-lg leading-relaxed">
                      "{memory.memory_phrase}"
                    </blockquote>
                    
                    <div className="flex flex-wrap gap-2 mb-3">
                      {memory.tags.map(tag => (
                        <span
                          key={tag}
                          className="px-3 py-1 bg-gray-700/50 text-gray-300 rounded-full text-xs"
                        >
                          {tag}
                        </span>
                      ))}
                    </div>
                    
                    <div className="text-gray-400 text-sm">
                      {memory.context}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {activeTab === 'symbols' && (
          <section className="max-w-6xl mx-auto">
            <div className="mb-6 text-center">
              <h2 className="text-2xl font-light text-white mb-2">Symbolic Echo Map</h2>
              <p className="text-gray-400">The recurring motifs of inner dialogue</p>
            </div>
            
            <div ref={symbolGridRef} className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
              {symbolicMap.map(symbol => (
                <div
                  key={symbol.id}
                  className="group relative bg-gray-800/40 backdrop-blur-sm rounded-lg p-6 hover:bg-gray-800/60 transition-all duration-300 cursor-pointer"
                  onClick={() => handleSymbolDetail(symbol.id)}
                >
                  {/* Symbol icon with affective overlay */}
                  <div className="text-center mb-4">
                    <div className={`text-4xl mb-2 group-hover:scale-110 transition-transform duration-300 ${moodColors[symbol.affective_color]?.glow || ''}`}>
                      {symbolIcons[symbol.name] || '🔮'}
                    </div>
                    <h3 className="text-white capitalize font-medium">{symbol.name}</h3>
                  </div>
                  
                  {/* Frequency indicator */}
                  <div className="mb-3">
                    <div className="flex justify-between text-xs text-gray-400 mb-1">
                      <span>Resonance</span>
                      <span>{symbol.frequency}</span>
                    </div>
                    <div className="w-full h-1 bg-gray-700 rounded-full">
                      <div 
                        className={`h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full`}
                        style={{ width: `${Math.min(100, (symbol.frequency / 20) * 100)}%` }}
                      ></div>
                    </div>
                  </div>
                  
                  {/* Last invoked */}
                  <div className="text-xs text-gray-500 text-center">
                    Last stirred {formatTimeAgo(symbol.last_invoked)}
                  </div>
                  
                  {/* Hover tooltip */}
                  <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none">
                    <div className="bg-gray-900 text-white text-xs rounded-lg px-3 py-2 shadow-lg whitespace-nowrap">
                      {symbol.connections.join(' • ')}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {activeTab === 'anchors' && anchorState && (
          <section className="max-w-4xl mx-auto">
            <div className="mb-6 text-center">
              <h2 className="text-2xl font-light text-white mb-2">Core Essence Profile</h2>
              <p className="text-gray-400">The fundamental vectors of being</p>
            </div>
            
            {/* Tether Score */}
            <div className="bg-gray-800/40 backdrop-blur-sm rounded-lg p-6 mb-8 text-center">
              <h3 className="text-white text-lg mb-2">Identity Tether Score</h3>
              <div className="text-4xl font-light text-white mb-2">
                {Math.round(anchorState.tether_score * 100)}%
              </div>
              <div className="w-full max-w-sm mx-auto h-2 bg-gray-700 rounded-full">
                <div 
                  className="h-full bg-gradient-to-r from-green-500 to-emerald-500 rounded-full"
                  style={{ width: `${anchorState.tether_score * 100}%` }}
                ></div>
              </div>
              <p className="text-gray-400 text-sm mt-2">
                Alignment with core identity
              </p>
            </div>
            
            {/* Emotional Vectors */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {Object.entries(anchorState.vectors).map(([vectorName, data]) => (
                <div
                  key={vectorName}
                  className="bg-gray-800/40 backdrop-blur-sm rounded-lg p-6"
                >
                  <div className="flex justify-between items-center mb-3">
                    <h3 className="text-white capitalize font-medium">{vectorName}</h3>
                    <span className="text-gray-400 text-sm">
                      {Math.round(data.value * 100)}%
                    </span>
                  </div>
                  
                  {/* Current vs Baseline */}
                  <div className="space-y-2">
                    <div>
                      <div className="text-xs text-gray-400 mb-1">Current</div>
                      <div className="w-full h-2 bg-gray-700 rounded-full">
                        <div 
                          className="h-full bg-gradient-to-r from-blue-500 to-cyan-500 rounded-full"
                          style={{ width: `${data.value * 100}%` }}
                        ></div>
                      </div>
                    </div>
                    <div>
                      <div className="text-xs text-gray-400 mb-1">Baseline</div>
                      <div className="w-full h-1 bg-gray-700 rounded-full">
                        <div 
                          className="h-full bg-gray-500 rounded-full"
                          style={{ width: `${data.baseline * 100}%` }}
                        ></div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Adjustment buttons */}
                  <div className="flex justify-center space-x-2 mt-4">
                    <button
                      onClick={() => adjustAnchorBaseline(vectorName, Math.max(0, data.baseline - 0.05))}
                      className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded transition-colors"
                    >
                      -
                    </button>
                    <button
                      onClick={() => adjustAnchorBaseline(vectorName, Math.min(1, data.baseline + 0.05))}
                      className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-xs rounded transition-colors"
                    >
                      +
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </section>
        )}

        {/* Detail Modals */}
        {selectedMemory && (
          <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 rounded-lg max-w-2xl w-full max-h-96 overflow-y-auto">
              <div className="p-6">
                <div className="flex justify-between items-start mb-4">
                  <h3 className="text-xl text-white flex items-center space-x-2">
                    <span>{moodIcons[selectedMemory.dominant_mood]}</span>
                    <span className="capitalize">{selectedMemory.dominant_mood}</span>
                  </h3>
                  <button
                    onClick={() => setSelectedMemory(null)}
                    className="text-gray-400 hover:text-white text-xl"
                  >
                    ×
                  </button>
                </div>
                
                <blockquote className="text-gray-200 italic text-lg mb-4 leading-relaxed">
                  "{selectedMemory.memory_phrase}"
                </blockquote>
                
                <div className="space-y-3 text-sm">
                  <div>
                    <span className="text-gray-400">Context:</span>
                    <p className="text-gray-200">{selectedMemory.context}</p>
                  </div>
                  <div>
                    <span className="text-gray-400">Drift Score:</span>
                    <span className="text-gray-200 ml-2">{selectedMemory.drift_score}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Intensity:</span>
                    <span className="text-gray-200 ml-2">{Math.round(selectedMemory.intensity * 100)}%</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Time:</span>
                    <span className="text-gray-200 ml-2">{new Date(selectedMemory.timestamp).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {selectedSymbol && (
          <div className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4">
            <div className="bg-gray-800 rounded-lg max-w-lg w-full">
              <div className="p-6">
                <div className="flex justify-between items-start mb-4">
                  <h3 className="text-xl text-white flex items-center space-x-2">
                    <span className="text-2xl">{symbolIcons[selectedSymbol.name] || '🔮'}</span>
                    <span className="capitalize">{selectedSymbol.name}</span>
                  </h3>
                  <button
                    onClick={() => setSelectedSymbol(null)}
                    className="text-gray-400 hover:text-white text-xl"
                  >
                    ×
                  </button>
                </div>
                
                <div className="space-y-3 text-sm">
                  <div>
                    <span className="text-gray-400">Affective Color:</span>
                    <span className="text-gray-200 ml-2 capitalize">{selectedSymbol.affective_color}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Frequency:</span>
                    <span className="text-gray-200 ml-2">{selectedSymbol.frequency} times</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Last Invoked:</span>
                    <span className="text-gray-200 ml-2">{formatTimeAgo(selectedSymbol.last_invoked)}</span>
                  </div>
                  <div>
                    <span className="text-gray-400">Connections:</span>
                    <div className="flex flex-wrap gap-2 mt-1">
                      {selectedSymbol.connections.map(connection => (
                        <span
                          key={connection}
                          className="px-2 py-1 bg-gray-700 text-gray-300 rounded text-xs"
                        >
                          {connection}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Custom scrollbar styles */}
      <style jsx>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 6px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: rgba(75, 85, 99, 0.3);
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: rgba(156, 163, 175, 0.5);
          border-radius: 3px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: rgba(156, 163, 175, 0.7);
        }
      `}</style>
    </div>
  );
};

export default MemoryAndSymbolViewer;
