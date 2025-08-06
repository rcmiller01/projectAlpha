#!/usr/bin/env python3
"""
CoreArbiter API Integration

Flask API endpoints for integrating CoreArbiter with the existing system.
Provides REST API access to the CoreArbiter functionality.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
import traceback

from core_arbiter import CoreArbiter, WeightingStrategy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global CoreArbiter instance
core_arbiter = None

def get_arbiter():
    """Get or create CoreArbiter instance"""
    global core_arbiter
    if core_arbiter is None:
        core_arbiter = CoreArbiter()
    return core_arbiter

@app.route('/api/arbiter/process', methods=['POST'])
def process_input():
    """Process user input through CoreArbiter"""
    try:
        data = request.json
        user_input = data.get('message', '')
        state = data.get('state', {})
        
        if not user_input:
            return jsonify({'error': 'Message is required'}), 400
        
        # Run async function in new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            arbiter = get_arbiter()
            response = loop.run_until_complete(arbiter.process_input(user_input, state))
            
            # Convert response to JSON-serializable format
            response_data = {
                'final_output': response.final_output,
                'reflection': response.reflection,
                'action': response.action,
                'mood_inflected': response.mood_inflected,
                'tone': response.tone,
                'priority': response.priority,
                'source_weights': response.source_weights,
                'confidence': response.confidence,
                'emotional_override': response.emotional_override,
                'symbolic_context': response.symbolic_context,
                'resolution_strategy': response.resolution_strategy,
                'timestamp': response.timestamp.isoformat(),
                'metadata': response.metadata
            }
            
            return jsonify(response_data)
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/api/arbiter/status', methods=['GET'])
def get_status():
    """Get current arbiter system status"""
    try:
        arbiter = get_arbiter()
        status = arbiter.get_system_status()
        return jsonify(status)
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/arbiter/strategy', methods=['POST'])
def set_strategy():
    """Change weighting strategy"""
    try:
        data = request.json
        strategy_name = data.get('strategy', 'harmonic')
        
        # Validate strategy
        try:
            strategy = WeightingStrategy(strategy_name)
        except ValueError:
            return jsonify({'error': f'Invalid strategy: {strategy_name}'}), 400
        
        arbiter = get_arbiter()
        arbiter.set_weighting_strategy(strategy)
        
        return jsonify({
            'status': 'success',
            'strategy': strategy.value,
            'message': f'Strategy changed to {strategy.value}'
        })
        
    except Exception as e:
        logger.error(f"Error setting strategy: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/arbiter/regulate', methods=['POST'])
def regulate_system():
    """Perform system regulation"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            arbiter = get_arbiter()
            loop.run_until_complete(arbiter.regulate_system())
            
            # Get updated status
            status = arbiter.get_system_status()
            
            return jsonify({
                'status': 'success',
                'message': 'System regulation completed',
                'system_status': status
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error regulating system: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/emotional_state', methods=['GET'])
def get_emotional_state():
    """Get current emotional state for UI"""
    try:
        # Load from file or generate current state
        emotional_state_path = Path("data/emotional_state.json")
        
        if emotional_state_path.exists():
            with open(emotional_state_path, 'r') as f:
                state = json.load(f)
        else:
            # Generate default state
            state = {
                "valence": 0.2,
                "arousal": 0.4,
                "dominant_emotion": "contemplative",
                "stability": 0.85,
                "mood_signals": {
                    "warmth": 0.7,
                    "empathy": 0.8,
                    "curiosity": 0.6,
                    "concern": 0.3
                }
            }
        
        # Add arbiter status if available
        if core_arbiter:
            arbiter_status = core_arbiter.get_system_status()
            state['arbiter_status'] = arbiter_status
        
        return jsonify(state)
        
    except Exception as e:
        logger.error(f"Error getting emotional state: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/symbolic_response', methods=['POST'])
def generate_symbolic_response():
    """Generate symbolic/ritual response"""
    try:
        data = request.json
        current_state = data.get('current_state', {})
        context = data.get('context', [])
        
        # Create symbolic input for arbiter
        symbolic_input = "Express the deeper symbolic meaning of our connection"
        state = {
            'context': 'symbolic_expression',
            'emotional_state': current_state,
            'recent_context': context,
            'ritual_request': True
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            arbiter = get_arbiter()
            response = loop.run_until_complete(arbiter.process_input(symbolic_input, state))
            
            return jsonify({
                'symbolic_output': response.final_output,
                'reflection': response.reflection,
                'symbolic_context': response.symbolic_context,
                'ritual_strength': response.symbolic_context.get('ritual_strength', 0.5)
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error generating symbolic response: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/log_emotional_message', methods=['POST'])
def log_emotional_message():
    """Log message with emotional context"""
    try:
        data = request.json
        
        # Create log entry
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'message': data.get('message'),
            'role': data.get('role'),
            'emotional_state': data.get('emotional_state'),
            'mood_profile': data.get('mood_profile')
        }
        
        # Append to emotional conversation log
        log_path = Path("logs/emotional_conversations.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        if log_path.exists():
            with open(log_path, 'r') as f:
                logs = json.load(f)
        else:
            logs = []
        
        logs.append(log_entry)
        
        # Keep only last 1000 entries
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return jsonify({'status': 'logged'})
        
    except Exception as e:
        logger.error(f"Error logging message: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/arbiter/traces', methods=['GET'])
def get_traces():
    """Get arbiter decision traces"""
    try:
        trace_path = Path("logs/core_arbiter_trace.json")
        
        if trace_path.exists():
            with open(trace_path, 'r') as f:
                traces = json.load(f)
            
            # Return last N traces
            limit = request.args.get('limit', 50, type=int)
            return jsonify({
                'traces': traces[-limit:],
                'total_count': len(traces)
            })
        else:
            return jsonify({
                'traces': [],
                'total_count': 0
            })
            
    except Exception as e:
        logger.error(f"Error getting traces: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_arbiter():
    """Main chat endpoint using CoreArbiter"""
    try:
        data = request.json
        message = data.get('message', '')
        emotional_context = data.get('emotional_context', {})
        mood_profile = data.get('mood_profile', {})
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Prepare state for arbiter
        state = {
            'context': 'conversational_chat',
            'emotional_context': emotional_context,
            'mood_profile': mood_profile,
            'user_message': message
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            arbiter = get_arbiter()
            response = loop.run_until_complete(arbiter.process_input(message, state))
            
            # Generate mood profile based on response
            mood_colors = {
                'emotional': {'primary': '#EC4899', 'secondary': '#F9A8D4'},
                'balanced': {'primary': '#8B5CF6', 'secondary': '#C4B5FD'},
                'objective': {'primary': '#06B6D4', 'secondary': '#67E8F9'}
            }
            
            colors = mood_colors.get(response.tone, mood_colors['balanced'])
            
            response_mood = {
                'emotion': response.symbolic_context.get('mood_primary', 'contemplative'),
                'intensity': response.confidence,
                'colors': colors,
                'icon': '🤔' if response.tone == 'objective' else '💭' if response.tone == 'balanced' else '💖'
            }
            
            return jsonify({
                'response': response.final_output,
                'mood_profile': response_mood,
                'metadata': {
                    'confidence': response.confidence,
                    'tone': response.tone,
                    'priority': response.priority,
                    'emotional_override': response.emotional_override,
                    'resolution_strategy': response.resolution_strategy,
                    'symbolic_context': response.symbolic_context
                },
                'reflection': response.reflection,
                'timestamp': response.timestamp.isoformat()
            })
            
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error in chat: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        status = "healthy"
        if core_arbiter:
            arbiter_status = core_arbiter.get_system_status()
            if arbiter_status['health_status'] in ['critical', 'concerning']:
                status = "degraded"
        
        return jsonify({
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'arbiter_initialized': core_arbiter is not None
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # Ensure directories exist
    Path("data").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Initialize arbiter
    get_arbiter()
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=True
    )
