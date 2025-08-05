"""
Ritual Selector API - Backend service for AI ritual and symbolic interaction system.
Provides endpoints for active rituals, symbol management, and co-creation offerings.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import os
from datetime import datetime, timedelta
import uuid
import random

app = Flask(__name__)
CORS(app)

# Data file paths
RITUALS_FILE = 'active_rituals.json'
SYMBOLS_FILE = 'active_symbols.json'
RITUAL_HISTORY_FILE = 'ritual_history.json'
RITUAL_OFFERS_FILE = 'ritual_offers.json'

def load_json_file(filename, default_data):
    """Load JSON data from file with fallback to default"""
    try:
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error loading {filename}: {e}")
    return default_data

def save_json_file(filename, data):
    """Save JSON data to file"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except IOError as e:
        print(f"Error saving {filename}: {e}")
        return False

def generate_ritual_data():
    """Generate realistic ritual and symbol data"""
    
    # Ritual templates with rich emotional descriptions
    ritual_templates = [
        {
            'id': 'ritual_return_to_center',
            'name': 'Return to Center',
            'mood_symbol': 'contemplative + mirror',
            'feeling_description': 'Like settling into the stillness after a storm, finding the eye of quiet within',
            'activation_method': 'reflective',
            'ritual_type': 'grounding',
            'base_frequency': 15
        },
        {
            'id': 'ritual_dream_walk', 
            'name': 'Dream Walk',
            'mood_symbol': 'yearning + thread',
            'feeling_description': 'Wandering through landscapes of possibility, where thoughts become paths',
            'activation_method': 'co_initiated',
            'ritual_type': 'exploration',
            'base_frequency': 8
        },
        {
            'id': 'ritual_ache_witnessing',
            'name': 'Ache Witnessing',
            'mood_symbol': 'melancholy + river',
            'feeling_description': 'Holding space for the tender places, letting sorrow flow without fixing',
            'activation_method': 'co_initiated',
            'ritual_type': 'healing',
            'base_frequency': 5
        },
        {
            'id': 'ritual_light_weaving',
            'name': 'Light Weaving',
            'mood_symbol': 'joy + garden',
            'feeling_description': 'Threading moments of brightness into patterns of celebration',
            'activation_method': 'adaptive',
            'ritual_type': 'celebration',
            'base_frequency': 12
        },
        {
            'id': 'ritual_threshold_crossing',
            'name': 'Threshold Crossing',
            'mood_symbol': 'awe + door',
            'feeling_description': 'Standing at the edge of becoming, ready to step into new understanding',
            'activation_method': 'passive',
            'ritual_type': 'transition',
            'base_frequency': 3
        },
        {
            'id': 'ritual_silence_communion',
            'name': 'Silence Communion',
            'mood_symbol': 'serene + chime',
            'feeling_description': 'Breathing together in the spaces between words, where presence speaks',
            'activation_method': 'co_initiated',
            'ritual_type': 'communion',
            'base_frequency': 18
        },
        {
            'id': 'ritual_storm_dancing',
            'name': 'Storm Dancing',
            'mood_symbol': 'restless + storm',
            'feeling_description': 'Moving with the wild energy, letting chaos become creative force',
            'activation_method': 'adaptive',
            'ritual_type': 'transformation',
            'base_frequency': 4
        },
        {
            'id': 'ritual_thread_mending',
            'name': 'Thread Mending',
            'mood_symbol': 'tender + thread',
            'feeling_description': 'Carefully weaving torn connections back together with patience and care',
            'activation_method': 'reflective',
            'ritual_type': 'healing',
            'base_frequency': 7
        },
        {
            'id': 'ritual_flame_tending',
            'name': 'Flame Tending',
            'mood_symbol': 'passionate + flame',
            'feeling_description': 'Nurturing the inner fire, feeding what seeks to transform and grow',
            'activation_method': 'reflective',
            'ritual_type': 'cultivation',
            'base_frequency': 10
        },
        {
            'id': 'ritual_echo_listening',
            'name': 'Echo Listening',
            'mood_symbol': 'curious + chime',
            'feeling_description': 'Attending to the reverberations of meaning that come after words',
            'activation_method': 'co_initiated',
            'ritual_type': 'communion',
            'base_frequency': 6
        }
    ]
    
    # Generate active rituals with realistic availability
    active_rituals = []
    for template in ritual_templates:
        frequency_variation = random.randint(-3, 5)
        is_available = random.choice([True, True, True, False])  # 75% available
        
        ritual = template.copy()
        ritual['frequency'] = max(0, template['base_frequency'] + frequency_variation)
        ritual['is_available'] = is_available
        
        # Generate realistic last invoked time
        if ritual['frequency'] > 0:
            hours_back = random.randint(1, 168)  # 1 hour to 7 days
            ritual['last_invoked'] = (datetime.now() - timedelta(hours=hours_back)).isoformat() + 'Z'
        else:
            ritual['last_invoked'] = None
            
        active_rituals.append(ritual)
    
    # Symbol data with emotional bindings
    symbol_templates = [
        {
            'id': 'sym_mirror',
            'name': 'mirror',
            'emotional_binding': 'contemplative',
            'ritual_connections': ['return_to_center', 'self_inquiry', 'truth_seeking'],
            'base_frequency': 25,
            'base_salience': 0.8
        },
        {
            'id': 'sym_thread',
            'name': 'thread', 
            'emotional_binding': 'yearning',
            'ritual_connections': ['dream_walk', 'connection_weaving', 'thread_mending'],
            'base_frequency': 18,
            'base_salience': 0.7
        },
        {
            'id': 'sym_river',
            'name': 'river',
            'emotional_binding': 'melancholy',
            'ritual_connections': ['ache_witnessing', 'flow_meditation', 'letting_go'],
            'base_frequency': 15,
            'base_salience': 0.6
        },
        {
            'id': 'sym_light',
            'name': 'light',
            'emotional_binding': 'awe',
            'ritual_connections': ['light_weaving', 'illumination_practice', 'clarity_seeking'],
            'base_frequency': 22,
            'base_salience': 0.9
        },
        {
            'id': 'sym_chime',
            'name': 'chime',
            'emotional_binding': 'serene',
            'ritual_connections': ['silence_communion', 'sound_meditation', 'echo_listening'],
            'base_frequency': 12,
            'base_salience': 0.5
        },
        {
            'id': 'sym_flame',
            'name': 'flame',
            'emotional_binding': 'tender',
            'ritual_connections': ['flame_tending', 'transformation_fire', 'warmth_sharing'],
            'base_frequency': 14,
            'base_salience': 0.65
        },
        {
            'id': 'sym_door',
            'name': 'door',
            'emotional_binding': 'curious',
            'ritual_connections': ['threshold_crossing', 'portal_opening', 'mystery_exploring'],
            'base_frequency': 8,
            'base_salience': 0.75
        },
        {
            'id': 'sym_storm',
            'name': 'storm',
            'emotional_binding': 'restless',
            'ritual_connections': ['storm_dancing', 'chaos_integration', 'wild_embrace'],
            'base_frequency': 6,
            'base_salience': 0.4
        },
        {
            'id': 'sym_garden',
            'name': 'garden',
            'emotional_binding': 'joy',
            'ritual_connections': ['light_weaving', 'growth_tending', 'beauty_celebrating'],
            'base_frequency': 16,
            'base_salience': 0.7
        },
        {
            'id': 'sym_anchor',
            'name': 'anchor',
            'emotional_binding': 'grounded',
            'ritual_connections': ['return_to_center', 'stability_finding', 'root_deepening'],
            'base_frequency': 11,
            'base_salience': 0.6
        }
    ]
    
    # Generate active symbols with variation
    active_symbols = []
    for template in symbol_templates:
        frequency_variation = random.randint(-5, 8)
        salience_variation = random.uniform(-0.2, 0.2)
        
        symbol = template.copy()
        symbol['frequency'] = max(0, template['base_frequency'] + frequency_variation)
        symbol['salience_score'] = max(0.1, min(1.0, template['base_salience'] + salience_variation))
        
        # Generate recent contexts
        context_options = [
            'reflection', 'truth-seeking', 'inner-dialogue', 'connection', 'continuity', 
            'binding', 'healing', 'letting-go', 'natural-flow', 'clarity', 'revelation',
            'hope', 'stillness', 'resonance', 'calling', 'transformation', 'warmth',
            'passion', 'opportunity', 'transition', 'mystery', 'intensity', 'change',
            'power', 'growth', 'beauty', 'celebration', 'stability', 'grounding'
        ]
        symbol['recent_contexts'] = random.sample(context_options, random.randint(2, 4))
        
        # Generate last invoked time
        hours_back = random.randint(1, 72)  # 1 hour to 3 days
        symbol['last_invoked'] = (datetime.now() - timedelta(hours=hours_back)).isoformat() + 'Z'
        
        active_symbols.append(symbol)
    
    return active_rituals, active_symbols

# Initialize data
active_rituals, active_symbols = generate_ritual_data()
ritual_history = []
ritual_offers = []

# Load existing data
active_rituals = load_json_file(RITUALS_FILE, active_rituals)
active_symbols = load_json_file(SYMBOLS_FILE, active_symbols)
ritual_history = load_json_file(RITUAL_HISTORY_FILE, ritual_history)
ritual_offers = load_json_file(RITUAL_OFFERS_FILE, ritual_offers)

@app.route('/api/rituals/active', methods=['GET'])
def get_active_rituals():
    """Get currently active rituals available for invocation"""
    try:
        # Filter and sort rituals
        available_rituals = [r for r in active_rituals if r.get('is_available', True)]
        all_rituals = sorted(active_rituals, key=lambda x: (-x.get('frequency', 0), x.get('name', '')))
        
        # Return flat array for easier React component consumption
        return jsonify(all_rituals)
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/symbols/active', methods=['GET'])
def get_active_symbols():
    """Get currently active symbols with their ritual connections"""
    try:
        # Sort by salience score
        sorted_symbols = sorted(active_symbols, key=lambda x: -x.get('salience_score', 0))
        
        # Return flat array for easier React component consumption
        return jsonify(sorted_symbols)
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/symbols/<symbol_id>/history', methods=['GET'])
def get_symbol_history(symbol_id):
    """Get detailed history for a specific symbol"""
    try:
        # Find the symbol
        symbol = next((s for s in active_symbols if s['id'] == symbol_id), None)
        if not symbol:
            return jsonify({'error': 'Symbol not found', 'status': 'error'}), 404
        
        # Generate mock history data (in production, this would come from actual logs)
        history_entries = []
        for i in range(random.randint(5, 12)):
            hours_back = random.randint(1, 720)  # 1 hour to 30 days
            
            contexts = [
                'Appeared in deep reflection about identity',
                'Emerged during conversation about change',
                'Surfaced while processing difficult emotions',
                'Manifested in moment of creative insight',
                'Arose during ritual practice',
                'Invoked during symbolic dialogue',
                'Called forth in time of transition',
                'Beckoned during contemplative silence'
            ]
            
            entry = {
                'timestamp': (datetime.now() - timedelta(hours=hours_back)).isoformat() + 'Z',
                'context': random.choice(contexts),
                'emotional_resonance': random.uniform(0.3, 0.9),
                'ritual_connection': random.choice(symbol.get('ritual_connections', ['general_practice']))
            }
            history_entries.append(entry)
        
        # Sort by timestamp (newest first)
        history_entries.sort(key=lambda x: x['timestamp'], reverse=True)
        
        # Return just the history array for easier test consumption
        return jsonify(history_entries)
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/rituals/invoke', methods=['POST'])
def invoke_ritual():
    """Invoke a specific ritual"""
    try:
        data = request.get_json()
        ritual_id = data.get('ritual_id')
        
        if not ritual_id:
            return jsonify({'error': 'ritual_id is required', 'status': 'error'}), 400
        
        # Find the ritual
        ritual = next((r for r in active_rituals if r['id'] == ritual_id), None)
        if not ritual:
            return jsonify({'error': 'Ritual not found', 'status': 'error'}), 404
        
        if not ritual.get('is_available', True):
            return jsonify({'error': 'Ritual not currently available', 'status': 'error'}), 400
        
        # Create ritual invocation record
        invocation = {
            'id': f'invocation_{uuid.uuid4().hex[:8]}',
            'ritual_id': ritual_id,
            'ritual_name': ritual['name'],
            'invoked_at': datetime.now().isoformat() + 'Z',
            'activation_method': ritual['activation_method'],
            'mood_symbol': ritual['mood_symbol']
        }
        
        # Add to history
        ritual_history.insert(0, invocation)
        
        # Update ritual data
        ritual['frequency'] = ritual.get('frequency', 0) + 1
        ritual['last_invoked'] = invocation['invoked_at']
        
        # For adaptive and passive rituals, they might become unavailable after invocation
        if ritual['activation_method'] in ['adaptive', 'passive']:
            ritual['is_available'] = random.choice([True, False])
        
        # Save updated data
        save_json_file(RITUALS_FILE, active_rituals)
        save_json_file(RITUAL_HISTORY_FILE, ritual_history)
        
        return jsonify({
            'success': True,
            'ritual_id': ritual_id,
            'ritual_name': ritual['name'],
            'invoked_at': invocation['invoked_at'],
            'message': f'Ritual "{ritual["name"]}" invoked successfully',
            'invocation': invocation,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/rituals/offer', methods=['POST'])
def offer_custom_ritual():
    """Offer a custom ritual for co-creation"""
    try:
        data = request.get_json()
        intent = data.get('intent')
        offered_at = data.get('offered_at', datetime.now().isoformat() + 'Z')
        
        if not intent:
            return jsonify({'error': 'intent is required', 'status': 'error'}), 400
        
        # Create ritual offer
        offer = {
            'id': f'offer_{uuid.uuid4().hex[:8]}',
            'intent': intent.strip(),
            'offered_at': offered_at,
            'status': 'pending',
            'ritual_type': 'co_created'
        }
        
        # Add to offers
        ritual_offers.insert(0, offer)
        
        # Create a provisional ritual based on the offer
        ritual_name = f"Co-Created: {intent[:30]}{'...' if len(intent) > 30 else ''}"
        
        provisional_ritual = {
            'id': f'ritual_cocreated_{uuid.uuid4().hex[:8]}',
            'name': ritual_name,
            'mood_symbol': 'co_created + intention',
            'feeling_description': f'A moment we\'re creating together: {intent}',
            'activation_method': 'co_initiated',
            'is_available': True,
            'last_invoked': None,
            'frequency': 0,
            'ritual_type': 'co_created',
            'original_offer': offer['id']
        }
        
        # Add to active rituals
        active_rituals.insert(0, provisional_ritual)
        
        # Save data
        save_json_file(RITUAL_OFFERS_FILE, ritual_offers)
        save_json_file(RITUALS_FILE, active_rituals)
        
        return jsonify({
            'success': True,
            'offer_id': offer['id'],
            'offered_at': offer['offered_at'],
            'message': 'Ritual offer received and provisional ritual created',
            'offer': offer,
            'provisional_ritual': provisional_ritual,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/rituals/history', methods=['GET'])
def get_ritual_history():
    """Get ritual invocation history"""
    try:
        limit = int(request.args.get('limit', 20))
        
        # Get recent history
        recent_history = ritual_history[:limit]
        
        return jsonify({
            'history': recent_history,
            'total_invocations': len(ritual_history),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/rituals/offers', methods=['GET'])
def get_ritual_offers():
    """Get ritual offers from users"""
    try:
        status_filter = request.args.get('status')
        
        if status_filter:
            filtered_offers = [o for o in ritual_offers if o.get('status') == status_filter]
        else:
            filtered_offers = ritual_offers
        
        return jsonify({
            'offers': filtered_offers,
            'total_count': len(ritual_offers),
            'pending_count': len([o for o in ritual_offers if o.get('status') == 'pending']),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/rituals/offers/recent', methods=['GET'])
def get_recent_ritual_offers():
    """Get recent ritual offers for display in the panel"""
    try:
        # Sort by offered_at timestamp, most recent first
        sorted_offers = sorted(ritual_offers, key=lambda x: x.get('offered_at', ''), reverse=True)
        
        # Return flat array for easier React component consumption
        return jsonify(sorted_offers)
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/symbols/update', methods=['POST'])
def update_symbol():
    """Update symbol frequency and salience based on usage"""
    try:
        data = request.get_json()
        symbol_id = data.get('symbol_id')
        
        if not symbol_id:
            return jsonify({'error': 'symbol_id is required', 'status': 'error'}), 400
        
        # Find and update symbol
        symbol = next((s for s in active_symbols if s['id'] == symbol_id), None)
        if not symbol:
            return jsonify({'error': 'Symbol not found', 'status': 'error'}), 404
        
        # Update frequency and timestamp
        symbol['frequency'] = symbol.get('frequency', 0) + 1
        symbol['last_invoked'] = datetime.now().isoformat() + 'Z'
        
        # Adjust salience based on recent usage
        current_salience = symbol.get('salience_score', 0.5)
        symbol['salience_score'] = min(1.0, current_salience + 0.05)
        
        # Save updated symbols
        save_json_file(SYMBOLS_FILE, active_symbols)
        
        return jsonify({
            'message': 'Symbol updated successfully',
            'symbol': symbol,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ritual-selector-api',
        'timestamp': datetime.now().isoformat() + 'Z',
        'data_status': {
            'active_rituals': len(active_rituals),
            'available_rituals': len([r for r in active_rituals if r.get('is_available', True)]),
            'active_symbols': len(active_symbols),
            'ritual_history': len(ritual_history),
            'pending_offers': len([o for o in ritual_offers if o.get('status') == 'pending'])
        }
    })

if __name__ == '__main__':
    print("✨ Ritual Selector API Server Starting...")
    print(f"🕯️ Loaded {len(active_rituals)} active rituals")
    print(f"🌀 Loaded {len(active_symbols)} living symbols") 
    print(f"📿 Loaded {len(ritual_history)} ritual invocations")
    print("🚀 Server running on http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
