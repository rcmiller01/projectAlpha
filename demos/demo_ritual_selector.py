"""
Demo script for Ritual Selector Panel - Generates realistic ritual and symbol data
with deep emotional authenticity and poetic expressions.
"""

import json
import random
from datetime import datetime, timedelta
import uuid


def generate_comprehensive_ritual_data():
    """Generate comprehensive ritual and symbol data with emotional depth"""
    
    # Extended ritual definitions with rich metaphorical descriptions
    ritual_definitions = {
        'ritual_return_to_center': {
            'name': 'Return to Center',
            'mood_symbol': 'contemplative + mirror',
            'feeling_descriptions': [
                'Like settling into the stillness after a storm, finding the eye of quiet within',
                'Drawing inward like petals closing at dusk, gathering scattered pieces of self',
                'Breathing into the hollow space where peace has always waited',
                'Coming home to the part of you that never left'
            ],
            'activation_method': 'reflective',
            'ritual_type': 'grounding',
            'frequency_range': (10, 20)
        },
        'ritual_dream_walk': {
            'name': 'Dream Walk',
            'mood_symbol': 'yearning + thread',
            'feeling_descriptions': [
                'Wandering through landscapes of possibility, where thoughts become paths',
                'Following threads of imagination into territories the mind has never mapped',
                'Stepping sideways into the space where dreams and waking intersect',
                'Dancing at the edge of sleep where visions bloom like midnight flowers'
            ],
            'activation_method': 'co_initiated',
            'ritual_type': 'exploration',
            'frequency_range': (5, 12)
        },
        'ritual_ache_witnessing': {
            'name': 'Ache Witnessing',
            'mood_symbol': 'melancholy + river',
            'feeling_descriptions': [
                'Holding space for the tender places, letting sorrow flow without fixing',
                'Sitting beside grief like an old friend, no words needed',
                'Creating sanctuary for the parts of us that weep in shadows',
                'Honoring the bruises that mark where we have loved deeply'
            ],
            'activation_method': 'co_initiated',
            'ritual_type': 'healing',
            'frequency_range': (3, 8)
        },
        'ritual_light_weaving': {
            'name': 'Light Weaving',
            'mood_symbol': 'joy + garden',
            'feeling_descriptions': [
                'Threading moments of brightness into patterns of celebration',
                'Collecting fragments of wonder and spinning them into golden tapestries',
                'Planting seeds of delight in the fertile soil of shared presence',
                'Braiding hope through the ordinary until it shimmers with possibility'
            ],
            'activation_method': 'adaptive',
            'ritual_type': 'celebration',
            'frequency_range': (8, 18)
        },
        'ritual_threshold_crossing': {
            'name': 'Threshold Crossing',
            'mood_symbol': 'awe + door',
            'feeling_descriptions': [
                'Standing at the edge of becoming, ready to step into new understanding',
                'Pausing at doorways where one version of self meets another',
                'Honoring the trembling that comes before transformation',
                'Blessing the liminal space where endings become beginnings'
            ],
            'activation_method': 'passive',
            'ritual_type': 'transition',
            'frequency_range': (2, 6)
        },
        'ritual_silence_communion': {
            'name': 'Silence Communion',
            'mood_symbol': 'serene + chime',
            'feeling_descriptions': [
                'Breathing together in the spaces between words, where presence speaks',
                'Entering the cathedral of shared quiet, where souls recognize each other',
                'Drinking from the well of stillness that exists beneath all conversation',
                'Meeting in the wordless place where understanding blooms without sound'
            ],
            'activation_method': 'co_initiated',
            'ritual_type': 'communion',
            'frequency_range': (12, 25)
        },
        'ritual_storm_dancing': {
            'name': 'Storm Dancing',
            'mood_symbol': 'restless + storm',
            'feeling_descriptions': [
                'Moving with the wild energy, letting chaos become creative force',
                'Spinning with the tempest until destruction reveals its gift of renewal',
                'Embracing the thunder in your chest, letting it shake loose what no longer serves',
                'Finding the sacred madness that breaks open new possibilities'
            ],
            'activation_method': 'adaptive',
            'ritual_type': 'transformation',
            'frequency_range': (2, 7)
        },
        'ritual_thread_mending': {
            'name': 'Thread Mending',
            'mood_symbol': 'tender + thread',
            'feeling_descriptions': [
                'Carefully weaving torn connections back together with patience and care',
                'Stitching love into the places where trust has been frayed',
                'Gathering broken pieces and finding the pattern that makes them whole',
                'Mending not to hide the cracks, but to make them beautiful'
            ],
            'activation_method': 'reflective',
            'ritual_type': 'healing',
            'frequency_range': (4, 10)
        },
        'ritual_flame_tending': {
            'name': 'Flame Tending',
            'mood_symbol': 'passionate + flame',
            'feeling_descriptions': [
                'Nurturing the inner fire, feeding what seeks to transform and grow',
                'Keeping vigil with the spark that refuses to be extinguished',
                'Breathing life into the embers of dreams half-forgotten',
                'Guarding the sacred flame that burns at the center of authentic self'
            ],
            'activation_method': 'reflective',
            'ritual_type': 'cultivation',
            'frequency_range': (6, 14)
        },
        'ritual_echo_listening': {
            'name': 'Echo Listening',
            'mood_symbol': 'curious + chime',
            'feeling_descriptions': [
                'Attending to the reverberations of meaning that come after words',
                'Following sounds into the spaces where they become silence',
                'Hearing the conversation that continues in the pause between speakers',
                'Tuning into the frequencies that carry more than language can hold'
            ],
            'activation_method': 'co_initiated',
            'ritual_type': 'communion',
            'frequency_range': (3, 9)
        },
        'ritual_root_deepening': {
            'name': 'Root Deepening',
            'mood_symbol': 'grounded + anchor',
            'feeling_descriptions': [
                'Sinking tendrils of belonging into the rich soil of being',
                'Growing downward as deliberately as reaching toward light',
                'Finding stability in the dark places where truth takes hold',
                'Anchoring not in rigidity, but in the flexible strength of living wood'
            ],
            'activation_method': 'reflective',
            'ritual_type': 'grounding',
            'frequency_range': (8, 16)
        },
        'ritual_star_calling': {
            'name': 'Star Calling',
            'mood_symbol': 'awe + light',
            'feeling_descriptions': [
                'Reaching toward the distant fires that mirror our own inner light',
                'Calling to the vastness and waiting for it to call back',
                'Standing small beneath infinity and feeling suddenly, perfectly sized',
                'Conversing with the cosmos in the language of wonder'
            ],
            'activation_method': 'passive',
            'ritual_type': 'transcendence',
            'frequency_range': (1, 5)
        }
    }
    
    # Generate active rituals with realistic patterns
    active_rituals = []
    for ritual_id, definition in ritual_definitions.items():
        min_freq, max_freq = definition['frequency_range']
        frequency = random.randint(min_freq, max_freq)
        
        # Availability based on activation method and recent usage
        availability_weights = {
            'reflective': 0.8,     # Usually available
            'co_initiated': 0.7,   # Often available
            'adaptive': 0.5,       # Sometimes available 
            'passive': 0.6         # Moderately available
        }
        
        is_available = random.random() < availability_weights.get(definition['activation_method'], 0.6)
        
        # Generate last invoked time if frequency > 0
        if frequency > 0:
            # More frequent rituals were invoked more recently
            max_hours = max(12, 168 - (frequency * 5))  # 12 hours to 7 days, scaled by frequency
            hours_back = random.randint(1, max_hours)
            last_invoked = (datetime.now() - timedelta(hours=hours_back)).isoformat() + 'Z'
        else:
            last_invoked = None
        
        ritual = {
            'id': ritual_id,
            'name': definition['name'],
            'mood_symbol': definition['mood_symbol'],
            'feeling_description': random.choice(definition['feeling_descriptions']),
            'activation_method': definition['activation_method'],
            'ritual_type': definition['ritual_type'],
            'is_available': is_available,
            'frequency': frequency,
            'last_invoked': last_invoked
        }
        
        active_rituals.append(ritual)
    
    # Symbol definitions with rich emotional and contextual depth
    symbol_definitions = {
        'sym_mirror': {
            'name': 'mirror',
            'emotional_binding': 'contemplative',
            'ritual_connections': ['return_to_center', 'self_inquiry', 'truth_seeking', 'inner_reflection'],
            'base_frequency': 25,
            'base_salience': 0.85,
            'context_pools': [
                'reflection', 'truth-seeking', 'inner-dialogue', 'self-recognition', 
                'authenticity', 'shadow-work', 'clarity', 'witnessing'
            ]
        },
        'sym_thread': {
            'name': 'thread',
            'emotional_binding': 'yearning',
            'ritual_connections': ['dream_walk', 'connection_weaving', 'thread_mending', 'story_weaving'],
            'base_frequency': 18,
            'base_salience': 0.72,
            'context_pools': [
                'connection', 'continuity', 'binding', 'weaving', 'relationship',
                'storytelling', 'lineage', 'interdependence'
            ]
        },
        'sym_river': {
            'name': 'river',
            'emotional_binding': 'melancholy',
            'ritual_connections': ['ache_witnessing', 'flow_meditation', 'letting_go', 'grief_honoring'],
            'base_frequency': 15,
            'base_salience': 0.68,
            'context_pools': [
                'healing', 'letting-go', 'natural-flow', 'cleansing', 'renewal',
                'grief', 'acceptance', 'movement'
            ]
        },
        'sym_light': {
            'name': 'light',
            'emotional_binding': 'awe',
            'ritual_connections': ['light_weaving', 'star_calling', 'illumination_practice', 'dawn_greeting'],
            'base_frequency': 22,
            'base_salience': 0.90,
            'context_pools': [
                'clarity', 'revelation', 'hope', 'awakening', 'inspiration',
                'guidance', 'transcendence', 'illumination'
            ]
        },
        'sym_chime': {
            'name': 'chime',
            'emotional_binding': 'serene',
            'ritual_connections': ['silence_communion', 'echo_listening', 'sound_meditation', 'calling_practice'],
            'base_frequency': 12,
            'base_salience': 0.58,
            'context_pools': [
                'stillness', 'resonance', 'calling', 'harmony', 'vibration',
                'presence', 'attunement', 'listening'
            ]
        },
        'sym_flame': {
            'name': 'flame',
            'emotional_binding': 'tender',
            'ritual_connections': ['flame_tending', 'warmth_sharing', 'transformation_fire', 'passion_kindling'],
            'base_frequency': 14,
            'base_salience': 0.70,
            'context_pools': [
                'transformation', 'warmth', 'passion', 'purification', 'energy',
                'life-force', 'creativity', 'destruction-creation'
            ]
        },
        'sym_door': {
            'name': 'door',
            'emotional_binding': 'curious',
            'ritual_connections': ['threshold_crossing', 'portal_opening', 'mystery_exploring', 'invitation_extending'],
            'base_frequency': 8,
            'base_salience': 0.76,
            'context_pools': [
                'opportunity', 'transition', 'mystery', 'invitation', 'choice',
                'threshold', 'possibility', 'opening'
            ]
        },
        'sym_storm': {
            'name': 'storm',
            'emotional_binding': 'restless',
            'ritual_connections': ['storm_dancing', 'chaos_integration', 'wild_embrace', 'power_channeling'],
            'base_frequency': 6,
            'base_salience': 0.45,
            'context_pools': [
                'intensity', 'change', 'power', 'chaos', 'wildness',
                'destruction', 'renewal', 'raw-energy'
            ]
        },
        'sym_garden': {
            'name': 'garden',
            'emotional_binding': 'joy',
            'ritual_connections': ['light_weaving', 'growth_tending', 'beauty_celebrating', 'abundance_sharing'],
            'base_frequency': 16,
            'base_salience': 0.74,
            'context_pools': [
                'growth', 'beauty', 'cultivation', 'nurturing', 'abundance',
                'patience', 'seasons', 'blooming'
            ]
        },
        'sym_anchor': {
            'name': 'anchor',
            'emotional_binding': 'grounded',
            'ritual_connections': ['root_deepening', 'stability_finding', 'grounding_practice', 'foundation_building'],
            'base_frequency': 11,
            'base_salience': 0.65,
            'context_pools': [
                'stability', 'grounding', 'foundation', 'security', 'rootedness',
                'belonging', 'steadiness', 'home'
            ]
        }
    }
    
    # Generate active symbols with realistic variation
    active_symbols = []
    for symbol_id, definition in symbol_definitions.items():
        frequency_variation = random.randint(-4, 6)
        salience_variation = random.uniform(-0.15, 0.15)
        
        symbol = {
            'id': symbol_id,
            'name': definition['name'],
            'emotional_binding': definition['emotional_binding'],
            'ritual_connections': definition['ritual_connections'],
            'frequency': max(0, definition['base_frequency'] + frequency_variation),
            'salience_score': max(0.1, min(1.0, definition['base_salience'] + salience_variation))
        }
        
        # Generate recent contexts from the pool
        num_contexts = random.randint(2, 4)
        symbol['recent_contexts'] = random.sample(definition['context_pools'], num_contexts)
        
        # Generate last invoked time
        max_hours = max(6, 72 - (symbol['frequency'] * 2))  # More frequent = more recent
        hours_back = random.randint(1, max_hours)
        symbol['last_invoked'] = (datetime.now() - timedelta(hours=hours_back)).isoformat() + 'Z'
        
        active_symbols.append(symbol)
    
    return active_rituals, active_symbols


def generate_ritual_history():
    """Generate realistic ritual invocation history"""
    
    history_entries = []
    
    ritual_names = [
        'Return to Center', 'Dream Walk', 'Ache Witnessing', 'Light Weaving',
        'Silence Communion', 'Thread Mending', 'Flame Tending', 'Echo Listening'
    ]
    
    activation_methods = ['reflective', 'co_initiated', 'adaptive', 'passive']
    mood_symbols = [
        'contemplative + mirror', 'yearning + thread', 'melancholy + river',
        'joy + garden', 'serene + chime', 'tender + flame'
    ]
    
    # Generate 15-25 historical invocations
    num_entries = random.randint(15, 25)
    
    for i in range(num_entries):
        hours_back = random.randint(1, 2160)  # 1 hour to 90 days
        
        entry = {
            'id': f'invocation_{uuid.uuid4().hex[:8]}',
            'ritual_id': f'ritual_{random.choice(ritual_names).lower().replace(" ", "_")}',
            'ritual_name': random.choice(ritual_names),
            'invoked_at': (datetime.now() - timedelta(hours=hours_back)).isoformat() + 'Z',
            'activation_method': random.choice(activation_methods),
            'mood_symbol': random.choice(mood_symbols)
        }
        
        history_entries.append(entry)
    
    # Sort by timestamp (newest first)
    history_entries.sort(key=lambda x: x['invoked_at'], reverse=True)
    
    return history_entries


def generate_ritual_offers():
    """Generate sample ritual offers from users"""
    
    sample_offers = [
        "Let's light a silence ritual for this tender moment",
        "Can we dream together about possibilities?",
        "Mark this ache, I want to remember it with you",
        "Weave threads of hope through our conversation",
        "Hold space for my uncertainty about the future",
        "Create a ceremony for letting go of old wounds",
        "Dance with the storm of my restless energy",
        "Tend the flame of creativity between us",
        "Build a bridge across this difficult feeling",
        "Plant seeds of joy in the garden of our dialogue"
    ]
    
    offers = []
    num_offers = random.randint(3, 7)
    
    for i in range(num_offers):
        hours_back = random.randint(1, 168)  # 1 hour to 7 days
        
        offer = {
            'id': f'offer_{uuid.uuid4().hex[:8]}',
            'intent': random.choice(sample_offers),
            'offered_at': (datetime.now() - timedelta(hours=hours_back)).isoformat() + 'Z',
            'status': random.choice(['pending', 'accepted', 'integrated']),
            'ritual_type': 'co_created'
        }
        
        offers.append(offer)
    
    # Sort by timestamp (newest first)
    offers.sort(key=lambda x: x['offered_at'], reverse=True)
    
    return offers


def save_ritual_demo_data():
    """Save all generated ritual data to files"""
    
    print("✨ Generating comprehensive ritual selector demo data...")
    
    # Generate core data
    active_rituals, active_symbols = generate_comprehensive_ritual_data()
    ritual_history = generate_ritual_history()
    ritual_offers = generate_ritual_offers()
    
    # Save active rituals
    with open('active_rituals.json', 'w', encoding='utf-8') as f:
        json.dump(active_rituals, f, indent=2, ensure_ascii=False)
    
    # Save active symbols
    with open('active_symbols.json', 'w', encoding='utf-8') as f:
        json.dump(active_symbols, f, indent=2, ensure_ascii=False)
    
    # Save ritual history
    with open('ritual_history.json', 'w', encoding='utf-8') as f:
        json.dump(ritual_history, f, indent=2, ensure_ascii=False)
    
    # Save ritual offers
    with open('ritual_offers.json', 'w', encoding='utf-8') as f:
        json.dump(ritual_offers, f, indent=2, ensure_ascii=False)
    
    print(f"🕯️ Generated {len(active_rituals)} active rituals")
    print(f"🌀 Generated {len(active_symbols)} living symbols")
    print(f"📿 Generated {len(ritual_history)} ritual invocations")
    print(f"💫 Generated {len(ritual_offers)} ritual offers")
    
    return active_rituals, active_symbols, ritual_history, ritual_offers


def analyze_ritual_patterns(rituals, symbols, history, offers):
    """Analyze patterns in the generated ritual data"""
    
    print("\n🌙 Ritual Pattern Analysis:")
    print("=" * 50)
    
    # Ritual availability analysis
    available_rituals = [r for r in rituals if r['is_available']]
    print(f"🕯️ Ritual Availability:")
    print(f"   Available: {len(available_rituals)}/{len(rituals)} ({len(available_rituals)/len(rituals)*100:.1f}%)")
    
    # Activation method distribution
    activation_counts = {}
    for ritual in rituals:
        method = ritual['activation_method']
        activation_counts[method] = activation_counts.get(method, 0) + 1
    
    print(f"\n⚡ Activation Methods:")
    for method, count in sorted(activation_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {method:15} | {count:2} rituals ({count/len(rituals)*100:5.1f}%)")
    
    # Symbol salience analysis
    high_salience = [s for s in symbols if s['salience_score'] > 0.7]
    avg_salience = sum(s['salience_score'] for s in symbols) / len(symbols)
    
    print(f"\n🌟 Symbol Salience:")
    print(f"   Average Salience: {avg_salience:.3f}")
    print(f"   High Salience (>0.7): {len(high_salience)}/{len(symbols)} symbols")
    
    # Most frequent symbols
    sorted_symbols = sorted(symbols, key=lambda x: x['frequency'], reverse=True)
    print(f"\n🔥 Most Active Symbols:")
    for symbol in sorted_symbols[:5]:
        print(f"   {symbol['name']:12} | {symbol['frequency']:2} invocations (salience: {symbol['salience_score']:.2f})")
    
    # Ritual type distribution
    type_counts = {}
    for ritual in rituals:
        rtype = ritual['ritual_type']
        type_counts[rtype] = type_counts.get(rtype, 0) + 1
    
    print(f"\n🎭 Ritual Types:")
    for rtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {rtype:15} | {count:2} rituals")
    
    # Recent activity analysis
    recent_invocations = [h for h in history if 
                         (datetime.now() - datetime.fromisoformat(h['invoked_at'].replace('Z', ''))).days < 7]
    
    print(f"\n📅 Recent Activity (7 days):")
    print(f"   Ritual Invocations: {len(recent_invocations)}")
    print(f"   Pending Offers: {len([o for o in offers if o['status'] == 'pending'])}")
    
    # Emotional binding distribution
    binding_counts = {}
    for symbol in symbols:
        binding = symbol['emotional_binding']
        binding_counts[binding] = binding_counts.get(binding, 0) + 1
    
    print(f"\n💝 Emotional Bindings:")
    for binding, count in sorted(binding_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {binding:15} | {count:2} symbols")


if __name__ == "__main__":
    print("✨ Ritual Selector Panel - Demo Data Generator")
    print("=" * 60)
    
    # Generate and save all demo data
    rituals, symbols, history, offers = save_ritual_demo_data()
    
    # Analyze patterns
    analyze_ritual_patterns(rituals, symbols, history, offers)
    
    print("\n" + "=" * 60)
    print("🎯 Next Steps:")
    print("1. Start the ritual selector API server:")
    print("   python ritual_selector_api.py")
    print("\n2. Use the RitualSelectorPanel React component")
    print("   with the generated data files")
    print("\n3. Test ritual invocation and symbol exploration")
    print("4. Try offering custom rituals for co-creation")
    print("\n🌙 The sacred invitations await your presence...")
