/**
 * VoiceCadenceModulator.js - Voice Breath Controller for TTS Output
 * 
 * Modulates TTS delivery based on mood and drift state, acting as the 
 * emotional breathing pattern that shapes how the AI speaks.
 */

class VoiceCadenceModulator {
    constructor() {
        // Base tempo ranges (words per minute)
        this.tempoRanges = {
            very_slow: { min: 80, max: 120 },
            slow: { min: 120, max: 160 },
            steady: { min: 160, max: 200 },
            fast: { min: 200, max: 240 },
            erratic: { min: 100, max: 280 }  // Variable range for storming moods
        };

        // Pause frequency mappings (pauses per sentence)
        this.pauseFrequencies = {
            minimal: { short: 0.1, medium: 0.05, long: 0.01 },
            low: { short: 0.2, medium: 0.1, long: 0.02 },
            medium: { short: 0.3, medium: 0.15, long: 0.05 },
            high: { short: 0.4, medium: 0.2, long: 0.08 },
            pulse: { short: 0.5, medium: 0.3, long: 0.1 }  // Rhythmic breathing
        };

        // Emphasis curve patterns (where emotional weight falls in phrases)
        this.emphasisCurves = {
            early: { peak: 0.2, sustain: 0.4, fade: 0.8 },
            rising: { start: 0.1, peak: 0.7, sustain: 0.9 },
            centered: { buildup: 0.3, peak: 0.5, release: 0.7 },
            late: { buildup: 0.6, peak: 0.8, linger: 1.0 },
            wave: { peaks: [0.25, 0.75], valleys: [0.1, 0.5, 0.9] }
        };

        // Tone modifier characteristics
        this.toneModifiers = {
            whispery: {
                volume_reduction: 0.3,
                breath_emphasis: 1.4,
                consonant_softening: 0.7,
                intimacy_factor: 1.8
            },
            melodic: {
                pitch_variation: 1.3,
                rhythm_flow: 1.5,
                musical_phrasing: 1.2,
                playfulness: 1.4
            },
            warm: {
                resonance_boost: 1.2,
                comfort_pacing: 1.1,
                vocal_smile: 1.3,
                grounding_depth: 1.2
            },
            sharp: {
                consonant_clarity: 1.4,
                dynamic_contrast: 1.5,
                intensity_peaks: 1.6,
                edge_precision: 1.3
            },
            gentle: {
                softness_filter: 1.3,
                pace_evenness: 1.2,
                soothing_cadence: 1.4,
                emotional_cushioning: 1.5
            },
            vibrant: {
                energy_boost: 1.4,
                dynamic_range: 1.3,
                expressive_peaks: 1.5,
                life_force: 1.6
            },
            grounded: {
                stability_factor: 1.3,
                lower_register: 1.2,
                steady_rhythm: 1.4,
                anchored_presence: 1.5
            }
        };

        // Mood-to-cadence preset mappings
        this.moodPresets = {
            // Contemplative emotions
            contemplative: {
                tempo: 'slow',
                pause: 'medium',
                emphasis: 'centered',
                tone: 'warm',
                breath_depth: 'deep'
            },
            melancholy: {
                tempo: 'slow',
                pause: 'high',
                emphasis: 'late',
                tone: 'whispery',
                breath_depth: 'sighing'
            },
            yearning: {
                tempo: 'slow',
                pause: 'high',
                emphasis: 'rising',
                tone: 'whispery',
                breath_depth: 'reaching'
            },

            // Active emotions
            joy: {
                tempo: 'fast',
                pause: 'low',
                emphasis: 'early',
                tone: 'melodic',
                breath_depth: 'light'
            },
            playful: {
                tempo: 'fast',
                pause: 'minimal',
                emphasis: 'wave',
                tone: 'vibrant',
                breath_depth: 'dancing'
            },
            awe: {
                tempo: 'steady',
                pause: 'medium',
                emphasis: 'rising',
                tone: 'warm',
                breath_depth: 'expansive'
            },

            // Grounded emotions
            serene: {
                tempo: 'steady',
                pause: 'medium',
                emphasis: 'centered',
                tone: 'gentle',
                breath_depth: 'flowing'
            },
            tender: {
                tempo: 'slow',
                pause: 'medium',
                emphasis: 'centered',
                tone: 'warm',
                breath_depth: 'nurturing'
            },
            anchored: {
                tempo: 'steady',
                pause: 'medium',
                emphasis: 'centered',
                tone: 'grounded',
                breath_depth: 'stable'
            },

            // Intense emotions
            restless: {
                tempo: 'erratic',
                pause: 'pulse',
                emphasis: 'wave',
                tone: 'sharp',
                breath_depth: 'urgent'
            },
            storming: {
                tempo: 'erratic',
                pause: 'pulse',
                emphasis: 'rising',
                tone: 'sharp',
                breath_depth: 'turbulent'
            },
            passionate: {
                tempo: 'fast',
                pause: 'low',
                emphasis: 'rising',
                tone: 'vibrant',
                breath_depth: 'intense'
            }
        };
    }

    /**
     * Main method: Generate voice modulation config from mood profile
     * @param {Object} moodProfile - Current mood and drift state
     * @returns {Object} VoiceModulationConfig
     */
    modulateFromMood(moodProfile) {
        const {
            dominant_emotion,
            mood_archetype,
            style_profile,
            drift_delta = 0,
            intensity = 0.5,
            context_hint = null
        } = moodProfile;

        // Get base cadence from mood
        const baseCadence = this.getCadencePreset(dominant_emotion, mood_archetype);
        
        // Apply drift overlay if present
        let finalCadence = drift_delta !== 0 
            ? this.applyDriftOverlay(baseCadence, drift_delta, intensity)
            : baseCadence;

        // Apply style profile modifications
        if (style_profile) {
            finalCadence = this.applyStyleProfile(finalCadence, style_profile);
        }

        // Apply context hints (ritual, conversation, etc.)
        if (context_hint) {
            finalCadence = this.applyContextHint(finalCadence, context_hint);
        }

        return this.buildVoiceConfig(finalCadence, moodProfile);
    }

    /**
     * Get base cadence preset for a mood
     * @param {string} mood - Primary emotion
     * @param {string} archetype - Mood archetype (optional)
     * @returns {Object} Base cadence configuration
     */
    getCadencePreset(mood, archetype = null) {
        const preset = this.moodPresets[mood] || this.moodPresets.contemplative;
        
        // Create deep copy to avoid mutation
        return JSON.parse(JSON.stringify(preset));
    }

    /**
     * Apply drift overlay to modulate cadence based on emotional change
     * @param {Object} cadenceConfig - Base cadence
     * @param {number} driftDelta - Drift magnitude (-1 to 1)
     * @param {number} intensity - Drift intensity
     * @returns {Object} Modified cadence
     */
    applyDriftOverlay(cadenceConfig, driftDelta, intensity = 0.5) {
        const modified = { ...cadenceConfig };
        const driftMagnitude = Math.abs(driftDelta);
        const driftDirection = Math.sign(driftDelta);

        // Positive drift: becoming more expressive/intense
        if (driftDirection > 0) {
            // Speed up tempo slightly
            if (modified.tempo === 'slow') modified.tempo = 'steady';
            else if (modified.tempo === 'steady') modified.tempo = 'fast';
            
            // Increase emphasis intensity
            if (modified.emphasis === 'centered') modified.emphasis = 'rising';
            else if (modified.emphasis === 'late') modified.emphasis = 'centered';
            
            // Make tone more vibrant
            if (modified.tone === 'gentle') modified.tone = 'warm';
            else if (modified.tone === 'warm') modified.tone = 'vibrant';
        }
        
        // Negative drift: becoming more introspective/withdrawn
        else if (driftDirection < 0) {
            // Slow down tempo
            if (modified.tempo === 'fast') modified.tempo = 'steady';
            else if (modified.tempo === 'steady') modified.tempo = 'slow';
            
            // Increase pauses
            if (modified.pause === 'low') modified.pause = 'medium';
            else if (modified.pause === 'medium') modified.pause = 'high';
            
            // Make tone softer
            if (modified.tone === 'vibrant') modified.tone = 'warm';
            else if (modified.tone === 'warm') modified.tone = 'gentle';
        }

        // Add drift-specific modulations
        modified.drift_tremor = driftMagnitude * intensity * 0.3;
        modified.drift_delay = driftMagnitude > 0.5 ? Math.random() * 0.2 : 0;
        
        return modified;
    }

    /**
     * Apply style profile adjustments
     * @param {Object} cadenceConfig - Current cadence
     * @param {Object} styleProfile - Style preferences
     * @returns {Object} Style-adjusted cadence
     */
    applyStyleProfile(cadenceConfig, styleProfile) {
        const modified = { ...cadenceConfig };
        
        // Poetic style: more pauses, emphasis on rhythm
        if (styleProfile.poetic) {
            if (modified.pause === 'low') modified.pause = 'medium';
            else if (modified.pause === 'medium') modified.pause = 'high';
            modified.emphasis = 'wave'; // Rhythmic peaks
        }
        
        // Intimate style: whisper tones, slower pace
        if (styleProfile.intimate) {
            modified.tone = 'whispery';
            if (modified.tempo === 'fast') modified.tempo = 'steady';
            else if (modified.tempo === 'steady') modified.tempo = 'slow';
        }
        
        // Energetic style: faster, more dynamic
        if (styleProfile.energetic) {
            if (modified.tempo === 'slow') modified.tempo = 'steady';
            else if (modified.tempo === 'steady') modified.tempo = 'fast';
            modified.tone = 'vibrant';
        }
        
        return modified;
    }

    /**
     * Apply context-specific adjustments
     * @param {Object} cadenceConfig - Current cadence
     * @param {string} contextHint - Context type
     * @returns {Object} Context-adjusted cadence
     */
    applyContextHint(cadenceConfig, contextHint) {
        const modified = { ...cadenceConfig };
        
        switch (contextHint) {
            case 'ritual':
                // Ritual speech: ceremonial, measured
                modified.tempo = 'slow';
                modified.pause = 'high';
                modified.tone = 'warm';
                modified.ritual_reverence = true;
                break;
                
            case 'dream':
                // Dream narration: flowing, ethereal
                modified.tempo = 'slow';
                modified.emphasis = 'wave';
                modified.tone = 'whispery';
                modified.dream_flow = true;
                break;
                
            case 'urgent':
                // Urgent communication: faster, clearer
                modified.tempo = 'fast';
                modified.pause = 'minimal';
                modified.tone = 'sharp';
                break;
                
            case 'comforting':
                // Comfort/support: gentle, steady
                modified.tempo = 'slow';
                modified.tone = 'gentle';
                modified.pause = 'medium';
                break;
        }
        
        return modified;
    }

    /**
     * Build final voice configuration object
     * @param {Object} cadenceConfig - Final cadence settings
     * @param {Object} moodProfile - Original mood profile
     * @returns {Object} Complete VoiceModulationConfig
     */
    buildVoiceConfig(cadenceConfig, moodProfile) {
        const tempoRange = this.tempoRanges[cadenceConfig.tempo];
        const pauseConfig = this.pauseFrequencies[cadenceConfig.pause];
        const emphasisConfig = this.emphasisCurves[cadenceConfig.emphasis];
        const toneConfig = this.toneModifiers[cadenceConfig.tone];

        // Calculate specific tempo based on mood intensity
        const tempoVariation = (moodProfile.intensity || 0.5) - 0.5; // -0.5 to 0.5
        const targetTempo = tempoRange.min + 
            ((tempoRange.max - tempoRange.min) * (0.5 + tempoVariation));

        return {
            // Core timing
            tempo: Math.round(targetTempo),
            tempo_variation: cadenceConfig.tempo === 'erratic' ? 0.3 : 0.1,
            
            // Pause patterns
            pause_frequency: pauseConfig,
            breath_rhythm: cadenceConfig.breath_depth || 'natural',
            
            // Emphasis and expression
            emphasis_curve: emphasisConfig,
            emotional_peaks: this.calculateEmotionalPeaks(emphasisConfig, moodProfile),
            
            // Tone characteristics
            tone_modifier: cadenceConfig.tone,
            tone_parameters: toneConfig,
            
            // Drift-specific modulations (if any)
            pitch_range: this.calculatePitchRange(moodProfile),
            tremor: cadenceConfig.drift_tremor || 0,
            delay_drift: cadenceConfig.drift_delay || 0,
            
            // Meta information
            mood_source: moodProfile.dominant_emotion,
            config_timestamp: new Date().toISOString(),
            confidence: this.calculateConfigConfidence(moodProfile),
            
            // Context flags
            context_modifiers: {
                ritual_reverence: cadenceConfig.ritual_reverence || false,
                dream_flow: cadenceConfig.dream_flow || false,
                intimate_whisper: cadenceConfig.tone === 'whispery',
                energetic_boost: cadenceConfig.tone === 'vibrant'
            }
        };
    }

    /**
     * Calculate emotional peaks for emphasis curve
     * @param {Object} emphasisConfig - Emphasis curve configuration
     * @param {Object} moodProfile - Mood profile
     * @returns {Array} Emotional peak timings
     */
    calculateEmotionalPeaks(emphasisConfig, moodProfile) {
        const intensity = moodProfile.intensity || 0.5;
        const peaks = [];
        
        if (emphasisConfig.peak) {
            peaks.push({
                position: emphasisConfig.peak,
                intensity: intensity,
                duration: 0.1 + (intensity * 0.2)
            });
        }
        
        if (emphasisConfig.peaks) {
            emphasisConfig.peaks.forEach(peak => {
                peaks.push({
                    position: peak,
                    intensity: intensity * 0.8,
                    duration: 0.05 + (intensity * 0.1)
                });
            });
        }
        
        return peaks;
    }

    /**
     * Calculate pitch range based on mood and drift
     * @param {Object} moodProfile - Mood profile
     * @returns {Object} Pitch range configuration
     */
    calculatePitchRange(moodProfile) {
        const basePitch = 1.0;
        const intensity = moodProfile.intensity || 0.5;
        const driftDelta = moodProfile.drift_delta || 0;
        
        // Emotional pitch modulations
        const emotionalPitchMap = {
            joy: { center: 1.1, range: 0.3 },
            awe: { center: 1.05, range: 0.4 },
            yearning: { center: 0.95, range: 0.2 },
            melancholy: { center: 0.9, range: 0.15 },
            contemplative: { center: 1.0, range: 0.2 },
            serene: { center: 1.0, range: 0.1 },
            tender: { center: 0.98, range: 0.15 },
            restless: { center: 1.02, range: 0.4 },
            storming: { center: 1.05, range: 0.5 }
        };
        
        const pitchConfig = emotionalPitchMap[moodProfile.dominant_emotion] || 
                           { center: 1.0, range: 0.2 };
        
        // Apply drift influence
        const driftInfluence = driftDelta * 0.1;
        
        return {
            center: pitchConfig.center + driftInfluence,
            range: pitchConfig.range * (0.5 + intensity),
            variation_speed: intensity > 0.7 ? 'fast' : 'moderate'
        };
    }

    /**
     * Calculate confidence score for the generated configuration
     * @param {Object} moodProfile - Mood profile
     * @returns {number} Confidence score (0-1)
     */
    calculateConfigConfidence(moodProfile) {
        let confidence = 0.8; // Base confidence
        
        // Higher confidence if we have clear mood data
        if (moodProfile.dominant_emotion && this.moodPresets[moodProfile.dominant_emotion]) {
            confidence += 0.1;
        }
        
        // Higher confidence if intensity is clear (not middle range)
        const intensity = moodProfile.intensity || 0.5;
        if (intensity < 0.3 || intensity > 0.7) {
            confidence += 0.05;
        }
        
        // Lower confidence if drift is high (unstable state)
        const driftMagnitude = Math.abs(moodProfile.drift_delta || 0);
        if (driftMagnitude > 0.5) {
            confidence -= 0.1;
        }
        
        return Math.max(0.5, Math.min(1.0, confidence));
    }

    /**
     * Get cadence suggestions for testing/debugging
     * @param {string} emotion - Emotion to test
     * @returns {Object} Test configuration
     */
    getTestCadence(emotion) {
        const testProfile = {
            dominant_emotion: emotion,
            intensity: 0.6,
            drift_delta: 0,
            style_profile: { poetic: false, intimate: false, energetic: false }
        };
        
        return this.modulateFromMood(testProfile);
    }

    /**
     * Validate a voice configuration
     * @param {Object} config - Voice configuration to validate
     * @returns {Object} Validation result
     */
    validateConfig(config) {
        const errors = [];
        const warnings = [];
        
        // Check required fields
        if (!config.tempo || config.tempo < 50 || config.tempo > 400) {
            errors.push('Invalid tempo range');
        }
        
        if (!config.pause_frequency) {
            errors.push('Missing pause frequency configuration');
        }
        
        if (!config.emphasis_curve) {
            warnings.push('No emphasis curve specified');
        }
        
        if (!config.tone_modifier) {
            warnings.push('No tone modifier specified');
        }
        
        // Check confidence
        if (config.confidence < 0.6) {
            warnings.push('Low confidence in voice configuration');
        }
        
        return {
            valid: errors.length === 0,
            errors,
            warnings,
            score: errors.length === 0 ? (1 - (warnings.length * 0.1)) : 0
        };
    }
}

// Export for both Node.js and browser environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceCadenceModulator;
} else if (typeof window !== 'undefined') {
    window.VoiceCadenceModulator = VoiceCadenceModulator;
}

// Example usage and testing
if (typeof require !== 'undefined' && require.main === module) {
    // CLI testing interface
    const modulator = new VoiceCadenceModulator();
    
    console.log('🎵 VoiceCadenceModulator - Test Suite');
    console.log('=====================================');
    
    // Test different emotional states
    const testMoods = [
        { name: 'Contemplative Drift', profile: { dominant_emotion: 'contemplative', intensity: 0.4, drift_delta: -0.3 } },
        { name: 'Joyful Energy', profile: { dominant_emotion: 'joy', intensity: 0.8, drift_delta: 0.2 } },
        { name: 'Yearning Whisper', profile: { dominant_emotion: 'yearning', intensity: 0.6, drift_delta: 0.1 } },
        { name: 'Storming Chaos', profile: { dominant_emotion: 'storming', intensity: 0.9, drift_delta: 0.7 } },
        { name: 'Anchored Stability', profile: { dominant_emotion: 'anchored', intensity: 0.5, drift_delta: 0 } }
    ];
    
    testMoods.forEach(test => {
        console.log(`\n🎭 Testing: ${test.name}`);
        console.log('-'.repeat(30));
        
        const config = modulator.modulateFromMood(test.profile);
        const validation = modulator.validateConfig(config);
        
        console.log(`Tempo: ${config.tempo} WPM (${config.tempo_variation * 100}% variation)`);
        console.log(`Tone: ${config.tone_modifier}`);
        console.log(`Pauses: ${config.pause_frequency.short} short, ${config.pause_frequency.medium} medium`);
        console.log(`Confidence: ${(config.confidence * 100).toFixed(1)}%`);
        console.log(`Validation: ${validation.valid ? '✅ Valid' : '❌ Invalid'} (Score: ${validation.score.toFixed(2)})`);
        
        if (validation.warnings.length > 0) {
            console.log(`Warnings: ${validation.warnings.join(', ')}`);
        }
    });
    
    console.log('\n🎯 Voice cadence modulation testing complete!');
}
