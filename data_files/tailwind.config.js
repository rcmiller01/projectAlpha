/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./ui/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Emotional color palette
        'emotion-joy': '#F59E0B',
        'emotion-calm': '#10B981',
        'emotion-contemplative': '#8B5CF6',
        'emotion-melancholy': '#6366F1',
        'emotion-excited': '#EF4444',
        'emotion-peaceful': '#06B6D4',
        'emotion-longing': '#EC4899',
        'emotion-warmth': '#F97316',
        'emotion-empathy': '#84CC16',
        'emotion-curiosity': '#A855F7',
        'emotion-concern': '#6B7280',
      },
      gradients: {
        'emotional-joy': 'linear-gradient(135deg, #F59E0B 0%, #FCD34D 100%)',
        'emotional-calm': 'linear-gradient(135deg, #10B981 0%, #6EE7B7 100%)',
        'emotional-contemplative': 'linear-gradient(135deg, #8B5CF6 0%, #C4B5FD 100%)',
        'emotional-melancholy': 'linear-gradient(135deg, #6366F1 0%, #A5B4FC 100%)',
        'emotional-excited': 'linear-gradient(135deg, #EF4444 0%, #FCA5A5 100%)',
        'emotional-peaceful': 'linear-gradient(135deg, #06B6D4 0%, #67E8F9 100%)',
        'emotional-longing': 'linear-gradient(135deg, #EC4899 0%, #F9A8D4 100%)',
      },
      animation: {
        'fadeIn': 'fadeIn 0.5s ease-in-out',
        'slideIn': 'slideIn 0.3s ease-out',
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'bounce-gentle': 'bounce 2s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        slideIn: {
          '0%': { transform: 'translateX(-100%)' },
          '100%': { transform: 'translateX(0)' },
        },
        glow: {
          '0%': { boxShadow: '0 0 5px rgba(139, 92, 246, 0.5)' },
          '100%': { boxShadow: '0 0 20px rgba(139, 92, 246, 0.8)' },
        },
      },
      backdropBlur: {
        'xs': '2px',
      },
      fontFamily: {
        'emotion': ['Inter', 'system-ui', 'sans-serif'],
        'symbolic': ['Crimson Text', 'serif'],
      },
      spacing: {
        '18': '4.5rem',
        '88': '22rem',
      },
      borderRadius: {
        'xl': '1rem',
        '2xl': '1.5rem',
      },
      boxShadow: {
        'emotional': '0 4px 20px rgba(139, 92, 246, 0.2)',
        'mood-ring': '0 0 30px rgba(139, 92, 246, 0.4)',
        'inner-glow': 'inset 0 2px 4px rgba(139, 92, 246, 0.1)',
      },
    },
  },
  plugins: [
    // Custom plugin for emotional utilities
    function({ addUtilities, theme }) {
      const emotionalUtilities = {
        '.mood-ring': {
          borderRadius: '50%',
          transition: 'all 1s ease-in-out',
          backdropFilter: 'blur(4px)',
        },
        '.drift-notification': {
          background: 'linear-gradient(90deg, #8B5CF6 0%, #EC4899 100%)',
          animation: 'fadeIn 0.5s ease-in-out',
        },
        '.message-emotional': {
          background: 'linear-gradient(135deg, #EC4899 0%, #F9A8D4 100%)',
          color: 'white',
        },
        '.message-balanced': {
          background: 'linear-gradient(135deg, #8B5CF6 0%, #C4B5FD 100%)',
          color: 'white',
        },
        '.message-objective': {
          background: 'linear-gradient(135deg, #06B6D4 0%, #67E8F9 100%)',
          color: 'white',
        },
        '.symbolic-response': {
          fontFamily: theme('fontFamily.symbolic'),
          fontStyle: 'italic',
          background: 'linear-gradient(135deg, #8B5CF6 0%, #EC4899 100%)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          color: 'transparent',
        },
        '.typing-indicator': {
          animation: 'pulse 1.5s ease-in-out infinite',
        },
        '.ambient-background': {
          background: 'linear-gradient(135deg, var(--emotion-primary, #8B5CF6)10 0%, var(--emotion-secondary, #C4B5FD)05 100%)',
          transition: 'background 2s ease-in-out',
        },
      };
      
      addUtilities(emotionalUtilities);
    },
  ],
}
