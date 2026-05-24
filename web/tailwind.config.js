/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Victorian Gothic palette
        coal: {
          50: '#f5f4f1',
          100: '#e8e5df',
          200: '#d4cfc4',
          300: '#b8b0a1',
          400: '#9a8d7a',
          500: '#7d7060',
          600: '#655a4e',
          700: '#50483f',
          800: '#3d3630',
          900: '#2a2420',
          950: '#1a1613',
        },
        brass: {
          50: '#fdf8ed',
          100: '#f9edce',
          200: '#f2d89a',
          300: '#e9be61',
          400: '#e0a538',
          500: '#c48722',
          600: '#a86a19',
          700: '#8a5117',
          800: '#724219',
          900: '#5f3618',
          950: '#341b09',
        },
        blood: {
          50: '#fef2f2',
          100: '#fde2e2',
          200: '#fcc8c8',
          300: '#f9a3a3',
          400: '#f36f6f',
          500: '#e84141',
          600: '#d22424',
          700: '#b01c1c',
          800: '#8f1a1a',
          900: '#751a1a',
          950: '#3f0b0b',
        },
        parchment: {
          50: '#fdfbf7',
          100: '#f9f4e8',
          200: '#f2e6cc',
          300: '#e8d2a3',
          400: '#d9b574',
          500: '#cfa35a',
          600: '#c08a48',
          700: '#a06d3c',
          800: '#815835',
          900: '#6a492e',
          950: '#392515',
        },
      },
      fontFamily: {
        serif: ['Georgia', 'Times New Roman', 'serif'],
        body: ['system-ui', '-apple-system', 'sans-serif'],
      },
      animation: {
        'fade-in': 'fadeIn 0.5s ease-out',
        'slide-up': 'slideUp 0.3s ease-out',
        'pulse-glow': 'pulseGlow 2s ease-in-out infinite',
        'typing': 'typing 1.5s ease-in-out infinite',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0' },
          '100%': { opacity: '1' },
        },
        slideUp: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        pulseGlow: {
          '0%, 100%': { opacity: '0.6' },
          '50%': { opacity: '1' },
        },
        typing: {
          '0%, 100%': { opacity: '0.3' },
          '50%': { opacity: '1' },
        },
      },
    },
  },
  plugins: [],
}
