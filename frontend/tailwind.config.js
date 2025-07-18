/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./src/**/*.{js,jsx,ts,tsx}",
    ],
    theme: {
        extend: {
            animation: {
                'border-flow': 'borderFlow 2s ease-in-out infinite',
                'typing': 'typing 1.5s ease-in-out infinite',
                'fade-in': 'fadeIn 0.3s ease-out',
                'slide-up': 'slideUp 0.3s ease-out',
            },
            keyframes: {
                borderFlow: {
                    '0%': {
                        background: 'linear-gradient(90deg, #3b82f6 0%, transparent 0%)',
                        backgroundSize: '100% 2px'
                    },
                    '25%': {
                        background: 'linear-gradient(90deg, #3b82f6 25%, transparent 25%)',
                        backgroundSize: '100% 2px'
                    },
                    '50%': {
                        background: 'linear-gradient(90deg, #3b82f6 50%, transparent 50%)',
                        backgroundSize: '100% 2px'
                    },
                    '75%': {
                        background: 'linear-gradient(90deg, #3b82f6 75%, transparent 75%)',
                        backgroundSize: '100% 2px'
                    },
                    '100%': {
                        background: 'linear-gradient(90deg, #3b82f6 100%, transparent 100%)',
                        backgroundSize: '100% 2px'
                    }
                },
                typing: {
                    '0%, 100%': { opacity: '0.3' },
                    '50%': { opacity: '1' }
                },
                fadeIn: {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' }
                },
                slideUp: {
                    '0%': { transform: 'translateY(10px)', opacity: '0' },
                    '100%': { transform: 'translateY(0)', opacity: '1' }
                }
            },
            colors: {
                primary: {
                    50: '#eff6ff',
                    100: '#dbeafe',
                    200: '#bfdbfe',
                    300: '#93c5fd',
                    400: '#60a5fa',
                    500: '#3b82f6',
                    600: '#2563eb',
                    700: '#1d4ed8',
                    800: '#1e40af',
                    900: '#1e3a8a',
                }
            }
        },
    },
    plugins: [],
} 