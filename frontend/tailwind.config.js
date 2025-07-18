/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
        "./src/**/*.{js,jsx,ts,tsx}",
    ],
    darkMode: 'class', // Enable class-based dark mode
    theme: {
        extend: {
            animation: {
                'border-flow': 'borderFlow 3s ease-in-out infinite',
                'typing': 'typing 1.2s ease-in-out infinite',
                'fade-in': 'fadeIn 0.3s ease-out',
                'slide-up': 'slideUp 0.3s ease-out',
                'shimmer': 'shimmer 2s linear infinite',
                'pulse-soft': 'pulseSoft 2s ease-in-out infinite',
                'float': 'float 3s ease-in-out infinite',
            },
            keyframes: {
                borderFlow: {
                    '0%': {
                        transform: 'translateX(-100%)',
                        opacity: '0'
                    },
                    '50%': {
                        opacity: '1'
                    },
                    '100%': {
                        transform: 'translateX(100%)',
                        opacity: '0'
                    }
                },
                shimmer: {
                    '0%': {
                        transform: 'translateX(-100%)'
                    },
                    '100%': {
                        transform: 'translateX(100%)'
                    }
                },
                pulseSoft: {
                    '0%, 100%': {
                        opacity: '0.6',
                        transform: 'scale(1)'
                    },
                    '50%': {
                        opacity: '1',
                        transform: 'scale(1.05)'
                    }
                },
                float: {
                    '0%, 100%': {
                        transform: 'translateY(0px)'
                    },
                    '50%': {
                        transform: 'translateY(-10px)'
                    }
                },
                typing: {
                    '0%, 100%': {
                        opacity: '0.4',
                        transform: 'scale(1)'
                    },
                    '50%': {
                        opacity: '1',
                        transform: 'scale(1.2)'
                    }
                },
                fadeIn: {
                    '0%': { opacity: '0' },
                    '100%': { opacity: '1' }
                },
                slideUp: {
                    '0%': { transform: 'translateY(20px)', opacity: '0' },
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
                },
                glass: {
                    light: 'rgba(255, 255, 255, 0.25)',
                    medium: 'rgba(255, 255, 255, 0.18)',
                    dark: 'rgba(255, 255, 255, 0.1)',
                },
                // Dark mode specific colors
                dark: {
                    50: '#f8fafc',
                    100: '#f1f5f9',
                    200: '#e2e8f0',
                    300: '#cbd5e1',
                    400: '#94a3b8',
                    500: '#64748b',
                    600: '#475569',
                    700: '#334155',
                    800: '#1e293b',
                    900: '#0f172a',
                    950: '#020617',
                }
            },
            backdropBlur: {
                xs: '2px',
            },
            fontFamily: {
                sans: ['-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
            },
            spacing: {
                '18': '4.5rem',
                '88': '22rem',
            },
            borderRadius: {
                '4xl': '2rem',
            },
            boxShadow: {
                'glass': '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
                'glass-inset': 'inset 0 1px 0 0 rgba(255, 255, 255, 0.05)',
                'soft': '0 2px 15px -3px rgba(0, 0, 0, 0.07), 0 10px 20px -2px rgba(0, 0, 0, 0.04)',
                'soft-lg': '0 10px 40px -10px rgba(0, 0, 0, 0.1), 0 2px 10px -2px rgba(0, 0, 0, 0.04)',
                // Dark mode shadows
                'glass-dark': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
                'soft-dark': '0 2px 15px -3px rgba(0, 0, 0, 0.3), 0 10px 20px -2px rgba(0, 0, 0, 0.2)',
            }
        },
    },
    plugins: [],
} 