/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        consistent: '#10b981',
        inconsistent: '#ef4444',
        neutral: '#6b7280',
        support: '#10b981',
        contradict: '#ef4444',
        neutralGray: '#9ca3af',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
