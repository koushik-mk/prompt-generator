import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/tests/setup.js',
    css: false,
    // Pinned so a developer .env.local cannot alter test expectations.
    env: {
      VITE_API_BASE_URL: '',
    },
  },
})
