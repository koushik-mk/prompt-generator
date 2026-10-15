import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: './src/tests/setup.js',
    css: false,
    // Pin the API base URL so a developer's .env.local cannot change
    // test expectations (tests always exercise the same-origin default).
    env: {
      VITE_API_BASE_URL: '',
    },
  },
})
