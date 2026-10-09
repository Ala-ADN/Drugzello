import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(
      ['/predict', '/molecules', '/solvents', '/solubility', '/health', '/ready', '/model']
        .map(path => [path, 'http://127.0.0.1:8000'])
    )
  },
  define: {
    'process.env': {},
    global: 'window'
  }
})
