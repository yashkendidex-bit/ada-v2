import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
    plugins: [react()],
    base: './', // Important for Electron
    server: {
        port: 5174,
        strictPort: true, // 5173 is reserved by the trade-log dev server (launchd agent)
    }
})
