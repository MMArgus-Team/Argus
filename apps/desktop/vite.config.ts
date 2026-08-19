import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss()],
  css: {
    // Pin an explicit (empty) PostCSS config. Tailwind is handled entirely by
    // `@tailwindcss/vite`, so the renderer needs no PostCSS plugins — and
    // without this, Vite's `postcss-load-config` walks UP the filesystem
    // looking for a stray `postcss.config.*` / `tailwind.config.*`. The desktop
    // build runs from inside the user's home tree (e.g.
    // `C:\Users\<name>\AppData\Local\hermes\hermes-agent\apps\desktop`), so an
    // unrelated Tailwind v3 config higher up the tree gets picked up and
    // reprocesses our v4 stylesheet, failing the build with
    // "`@layer base` is used but no matching `@tailwind base` directive is
    // present." Pinning the config makes the build hermetic.
    postcss: { plugins: [] }
  },
  build: {
    // Keep desktop packaging stable: Shiki ships many dynamic chunks by
    // default, and electron-builder can OOM scanning thousands of files.
    // Collapsing to a single chunk is intentional, so the renderer bundle is
    // large by design (~22 MB). Raise the warning ceiling above that so the
    // cosmetic "chunk larger than 500 kB" nag stays quiet, while still acting
    // as a regression alarm if the bundle balloons well past today's size.
    chunkSizeWarningLimit: 25000,
    rolldownOptions: {
      output: {
        codeSplitting: false
      }
    }
  },
  optimizeDeps: {
    // Dev cold-start: without this, Vite discovers third-party deps lazily as
    // the module graph unfolds. The renderer's eager graph is ~400 local
    // modules deep (10 request round-trips) and reaches 46 packages, so
    // discovery arrives in waves — and every wave that finds a NEW dep triggers
    // a re-optimize + full page reload. Listing the big ones up front collapses
    // that into a single prebundle pass before the first request is served.
    //
    // Only the expensive/deep ones are listed; Vite still auto-discovers the
    // rest. `@tabler/icons-react` is the standout — a 12k-module, 91 MB barrel
    // that `lib/icons.ts` re-exports, so unbundled it means thousands of dev
    // requests. Keep this list in sync when a heavy dep is added to the eager
    // graph (check with: does it appear in the first-paint waterfall?).
    include: [
      '@tabler/icons-react',
      '@icons-pack/react-simple-icons',
      'shiki',
      'react-shiki',
      'katex',
      '@xterm/xterm',
      '@xterm/addon-fit',
      '@xterm/addon-unicode11',
      '@xterm/addon-web-links',
      '@xterm/addon-webgl',
      'motion',
      'streamdown',
      '@assistant-ui/react',
      '@assistant-ui/core',
      '@tanstack/react-query',
      'react-router-dom',
      'radix-ui'
    ]
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@hermes/shared': path.resolve(__dirname, '../shared/src'),
      react: path.resolve(__dirname, '../../node_modules/react'),
      'react-dom': path.resolve(__dirname, '../../node_modules/react-dom'),
      'react/jsx-dev-runtime': path.resolve(__dirname, '../../node_modules/react/jsx-dev-runtime.js'),
      'react/jsx-runtime': path.resolve(__dirname, '../../node_modules/react/jsx-runtime.js')
    },
    dedupe: ['react', 'react-dom']
  },
  server: {
    host: '127.0.0.1',
    port: 5174,
    strictPort: true
  },
  preview: {
    host: '127.0.0.1',
    port: 4174
  }
})
