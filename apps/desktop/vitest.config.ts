import { defineConfig, mergeConfig } from 'vitest/config'

import viteConfig from './vite.config'

// Kept separate from vite.config.ts so the renderer build stays on plain
// `vite`'s types and never imports vitest — this file is the only place that
// knows about tests. Merging (rather than redeclaring) inherits the `@` /
// `@hermes/shared` aliases and the react dedupe, which the component tests need.
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      // jsdom is also passed by `test:ui`'s --environment flag; setting it here
      // means a bare `npx vitest run` behaves the same, instead of failing with
      // `document is not defined`.
      environment: 'jsdom',
      // `build/hermes-src/` is a staged copy of the whole repo source (written
      // by scripts/stage-hermes-source.cjs and gitignored). Without excluding
      // it, vitest globs those stale duplicates too — they fail on missing
      // aliases and on `node:test` files, drowning the real results.
      // `electron/` and `scripts/` are `node:test` suites, not vitest ones;
      // they run via `npm run test:desktop:platforms`.
      exclude: [
        '**/node_modules/**',
        '**/dist/**',
        'build/**',
        'electron/**',
        'scripts/**'
      ],
      // Loads the jsdom shims (CSS.escape) before any test module evaluates.
      setupFiles: ['./src/test-setup.ts']
    }
  })
)
