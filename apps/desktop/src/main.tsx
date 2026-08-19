import './styles.css'
// Side-effect: applies the persisted window translucency on load.
import './store/translucency'

import { QueryClientProvider } from '@tanstack/react-query'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { HashRouter } from 'react-router-dom'

import App from './app'
import { ErrorBoundary } from './components/error-boundary'
import { HapticsProvider } from './components/haptics-provider'
import { I18nProvider } from './i18n'
import { installClipboardShim } from './lib/clipboard'
import { queryClient } from './lib/query-client'
import { ThemeProvider } from './themes/context'

installClipboardShim()

// Dismiss the pre-JS splash from index.html. By the time this module executes,
// the whole eager graph has already been fetched + run — which is precisely the
// gap the splash covers — so we tear it down as the React tree mounts.
//
// Fade out rather than yanking it: React's first commit and the splash removal
// land in different frames, and a hard removal shows one frame of empty
// background between them. `pointer-events: none` goes on immediately (via the
// class) so the doomed splash can't eat a click during the fade.
function dismissBootSplash() {
  const splash = document.getElementById('boot-splash')

  if (!splash) {
    return
  }

  splash.classList.add('is-leaving')
  const drop = () => splash.remove()
  splash.addEventListener('transitionend', drop, { once: true })
  // Belt-and-braces: if the transition never fires (reduced-motion disables it,
  // or the element is display:none'd by something else), still clean up.
  window.setTimeout(drop, 400)
}

// Dev-only: install __PERF_DRIVE__ + __PERF_PROBE__ on window so the
// scripts/ harnesses can drive a synthetic stream + record render cost.
// Tree-shaken out of production builds. (Uses MODE rather than DEV because
// our Vite setup currently bundles with PROD=true even in `vite dev`; see
// scripts/dev-no-hmr.mjs for the surrounding workarounds.)
if (import.meta.env.MODE !== 'production') {
  import('./app/chat/perf-probe')
}

// The pet overlay rides this same bundle (`?win=overlay`) but mounts a tiny,
// transparent, gateway-less surface instead of the full app. Branch before any
// app-shell work so the overlay window stays cheap.
if (new URLSearchParams(window.location.search).get('win') === 'overlay') {
  // The overlay is a small transparent always-on-top surface — a centered
  // "HERMES" splash would be wrong there, so drop it immediately instead of
  // fading (nothing was ever meant to be visible in this window).
  document.getElementById('boot-splash')?.remove()
  void import('./app/pet-overlay/overlay-root').then(({ mountPetOverlay }) => mountPetOverlay())
} else {
  dismissBootSplash()
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <ErrorBoundary label="root">
        <QueryClientProvider client={queryClient}>
          <I18nProvider>
            <ThemeProvider>
              <HapticsProvider>
                <HashRouter>
                  <App />
                </HashRouter>
              </HapticsProvider>
            </ThemeProvider>
          </I18nProvider>
        </QueryClientProvider>
      </ErrorBoundary>
    </StrictMode>
  )
}
