'use strict'

const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')
const vm = require('node:vm')

const {
  buildDisplayMediaStreams,
  resolveHostSystemVersion,
  supportsDisplayMediaSystemAudio
} = require('./display-media-streams.cjs')

const source = { id: 'screen:1:0' }

for (const row of [
  { audioRequested: true, platform: 'win32', systemVersion: '', wantsLoopback: true },
  { audioRequested: false, platform: 'win32', systemVersion: '', wantsLoopback: false },
  { audioRequested: true, platform: 'darwin', systemVersion: '12.7.6', wantsLoopback: false },
  { audioRequested: true, platform: 'darwin', systemVersion: '13.0', wantsLoopback: true },
  { audioRequested: true, platform: 'darwin', systemVersion: '14.2.1', wantsLoopback: true },
  { audioRequested: true, platform: 'darwin', systemVersion: '26.5.1', wantsLoopback: true },
  { audioRequested: true, platform: 'linux', systemVersion: '', wantsLoopback: false }
]) {
  test(`${row.platform} ${row.systemVersion || '-'} audioRequested=${row.audioRequested}`, () => {
    const streams = buildDisplayMediaStreams(
      source,
      row.audioRequested,
      row.platform,
      row.systemVersion
    )

    assert.equal(streams.video, source)
    assert.equal(
      supportsDisplayMediaSystemAudio(row.platform, row.systemVersion),
      row.platform === 'win32' || (row.platform === 'darwin' && row.wantsLoopback)
    )
    if (row.wantsLoopback) {
      assert.equal(streams.audio, 'loopback')
    } else {
      assert.equal(Object.hasOwn(streams, 'audio'), false)
    }
  })
}

test('Electron system version is authoritative for the macOS capability', () => {
  const fakeProcess = { getSystemVersion: () => '14.6.1' }
  assert.equal(resolveHostSystemVersion('darwin', fakeProcess, '21.6.0'), '14.6.1')
})

test('Darwin release is a safe fallback when Electron system version is unavailable', () => {
  assert.equal(resolveHostSystemVersion('darwin', {}, '22.6.0'), '13.0')
  assert.equal(resolveHostSystemVersion('darwin', {}, '25.5.0'), '26.0')
  assert.equal(resolveHostSystemVersion('linux', {}, '6.8.0'), '')
})

test('packaged macOS app declares the system-audio privacy usage description', () => {
  const packagePath = path.join(__dirname, '..', 'package.json')
  const packageJson = JSON.parse(fs.readFileSync(packagePath, 'utf8'))
  const usage = packageJson?.build?.mac?.extendInfo?.NSAudioCaptureUsageDescription

  assert.equal(typeof usage, 'string')
  assert.ok(usage.trim().length > 0)
})

test('sandboxed preload does not require a local capability helper', () => {
  const preloadPath = path.join(__dirname, 'preload.cjs')
  const preloadSource = fs.readFileSync(preloadPath, 'utf8')

  // Sandboxed preload scripts can require Electron and a small built-in module
  // subset, but Electron explicitly disallows splitting them into local CJS
  // modules. Requiring display-media-streams.cjs here would break the complete
  // hermesDesktop bridge, not just the audio capability.
  assert.doesNotMatch(preloadSource, /require\(['"]\.\/display-media-streams\.cjs['"]\)/)
  assert.match(preloadSource, /process\.getSystemVersion/)
})

test('sandboxed preload exposes the same Windows/macOS/Linux boundary', () => {
  const preloadPath = path.join(__dirname, 'preload.cjs')
  const preloadSource = fs.readFileSync(preloadPath, 'utf8')

  for (const row of [
    { platform: 'win32', systemVersion: '10.0.26100', supported: true },
    { platform: 'darwin', systemVersion: '12.7.6', supported: false },
    { platform: 'darwin', systemVersion: '13.0', supported: true },
    { platform: 'darwin', systemVersion: '26.5.1', supported: true },
    { platform: 'linux', systemVersion: '6.8.0', supported: false }
  ]) {
    let exposedApi = null
    const electron = {
      contextBridge: {
        exposeInMainWorld(name, value) {
          if (name === 'hermesDesktop') exposedApi = value
        }
      },
      ipcRenderer: {},
      webUtils: {}
    }

    vm.runInNewContext(preloadSource, {
      process: {
        getSystemVersion: () => row.systemVersion,
        platform: row.platform
      },
      require(specifier) {
        assert.equal(specifier, 'electron')
        return electron
      }
    })

    assert.equal(exposedApi?.screenShareSystemAudio, row.supported, row.platform)
  }
})
