'use strict'

const os = require('node:os')

const MIN_MACOS_SYSTEM_AUDIO_MAJOR = 13

function numericMajor(version) {
  const match = String(version || '').trim().match(/^(\d+)/)
  return match ? Number.parseInt(match[1], 10) : null
}

/**
 * Return the user-facing OS version Electron reports.
 *
 * `process.getSystemVersion()` is the authority on macOS.  Plain Node test
 * runners do not expose it, so keep a Darwin-kernel fallback. Darwin 22–24
 * map to macOS 13–15; Apple then aligned Darwin 25 with macOS 26.
 */
function resolveHostSystemVersion(
  platform = process.platform,
  electronProcess = process,
  kernelRelease = os.release()
) {
  try {
    const version = electronProcess?.getSystemVersion?.()
    if (typeof version === 'string' && version.trim()) return version.trim()
  } catch {
    // Fall through to the Darwin release mapping below.
  }

  if (platform !== 'darwin') return ''
  const darwinMajor = numericMajor(kernelRelease)
  if (darwinMajor === null || darwinMajor < 20) return ''
  const macOSMajor = darwinMajor >= 25 ? darwinMajor + 1 : darwinMajor - 9
  return `${macOSMajor}.0`
}

/** Electron 40 supports display-media system audio on Windows and macOS 13+. */
function supportsDisplayMediaSystemAudio(
  platform = process.platform,
  systemVersion = resolveHostSystemVersion(platform)
) {
  if (platform === 'win32') return true
  if (platform !== 'darwin') return false

  const macOSMajor = numericMajor(systemVersion)
  return macOSMajor !== null && macOSMajor >= MIN_MACOS_SYSTEM_AUDIO_MAJOR
}

/** Build the streams object accepted by Electron's display-media callback. */
function buildDisplayMediaStreams(
  video,
  audioRequested,
  platform = process.platform,
  systemVersion = resolveHostSystemVersion(platform)
) {
  if (audioRequested === true && supportsDisplayMediaSystemAudio(platform, systemVersion)) {
    return { audio: 'loopback', video }
  }

  return { video }
}

module.exports = {
  buildDisplayMediaStreams,
  resolveHostSystemVersion,
  supportsDisplayMediaSystemAudio
}
