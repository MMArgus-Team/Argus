'use strict'

const API_PENDING_LOG_MS = 5_000
// Keep routine cold-start calls out of desktop.log. Anything that reaches the
// pending threshold is worth pairing with a completion/failure line.
const API_SLOW_LOG_MS = API_PENDING_LOG_MS

function safeApiMethod(method) {
  const normalized = String(method || 'GET').trim().toUpperCase()
  return /^[A-Z]{1,16}$/.test(normalized) ? normalized : 'UNKNOWN'
}

function safeApiPath(rawPath) {
  try {
    const parsed = new URL(String(rawPath || '/'), 'http://hermes.invalid')
    return parsed.pathname || '/'
  } catch {
    return '<invalid-path>'
  }
}

function safeProfileLabel(profile) {
  const raw = String(profile || '').trim()
  if (!raw) return '<default>'
  const normalized = raw.replace(/[^A-Za-z0-9_.-]/g, '_').slice(0, 64)
  return normalized || '<default>'
}

function safeDiagnosticToken(value, fallback) {
  const normalized = String(value || '')
    .trim()
    .replace(/[^A-Za-z0-9_.-]/g, '_')
    .slice(0, 64)
  return normalized || fallback
}

function classifyApiRequestError(error) {
  const message = String(error?.message || '')
  const code = String(error?.code || '').toUpperCase()
  if (/timed out|timeout/i.test(message) || code === 'ETIMEDOUT') return 'timeout'
  if (code === 'ECONNREFUSED') return 'connection_refused'
  if (code === 'ECONNRESET') return 'connection_reset'
  if (error?.name === 'AbortError' || /abort/i.test(message)) return 'aborted'
  return 'request_failed'
}

function formatApiRequestDiagnostic({
  phase,
  requestId,
  method,
  requestPath,
  profile,
  stage,
  elapsedMs,
  timeoutMs,
  errorKind
}) {
  const parts = [
    '[api-debug]',
    safeDiagnosticToken(phase, 'event'),
    `id=${safeDiagnosticToken(requestId, 'unknown')}`,
    `method=${safeApiMethod(method)}`,
    `path=${safeApiPath(requestPath)}`,
    `profile=${safeProfileLabel(profile)}`,
    `stage=${safeDiagnosticToken(stage, 'unknown')}`,
    `elapsed_ms=${Math.max(0, Math.round(Number(elapsedMs) || 0))}`
  ]
  if (Number.isFinite(Number(timeoutMs)) && Number(timeoutMs) > 0) {
    parts.push(`timeout_ms=${Math.round(Number(timeoutMs))}`)
  }
  if (errorKind) {
    parts.push(`error=${safeDiagnosticToken(errorKind, 'request_failed')}`)
  }
  return parts.join(' ')
}

module.exports = {
  API_PENDING_LOG_MS,
  API_SLOW_LOG_MS,
  classifyApiRequestError,
  formatApiRequestDiagnostic,
  safeApiMethod,
  safeApiPath,
  safeProfileLabel
}
