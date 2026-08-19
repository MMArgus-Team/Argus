const assert = require('node:assert/strict')
const test = require('node:test')

const {
  classifyApiRequestError,
  formatApiRequestDiagnostic,
  safeApiMethod,
  safeApiPath,
  safeProfileLabel
} = require('./api-request-diagnostics.cjs')

test('API diagnostics expose only the method and pathname', () => {
  assert.equal(safeApiMethod('post'), 'POST')
  assert.equal(safeApiMethod('GET /secret'), 'UNKNOWN')
  assert.equal(safeApiPath('/api/status?token=secret#private'), '/api/status')
  assert.equal(safeApiPath('https://user:password@backend.invalid/api/config?api_key=secret'), '/api/config')
  assert.equal(safeApiPath('http://['), '<invalid-path>')
})

test('API diagnostics sanitize profile labels instead of logging arbitrary text', () => {
  assert.equal(safeProfileLabel(undefined), '<default>')
  assert.equal(safeProfileLabel('work-profile'), 'work-profile')
  assert.equal(safeProfileLabel('work profile\ntoken=secret'), 'work_profile_token_secret')
})

test('API diagnostic lines omit queries, fragments, bodies, and exception messages', () => {
  const line = formatApiRequestDiagnostic({
    phase: 'failed',
    requestId: 'api-7',
    method: 'get',
    requestPath: '/api/profiles/sessions?token=secret#private',
    profile: 'default',
    stage: 'http',
    elapsedMs: 15_004.6,
    timeoutMs: 15_000,
    errorKind: classifyApiRequestError(new Error('Timed out with token=secret'))
  })

  assert.equal(
    line,
    '[api-debug] failed id=api-7 method=GET path=/api/profiles/sessions profile=default stage=http elapsed_ms=15005 timeout_ms=15000 error=timeout'
  )
  assert.doesNotMatch(line, /secret|token=|private/)
})

test('API diagnostic error kinds distinguish common transport failures', () => {
  assert.equal(classifyApiRequestError(Object.assign(new Error('socket failed'), { code: 'ECONNREFUSED' })), 'connection_refused')
  assert.equal(classifyApiRequestError(Object.assign(new Error('socket failed'), { code: 'ECONNRESET' })), 'connection_reset')
  assert.equal(classifyApiRequestError(Object.assign(new Error('cancelled'), { name: 'AbortError' })), 'aborted')
  assert.equal(classifyApiRequestError(new Error('bad response body')), 'request_failed')
})
