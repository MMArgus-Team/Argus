import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'

import { I18nProvider } from '@/i18n'
import type { ToolArgField } from '@/types/hermes'

import { ToolArgsPanel } from './tool-fallback'

// Renders under en-US so the assertions can pin the English copy directly;
// the keys themselves are checked for all locales by the i18n catalog tests.
const renderPanel = (fields: ToolArgField[]) =>
  render(
    <I18nProvider configClient={null} initialLocale="en">
      <ToolArgsPanel fields={fields} />
    </I18nProvider>
  )

afterEach(cleanup)

describe('ToolArgsPanel', () => {
  it('shows literal args as key/value rows', () => {
    // The gap this closes: a `computer_use` card used to show only the tool
    // name, so the user could see a skill ran but never what it was told to do.
    renderPanel([
      { key: 'action', kind: 'literal', value: 'capture' },
      { key: 'app', kind: 'literal', value: 'Google Chrome' }
    ])

    expect(screen.getByText('action')).toBeTruthy()
    expect(screen.getByText('capture')).toBeTruthy()
    expect(screen.getByText('Google Chrome')).toBeTruthy()
  })

  it('shows a payload field as a length and never its content', () => {
    renderPanel([
      { key: 'group_code', kind: 'literal', value: 'G9' },
      { chars: 22, key: 'message', kind: 'freeform' }
    ])

    expect(screen.getByText('G9')).toBeTruthy()
    expect(screen.getByText('22 chars (not shown)')).toBeTruthy()
  })

  it('withholds a credential entirely, including its length', () => {
    // Stricter than `freeform`: a password's length is itself a clue, so the
    // backend sends the key alone. The row must not borrow the `chars` wording.
    const { container } = renderPanel([
      { key: 'user', kind: 'literal', value: 'alice' },
      { key: 'password', kind: 'credential' }
    ])

    expect(screen.getByText('password')).toBeTruthy()
    expect(screen.getByText('withheld (credential)')).toBeTruthy()
    expect(container.textContent).not.toContain('chars')
  })

  it('renders array/object args as an item count', () => {
    renderPanel([{ count: 3, key: 'urls', kind: 'shape' }])

    expect(screen.getByText('urls')).toBeTruthy()
    expect(screen.getByText('3 items')).toBeTruthy()
  })

  it('renders the elided tail as a bare count with no key', () => {
    renderPanel([
      { key: 'name', kind: 'literal', value: 'search' },
      { count: 4, key: '', kind: 'elided' }
    ])

    expect(screen.getByText('+4 more fields (not shown)')).toBeTruthy()
  })

  it('renders nothing when the backend sent no fields', () => {
    const { container } = renderPanel([])

    expect(container.textContent).toBe('')
  })
})
