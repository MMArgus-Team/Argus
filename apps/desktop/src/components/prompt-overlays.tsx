'use client'

import { useStore } from '@nanostores/react'
import { type FormEvent, useCallback, useEffect, useState } from 'react'

import { PendingApprovalFallback } from '@/components/assistant-ui/tool-approval'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/i18n'
import { triggerHaptic } from '@/lib/haptics'
import { Textarea } from '@/components/ui/textarea'
import { KeyRound, Loader2, Lock, MessageQuestion } from '@/lib/icons'
import { $unanchoredClarifyRequest, clearClarifyRequest } from '@/store/clarify'
import { $gateway } from '@/store/gateway'
import { notifyError } from '@/store/notifications'
import { $secretRequest, $sudoRequest, clearSecretRequest, clearSudoRequest } from '@/store/prompts'

// Renders the modal mid-turn prompts the gateway raises and waits on: sudo
// password and skill secret capture. Dangerous-command / execute_code approval
// prefers the pending tool row, but also has a chat-level fallback when no row
// is mounted (remote gateway sessions can raise the request before the matching
// tool call is visible). Each Python-side caller blocks the agent thread until
// the matching `*.respond` RPC lands; without a renderer the agent stalls until
// its timeout and the tool is BLOCKED. Any close path (Esc, backdrop
// click) funnels through Radix's single `onOpenChange(false)` and maps to a
// refusal, so silence is never mistaken for consent, matching the TUI. We
// deliberately do NOT add onEscapeKeyDown / onInteractOutside handlers — they'd
// fire a second `*.respond` alongside onOpenChange (double-send) or block the
// backdrop-dismiss path.

function SudoDialog() {
  const { t } = useI18n()
  const copy = t.prompts
  const request = useStore($sudoRequest)
  const gateway = useStore($gateway)
  const [password, setPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    setPassword('')
    setSubmitting(false)
  }, [request?.requestId])

  const send = useCallback(
    async (value: string) => {
      if (!request) {
        return
      }

      if (!gateway) {
        notifyError(new Error(copy.gatewayDisconnected), copy.sudoSendFailed)

        return
      }

      setSubmitting(true)

      try {
        await gateway.request<{ status?: string }>('sudo.respond', {
          password: value,
          request_id: request.requestId
        })
        triggerHaptic('submit')
        clearSudoRequest(request.sessionId, request.requestId)
      } catch (error) {
        notifyError(error, copy.sudoSendFailed)
        setSubmitting(false)
      }
    },
    [copy.gatewayDisconnected, copy.sudoSendFailed, gateway, request]
  )

  // Cancel → empty password. The backend treats an empty sudo response as a
  // failed sudo (no command runs), so closing the dialog is a safe refusal.
  const onOpenChange = useCallback(
    (open: boolean) => {
      if (!open && !submitting && request) {
        void send('')
      }
    },
    [request, send, submitting]
  )

  const onSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      void send(password)
    },
    [password, send]
  )

  if (!request) {
    return null
  }

  return (
    <Dialog onOpenChange={onOpenChange} open>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle icon={Lock}>{copy.sudoTitle}</DialogTitle>
          <DialogDescription>{copy.sudoDesc}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-3" onSubmit={onSubmit}>
          <Input
            autoFocus
            disabled={submitting}
            onChange={event => setPassword(event.target.value)}
            placeholder={copy.sudoPlaceholder}
            type="password"
            value={password}
          />
          <DialogFooter>
            <Button disabled={submitting} onClick={() => void send('')} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={submitting} type="submit">
              {submitting ? <Loader2 className="size-3.5 animate-spin" /> : t.common.send}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

function SecretDialog() {
  const { t } = useI18n()
  const copy = t.prompts
  const request = useStore($secretRequest)
  const gateway = useStore($gateway)
  const [value, setValue] = useState('')
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    setValue('')
    setSubmitting(false)
  }, [request?.requestId])

  const send = useCallback(
    async (secret: string) => {
      if (!request) {
        return
      }

      if (!gateway) {
        notifyError(new Error(copy.gatewayDisconnected), copy.secretSendFailed)

        return
      }

      setSubmitting(true)

      try {
        await gateway.request<{ status?: string }>('secret.respond', {
          request_id: request.requestId,
          value: secret
        })
        triggerHaptic('submit')
        clearSecretRequest(request.sessionId, request.requestId)
      } catch (error) {
        notifyError(error, copy.secretSendFailed)
        setSubmitting(false)
      }
    },
    [copy.gatewayDisconnected, copy.secretSendFailed, gateway, request]
  )

  const onOpenChange = useCallback(
    (open: boolean) => {
      if (!open && !submitting && request) {
        void send('')
      }
    },
    [request, send, submitting]
  )

  const onSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault()
      void send(value)
    },
    [send, value]
  )

  if (!request) {
    return null
  }

  return (
    <Dialog onOpenChange={onOpenChange} open>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle icon={KeyRound}>{request.envVar || copy.secretTitle}</DialogTitle>
          <DialogDescription>{request.prompt || copy.secretDesc}</DialogDescription>
        </DialogHeader>

        <form className="grid gap-3" onSubmit={onSubmit}>
          <Input
            autoFocus
            disabled={submitting}
            onChange={event => setValue(event.target.value)}
            placeholder={request.envVar || copy.secretPlaceholder}
            type="password"
            value={value}
          />
          <DialogFooter>
            <Button disabled={submitting} onClick={() => void send('')} type="button" variant="ghost">
              {t.common.cancel}
            </Button>
            <Button disabled={submitting || !value} type="submit">
              {submitting ? <Loader2 className="size-3.5 animate-spin" /> : t.common.send}
            </Button>
          </DialogFooter>
        </form>
      </DialogContent>
    </Dialog>
  )
}

// Standalone clarify overlay for requests with no inline ClarifyTool row.
// The inline tool only mounts against a `clarify` tool-call part; a raw
// clarify.request (e.g. set_monitor's clarify_callback asking the alert mode)
// has no such part, so without this modal the agent blocks on `clarify.respond`
// until the backend `_block` 300s timeout. $unanchoredClarifyRequest is already
// scoped to the active session AND excludes any request an inline row is
// handling, so this never double-renders a normal clarify tool call.
function ClarifyDialog() {
  const { t } = useI18n()
  const copy = t.assistant.clarify
  const request = useStore($unanchoredClarifyRequest)
  const gateway = useStore($gateway)
  const [draft, setDraft] = useState('')
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    setDraft('')
    setSubmitting(false)
  }, [request?.requestId])

  const send = useCallback(
    async (answer: string) => {
      if (!request) {
        return
      }

      if (!gateway) {
        notifyError(new Error(copy.gatewayDisconnected), copy.sendFailed)

        return
      }

      setSubmitting(true)

      try {
        await gateway.request<{ ok?: boolean }>('clarify.respond', {
          answer,
          request_id: request.requestId
        })
        triggerHaptic('submit')
        clearClarifyRequest(request.requestId, request.sessionId)
      } catch (error) {
        notifyError(error, copy.sendFailed)
        setSubmitting(false)
      }
    },
    [copy.gatewayDisconnected, copy.sendFailed, gateway, request]
  )

  // Dismiss (Esc / backdrop) → empty answer. The backend treats "" as a
  // no-choice timeout fallback (see tools/monitor_tool._resolve_silent), so
  // closing is a safe non-answer rather than a hang.
  const onOpenChange = useCallback(
    (open: boolean) => {
      if (!open && !submitting && request) {
        void send('')
      }
    },
    [request, send, submitting]
  )

  if (!request) {
    return null
  }

  const choices = request.choices ?? []

  return (
    <Dialog onOpenChange={onOpenChange} open>
      <DialogContent showCloseButton={false}>
        <DialogHeader>
          <DialogTitle icon={MessageQuestion}>{request.question}</DialogTitle>
        </DialogHeader>

        <div className="grid gap-3">
          {choices.length > 0 && (
            <div className="grid gap-2">
              {choices.map(choice => (
                <Button
                  className="justify-start"
                  disabled={submitting}
                  key={choice}
                  onClick={() => void send(choice)}
                  type="button"
                  variant="outline"
                >
                  {choice}
                </Button>
              ))}
            </div>
          )}

          <form
            className="grid gap-3"
            onSubmit={event => {
              event.preventDefault()
              if (draft.trim()) {
                void send(draft.trim())
              }
            }}
          >
            <Textarea
              autoFocus={choices.length === 0}
              disabled={submitting}
              onChange={event => setDraft(event.target.value)}
              onKeyDown={event => {
                if (event.key === 'Enter' && !event.shiftKey && draft.trim()) {
                  event.preventDefault()
                  void send(draft.trim())
                }
              }}
              placeholder={copy.placeholder}
              rows={2}
              value={draft}
            />
            <DialogFooter>
              <Button disabled={submitting} onClick={() => void send('')} type="button" variant="ghost">
                {copy.skip}
              </Button>
              <Button disabled={submitting || !draft.trim()} type="submit">
                {submitting ? <Loader2 className="size-3.5 animate-spin" /> : copy.continueLabel}
              </Button>
            </DialogFooter>
          </form>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export function PromptOverlays() {
  return (
    <>
      <PendingApprovalFallback />
      <SudoDialog />
      <SecretDialog />
      <ClarifyDialog />
    </>
  )
}
