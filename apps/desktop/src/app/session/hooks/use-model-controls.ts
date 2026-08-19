import { type QueryClient } from '@tanstack/react-query'
import { useCallback, useRef } from 'react'

import { getGlobalModelInfo } from '@/hermes'
import { useI18n } from '@/i18n'
import { notify, notifyError } from '@/store/notifications'
import { $activeSessionId, $currentModel, $currentProvider, setCurrentModel, setCurrentProvider } from '@/store/session'
import type { ModelOptionsResponse } from '@/types/hermes'

interface ModelSelection {
  model: string
  provider: string
}

/** `config.set{key:"model"}` reply. `confirm_required` means the backend
 *  returned before switching — see the guard in selectModel. */
interface ModelSwitchResponse {
  confirm_message?: string
  confirm_required?: boolean
  value?: string
  warning?: string
}

interface ModelControlsOptions {
  activeSessionId: string | null
  queryClient: QueryClient
  requestGateway: <T = unknown>(method: string, params?: Record<string, unknown>) => Promise<T>
}

export function useModelControls({ activeSessionId, queryClient, requestGateway }: ModelControlsOptions) {
  const { t } = useI18n()
  const copy = t.desktop
  const selectModelRef = useRef<((selection: ModelSelection, confirm?: boolean) => Promise<boolean>) | null>(null)

  const updateModelOptionsCache = useCallback(
    (provider: string, model: string, includeGlobal: boolean) => {
      const patch = (prev: ModelOptionsResponse | undefined) => ({ ...(prev ?? {}), provider, model })

      queryClient.setQueryData<ModelOptionsResponse>(['model-options', activeSessionId || 'global'], patch)

      if (includeGlobal) {
        queryClient.setQueryData<ModelOptionsResponse>(['model-options', 'global'], patch)
      }
    },
    [activeSessionId, queryClient]
  )

  // Seed the composer's model state from the profile default. `force` reseeds
  // for a profile swap (the new profile has its own default); otherwise this
  // only fills an EMPTY selection so a user's pick (plain UI state in
  // $currentModel) survives the lifecycle refreshes that fire on boot / fresh
  // draft / session events. A live session owns the footer, so skip entirely.
  const refreshCurrentModel = useCallback(async (force = false) => {
    try {
      if ($activeSessionId.get()) {
        return
      }

      if (!force && $currentModel.get()) {
        return
      }

      const result = await getGlobalModelInfo()

      if ($activeSessionId.get() || (!force && $currentModel.get())) {
        return
      }

      if (typeof result.model === 'string') {
        setCurrentModel(result.model)
      }

      if (typeof result.provider === 'string') {
        setCurrentProvider(result.provider)
      }
    } catch {
      // The delayed session.info event still updates this once the agent is ready.
    }
  }, [])

  // Returns whether the switch succeeded so callers can await it before applying
  // follow-up changes. The composer model is plain UI state: with no live
  // session it's just stored (and shipped on the next session.create); with one
  // it's a SESSION-scoped hot swap via config.set — the backend calls
  // agent.switch_model(), so it lands on the next turn without losing context.
  // It never writes the profile default (that lives in Settings → Model), which
  // is what the explicit `--session` flag below guarantees.
  const selectModel = useCallback(
    async (selection: ModelSelection, confirmExpensiveModel = false): Promise<boolean> => {
      // Snapshot for rollback: the switch is applied optimistically, so a
      // failure must restore the prior model/provider (store + query cache)
      // rather than leave the UI showing a model the backend never selected.
      const prevModel = $currentModel.get()
      const prevProvider = $currentProvider.get()

      const rollback = () => {
        setCurrentModel(prevModel)
        setCurrentProvider(prevProvider)
        updateModelOptionsCache(prevProvider, prevModel, !activeSessionId)
      }

      setCurrentModel(selection.model)
      setCurrentProvider(selection.provider)
      updateModelOptionsCache(selection.provider, selection.model, !activeSessionId)

      // No live session yet: the pick is pure UI state. session.create reads
      // $currentModel/$currentProvider and applies it as that session's override.
      if (!activeSessionId) {
        return true
      }

      try {
        const result = await requestGateway<ModelSwitchResponse>('config.set', {
          confirm_expensive_model: confirmExpensiveModel,
          session_id: activeSessionId,
          key: 'model',
          // ★ `--session` is load-bearing. Without it the backend's
          //   resolve_persist_behavior() falls through to
          //   `model.persist_switch_by_default`, which DEFAULTS TO TRUE — so a
          //   casual pick in the composer silently rewrote the global profile
          //   default in config.yaml, the exact opposite of what this hook's
          //   contract promises. Scope it to this session explicitly.
          value: `${selection.model} --provider ${selection.provider} --session`
        })

        // ★ `confirm_required` means the backend returned BEFORE calling
        //   agent.switch_model() (expensive-model guard) — the switch did NOT
        //   happen. Ignoring it left the UI showing a model the agent was not
        //   running. Roll back and ask; the retry passes
        //   confirm_expensive_model so the guard is satisfied.
        if (result?.confirm_required) {
          rollback()
          notify({
            action: {
              label: copy.modelSwitchConfirmAction,
              onClick: () => void selectModelRef.current?.(selection, true)
            },
            durationMs: 0,
            kind: 'warning',
            message: result.confirm_message || result.warning || copy.modelSwitchConfirmMessage,
            title: copy.modelSwitchConfirmTitle
          })

          return false
        }

        // ★ The backend may have auto-corrected a near-miss name (a typo, or a
        //   variant the endpoint doesn't serve: `glm-5v-turbo` → `glm-5-turbo`).
        //   `value` is what the agent ACTUALLY switched to, so it wins over the
        //   optimistic guess — otherwise the composer keeps advertising a model
        //   that was never selected.
        if (result?.value && result.value !== selection.model) {
          setCurrentModel(result.value)
          updateModelOptionsCache(selection.provider, result.value, !activeSessionId)
        }

        // A non-blocking warning: the switch DID apply, but something is worth
        // knowing — the auto-correction above, or `recognized: false` ("couldn't
        // verify this model name"), which is the only early signal that api_mode
        // inference guessed the wire protocol wrong. Swallowing it meant the user
        // got a bare 404 on the next turn with no idea why.
        if (result?.warning) {
          notify({ kind: 'warning', message: result.warning, title: copy.modelSwitchWarningTitle })
        }

        void queryClient.invalidateQueries({ queryKey: ['model-options', activeSessionId] })

        return true
      } catch (err) {
        rollback()
        notifyError(err, copy.modelSwitchFailed)

        return false
      }
    },
    [
      activeSessionId,
      copy.modelSwitchConfirmAction,
      copy.modelSwitchConfirmMessage,
      copy.modelSwitchConfirmTitle,
      copy.modelSwitchFailed,
      copy.modelSwitchWarningTitle,
      queryClient,
      requestGateway,
      updateModelOptionsCache
    ]
  )

  // The confirm action retries `selectModel` from inside its own notification,
  // so it needs a stable handle on the latest callback without making the
  // callback depend on itself.
  selectModelRef.current = selectModel

  return { refreshCurrentModel, selectModel, updateModelOptionsCache }
}
