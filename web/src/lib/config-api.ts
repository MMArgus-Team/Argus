/**
 * Shared helpers for reading/writing the thinking-effort dial.
 *
 * Two scopes, deliberately different destinations:
 *
 *   setSessionReasoningEffort() — THIS session only. Goes through the
 *     `config.set` RPC with scope="session", which sets the live agent's
 *     reasoning_config (re-read on every API call, so it lands on the next
 *     turn) and persists it into the session row. Use this wherever a live
 *     session exists; it is the only variant that affects the running turn.
 *
 *   setReasoningEffort() — the config.yaml baseline for NEW sessions. Only for
 *     surfaces with no session (dashboard Config/Models page). Note that
 *     HERMES_HOME's config.yaml is overwritten from the git-tracked project
 *     config on every dashboard start (sync_project_config), so a value written
 *     here does not necessarily survive a restart — which is exactly why the
 *     live dial must not rely on it.
 *
 * Semantics mirror hermes_constants.parse_reasoning_effort:
 *   ""            → Hermes default (treated as "medium" client-side)
 *   "none"        → thinking OFF
 *   valid level   → thinking ON at that level
 */

import { api } from "@/lib/api";
import type { GatewayClient } from "@/lib/gatewayClient";
import { normalizeEffort, VALID_EFFORTS } from "@/lib/reasoning-effort";

export async function getReasoningEffort(): Promise<string> {
  try {
    const cfg = await api.getConfig();
    const agent = (cfg?.agent as Record<string, unknown> | undefined) ?? {};
    return normalizeEffort(agent.reasoning_effort);
  } catch {
    return "medium";
  }
}

/** Read-modify-write the config so sibling keys are preserved. Returns the
 *  applied effort on success; throws on failure so callers can revert. */
export async function setReasoningEffort(next: string): Promise<string> {
  if (!VALID_EFFORTS.has(next)) {
    throw new Error(`invalid reasoning effort: ${next}`);
  }
  const cfg = await api.getConfig();
  const base = (cfg ?? {}) as Record<string, unknown>;
  const agent =
    base.agent && typeof base.agent === "object"
      ? { ...(base.agent as Record<string, unknown>) }
      : {};
  agent.reasoning_effort = next;
  await api.saveConfig({ ...base, agent });
  return next;
}

/** Apply an effort to ONE live session. Takes effect on that session's next
 *  turn and is persisted with the session (restored on resume), leaving
 *  config.yaml — the baseline for new sessions — untouched. Throws on failure
 *  so callers can revert their optimistic update. */
export async function setSessionReasoningEffort(
  gw: GatewayClient,
  sessionId: string,
  next: string,
): Promise<string> {
  if (!VALID_EFFORTS.has(next)) {
    throw new Error(`invalid reasoning effort: ${next}`);
  }
  await gw.request<{ key: string; value: string; scope: string }>("config.set", {
    key: "reasoning",
    scope: "session",
    session_id: sessionId,
    value: next,
  });
  return next;
}

/** `config.set{key:"model"}` reply. */
export interface SessionModelSwitchResult {
  confirm_message?: string;
  /** True ⇒ the backend returned BEFORE calling agent.switch_model(); the
   *  switch did NOT happen. Re-issue with confirmExpensive to go through. */
  confirm_required?: boolean;
  value?: string;
  /** Applied, but worth reporting — notably `recognized: false` ("couldn't
   *  verify this model name"), the only early signal that api_mode inference
   *  guessed the wire protocol wrong. */
  warning?: string;
}

/** Hot-swap the model on ONE live session. The backend calls
 *  ``agent.switch_model()``, which rebuilds the client in place (with atomic
 *  rollback) — so this lands on the session's next turn WITHOUT discarding the
 *  conversation, and emits ``session.info`` so every surface re-renders.
 *
 *  ★ Why not `POST /api/model/set` (what every web picker used to do, and the
 *    reason switching "did nothing"): that endpoint only does
 *    load_config/save_config. An agent is built once per session and then reads
 *    its model off in-memory attributes, so writing config.yaml has ZERO effect
 *    on a live session — and HERMES_HOME/config.yaml is overwritten from the
 *    git-tracked project config on every start (sync_project_config), so the
 *    value need not even survive a restart. REST is right for "the default for
 *    NEW sessions" and nothing else. Same reasoning as the effort dial above.
 *
 *  ``--session`` is load-bearing: without it the backend's
 *  resolve_persist_behavior() falls through to `model.persist_switch_by_default`
 *  (which DEFAULTS TO TRUE), so a casual pick would rewrite the global profile
 *  default. Throws on transport failure so callers can revert. */
export async function setSessionModel(
  gw: GatewayClient,
  sessionId: string,
  {
    confirmExpensive = false,
    model,
    provider,
  }: { confirmExpensive?: boolean; model: string; provider: string },
): Promise<SessionModelSwitchResult> {
  return gw.request<SessionModelSwitchResult>("config.set", {
    confirm_expensive_model: confirmExpensive,
    key: "model",
    session_id: sessionId,
    value: `${model} --provider ${provider} --session`,
  });
}
