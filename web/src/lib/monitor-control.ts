export type MonitorTriggerMode = "once" | "continuous";

export interface MonitorRegistryItem {
  monitor_id: string;
  brief: string;
  label?: string;
  enabled?: boolean;
  status?: string;
  trigger_mode?: MonitorTriggerMode;
  created_at: number;
}

export interface EphemeralControlFields {
  history_policy?: unknown;
  ephemeral_control?: unknown;
  ephemeral?: unknown;
}

export type MonitorStatusToken = "active" | "done" | "interrupted";

export interface MonitorPresentation {
  active: boolean;
  done: boolean;
  canToggle: boolean;
  mode: MonitorTriggerMode;
  statusToken: MonitorStatusToken;
}

/**
 * Older registry rows predate trigger_mode and behaved continuously. Keeping
 * that fallback here prevents an upgraded dashboard from mislabelling or
 * changing the semantics of already-running monitors.
 */
export function monitorPresentation(monitor: MonitorRegistryItem): MonitorPresentation {
  const mode: MonitorTriggerMode = monitor.trigger_mode === "once" ? "once" : "continuous";
  const done = monitor.status === "done" || monitor.status === "complete";
  const interrupted = monitor.status === "interrupted" || monitor.enabled === false;
  const deleted = monitor.status === "deleted";
  const active = !done && !interrupted && !deleted;

  return {
    active,
    done,
    canToggle: !done && !deleted,
    mode,
    statusToken: done ? "done" : active ? "active" : "interrupted",
  };
}

/**
 * Merge a registry pull with a possibly newer push already rendered locally.
 *
 * During a cold resume the non-blocking registry endpoint may temporarily
 * return an empty list before the agent is built.  That legacy/unknown empty
 * response must not erase a push that already arrived.  Once the backend says
 * `ready=true`, however, even an empty list is authoritative (for example,
 * when the last Monitor was deleted while the browser was disconnected).
 */
export function resolveRegistryPull<T>(
  current: T[],
  incoming: T[] | undefined,
  ready: boolean | undefined,
): T[] {
  if (!Array.isArray(incoming)) return current;
  if (ready === true || incoming.length > 0 || current.length === 0) return incoming;
  return current;
}

/**
 * Compatibility reader for the control-plane completion contract. Backends
 * may roll out either the explicit boolean or the history policy first.
 */
export function isEphemeralControl(payload: EphemeralControlFields | null | undefined): boolean {
  return payload?.ephemeral === true
    || payload?.ephemeral_control === true
    || payload?.history_policy === "ephemeral_control";
}

export interface EphemeralTurnItem {
  id: string;
  requestId?: string;
}

/**
 * Remove a finalized pure Monitor-control turn from the center transcript.
 * A correlated completion removes every center item owned by that request,
 * including all tool cards from the batch. Legacy uncorrelated completions can
 * still remove their known assistant bubble without guessing at nearby items.
 * The right-side registry is separate state and is deliberately unaffected.
 */
export function removeEphemeralControlTurn<T extends EphemeralTurnItem>(
  messages: T[],
  requestId: string,
  assistantId: string,
): T[] {
  return messages.filter((message) => (
    message.id !== assistantId
    && (!requestId || message.requestId !== requestId)
  ));
}
