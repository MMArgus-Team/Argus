/**
 * Slash command execution pipeline for the web chat.
 *
 * Mirrors the Ink TUI's createSlashHandler.ts:
 *
 *   1. Parse the command into `name` + `arg`.
 *   2. Try `slash.exec` — covers every registry-backed command the terminal
 *      UI knows about (/help, /resume, /compact, /model, …). Output is
 *      rendered into the transcript.
 *   3. The gateway routes a set of commands (retry/queue/q/steer/plan/goal/
 *      moa/undo/learn — see tui_gateway/server.py::_PENDING_INPUT_COMMANDS)
 *      straight to command.dispatch and returns a STRUCTURED DIRECTIVE from
 *      slash.exec itself. So before treating the response as plain output we
 *      parse it as a dispatch — otherwise /undo would render "/undo: no
 *      output" and do nothing.
 *   4. If `slash.exec` errors (command rejected, unknown, or needs client
 *      behaviour), fall back to `command.dispatch` which returns a typed
 *      directive: `exec` | `plugin` | `alias` | `skill` | `send` | `prefill`.
 *   5. Each directive is dispatched to the appropriate callback.
 *
 * Keeping the pipeline here (instead of inline in ChatPage) lets future
 * clients (SwiftUI, Android) implement the same logic by reading the same
 * contract.
 */

import type { GatewayClient } from "@/lib/gatewayClient";

export interface SlashExecResponse {
  output?: string;
  warning?: string;
}

export type CommandDispatchResponse =
  | { type: "exec" | "plugin"; output?: string }
  | { type: "alias"; target: string }
  | { type: "skill"; name: string; message?: string }
  | { type: "send"; message: string; notice?: string }
  | { type: "prefill"; message: string; notice?: string };

export interface SlashExecCallbacks {
  /** Render a transcript system message. */
  sys(text: string): void;
  /** Submit a user message to the agent (prompt.submit). */
  send(message: string): Promise<void> | void;
  /** Drop text into the composer for editing (e.g. /undo backs up the last
   *  user turn so it can be edited and resubmitted). Optional: callers that
   *  cannot prefill fall back to surfacing the text as a system line. */
  prefill?(message: string): void;
}

export interface SlashExecOptions {
  /** Raw command including the leading slash (e.g. "/model opus-4.6"). */
  command: string;
  /** Session id. If empty the call is still issued — some commands are session-less. */
  sessionId: string;
  gw: GatewayClient;
  callbacks: SlashExecCallbacks;
}

export type SlashExecResult = "done" | "sent" | "error";

/** Human-readable message from an unknown rejection (mirrors Ink's rpc.ts). */
export function rpcErrorMessage(err: unknown): string {
  return err instanceof Error && err.message
    ? err.message
    : typeof err === "string" && err.trim()
      ? err
      : "request failed";
}

/**
 * Run a slash command. Returns the terminal state so callers can decide
 * whether to clear the composer, queue retries, etc.
 */
export async function executeSlash({
  command,
  sessionId,
  gw,
  callbacks: { sys, send, prefill },
}: SlashExecOptions): Promise<SlashExecResult> {
  const { name, arg } = parseSlash(command);

  if (!name) {
    sys("empty slash command");
    return "error";
  }

  // Shared directive handler. Handles every type command.dispatch (or
  // slash.exec via the _PENDING_INPUT_COMMANDS route) can return.
  const dispatchDirective = async (
    d: CommandDispatchResponse,
  ): Promise<SlashExecResult> => {
    switch (d.type) {
      case "exec":
      case "plugin":
        sys(d.output ?? "(no output)");
        return "done";

      case "alias":
        return executeSlash({
          command: `/${d.target}${arg ? ` ${arg}` : ""}`,
          sessionId,
          gw,
          callbacks: { sys, send, prefill },
        });

      case "skill":
        sys(`⚡ loading skill: ${d.name}`);
        return submitOrError(name, d.message, "skill payload missing message");

      case "send":
        // /goal, /moa, /retry, /steer…: the gateway may attach a `notice`
        // that must render as a sys line BEFORE the message is submitted.
        if (d.notice?.trim()) sys(d.notice);
        return submitOrError(name, d.message, "empty message");

      case "prefill": {
        // /undo returns prefill: drop the backed-up message text into the
        // composer so the user can edit and resubmit, instead of submitting
        // it immediately like 'send'.
        if (d.notice?.trim()) sys(d.notice);
        if (d.message) {
          if (prefill) prefill(d.message);
          else sys(`/undo: ${d.message}`);
        }
        return "done";
      }
    }
  };

  const submitOrError = (
    cmdName: string,
    message: string | undefined,
    emptyLabel: string,
  ): SlashExecResult => {
    const msg = message?.trim() ?? "";
    if (!msg) {
      sys(`/${cmdName}: ${emptyLabel}`);
      return "error";
    }
    void send(msg);
    return "sent";
  };

  // Primary dispatcher.
  try {
    const r = await gw.request<SlashExecResponse>("slash.exec", {
      command: command.replace(/^\/+/, ""),
      session_id: sessionId,
    });
    // The gateway may have routed this command to command.dispatch and
    // returned a structured directive straight from slash.exec. Parse it
    // FIRST — only a plain `{output}` response renders as a system line.
    const d = parseCommandDispatch(r);
    if (d) return dispatchDirective(d);
    const body = r?.output || `/${name}: no output`;
    sys(r?.warning ? `warning: ${r.warning}\n${body}` : body);
    return "done";
  } catch {
    /* fall through to command.dispatch */
  }

  try {
    const d = parseCommandDispatch(
      await gw.request<unknown>("command.dispatch", {
        name,
        arg,
        session_id: sessionId,
      }),
    );

    if (!d) {
      sys("error: invalid response: command.dispatch");
      return "error";
    }

    return dispatchDirective(d);
  } catch (err) {
    sys(`error: ${rpcErrorMessage(err)}`);
    return "error";
  }
}

export function parseSlash(command: string): { name: string; arg: string } {
  const m = command.replace(/^\/+/, "").match(/^(\S+)\s*(.*)$/);
  return m ? { name: m[1], arg: m[2].trim() } : { name: "", arg: "" };
}

export function parseCommandDispatch(raw: unknown): CommandDispatchResponse | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;

  const r = raw as Record<string, unknown>;
  const str = (v: unknown) => (typeof v === "string" ? v : undefined);

  switch (r.type) {
    case "exec":
    case "plugin":
      return { type: r.type, output: str(r.output) };

    case "alias":
      return typeof r.target === "string"
        ? { type: "alias", target: r.target }
        : null;

    case "skill":
      return typeof r.name === "string"
        ? { type: "skill", name: r.name, message: str(r.message) }
        : null;

    case "send":
      return typeof r.message === "string"
        ? { type: "send", message: r.message, notice: str(r.notice) }
        : null;

    case "prefill":
      return typeof r.message === "string"
        ? { type: "prefill", message: r.message, notice: str(r.notice) }
        : null;

    default:
      return null;
  }
}
