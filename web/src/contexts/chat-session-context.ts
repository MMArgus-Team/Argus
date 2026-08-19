import { createContext } from "react";

import type { GatewayClient } from "@/lib/gatewayClient";

/**
 * The live gateway + session id of the surrounding chat, for leaf controls that
 * need to issue session-scoped RPCs (currently the composer's thinking dial).
 *
 * Exposed as a GETTER rather than plain fields on purpose. The chat page keeps
 * `gw`/`sessionId` in a mutable ref (they change on connect / session switch
 * without a re-render), and the whole composer subtree is memoized so that
 * frame ticks and streaming updates skip it. A value object holding the ids
 * directly would need a new identity on every reconnect — and threading them as
 * props would defeat ChatColumn/ChatComposer's memoization, which exists
 * precisely to keep keystrokes and frame ticks off the message list.
 *
 * `resolve()` returns null when there is no usable session yet (not connected,
 * or a standalone surface with no chat at all), so callers can fall back to the
 * global config write.
 */
export interface ChatSessionContextValue {
  resolve: () => { gw: GatewayClient; sessionId: string } | null;
}

export const ChatSessionContext = createContext<ChatSessionContextValue | null>(
  null,
);
