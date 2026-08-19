import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api";
import type { ActionStatusResponse } from "@/lib/api";
import { Toast } from "@nous-research/ui/ui/components/toast";
import { useI18n } from "@/i18n";
import {
  SystemActionsContext,
  type SystemAction,
} from "./system-actions-context";

const ACTION_NAMES: Record<SystemAction, string> = {
  restart: "gateway-restart",
  update: "hermes-update",
};

export function SystemActionsProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const [pendingAction, setPendingAction] = useState<SystemAction | null>(null);
  const [activeAction, setActiveAction] = useState<SystemAction | null>(null);
  const [actionStatus, setActionStatus] = useState<ActionStatusResponse | null>(
    null,
  );
  const [toast, setToast] = useState<ToastState | null>(null);
  const { t } = useI18n();

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 4000);
    return () => clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (!activeAction) return;
    const name = ACTION_NAMES[activeAction];
    let cancelled = false;
    const startedAt = Date.now();
    let healthyStreak = 0;

    const poll = async () => {
      try {
        // For ``restart`` we also probe ``/api/status`` in parallel so the
        // UI doesn't wedge on "Restarting gateway…" forever in the
        // no-service-manager fallback: on that path the ``argus gateway
        // restart`` command exec-hosts the new gateway itself, so its
        // Popen never exits and ``resp.running`` stays ``true`` for the
        // whole lifetime of the new gateway. Once the heartbeat has
        // reported ``gateway_running=true`` a few polls in a row past a
        // short grace window, treat the restart as done.
        const [resp, gwStatus] = await Promise.all([
          api.getActionStatus(name),
          activeAction === "restart"
            ? api.getStatus().catch(() => null)
            : Promise.resolve(null),
        ]);
        if (cancelled) return;
        setActionStatus(resp);
        if (!resp.running) {
          const ok = resp.exit_code === 0;
          setToast({
            type: ok ? "success" : "error",
            message: ok
              ? t.status.actionFinished
              : `${t.status.actionFailed} (exit ${resp.exit_code ?? "?"})`,
          });
          return;
        }
        if (
          activeAction === "restart" &&
          gwStatus?.gateway_running === true &&
          Date.now() - startedAt > 5000
        ) {
          // Require the heartbeat to stay healthy across three
          // consecutive polls (~4.5s) — one green tick right after we
          // kicked off the restart could just be the old gateway before
          // the kill has taken effect.
          healthyStreak += 1;
          if (healthyStreak >= 3) {
            setActionStatus({ ...resp, running: false, exit_code: 0 });
            setToast({
              type: "success",
              message: t.status.actionFinished,
            });
            return;
          }
        } else {
          healthyStreak = 0;
        }
      } catch {
        // transient fetch error; keep polling
      }
      if (!cancelled) setTimeout(poll, 1500);
    };

    poll();
    return () => {
      cancelled = true;
    };
  }, [activeAction, t.status.actionFinished, t.status.actionFailed]);

  const runAction = useCallback(
    async (action: SystemAction) => {
      setPendingAction(action);
      setActionStatus(null);
      try {
        if (action === "restart") {
          await api.restartGateway();
          setActiveAction(action);
        } else {
          const resp = await api.updateHermes();
          // Some installs cannot apply updates from inside the dashboard. The
          // endpoint returns a structured {ok:false, message, update_command}
          // envelope instead of spawning the action; surface that guidance
          // rather than polling a synthetic failed action.
          if (!resp.ok) {
            const cmd = resp.update_command ? `  ${resp.update_command}` : "";
            setToast({
              type: "success",
              message:
                (resp.message ??
                  "Updates don't apply from this dashboard.") +
                cmd,
            });
            return;
          }
          setActiveAction(action);
        }
      } catch (err) {
        const detail = err instanceof Error ? err.message : String(err);
        setToast({
          type: "error",
          message: `${t.status.actionFailed}: ${detail}`,
        });
      } finally {
        setPendingAction(null);
      }
    },
    [t.status.actionFailed],
  );

  const dismissLog = useCallback(() => {
    setActiveAction(null);
    setActionStatus(null);
  }, []);

  const isRunning = activeAction !== null && actionStatus?.running !== false;
  const isBusy = pendingAction !== null || isRunning;

  return (
    <SystemActionsContext.Provider
      value={{
        actionStatus,
        activeAction,
        dismissLog,
        isBusy,
        isRunning,
        pendingAction,
        runAction,
      }}
    >
      {children}
      <Toast toast={toast} />
    </SystemActionsContext.Provider>
  );
}

interface ToastState {
  message: string;
  type: "success" | "error";
}
