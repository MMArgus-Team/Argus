import { Button } from "@nous-research/ui/ui/components/button";
import { Checkbox } from "@nous-research/ui/ui/components/checkbox";
import { ListItem } from "@nous-research/ui/ui/components/list-item";
import { Spinner } from "@nous-research/ui/ui/components/spinner";
import { Input } from "@nous-research/ui/ui/components/input";
import { Label } from "@nous-research/ui/ui/components/label";
import { ConfirmDialog } from "@/components/ConfirmDialog";
import { ThinkingSlider } from "@/components/ThinkingSlider";
import {
  getReasoningEffort,
  setReasoningEffort,
} from "@/lib/config-api";
import { normalizeEffort, VALID_EFFORTS } from "@/lib/reasoning-effort";
import { Check, Search, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { cn, themedBody } from "@/lib/utils";
import { fuzzyRank } from "@/lib/fuzzy";

/**
 * Two-stage model picker modal.
 *
 * Mirrors ui-tui/src/components/modelPicker.tsx:
 *   Stage 1: pick provider (authenticated providers only)
 *   Stage 2: pick model within that provider
 *
 * One invocation mode: pass a `loader` and `onApply`. The picker fetches
 * options over REST and calls `onApply({provider, model, ...})`, which writes
 * the config.yaml baseline for NEW sessions (ModelsPage main/auxiliary/MoA).
 *
 * ★ This dialog is deliberately NOT a session-switch surface. Hot-swapping the
 *   model on a LIVE session is `config.set{key:"model"}` → agent.switch_model(),
 *   and it belongs to the composer's ChatModelPill (see lib/config-api
 *   setSessionModel). This picker used to carry two extra branches for that
 *   (`gw`+`sessionId` → config.set, and `onSubmit` → a /model slash command),
 *   but every caller passed `onApply`, so both were unreachable — and the one
 *   chat surface that used it wrote config.yaml over REST, which cannot affect a
 *   running agent. They were removed rather than left as a second path to drift.
 */

interface ModelOptionProvider {
  name: string;
  slug: string;
  models?: string[];
  total_models?: number;
  is_current?: boolean;
  warning?: string;
}

interface ModelOptionsResponse {
  model?: string;
  provider?: string;
  providers?: ModelOptionProvider[];
}

interface ExpensiveModelConfirmResponse {
  confirm_message?: string;
  confirm_required?: boolean;
  warning?: string;
}

interface PendingExpensiveConfirm {
  message: string;
  model: string;
  persistGlobal: boolean;
  provider: string;
}

interface Props {
  /** Fetches the provider/model catalog over REST. */
  loader(): Promise<ModelOptionsResponse>;
  /** Writes the choice as the config.yaml baseline for new sessions. */
  onApply(args: {
    confirmExpensiveModel?: boolean;
    provider: string;
    model: string;
    persistGlobal: boolean;
  }):
    | Promise<ExpensiveModelConfirmResponse | void>
    | ExpensiveModelConfirmResponse
    | void;

  onClose(): void;
  title?: string;
  /** If true, hides "Persist globally" checkbox — always saves to config.yaml. */
  alwaysGlobal?: boolean;
}

export function ModelPickerDialog(props: Props) {
  const {
    loader,
    onApply,
    onClose,
    title = "Switch Model",
    alwaysGlobal = false,
  } = props;

  const [providers, setProviders] = useState<ModelOptionProvider[]>([]);
  const [currentModel, setCurrentModel] = useState("");
  const [currentProviderSlug, setCurrentProviderSlug] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedSlug, setSelectedSlug] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [query, setQuery] = useState("");
  const [persistGlobal, setPersistGlobal] = useState(alwaysGlobal);
  const [applying, setApplying] = useState(false);
  const [pendingConfirm, setPendingConfirm] =
    useState<PendingExpensiveConfirm | null>(null);
  // Thinking tier — mirrors ChatModelPill. Reads `agent.reasoning_effort` on
  // open, writes it via the shared config-api helper so ChatModelPill/desktop
  // stay in lockstep. Optimistic update; revert on failure.
  const [effort, setEffort] = useState<string>("medium");
  const [savingEffort, setSavingEffort] = useState(false);
  const closedRef = useRef(false);

  // Load providers + models on open.
  useEffect(() => {
    closedRef.current = false;

    loader()
      .then((r) => {
        if (closedRef.current) return;
        const next = r?.providers ?? [];
        setProviders(next);
        setCurrentModel(String(r?.model ?? ""));
        setCurrentProviderSlug(String(r?.provider ?? ""));
        setSelectedSlug(
          (next.find((p) => p.is_current) ?? next[0])?.slug ?? "",
        );
        setSelectedModel("");
        setLoading(false);
      })
      .catch((e) => {
        if (closedRef.current) return;
        setError(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });

    // Seed the Thinking tier from config.yaml. Best-effort — if it fails we
    // keep the "medium" default so the bar still has a highlighted tier.
    void getReasoningEffort()
      .then((e) => {
        if (closedRef.current) return;
        setEffort(normalizeEffort(e));
      })
      .catch(() => {
        /* keep default */
      });

    return () => {
      closedRef.current = true;
    };
    // Deliberately omit props from deps — stable for the dialog's lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Esc closes.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const selectedProvider = useMemo(
    () => providers.find((p) => p.slug === selectedSlug) ?? null,
    [providers, selectedSlug],
  );

  const models = useMemo(
    () => selectedProvider?.models ?? [],
    [selectedProvider],
  );

  const trimmedQuery = query.trim();

  // Fuzzy-ranked providers: match on name + slug + the provider's model ids so
  // typing a model name surfaces its provider (preserves the prior behaviour
  // where a model match also revealed its provider).
  const filteredProviders = useMemo(
    () =>
      fuzzyRank(
        providers,
        trimmedQuery,
        (p) => `${p.name} ${p.slug} ${(p.models ?? []).join(" ")}`,
      ).map((r) => r.item),
    [providers, trimmedQuery],
  );

  // Fuzzy-ranked models carrying the matched character positions so the model
  // list can highlight why each entry matched.
  const filteredModels = useMemo(
    () =>
      fuzzyRank(models, trimmedQuery, (m) => m).map((r) => ({
        model: r.item,
        positions: r.positions,
      })),
    [models, trimmedQuery],
  );

  const canConfirm = !!selectedProvider && !!selectedModel && !applying;

  const applySelection = async (
    confirmExpensiveModel = false,
    forced?: PendingExpensiveConfirm,
  ) => {
    const providerSlug = forced?.provider ?? selectedProvider?.slug ?? "";
    const model = forced?.model ?? selectedModel;
    const shouldPersistGlobal = forced?.persistGlobal ?? persistGlobal;

    if (!providerSlug || !model || applying) return;

    setApplying(true);
    try {
      const result = await onApply({
        confirmExpensiveModel,
        provider: providerSlug,
        model,
        persistGlobal: shouldPersistGlobal,
      });
      if (result?.confirm_required) {
        setPendingConfirm({
          provider: providerSlug,
          model,
          persistGlobal: shouldPersistGlobal,
          message:
            result.confirm_message ||
            result.warning ||
            "This model has unusually high known pricing.",
        });
        return;
      }
      onClose();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setApplying(false);
    }
  };

  const confirm = () => {
    if (!canConfirm) return;
    void applySelection();
  };

  // This dialog has no session (see the header note), so the dial writes the
  // config.yaml baseline for NEW sessions. Changing a LIVE session's effort is
  // ChatModelPill's job, via setSessionReasoningEffort.
  // ★ No `next === effort` short-circuit: see the matching note in
  //   ChatModelPill.applyEffort — it made a persistence bug look like a dead
  //   control by swallowing clicks on the currently-displayed tier.
  const applyEffort = (next: string) => {
    if (!VALID_EFFORTS.has(next) || savingEffort) return;
    const prev = effort;
    setEffort(next); // optimistic
    setSavingEffort(true);
    setReasoningEffort(next)
      .catch(() => setEffort(prev))
      .finally(() => setSavingEffort(false));
  };

  // Portal to document.body: the main dashboard column in App.tsx is
  // `relative z-2`, which creates a stacking context that traps fixed
  // descendants below the app sidebar (z-50). Without the portal this
  // modal's z-[100] is scoped to z-2 and the sidebar covers its left
  // edge — visible especially in the Large theme variants where the
  // larger root font widens the dialog into the sidebar's column. See
  // Toast.tsx for the same pattern.
  return createPortal(
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-background/85 backdrop-blur-sm p-4"
      onClick={(e) => e.target === e.currentTarget && onClose()}
      role="dialog"
      aria-modal="true"
      aria-labelledby="model-picker-title"
    >
      <div className={cn(themedBody, "relative w-full max-w-3xl max-h-[80vh] border border-border bg-card shadow-2xl flex flex-col")}>
        <Button
          ghost
          size="icon"
          onClick={onClose}
          className="absolute right-2 top-2 text-muted-foreground hover:text-foreground"
          aria-label="Close"
        >
          <X />
        </Button>

        <header className="p-5 pb-3 border-b border-border">
          <h2
            id="model-picker-title"
            className="font-mondwest text-display text-base tracking-wider"
          >
            {title}
          </h2>
          <p className="text-xs text-muted-foreground mt-1 font-mono">
            current: {currentModel || "(unknown)"}
            {currentProviderSlug && ` · ${currentProviderSlug}`}
          </p>
        </header>

        <div className="px-5 pt-3 pb-2 border-b border-border">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              autoFocus
              placeholder="Filter providers and models…"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="pl-7 h-8 text-sm"
            />
          </div>
        </div>

        <div className="flex-1 min-h-0 grid grid-cols-[200px_1fr] overflow-hidden">
          <ProviderColumn
            loading={loading}
            error={error}
            providers={filteredProviders}
            total={providers.length}
            selectedSlug={selectedSlug}
            query={trimmedQuery}
            onSelect={(slug) => {
              setSelectedSlug(slug);
              setSelectedModel("");
            }}
          />

          <ModelColumn
            provider={selectedProvider}
            models={filteredModels}
            allModels={models}
            selectedModel={selectedModel}
            currentModel={currentModel}
            currentProviderSlug={currentProviderSlug}
            onSelect={setSelectedModel}
            onConfirm={(m) => {
              setSelectedModel(m);
              void applySelection(false, {
                provider: selectedProvider?.slug ?? "",
                model: m,
                persistGlobal,
                message: "",
              });
            }}
          />
        </div>

        {/* Thinking 6 档滑块 — 与 desktop 的 effort 档位/文案对齐, 换掉之前独立
            在 ChatSidebar 里的 ReasoningPicker (Brain 图标那张 Card)。写入
            `agent.reasoning_effort`, 后端按当前模型能力做映射。 */}
        <div className="border-t border-border px-4 py-2">
          <ThinkingSlider
            onChange={applyEffort}
            saving={savingEffort}
            value={effort}
          />
        </div>

        <footer className="border-t border-border p-3 flex items-center justify-between gap-3 flex-wrap">
          {alwaysGlobal ? (
            <span className="text-xs text-muted-foreground">
              Saves to config.yaml — applies to new sessions.
            </span>
          ) : (
            <div className="flex items-center gap-2">
              <Checkbox
                checked={persistGlobal}
                id="model-picker-persist-global"
                onCheckedChange={(checked) =>
                  setPersistGlobal(checked === true)
                }
              />

              <Label
                className="font-mondwest normal-case tracking-normal text-xs text-muted-foreground cursor-pointer"
                htmlFor="model-picker-persist-global"
              >
                Persist globally (otherwise this session only)
              </Label>
            </div>
          )}

          <div className="flex items-center gap-2 ml-auto">
            <Button outlined onClick={onClose} disabled={applying}>
              Cancel
            </Button>
            <Button onClick={confirm} disabled={!canConfirm}>
              {applying ? <Spinner /> : "Switch"}
            </Button>
          </div>
        </footer>
      </div>
      <ConfirmDialog
        open={!!pendingConfirm}
        title="Expensive Model Warning"
        description={pendingConfirm?.message}
        destructive
        confirmLabel="Switch anyway"
        cancelLabel="Cancel"
        loading={applying}
        onCancel={() => setPendingConfirm(null)}
        onConfirm={() => {
          const pending = pendingConfirm;
          if (!pending) return;
          setPendingConfirm(null);
          void applySelection(true, pending);
        }}
      />
    </div>,
    document.body,
  );
}

/* ------------------------------------------------------------------ */
/*  Provider column                                                    */
/* ------------------------------------------------------------------ */

function ProviderColumn({
  loading,
  error,
  providers,
  total,
  selectedSlug,
  query,
  onSelect,
}: {
  loading: boolean;
  error: string | null;
  providers: ModelOptionProvider[];
  total: number;
  selectedSlug: string;
  query: string;
  onSelect(slug: string): void;
}) {
  return (
    <div className="border-r border-border overflow-y-auto">
      {loading && (
        <div className="flex items-center gap-2 p-4 text-xs text-muted-foreground">
          <Spinner className="text-xs" /> loading…
        </div>
      )}

      {error && <div className="p-4 text-xs text-destructive">{error}</div>}

      {!loading && !error && providers.length === 0 && (
        <div className="p-4 text-xs text-muted-foreground italic">
          {query
            ? "no matches"
            : total === 0
              ? "no authenticated providers"
              : "no matches"}
        </div>
      )}

      {providers.map((p) => {
        const active = p.slug === selectedSlug;
        return (
          <ListItem
            key={p.slug}
            active={active}
            onClick={() => onSelect(p.slug)}
            className={`items-start text-xs border-l-2 ${
              active ? "border-l-primary" : "border-l-transparent"
            }`}
          >
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-1.5">
                <span className="font-medium truncate">{p.name}</span>
                {p.is_current && <CurrentTag />}
              </div>
              <div className="text-xs text-text-secondary font-mono truncate">
                {p.slug} · {p.total_models ?? p.models?.length ?? 0} models
              </div>
            </div>
          </ListItem>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  Model column                                                       */
/* ------------------------------------------------------------------ */

function ModelColumn({
  provider,
  models,
  allModels,
  selectedModel,
  currentModel,
  currentProviderSlug,
  onSelect,
  onConfirm,
}: {
  provider: ModelOptionProvider | null;
  models: { model: string; positions: number[] }[];
  allModels: string[];
  selectedModel: string;
  currentModel: string;
  currentProviderSlug: string;
  onSelect(model: string): void;
  onConfirm(model: string): void;
}) {
  if (!provider) {
    return (
      <div className="overflow-y-auto">
        <div className="p-4 text-xs text-muted-foreground italic">
          pick a provider →
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-y-auto">
      {provider.warning && (
        <div className="p-3 text-xs text-destructive border-b border-border">
          {provider.warning}
        </div>
      )}

      {models.length === 0 ? (
        <div className="p-4 text-xs text-muted-foreground italic">
          {allModels.length
            ? "no models match your filter"
            : "no models listed for this provider"}
        </div>
      ) : (
        models.map(({ model: m, positions }) => {
          const active = m === selectedModel;
          const isCurrent =
            m === currentModel && provider.slug === currentProviderSlug;

          return (
            <ListItem
              key={m}
              active={active}
              onClick={() => onSelect(m)}
              onDoubleClick={() => onConfirm(m)}
              className="px-3 py-1.5 text-xs font-mono"
            >
              <Check
                className={`h-3 w-3 shrink-0 ${active ? "text-primary" : "text-transparent"}`}
              />
              <span className="flex-1 truncate">
                <HighlightedText text={m} positions={positions} />
              </span>
              {isCurrent && <CurrentTag />}
            </ListItem>
          );
        })
      )}
    </div>
  );
}

function CurrentTag() {
  return (
    <span className="text-display text-xs tracking-wider text-primary shrink-0">
      current
    </span>
  );
}

/**
 * Render `text` with the characters at `positions` emphasised, so users can
 * see which characters their fuzzy query matched. Positions are indices into
 * `text`; out-of-range indices are ignored.
 */
function HighlightedText({
  text,
  positions,
}: {
  text: string;
  positions: number[];
}) {
  if (!positions.length) {
    return <>{text}</>;
  }

  const hit = new Set(positions);

  return (
    <>
      {Array.from(text).map((ch, i) =>
        hit.has(i) ? (
          <mark
            key={i}
            className="bg-transparent text-primary font-semibold underline underline-offset-2"
          >
            {ch}
          </mark>
        ) : (
          <span key={i}>{ch}</span>
        ),
      )}
    </>
  );
}
