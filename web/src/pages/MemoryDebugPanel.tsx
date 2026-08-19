/**
 * Memory-debug inspector for the multimodal page.
 *
 * Split out of MultimodalChatPage.tsx: this is a developer-facing panel behind
 * the header's "Memory" button, mounted only when the user opens it. It shares
 * nothing with the chat/video columns beyond its props, so keeping it in the
 * page module only inflated that file (~800 lines) and the main bundle.
 *
 * The MmTrajectory* types stay in MultimodalChatPage.tsx because the page's own
 * progress derivation uses them too; they are imported back here as types only.
 */
import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Activity, Database, FileText, RefreshCw, Search, Table2, X } from "lucide-react";
import { Button } from "@nous-research/ui/ui/components/button";
import { api } from "@/lib/api";
import type {
  MmMemoryDebugEvent,
  MmMemoryDebugFrameResponse,
  MmMemoryDebugSearchResult,
  MmMemoryDebugSessionResponse,
  MmMemoryDebugSessionSummary,
  MmMemoryDebugTraceResponse,
} from "@/lib/api";
import { translateNow, useLocaleRevision } from "@/i18n";
import type { MmTrajectoryEntry, MmTrajectoryFrame } from "./MultimodalChatPage";

type MemoryDebugTab = "memory" | "frame" | "search" | "debug";

function fmtDebugTime(seconds?: number): string {
  if (seconds == null || !isFinite(seconds)) return "";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

function fmtDebugWall(seconds?: number): string {
  if (!seconds) return "";
  return new Date(seconds * 1000).toLocaleString();
}

function fmtDebugBytes(bytes?: number): string {
  const n = Number(bytes || 0);
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function debugJson(v: unknown): string {
  if (typeof v === "string") return v;
  try { return JSON.stringify(v, null, 2); } catch { return String(v); }
}

function extractDebugBox(block: unknown): number[] | null {
  if (!block || typeof block !== "object") return null;
  const b = block as Record<string, unknown>;
  const raw = b.bbox || b.box || b.rect || b.points || b.polygon || b.poly;
  if (!Array.isArray(raw)) return null;
  if (raw.length >= 4 && raw.every((x) => typeof x === "number")) {
    const nums = raw.slice(0, 4) as number[];
    if (nums[2] > nums[0] && nums[3] > nums[1]) return nums;
    return [nums[0], nums[1], nums[0] + Math.max(0, nums[2]), nums[1] + Math.max(0, nums[3])];
  }
  const pts = raw.filter((p) => Array.isArray(p) && p.length >= 2) as unknown[][];
  if (pts.length >= 2) {
    const xs = pts.map((p) => Number(p[0])).filter(Number.isFinite);
    const ys = pts.map((p) => Number(p[1])).filter(Number.isFinite);
    if (xs.length && ys.length) return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  }
  return null;
}

function blockText(block: unknown): string {
  if (!block || typeof block !== "object") return "";
  return String((block as Record<string, unknown>).text || "");
}

const OcrOverlayImage = memo(function OcrOverlayImage({
  imageB64, blocks,
}: { imageB64: string; blocks: unknown[] }) {
  const [size, setSize] = useState<{ w: number; h: number } | null>(null);
  const boxes = useMemo(() => {
    const arr = Array.isArray(blocks) ? blocks : [];
    const rawBoxes = arr.map((b) => ({ block: b, box: extractDebugBox(b) })).filter((x) => x.box) as Array<{ block: unknown; box: number[] }>;
    if (!size || rawBoxes.length === 0) return [];
    const maxX = Math.max(...rawBoxes.map((x) => Math.max(x.box[0], x.box[2])));
    const maxY = Math.max(...rawBoxes.map((x) => Math.max(x.box[1], x.box[3])));
    const norm = maxX <= 1.5 && maxY <= 1.5;
    return rawBoxes.map(({ block, box }) => {
      const [x1, y1, x2, y2] = box;
      return {
        block,
        left: norm ? x1 * 100 : (x1 / Math.max(maxX, size.w)) * 100,
        top: norm ? y1 * 100 : (y1 / Math.max(maxY, size.h)) * 100,
        width: norm ? (x2 - x1) * 100 : ((x2 - x1) / Math.max(maxX, size.w)) * 100,
        height: norm ? (y2 - y1) * 100 : ((y2 - y1) / Math.max(maxY, size.h)) * 100,
      };
    });
  }, [blocks, size]);
  if (!imageB64) {
    return <div className="flex aspect-video items-center justify-center border bg-black text-xs text-muted-foreground">no frame image</div>;
  }
  return (
    <div className="relative overflow-hidden border bg-black">
      <img
        src={`data:image/jpeg;base64,${imageB64}`}
        alt="memory frame"
        className="max-h-[48vh] w-full object-contain"
        onLoad={(e) => setSize({ w: e.currentTarget.naturalWidth, h: e.currentTarget.naturalHeight })}
      />
      {boxes.map((b, i) => (
        <div
          key={i}
          title={blockText(b.block)}
          className="absolute border border-amber-300/90 bg-amber-300/10"
          style={{
            left: `${Math.max(0, Math.min(100, b.left))}%`,
            top: `${Math.max(0, Math.min(100, b.top))}%`,
            width: `${Math.max(0.4, Math.min(100, b.width))}%`,
            height: `${Math.max(0.4, Math.min(100, b.height))}%`,
          }}
        />
      ))}
    </div>
  );
});

const MemoryTableView = memo(function MemoryTableView({
  rows, columns,
}: { rows: unknown[]; columns: string[] }) {
  const displayRows = rows.slice(0, 30);
  if (columns.length === 0 && displayRows.length === 0) {
    return <div className="text-xs italic text-muted-foreground">(no structured rows)</div>;
  }
  if (columns.length === 0) {
    return <pre className="max-h-56 overflow-auto whitespace-pre-wrap border bg-background/50 p-2 text-[11px]">{debugJson(displayRows)}</pre>;
  }
  return (
    <div className="max-h-64 overflow-auto border">
      <table className="w-full border-collapse text-[11px]">
        <thead className="sticky top-0 bg-background">
          <tr>{columns.map((c) => <th key={c} className="border-b border-r px-2 py-1 text-left font-medium">{c}</th>)}</tr>
        </thead>
        <tbody>
          {displayRows.map((row, i) => {
            const obj = row && typeof row === "object" && !Array.isArray(row)
              ? row as Record<string, unknown> : {};
            return (
              <tr key={i}>
                {columns.map((c) => (
                  <td key={c} className="max-w-[220px] border-r border-t px-2 py-1 align-top">
                    <span className="whitespace-pre-wrap break-words">{String(obj[c] ?? "")}</span>
                  </td>
                ))}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
});

const MemoryEventCard = memo(function MemoryEventCard({
  event, level, onFrame,
}: {
  event: MmMemoryDebugEvent;
  level: "micro" | "macro" | "super";
  onFrame: (frameId: string) => void;
}) {
  useLocaleRevision();  // labels come from translateNow — see AsrBar
  const entities = event.entity_names || event.key_entities || [];
  const frameIds = event.frame_ids || [];
  const title = event.label || event.action || event.id;
  const description = event.summary || event.description || translateNow("multimodal.misc.noDescription");
  return (
    <details className="rounded border bg-background/50 p-2 text-xs" open={level !== "micro"}>
      <summary className="cursor-pointer list-none">
        <span className={`mr-2 rounded px-1.5 py-0.5 font-mono text-[10px] ${
          level === "super" ? "bg-violet-500/15 text-violet-200"
            : level === "macro" ? "bg-amber-500/15 text-amber-200"
              : "bg-cyan-500/15 text-cyan-200"
        }`}>{level}</span>
        <span className="font-semibold">{title}</span>
        <span className="ml-2 font-mono text-muted-foreground">
          {fmtDebugTime(event.t_start)}–{fmtDebugTime(event.t_end)}
        </span>
      </summary>
      <div className="mt-2 whitespace-pre-wrap leading-relaxed text-foreground/85">{description}</div>
      {(entities.length > 0 || frameIds.length > 0) && (
        <div className="mt-2 flex flex-wrap gap-1">
          {entities.map((name) => (
            <span key={name} className="rounded bg-muted px-1.5 py-0.5">entity: {name}</span>
          ))}
          {frameIds.map((fid) => (
            <button
              key={fid}
              type="button"
              onClick={() => onFrame(fid)}
              className="rounded border border-emerald-400/30 px-1.5 py-0.5 font-mono text-emerald-200 hover:bg-emerald-400/10"
            >
              {fid}
            </button>
          ))}
        </div>
      )}
      {((event.narrative_arc?.length || 0) > 0 || Object.keys(event.entity_arcs || {}).length > 0) && (
        <div className="mt-2 rounded bg-black/20 p-2">
          {event.narrative_arc?.map((phase, i) => (
            <div key={i} className="mb-1 last:mb-0">{debugJson(phase)}</div>
          ))}
          {Object.entries(event.entity_arcs || {}).map(([name, arc]) => (
            <div key={name}><span className="font-semibold">{name}</span> → {debugJson(arc)}</div>
          ))}
        </div>
      )}
    </details>
  );
});

export const MemoryDebugPanel = memo(function MemoryDebugPanel({
  open, onClose, currentSessionId, trajectory,
}: {
  open: boolean;
  onClose: () => void;
  currentSessionId: string;
  trajectory: MmTrajectoryEntry[];
}) {
  const [tab, setTab] = useState<MemoryDebugTab>("memory");
  const [sessions, setSessions] = useState<MmMemoryDebugSessionSummary[]>([]);
  const [selectedDb, setSelectedDb] = useState("");
  const [overview, setOverview] = useState<MmMemoryDebugSessionResponse | null>(null);
  const [trace, setTrace] = useState<MmMemoryDebugTraceResponse | null>(null);
  const [frame, setFrame] = useState<MmMemoryDebugFrameResponse | null>(null);
  const [selectedFrameId, setSelectedFrameId] = useState("");
  const [searchQ, setSearchQ] = useState("");
  const [searchScope, setSearchScope] = useState<"latest" | "today" | "all">("all");
  const [searchResults, setSearchResults] = useState<MmMemoryDebugSearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [workerFilter, setWorkerFilter] = useState("all");
  const [trajectoryDisplayLimit, setTrajectoryDisplayLimit] = useState(200);
  const selectedDbManually = useRef(false);

  useEffect(() => {
    selectedDbManually.current = false;
    setSelectedDb("");
    setSelectedFrameId("");
    setFrame(null);
  }, [currentSessionId]);

  const refreshSessions = useCallback(async () => {
    if (!open) return;
    setLoading(true);
    setError("");
    try {
      const res = await api.getMultimodalMemoryDebugSessions(80);
      setSessions(res.sessions);
      setSelectedDb((prev) => {
        const previousStillExists = Boolean(
          prev && res.sessions.some((s) => s.name === prev),
        );
        if (selectedDbManually.current && previousStillExists) return prev;
        const current = res.sessions.find(
          (s) => s.meta?.hermes_session_id === currentSessionId,
        );
        if (current) return current.name;
        if (previousStillExists) return prev;
        const newestNonEmpty = res.sessions.find((s) =>
          Object.values(s.counts || {}).some((n) => Number(n) > 0));
        return newestNonEmpty?.name || res.sessions[0]?.name || "";
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [open, currentSessionId]);

  useEffect(() => { void refreshSessions(); }, [refreshSessions]);

  const refreshOverview = useCallback(async () => {
    if (!open || !selectedDb) return;
    setLoading(true);
    setError("");
    try {
      const ov = await api.getMultimodalMemoryDebugSession(
        selectedDb, { session_id: currentSessionId, limit: 260 },
      );
      setOverview(ov);
      setSelectedFrameId((prev) => prev || ov.timeline[ov.timeline.length - 1]?.frame_id || "");
      if (tab === "debug") {
        const tr = await api.getMultimodalMemoryDebugTrace({
          session_id: currentSessionId, db: selectedDb, limit: 160,
        });
        setTrace(tr);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [open, selectedDb, currentSessionId, tab]);

  useEffect(() => { void refreshOverview(); }, [refreshOverview]);

  useEffect(() => {
    if (!open || tab !== "frame" || !selectedDb || !selectedFrameId) return;
    let cancelled = false;
    api.getMultimodalMemoryDebugFrame(selectedDb, selectedFrameId)
      .then((res) => { if (!cancelled) setFrame(res); })
      .catch((e) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)); });
    return () => { cancelled = true; };
  }, [open, tab, selectedDb, selectedFrameId]);

  const runSearch = useCallback(async () => {
    const q = searchQ.trim();
    if (!q) return;
    setLoading(true);
    setError("");
    try {
      const res = await api.searchMultimodalMemoryDebug(q, {
        scope: searchScope,
        session: searchScope === "latest" ? selectedDb : undefined,
        limit: 50,
      });
      setSearchResults(res.results);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, [searchQ, searchScope, selectedDb]);

  const workers = useMemo(() => Array.from(new Set(
    trajectory.map((it) => it.worker).filter(Boolean),
  )).sort(), [trajectory]);
  const visibleTrajectory = useMemo(() => (
    workerFilter === "all"
      ? trajectory
      : trajectory.filter((it) => it.worker === workerFilter)
  ), [trajectory, workerFilter]);
  const renderedTrajectory = useMemo(
    () => visibleTrajectory.slice(-trajectoryDisplayLimit),
    [visibleTrajectory, trajectoryDisplayLimit],
  );
  if (!open) return null;
  const counts = overview?.session.counts || {};
  const logs = trace?.logs || overview?.trace.logs || [];
  const activeFrame = frame?.frame_id === selectedFrameId ? frame : null;
  const blocks = activeFrame?.screen_text?.ocr_blocks || [];
  const health = overview?.health || {};
  const memory = overview?.memory;
  const entities = memory?.entities || [];
  const microEvents = memory?.events.micro || [];
  const macroEvents = memory?.events.macro || [];
  const superEvents = memory?.events.super || [];
  const entityStates = memory?.evolution.entity_states || [];
  const revisions = memory?.evolution.revisions || [];
  const eventCount = microEvents.length + macroEvents.length + superEvents.length;
  const evolutionCount = entityStates.length + revisions.length;
  const tabLabels: Record<MemoryDebugTab, string> = {
    memory: translateNow("multimodal.memoryDebug.tabMemory"),
    frame: translateNow("multimodal.memoryDebug.tabFrame"),
    search: translateNow("multimodal.memoryDebug.tabSearch"),
    debug: translateNow("multimodal.memoryDebug.tabDebug"),
  };

  return (
    <div className="fixed inset-y-0 right-0 z-50 flex w-[min(980px,96vw)] flex-col border-l border-border bg-background/95 shadow-2xl backdrop-blur">
      <div className="flex items-center gap-2 border-b px-3 py-2">
        <Database className="h-4 w-4 text-emerald-300" />
        <div className="min-w-0 flex-1">
          <div className="text-sm font-semibold">{translateNow("multimodal.memoryDebug.title")}</div>
          <div className="truncate text-[11px] text-muted-foreground">
            {translateNow("multimodal.memoryDebug.subtitlePrefix")}{overview?.session.meta?.summary || selectedDb || translateNow("multimodal.memoryDebug.noDbYet")}
          </div>
        </div>
        <Button size="icon" outlined title={translateNow("multimodal.memoryDebug.refresh")} onClick={() => { void refreshSessions(); void refreshOverview(); }}>
          <RefreshCw className={`h-4 w-4 ${loading ? "animate-spin" : ""}`} />
        </Button>
        <Button size="icon" outlined title={translateNow("multimodal.memoryDebug.close")} onClick={onClose}><X className="h-4 w-4" /></Button>
      </div>
      <div className="flex flex-wrap items-center gap-2 border-b px-3 py-2">
        <select
          value={selectedDb}
          onChange={(e) => {
            selectedDbManually.current = true;
            setSelectedDb(e.target.value);
            setSelectedFrameId("");
            setFrame(null);
          }}
          className="min-w-0 flex-1 rounded border bg-background px-2 py-1 text-xs"
        >
          {sessions.map((s) => (
            <option key={s.name} value={s.name}>
              {s.name} · frames {s.counts.memory_frames || 0} · OCR {s.counts.screen_texts || 0}
            </option>
          ))}
        </select>
        <div className="flex gap-1">
          {(["memory", "frame", "search", "debug"] as MemoryDebugTab[]).map((k) => (
            <button
              key={k}
              type="button"
              onClick={() => setTab(k)}
              className={`rounded border px-2 py-1 text-xs ${tab === k ? "border-emerald-300 text-emerald-200" : "border-border text-muted-foreground"}`}
            >
              {tabLabels[k]}
            </button>
          ))}
        </div>
      </div>
      {error && <div className="border-b border-red-400/30 bg-red-500/10 px-3 py-2 text-xs text-red-300">{error}</div>}
      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {tab === "memory" && (
          <div className="space-y-5">
            <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
              {[
                [translateNow("multimodal.memoryDebug.statFrames"), overview?.timeline.length || 0, counts.memory_frames || 0],
                [translateNow("multimodal.memoryDebug.statEntities"), entities.length, counts.entities || 0],
                [translateNow("multimodal.memoryDebug.statEvents"), eventCount, (counts.micro_events || 0) + (counts.macro_events || 0) + (counts.super_events || 0)],
                [translateNow("multimodal.memoryDebug.statEvolution"), evolutionCount, (counts.entity_states || 0) + (counts.revision_log || 0)],
              ].map(([label, shown, total]) => (
                <div key={String(label)} className="rounded border bg-background/50 p-3">
                  <div className="text-[11px] text-muted-foreground">{label}</div>
                  <div className="mt-1 font-mono text-xl text-emerald-200">{String(shown)}</div>
                  {Number(total) > Number(shown) && (
                    <div className="text-[10px] text-muted-foreground">{translateNow("multimodal.memoryDebug.totalInStore", Number(total))}</div>
                  )}
                </div>
              ))}
            </div>

            <section className="space-y-2">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-semibold">{translateNow("multimodal.memoryDebug.section1Title")}</h3>
                  <p className="text-[11px] text-muted-foreground">{translateNow("multimodal.memoryDebug.section1Hint")}</p>
                </div>
                <Button size="sm" outlined onClick={() => setTab("frame")}>{translateNow("multimodal.memoryDebug.viewAllOcr")}</Button>
              </div>
              {(overview?.timeline.length || 0) === 0 ? (
                <div className="rounded border p-3 text-xs italic text-muted-foreground">{translateNow("multimodal.memoryDebug.noFramesYet")}</div>
              ) : (
                <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-6">
                  {(overview?.timeline || []).slice(-48).reverse().map((it) => (
                    <button
                      key={it.frame_id}
                      type="button"
                      onClick={() => { setSelectedFrameId(it.frame_id); setTab("frame"); }}
                      className="overflow-hidden rounded border bg-background/50 text-left hover:border-emerald-300/60"
                    >
                      {it.thumb_b64 ? (
                        <img src={`data:image/jpeg;base64,${it.thumb_b64}`} alt={it.frame_id} className="h-24 w-full object-cover" />
                      ) : <div className="flex h-24 items-center justify-center bg-black/30 text-[10px] text-muted-foreground">no image</div>}
                      <div className="p-1.5">
                        <div className="truncate font-mono text-[10px] text-emerald-200">{fmtDebugTime(it.t_observed)} · {it.frame_id}</div>
                        <div className="truncate text-[10px] text-muted-foreground">{it.source || "unknown"} · {it.note || it.micro_id || "key frame"}</div>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </section>

            <section className="space-y-2">
              <div>
                <h3 className="text-sm font-semibold">{translateNow("multimodal.memoryDebug.section2Title")}</h3>
                <p className="text-[11px] text-muted-foreground">{translateNow("multimodal.memoryDebug.section2Hint")}</p>
              </div>
              {entities.length === 0 ? (
                <div className="rounded border p-3 text-xs italic text-muted-foreground">{translateNow("multimodal.memoryDebug.noEntitiesYet")}</div>
              ) : (
                <div className="grid grid-cols-1 gap-2 lg:grid-cols-2">
                  {entities.map((entity) => (
                    <details key={entity.id} className="rounded border bg-background/50 p-3 text-xs" open={entities.length <= 8}>
                      <summary className="cursor-pointer list-none">
                        <span className="rounded bg-emerald-500/15 px-1.5 py-0.5 font-mono text-[10px] text-emerald-200">{entity.type}</span>
                        <span className="ml-2 font-semibold">{entity.name}</span>
                        <span className="ml-2 text-muted-foreground">{translateNow("multimodal.memoryDebug.seenCount", entity.seen_count)} · {fmtDebugTime(entity.first_seen)}–{fmtDebugTime(entity.last_seen)}</span>
                      </summary>
                      {entity.aliases.length > 0 && (
                        <div className="mt-2 text-muted-foreground">{translateNow("multimodal.memoryDebug.aliasesLabel")}{entity.aliases.join(" / ")}</div>
                      )}
                      <div className="mt-2 flex flex-wrap gap-1">
                        {Object.entries(entity.attributes || {}).map(([key, value]) => (
                          <span key={key} className="rounded bg-muted px-1.5 py-0.5">
                            <span className="text-muted-foreground">{key}:</span> {debugJson(value)}
                          </span>
                        ))}
                      </div>
                      {entity.frame_ids.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {entity.frame_ids.map((fid) => (
                            <button
                              key={fid}
                              type="button"
                              onClick={() => { setSelectedFrameId(fid); setTab("frame"); }}
                              className="rounded border border-emerald-400/30 px-1.5 py-0.5 font-mono text-[10px] text-emerald-200"
                            >{fid}</button>
                          ))}
                        </div>
                      )}
                    </details>
                  ))}
                </div>
              )}
            </section>

            <section className="space-y-2">
              <div>
                <h3 className="text-sm font-semibold">{translateNow("multimodal.memoryDebug.section3Title")}</h3>
                <p className="text-[11px] text-muted-foreground">{translateNow("multimodal.memoryDebug.section3Hint")}</p>
              </div>
              {eventCount === 0 ? (
                <div className="rounded border p-3 text-xs italic text-muted-foreground">{translateNow("multimodal.memoryDebug.noEventsYet")}</div>
              ) : (
                <div className="space-y-2">
                  {[...superEvents, ...macroEvents, ...microEvents].map((event) => (
                    <MemoryEventCard
                      key={event.id}
                      event={event}
                      level={superEvents.includes(event) ? "super" : macroEvents.includes(event) ? "macro" : "micro"}
                      onFrame={(fid) => { setSelectedFrameId(fid); setTab("frame"); }}
                    />
                  ))}
                </div>
              )}
            </section>

            <section className="space-y-2">
              <div>
                <h3 className="text-sm font-semibold">{translateNow("multimodal.memoryDebug.section4Title")}</h3>
                <p className="text-[11px] text-muted-foreground">{translateNow("multimodal.memoryDebug.section4Hint")}</p>
              </div>
              {evolutionCount === 0 ? (
                <div className="rounded border p-3 text-xs italic text-muted-foreground">{translateNow("multimodal.memoryDebug.noEvolutionYet")}</div>
              ) : (
                <div className="space-y-2">
                  {entityStates.map((state) => (
                    <div key={`state-${state.id}`} className="rounded border-l-2 border-l-cyan-300 border-y border-r bg-background/50 p-2 text-xs">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="font-mono text-cyan-200">{fmtDebugTime(state.t_observed)}</span>
                        <span className="font-semibold">{state.entity_name}</span>
                        <span className="rounded bg-muted px-1.5 py-0.5">{state.state_label}</span>
                        <span className="text-muted-foreground">{state.source}</span>
                      </div>
                      <div className="mt-2 flex flex-wrap gap-1">
                        {Object.entries(state.attributes_delta || {}).map(([key, value]) => (
                          <span key={key} className="rounded bg-cyan-500/10 px-1.5 py-0.5"><span className="text-muted-foreground">{key} →</span> {debugJson(value)}</span>
                        ))}
                        {state.new_aliases.map((alias) => <span key={alias} className="rounded bg-violet-500/10 px-1.5 py-0.5">+ alias {alias}</span>)}
                      </div>
                      {state.evidence_frame_ids.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {state.evidence_frame_ids.map((fid) => (
                            <button key={fid} type="button" onClick={() => { setSelectedFrameId(fid); setTab("frame"); }} className="font-mono text-[10px] text-emerald-200 underline-offset-2 hover:underline">{fid}</button>
                          ))}
                        </div>
                      )}
                    </div>
                  ))}
                  {revisions.map((revision) => (
                    <details key={`revision-${revision.id}`} className={`rounded border-l-2 border-y border-r bg-background/50 p-2 text-xs ${revision.success ? "border-l-amber-300" : "border-l-red-300"}`}>
                      <summary className="cursor-pointer list-none">
                        <span className="font-mono text-amber-200">{fmtDebugWall(revision.t_applied)}</span>
                        <span className="ml-2 font-semibold">Reviewer: {revision.op}</span>
                        <span className="ml-2 text-muted-foreground">{revision.target_ids.join(", ")}</span>
                      </summary>
                      <div className="mt-2 whitespace-pre-wrap">
                        {revision.reason || revision.error || translateNow("multimodal.memoryDebug.noReason")}
                      </div>
                      <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap rounded bg-black/20 p-2 text-[11px]">{debugJson(revision.payload)}</pre>
                    </details>
                  ))}
                </div>
              )}
            </section>
          </div>
        )}

        {tab === "debug" && (
          <div className="space-y-3">
            <div className="sticky top-0 z-10 flex flex-wrap items-center gap-2 rounded border bg-background/95 p-2 text-xs backdrop-blur">
              <Activity className="h-3.5 w-3.5 text-cyan-300" />
              <span className="font-semibold">Live worker trajectory</span>
              <span className="text-muted-foreground">
                showing {renderedTrajectory.length} / {visibleTrajectory.length} events
              </span>
              <select
                value={workerFilter}
                onChange={(e) => setWorkerFilter(e.target.value)}
                className="ml-auto rounded border bg-background px-2 py-1 text-xs"
              >
                <option value="all">all workers</option>
                {workers.map((w) => <option key={w} value={w}>{w}</option>)}
              </select>
            </div>
            {visibleTrajectory.length === 0 ? (
              <div className="rounded border p-3 text-xs italic text-muted-foreground">
                {translateNow("multimodal.memoryDebug.noTrajectory")}
              </div>
            ) : [...renderedTrajectory].reverse().map((it) => {
              const rawFrames = (it.payload?.frames || []) as MmTrajectoryFrame[];
              const frames = Array.isArray(rawFrames) ? rawFrames : [];
              return (
                <details key={it.id} open={frames.length > 0} className="rounded border bg-background/50 p-2 text-xs">
                  <summary className="cursor-pointer list-none">
                    <span className="font-mono text-cyan-300">#{it.seq}</span>
                    <span className="ml-2 font-semibold text-emerald-200">{it.worker}</span>
                    <span className="ml-2 rounded bg-muted px-1.5 py-0.5 font-mono">{it.phase}</span>
                    <span className="ml-2 text-muted-foreground">{fmtDebugWall(it.ts)}</span>
                    <span className="ml-2 text-[10px] text-muted-foreground">{it.event}</span>
                  </summary>
                  {frames.length > 0 && (
                    <div className="mt-2 grid grid-cols-2 gap-2 sm:grid-cols-3 lg:grid-cols-4">
                      {frames.map((fr, i) => {
                        const b64 = fr.thumb_b64 || fr.jpeg_b64 || "";
                        const usableB64 = b64 && !b64.startsWith("<omitted");
                        return (
                          <figure key={`${fr.frame_id || fr.ts || i}-${i}`} className="overflow-hidden rounded border bg-black/20">
                            {usableB64
                              ? <img src={`data:image/jpeg;base64,${b64}`} alt="recall evidence" className="h-28 w-full object-contain" />
                              : <div className="flex h-28 items-center justify-center text-[10px] text-muted-foreground">thumbnail omitted</div>}
                            <figcaption className="px-1.5 py-1 font-mono text-[10px] text-muted-foreground">
                              {fr.frame_id || `frame ${i + 1}`} · {fmtDebugTime(fr.ts)}
                            </figcaption>
                          </figure>
                        );
                      })}
                    </div>
                  )}
                  <pre className="mt-2 max-h-96 overflow-auto whitespace-pre-wrap break-words rounded bg-black/20 p-2 text-[11px] leading-snug">
                    {debugJson(it.payload)}
                  </pre>
                </details>
              );
            })}
            {renderedTrajectory.length < visibleTrajectory.length && (
              <Button
                size="sm"
                outlined
                onClick={() => setTrajectoryDisplayLimit((n) => Math.min(n + 200, 2000))}
              >
                {translateNow("multimodal.memoryDebug.showMore200")}
              </Button>
            )}
          </div>
        )}

        {tab === "debug" && (
          <div className="grid min-h-0 grid-cols-1 gap-3 lg:grid-cols-[1fr_1fr]">
            <section className="min-w-0">
              <div className="mb-2 flex items-center gap-1 text-xs font-semibold text-muted-foreground">
                <Activity className="h-3.5 w-3.5" /> Recall Messages
              </div>
              <div className="space-y-2">
                {(trace?.messages || []).length === 0 ? (
                  <div className="rounded border p-2 text-xs italic text-muted-foreground">No persisted recall tool messages for this session.</div>
                ) : trace!.messages.map((m, i) => (
                  <details key={`${m.timestamp}-${i}`} className="rounded border bg-background/50 p-2 text-xs">
                    <summary className="cursor-pointer list-none">
                      <span className="font-mono text-emerald-300">{m.tool_name || m.role}</span>
                      <span className="ml-2 text-muted-foreground">{fmtDebugWall(m.timestamp)}</span>
                    </summary>
                    <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap break-words">{m.content || debugJson(m.tool_calls)}</pre>
                  </details>
                ))}
              </div>
            </section>
            <section className="min-w-0">
              <div className="mb-2 flex items-center gap-1 text-xs font-semibold text-muted-foreground">
                <FileText className="h-3.5 w-3.5" /> Relevant Logs
              </div>
              <pre className="max-h-[70vh] overflow-auto rounded border bg-black/30 p-2 text-[11px] leading-snug text-muted-foreground">
                {logs.join("\n") || "No recall/writer/OCR logs found."}
              </pre>
            </section>
          </div>
        )}

        {tab === "frame" && (
          <div className="grid min-h-0 grid-cols-1 gap-3 lg:grid-cols-[330px_minmax(0,1fr)]">
            <section className="min-w-0">
              <div className="mb-2 flex items-center justify-between text-xs text-muted-foreground">
                <span>All-scene memory frames · {overview?.timeline.length || 0}</span>
                <span>Indexed {counts.memory_frames || 0} · OCR {counts.screen_texts || 0} · Tables {counts.screen_tables || 0}</span>
              </div>
              <div className="max-h-[72vh] space-y-1 overflow-y-auto">
                {(overview?.timeline || []).map((it) => (
                  <button
                    key={it.frame_id}
                    type="button"
                    onClick={() => setSelectedFrameId(it.frame_id)}
                    className={`flex w-full gap-2 rounded border p-2 text-left text-xs ${selectedFrameId === it.frame_id ? "border-emerald-300/70 bg-emerald-400/10" : "border-border bg-background/40"}`}
                  >
                    {it.thumb_b64 ? <img src={`data:image/jpeg;base64,${it.thumb_b64}`} alt="" className="h-12 w-16 shrink-0 object-cover" /> : <div className="h-12 w-16 shrink-0 bg-black" />}
                    <span className="min-w-0 flex-1">
                      <span className="block font-mono text-emerald-300">{fmtDebugTime(it.t_observed)} · {it.frame_id}</span>
                      <span className="block truncate text-muted-foreground">
                        <span className="mr-1 rounded bg-muted px-1 py-0.5">{it.source_type || it.source || "unknown"}</span>
                        {it.window_title || it.app || it.note || it.micro_id}
                      </span>
                      <span className="line-clamp-2 text-foreground/80">{it.raw_preview || it.observation_preview || "(visual key frame; no OCR text)"}</span>
                    </span>
                    {it.table_count > 0 && <Table2 className="h-4 w-4 shrink-0 text-amber-300" />}
                  </button>
                ))}
              </div>
            </section>
            <section className="min-w-0 space-y-3">
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <span className="font-mono text-emerald-300">{selectedFrameId || "no frame selected"}</span>
                {activeFrame?.memory_frame && <span>{fmtDebugTime(activeFrame.memory_frame.t_observed)}</span>}
                {activeFrame?.memory_frame?.source_type && <span className="rounded bg-muted px-1.5 py-0.5">{activeFrame.memory_frame.source_type}</span>}
                {activeFrame?.screen_text && <span>{fmtDebugTime(activeFrame.screen_text.t_observed)}</span>}
                {activeFrame?.screen_text?.source && <span>{activeFrame.screen_text.source}</span>}
              </div>
              <OcrOverlayImage imageB64={activeFrame?.image_b64 || ""} blocks={blocks} />
              <div>
                <div className="mb-1 text-xs font-semibold text-muted-foreground">All-scene Memory Frame Metadata</div>
                <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded border bg-background/50 p-2 text-[11px]">{debugJson(activeFrame?.memory_frame || {})}</pre>
              </div>
              <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                <div>
                  <div className="mb-1 text-xs font-semibold text-muted-foreground">OCR Raw Text</div>
                  <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded border bg-background/50 p-2 text-[11px]">{activeFrame?.screen_text?.raw_text || "(empty)"}</pre>
                </div>
                <div>
                  <div className="mb-1 text-xs font-semibold text-muted-foreground">OCR Blocks · {Array.isArray(blocks) ? blocks.length : 0}</div>
                  <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded border bg-background/50 p-2 text-[11px]">{debugJson((Array.isArray(blocks) ? blocks : []).slice(0, 80))}</pre>
                </div>
              </div>
              {(activeFrame?.tables || []).map((t) => (
                <div key={`${t.table_id}-${t.frame_id}`} className="space-y-1">
                  <div className="text-xs font-semibold text-amber-200">{t.table_id} · {t.title || "table"}</div>
                  <MemoryTableView rows={t.rows} columns={t.columns} />
                </div>
              ))}
              <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                <pre className="max-h-52 overflow-auto whitespace-pre-wrap rounded border bg-background/50 p-2 text-[11px]">{debugJson(activeFrame?.micro_events || [])}</pre>
                <pre className="max-h-52 overflow-auto whitespace-pre-wrap rounded border bg-background/50 p-2 text-[11px]">{debugJson(activeFrame?.entities || [])}</pre>
              </div>
            </section>
          </div>
        )}

        {tab === "search" && (
          <div className="space-y-3">
            <div className="flex gap-2">
              <div className="relative min-w-0 flex-1">
                <Search className="pointer-events-none absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <input
                  value={searchQ}
                  onChange={(e) => setSearchQ(e.target.value)}
                  onKeyDown={(e) => { if (e.key === "Enter") void runSearch(); }}
                  placeholder="Search OCR / tables / events across sessions"
                  className="w-full rounded border bg-background py-2 pl-8 pr-3 text-sm"
                />
              </div>
              <select value={searchScope} onChange={(e) => setSearchScope(e.target.value as typeof searchScope)}
                className="rounded border bg-background px-2 text-xs">
                <option value="all">all sessions</option>
                <option value="today">today</option>
                <option value="latest">selected DB</option>
              </select>
              <Button size="sm" prefix={<Search />} onClick={() => void runSearch()}>Search</Button>
            </div>
            <div className="space-y-2">
              {searchResults.map((r, i) => (
                <details key={`${r.session}-${r.kind}-${r.frame_id}-${i}`} className="rounded border bg-background/50 p-2 text-xs" open={i < 3}>
                  <summary className="flex cursor-pointer list-none items-center gap-2">
                    {r.thumb_b64 ? <img src={`data:image/jpeg;base64,${r.thumb_b64}`} alt="" className="h-10 w-14 object-cover" /> : null}
                    <span className="min-w-0 flex-1">
                      <span className="block truncate font-medium">{r.title}</span>
                      <span className="font-mono text-muted-foreground">{r.session} · {r.kind} · score {r.score} · {r.frame_id || ""}</span>
                    </span>
                  </summary>
                  <pre className="mt-2 max-h-44 overflow-auto whitespace-pre-wrap rounded border bg-black/20 p-2">{r.snippet}</pre>
                  {r.table && <MemoryTableView rows={r.table.row_hits.map((x) => x.row)} columns={r.table.columns} />}
                </details>
              ))}
            </div>
          </div>
        )}

        {tab === "debug" && (
          <div className="space-y-3">
            <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
              {Object.entries(counts).map(([k, v]) => (
                <div key={k} className="rounded border bg-background/50 p-2">
                  <div className="text-[10px] uppercase text-muted-foreground">{k}</div>
                  <div className="font-mono text-lg text-emerald-200">{String(v)}</div>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
              <div className="rounded border bg-background/50 p-2 text-xs">
                <div className="mb-1 font-semibold text-muted-foreground">Session</div>
                <div>mtime: {overview?.session.mtime ? fmtDebugWall(overview.session.mtime) : ""}</div>
                <div>size: {fmtDebugBytes(overview?.session.size)}</div>
                <div>frame files: {String(health.frame_files ?? 0)}</div>
                <div>micro no frames: {String(health.micro_events_without_frames ?? 0)}</div>
                <div>empty OCR: {String(health.screen_texts_without_raw_text ?? 0)}</div>
              </div>
              <div className="rounded border bg-background/50 p-2 text-xs">
                <div className="mb-1 font-semibold text-muted-foreground">Meta</div>
                <pre className="max-h-40 overflow-auto whitespace-pre-wrap">{debugJson(overview?.session.meta || {})}</pre>
              </div>
            </div>
            <div>
              <div className="mb-1 text-xs font-semibold text-muted-foreground">Recent Writer / OCR / Recall Warnings</div>
              <pre className="max-h-[42vh] overflow-auto rounded border bg-black/30 p-2 text-[11px] leading-snug text-muted-foreground">
                {debugJson(health.recent_log_warnings || [])}
              </pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
});
