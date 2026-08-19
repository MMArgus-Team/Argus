export interface ConfigFieldSchema {
  category?: string
  description?: string
  options?: unknown[]
  type?: 'boolean' | 'list' | 'number' | 'select' | 'string' | 'text'
}

export interface ConfigSchemaResponse {
  category_order?: string[]
  fields: Record<string, ConfigFieldSchema>
}

export interface AudioTranscriptionResponse {
  ok: boolean
  provider?: string
  transcript: string
}

export interface AudioSpeakResponse {
  ok: boolean
  data_url: string
  mime_type: string
  provider?: string
}

export interface ElevenLabsVoice {
  label: string
  name: string
  voice_id: string
}

export interface ElevenLabsVoicesResponse {
  available: boolean
  voices: ElevenLabsVoice[]
}

export interface OAuthProviderStatus {
  error?: string
  expires_at?: null | string
  has_refresh_token?: boolean
  last_refresh?: null | string
  logged_in: boolean
  source?: null | string
  source_label?: null | string
  token_preview?: null | string
}

export interface OAuthProvider {
  cli_command: string
  /** Shell command that clears an external provider's credentials, run in the
   *  embedded terminal. Null when Hermes doesn't know how to remove it. */
  disconnect_command?: null | string
  disconnect_hint?: null | string
  disconnectable?: boolean
  docs_url: string
  flow: 'device_code' | 'external' | 'loopback' | 'pkce'
  id: string
  name: string
  status: OAuthProviderStatus
}

export interface OAuthProvidersResponse {
  providers: OAuthProvider[]
}

export type OAuthStartResponse =
  | {
      auth_url: string
      expires_in: number
      flow: 'pkce'
      session_id: string
    }
  | {
      expires_in: number
      flow: 'device_code'
      poll_interval: number
      session_id: string
      user_code: string
      verification_url: string
    }
  | {
      auth_url: string
      expires_in: number
      flow: 'loopback'
      session_id: string
    }

export interface OAuthSubmitResponse {
  message?: string
  ok: boolean
  status: 'approved' | 'error'
}

export interface OAuthPollResponse {
  error_message?: null | string
  expires_at?: null | number
  session_id: string
  status: 'approved' | 'denied' | 'error' | 'expired' | 'pending'
}

export interface MemoryProviderOAuthStatus {
  auth: 'apikey' | 'oauth' | null
  connected: boolean
  detail: string
  state: 'connected' | 'error' | 'idle' | 'pending'
}

export interface EnvVarInfo {
  advanced: boolean
  category: string
  // True when this var is a messaging-platform credential owned by a card on
  // the dedicated Messaging page. The Keys page hides these to avoid
  // duplicating the richer channel-configuration UI.
  channel_managed?: boolean
  description: string
  is_password: boolean
  is_set: boolean
  // Backend-derived provider grouping hints (from the unified provider catalog
  // in hermes_cli/provider_catalog.py). When present, the Keys tab groups by
  // this provider identity — the SAME one `argus model` uses — instead of
  // desktop-only env-var prefix guesses. Empty for non-provider env vars.
  provider?: string
  provider_label?: string
  redacted_value: null | string
  tools: string[]
  url: null | string
}

export type MemoryProviderFieldKind = 'secret' | 'select' | 'text'

export interface MemoryProviderFieldOption {
  description: string
  label: string
  value: string
}

export interface MemoryProviderField {
  description: string
  is_set: boolean
  key: string
  kind: MemoryProviderFieldKind
  label: string
  options: MemoryProviderFieldOption[]
  placeholder: string
  value: string
}

export interface MemoryProviderConfig {
  fields: MemoryProviderField[]
  label: string
  name: string
}

export interface MessagingEnvVarInfo {
  advanced: boolean
  description: string
  is_password: boolean
  is_set: boolean
  key: string
  prompt: string
  redacted_value: null | string
  required: boolean
  url: null | string
}

export interface MessagingHomeChannel {
  chat_id: string
  name: string
  platform: string
  thread_id?: string
}

export interface MessagingPlatformInfo {
  configured: boolean
  description: string
  docs_url: string
  enabled: boolean
  env_vars: MessagingEnvVarInfo[]
  error_code?: null | string
  error_message?: null | string
  gateway_running: boolean
  home_channel?: MessagingHomeChannel | null
  id: string
  name: string
  state?: null | string
  updated_at?: null | string
}

export interface MessagingPlatformsResponse {
  platforms: MessagingPlatformInfo[]
}

export interface MessagingPlatformUpdate {
  clear_env?: string[]
  enabled?: boolean
  env?: Record<string, string>
}

export interface MessagingPlatformTestResponse {
  message: string
  ok: boolean
  state?: null | string
}

export interface GatewayReadyPayload {
  skin?: unknown
}

export interface HermesConfig {
  agent?: {
    reasoning_effort?: string
    personalities?: Record<string, unknown>
    service_tier?: string
  }
  display?: {
    personality?: string
    skin?: string
  }
  terminal?: {
    cwd?: string
  }
  stt?: {
    enabled?: boolean
  }
  voice?: {
    max_recording_seconds?: number
  }
}

export type HermesConfigRecord = Record<string, unknown>

export interface ModelInfoResponse {
  auto_context_length?: number
  capabilities?: Record<string, unknown>
  config_context_length?: number
  effective_context_length?: number
  model: string
  provider: string
}

export interface ModelPricing {
  /** Formatted $/Mtok input price, e.g. "$3.00", or "free", or "" if unknown. */
  input: string
  /** Formatted $/Mtok output price. */
  output: string
  /** Formatted $/Mtok cached-input price, or null when the model has none. */
  cache: string | null
  /** True when the model costs nothing (free tier eligible). */
  free: boolean
}

export interface ModelOptionProvider {
  /** Backend's own verdict on "this row is the active provider" — authoritative.
   *  A `custom:` endpoint's slug is derived from its HOSTNAME
   *  (`custom:open.bigmodel.cn`) while the response's top-level `provider` is the
   *  bare configured name (`custom`), so comparing those two by string never
   *  matches. Use `isCurrentProviderRow`, never `slug === provider`. */
  is_current?: boolean
  models?: string[]
  name: string
  slug: string
  total_models?: number
  warning?: string
  /** True when the provider has usable credentials. False for canonical
   *  providers surfaced by `include_unconfigured` that the user hasn't set up
   *  yet — render these with a setup affordance instead of hiding them. */
  authenticated?: boolean
  /** Auth flow for an unconfigured provider: "api_key" can be activated inline
   *  by pasting `key_env`; anything else (oauth_*, external, aws_sdk, …) needs
   *  the `argus model` CLI / onboarding OAuth flow. */
  auth_type?: string
  /** Env var to paste an API key into, for unconfigured `api_key` providers. */
  key_env?: string
  /** True for providers defined via the user's `providers:` config block. */
  is_user_defined?: boolean
  /** Per-model pricing keyed by model id (present when the picker requested
   *  pricing and the provider supports live pricing). */
  pricing?: Record<string, ModelPricing>
  /** Nous only: whether the current account is on the free tier. */
  free_tier?: boolean
  /** Nous only: paid models a free-tier user cannot select (shown disabled). */
  unavailable_models?: string[]
  /** Per-model option support, keyed by model id (present when the picker
   *  requested capabilities). Lets the UI gate fast/reasoning controls. */
  capabilities?: Record<string, ModelCapabilities>
}

export interface ModelCapabilities {
  fast: boolean
  reasoning: boolean
}

export interface ModelOptionsResponse {
  model?: string
  provider?: string
  providers?: ModelOptionProvider[]
}

export interface PaginatedSessions {
  limit: number
  offset: number
  sessions: SessionInfo[]
  total: number
  /** Listable conversation count per profile (children excluded), keyed by
   *  profile name. Lets the sidebar scope its "Load more" footer to the active
   *  profile instead of the global total. Present only on
   *  `/api/profiles/sessions`. */
  profile_totals?: Record<string, number>
  /** Per-profile read failures from the cross-profile aggregator (e.g. a locked
   *  or corrupt state.db). Present only on `/api/profiles/sessions`. */
  errors?: Array<{ profile: string; error: string }>
}

export interface RpcEvent<T = unknown> {
  payload?: T
  session_id?: string
  type: string
}

export interface SessionCreateResponse {
  info?: SessionRuntimeInfo
  message_count?: number
  messages?: SessionMessage[]
  session_id: string
  stored_session_id?: string
}

export interface SessionInfo {
  archived?: boolean
  cwd?: null | string
  /** Git branch checked out in {@link cwd} when the session started/resumed.
   *  The sidebar groups main-checkout sessions by this so feature-branch work
   *  doesn't collapse under a single directory-named "main" row. Null for
   *  non-git workspaces and sessions created before branch capture landed. */
  git_branch?: null | string
  /** Git repo root that owns {@link cwd} — the authoritative project key,
   *  resolved server-side at cwd-set (and backfilled for history). The sidebar
   *  groups by this instead of probing git in the GUI. Null for non-git
   *  workspaces and not-yet-backfilled rows. */
  git_repo_root?: null | string
  ended_at: null | number
  id: string
  /** Original root id of a compression chain, when this entry is a projected
   *  continuation tip. Stable across compressions — used as the durable id for
   *  pins so a pinned conversation survives auto-compression. */
  _lineage_root_id?: null | string
  input_tokens: number
  is_active: boolean
  last_active: number
  message_count: number
  model: null | string
  output_tokens: number
  /** Parent conversation when this row is a /branch fork. */
  parent_session_id?: null | string
  preview: null | string
  source: null | string
  started_at: number
  title: null | string
  tool_call_count: number
  /** Origin platform when this session was handed off from a messaging
   *  platform (e.g. a Telegram thread continued in the desktop app). The live
   *  {@link source} becomes local (tui/desktop) after a handoff, so the origin
   *  is preserved here to surface the platform badge on the row. */
  handoff_platform?: null | string
  /** Handoff lifecycle: 'pending' | 'in_progress' | 'completed' | 'failed'. */
  handoff_state?: null | string
  handoff_error?: null | string
  /** Owning profile name, set by the cross-profile aggregator
   *  (`/api/profiles/sessions`). Absent on legacy single-profile responses,
   *  which the UI treats as the default profile. */
  profile?: string
  /** True when {@link profile} is the default profile. */
  is_default_profile?: boolean
}

/**
 * One privacy-classified tool-call argument, from the backend's
 * `agent.display.describe_arg_fields`. Shipped on every live `tool.start` and
 * again on every resumed tool row, so a card's inputs survive a reload.
 *
 * The classification is the backend's, and it is what makes displaying args
 * from an ARBITRARY tool (including MCP tools nobody wrote a renderer for)
 * safe by default:
 *  - `literal`    → `value` is safe to show, already truncated + redacted
 *  - `freeform`   → payload the call writes/sends; only `chars`, never content
 *  - `shape`      → array/object; only `count`
 *  - `credential` → a secret; key only, not even a length
 *  - `elided`     → synthetic tail (`key` is ''), `count` fields were dropped
 *
 * `value` is display-truncated (40 chars, 60 for intent prose), so it must
 * NEVER be fed back to logic that needs the real argument — see the note in
 * `toolArgFieldsFrom`.
 */
export interface ToolArgField {
  chars?: number
  count?: number
  key: string
  kind: 'credential' | 'elided' | 'freeform' | 'literal' | 'shape'
  value?: string
}

export interface SessionMessage {
  /** Classified call-side args for a resumed `role: 'tool'` row. */
  args_fields?: ToolArgField[]
  codex_reasoning_items?: unknown
  content: unknown
  context?: unknown
  name?: string
  reasoning?: null | string
  reasoning_content?: null | string
  reasoning_details?: unknown
  role: 'assistant' | 'system' | 'tool' | 'user'
  /** Tool-result summary (exit code / first error line) from the resume
   *  projection. `context` is the call-side args preview; `content` the output. */
  summary?: string
  text?: unknown
  timestamp?: number
  tool_call_id?: null | string
  tool_calls?: unknown
  tool_name?: string
  // ★ 功能1: monitor/watcher 通知元素 (type=mm_notice) 从后端 _history_to_messages
  //   重建时携带的字段, 供 toChatMessages 还原成 monitor/watcher_report 气泡。
  subRole?: 'monitor' | 'router' | 'watcher_report'
  monitorLabel?: string
  eventId?: string
  deepReportRid?: string
  deepRound?: number
}

export interface SessionMessagesResponse {
  messages: SessionMessage[]
  session_id: string
}

export interface SessionResumeResponse {
  info?: SessionRuntimeInfo
  message_count: number
  messages: SessionMessage[]
  resumed: string
  session_id: string
}

export interface SessionRuntimeInfo {
  branch?: string
  config_warning?: string
  credential_warning?: string
  cwd?: string
  desktop_contract?: number
  fast?: boolean
  model?: string
  personality?: string
  provider?: string
  reasoning_effort?: string
  /** Whether this ENDPOINT lets us change the model's reasoning — distinct from
   *  "the model reasons". False for thinking-only models, pre-toggle
   *  generations, and aggregators serving another vendor's model. Gate the
   *  thinking control on it so we never render a switch that silently does
   *  nothing. Absent (older backend) → treat as true. */
  can_toggle_reasoning?: boolean
  running?: boolean
  service_tier?: string
  skills?: Record<string, string[]> | string[]
  tools?: Record<string, string[]>
  usage?: Partial<UsageStats>
  version?: string
  yolo?: boolean
}

export interface UsageStats {
  calls: number
  context_max?: number
  context_percent?: number
  context_used?: number
  cost_usd?: number
  input: number
  output: number
  total: number
}

export interface AnalyticsDailyEntry {
  actual_cost: number
  api_calls: number
  cache_read_tokens: number
  day: string
  estimated_cost: number
  input_tokens: number
  output_tokens: number
  reasoning_tokens: number
  sessions: number
}

export interface AnalyticsModelEntry {
  api_calls: number
  estimated_cost: number
  input_tokens: number
  model: string
  output_tokens: number
  sessions: number
}

export interface AnalyticsResponse {
  by_model: AnalyticsModelEntry[]
  daily: AnalyticsDailyEntry[]
  period_days: number
  skills: {
    summary: AnalyticsSkillsSummary
    top_skills: AnalyticsSkillEntry[]
  }
  totals: AnalyticsTotals
}

export interface AnalyticsSkillEntry {
  last_used_at: null | number
  manage_count: number
  percentage: number
  skill: string
  total_count: number
  view_count: number
}

export interface AnalyticsSkillsSummary {
  distinct_skills_used: number
  total_skill_actions: number
  total_skill_edits: number
  total_skill_loads: number
}

export interface AnalyticsTotals {
  total_actual_cost: number
  total_api_calls: null | number
  total_cache_read: null | number
  total_estimated_cost: number
  total_input: null | number
  total_output: null | number
  total_reasoning: null | number
  total_sessions: number
}

export interface CronJob {
  deliver?: null | string
  enabled: boolean
  id: string
  last_error?: null | string
  last_run_at?: null | string
  name?: null | string
  next_run_at?: null | string
  prompt?: null | string
  schedule?: CronJobSchedule
  schedule_display?: null | string
  script?: null | string
  state?: null | string
}

export interface CronJobCreatePayload {
  deliver?: string
  name?: string
  prompt: string
  schedule: string
}

export interface CronJobSchedule {
  display?: string
  expr?: string
  kind?: string
}

export interface CronJobUpdates {
  deliver?: string
  enabled?: boolean
  name?: string
  prompt?: string
  schedule?: string
}

export interface ProfileCreatePayload {
  clone_all?: boolean
  clone_from?: null | string
  clone_from_default?: boolean
  name: string
  no_skills?: boolean
}

export interface ProfileInfo {
  has_env: boolean
  is_default: boolean
  model: null | string
  name: string
  path: string
  provider: null | string
  skill_count: number
}

export interface ProfileSetupCommand {
  command: string
}

// ── Projects ───────────────────────────────────────────────────────────────
// A first-class, per-profile, human-named workspace spanning one or more
// folders. Mirrors hermes_cli/projects_db.Project.to_dict().
export interface ProjectFolder {
  path: string
  label: null | string
  is_primary: boolean
  added_at: number
}

export interface ProjectInfo {
  id: string
  slug: string
  name: string
  description: null | string
  icon: null | string
  color: null | string
  board_slug: null | string
  primary_path: null | string
  archived: boolean
  created_at: number
  folders: ProjectFolder[]
}

export interface ProjectsPayload {
  projects: ProjectInfo[]
  active_id: null | string
}

export interface ProfileSoul {
  content: string
  exists: boolean
}

export interface ProfilesResponse {
  profiles: ProfileInfo[]
}

export interface SkillInfo {
  category: string
  description: string
  enabled: boolean
  name: string
}

export interface ToolsetInfo {
  configured: boolean
  description: string
  enabled: boolean
  label: string
  name: string
  tools: string[]
}

export interface ToolEnvVar {
  key: string
  prompt: string
  url: string | null
  default: string | null
  is_set: boolean
}

export interface ToolProvider {
  name: string
  badge: string
  tag: string
  env_vars: ToolEnvVar[]
  post_setup: string | null
  requires_nous_auth: boolean
  /** True when this is the provider currently written to config (mirrors the
   *  CLI `argus tools` active-provider detection). */
  is_active: boolean
}

export interface ToolsetConfig {
  name: string
  has_category: boolean
  providers: ToolProvider[]
  /** Name of the currently active provider, or null if none is configured. */
  active_provider: string | null
}

/** Shape of `GET /api/tools/computer-use/status`.
 *
 *  cua-driver runs on macOS, Windows, and Linux. `ready` is the single OS-aware
 *  readiness signal: on macOS both TCC grants (Accessibility + Screen
 *  Recording, which attach to cua-driver's own `com.trycua.driver` identity,
 *  not Hermes); elsewhere, driver health from `cua-driver doctor`. `null`
 *  means unknown (binary missing / probe failed). */
export interface ComputerUsePermissionSource {
  attribution?: string
  executable?: string
  note?: string
  pid?: number
  responsible_ppid?: number
}

export interface ComputerUseCheck {
  label: string
  status: string
  message: string
}

export interface ComputerUseStatus {
  /** `sys.platform`: "darwin" | "win32" | "linux" | ... */
  platform: string
  /** cua-driver has a runtime backend for this platform. */
  platform_supported: boolean
  /** cua-driver binary resolved on PATH. */
  installed: boolean
  /** e.g. "cua-driver 0.5.1", or null when unknown. */
  version: string | null
  /** Unified readiness — both TCC grants (macOS) or driver health (else). */
  ready: boolean | null
  /** Whether a permission grant flow exists (macOS-only TCC). */
  can_grant: boolean
  /** Cross-platform `cua-driver doctor` probes. */
  checks: ComputerUseCheck[]
  /** macOS TCC detail — `null` off macOS or when unknown. */
  accessibility: boolean | null
  screen_recording: boolean | null
  screen_recording_capturable: boolean | null
  source: ComputerUsePermissionSource | null
  /** Populated when the status probe itself failed. */
  error: string | null
}

export interface SessionSearchResult {
  /** Lineage root of the matched conversation. Stable across compression and
   *  used as the durable pin id; falls back to session_id when absent. */
  lineage_root?: string | null
  model: string | null
  role: string | null
  /** Live compression tip of the matched conversation — resume by this id. */
  session_id: string
  session_started: number | null
  snippet: string
  source: string | null
}

export interface SessionSearchResponse {
  results: SessionSearchResult[]
}

export interface LogsResponse {
  file: string
  lines: string[]
}

export interface MmMemoryDebugSessionSummary {
  name: string
  stem: string
  path: string
  mtime: number
  size: number
  counts: Record<string, number>
  meta: Record<string, string>
  latest_observed_ts?: number
  error?: string
}

export interface MmMemoryDebugSessionsResponse {
  root: string
  sessions: MmMemoryDebugSessionSummary[]
}

export interface MmMemoryDebugTimelineItem {
  kind: string
  frame_id: string
  t_observed: number
  wall_ts?: number
  micro_id?: string
  source_type?: string
  app: string
  window_title: string
  source: string
  ocr_source?: string
  note?: string
  raw_preview: string
  observation_preview?: string
  raw_len: number
  block_count: number
  table_count: number
  thumb_b64?: string
}

export interface MmMemoryDebugTablePayload {
  id?: number | null
  table_id: string
  frame_id: string
  t_observed: number
  title: string
  columns: string[]
  rows: unknown[]
  row_hits: Array<{ index: number; row: unknown }>
  raw_text: string
  snippet: string
  source: string
  confidence: number
}

export interface MmMemoryDebugEntity {
  id: string
  name: string
  type: string
  attributes: Record<string, unknown>
  aliases: string[]
  first_seen: number
  last_seen: number
  seen_count: number
  representative_frame_id: string
  frame_ids: string[]
  merged_into: string
  revised_at: number
  revision_count: number
}

export interface MmMemoryDebugEvent {
  id: string
  t_start: number
  t_end: number
  label?: string
  summary?: string
  description?: string
  subject?: string
  object?: string
  action?: string
  macro_id?: string
  super_id?: string
  frame_ids?: string[]
  macro_ids?: string[]
  entity_names?: string[]
  key_entities?: string[]
  facts_keys?: string[]
  narrative_arc?: unknown[]
  entity_arcs?: Record<string, unknown>
  superseded_by?: string
  revised_at?: number
  revision_count?: number
  is_root?: boolean
}

export interface MmMemoryDebugEntityState {
  id: number
  entity_id: string
  entity_name: string
  t_observed: number
  state_label: string
  attributes_delta: Record<string, unknown>
  new_aliases: string[]
  confidence: number
  evidence_frame_ids: string[]
  micro_id: string
  source: string
  note: string
}

export interface MmMemoryDebugRevision {
  id: number
  t_applied: number
  reviewer_round: number
  op: string
  target_ids: string[]
  new_ids: string[]
  payload: Record<string, unknown>
  reason: string
  success: boolean
  error: string
  actor: string
}

export interface MmMemoryDebugSessionResponse {
  session: MmMemoryDebugSessionSummary
  timeline: MmMemoryDebugTimelineItem[]
  memory?: {
    entities: MmMemoryDebugEntity[]
    events: {
      micro: MmMemoryDebugEvent[]
      macro: MmMemoryDebugEvent[]
      super: MmMemoryDebugEvent[]
    }
    evolution: {
      entity_states: MmMemoryDebugEntityState[]
      revisions: MmMemoryDebugRevision[]
    }
  }
  tables: MmMemoryDebugTablePayload[]
  health: Record<string, unknown>
  trace: { logs: string[] }
}

export interface MmMemoryDebugFrameResponse {
  frame_id: string
  image_b64: string
  thumb_b64: string
  memory_frame: null | {
    frame_id: string
    t_observed: number
    wall_ts: number
    micro_id: string
    source_type: string
    note: string
    created_at: number
  }
  screen_text: null | {
    frame_id: string
    t_observed: number
    app: string
    window_title: string
    ocr_blocks: unknown[]
    raw_text: string
    source: string
    created_at: number
  }
  tables: MmMemoryDebugTablePayload[]
  micro_events: Array<{
    id: string
    t_start: number
    t_end: number
    description: string
    frame_ids: string[]
  }>
  entities: Array<{
    entity_id: string
    name: string
    type: string
    attributes: Record<string, unknown>
    t_observed: number
  }>
}

export interface MmMemoryDebugSearchResult {
  session: string
  kind: string
  score: number
  frame_id?: string
  t_observed?: number
  title: string
  snippet: string
  source?: string
  frame_ids?: string[]
  thumb_b64?: string
  table?: MmMemoryDebugTablePayload
}

export interface MmMemoryDebugSearchResponse {
  query: string
  scope: string
  results: MmMemoryDebugSearchResult[]
}

export interface MmMemoryDebugTraceMessage {
  role: string
  tool_name: string
  timestamp: number
  finish_reason: string
  content: string
  tool_calls: unknown
}

export interface MmMemoryDebugTraceResponse {
  session_id: string
  db: string
  messages: MmMemoryDebugTraceMessage[]
  logs: string[]
}

export interface PlatformStatus {
  error_code?: string
  error_message?: string
  state: string
  updated_at: string
}

export interface StatusResponse {
  active_sessions: number
  config_path: string
  config_version: number
  env_path: string
  gateway_exit_reason: string | null
  gateway_health_url: string | null
  gateway_pid: number | null
  gateway_platforms: Record<string, PlatformStatus>
  gateway_running: boolean
  gateway_state: string | null
  gateway_updated_at: string | null
  hermes_home: string
  latest_config_version: number
  release_date: string
  version: string
}

export interface ActionResponse {
  name: string
  ok: boolean
  pid: number
}

export interface ActionStatusResponse {
  exit_code: number | null
  lines: string[]
  name: string
  pid: number | null
  running: boolean
}

export interface BackendUpdateCommit {
  sha: string
  summary: string
  author: string
  at: number
}

/** Shape of `GET /api/hermes/update/check` — the backend's own update state.
 *  Used by the desktop's remote update overlay so the backend version (not the
 *  Electron client clone) drives "what's changed + Install" in remote mode. */
export interface BackendUpdateCheckResponse {
  install_method: string
  current_version: string
  behind: number | null
  update_available: boolean
  can_apply: boolean
  update_command: string | null
  message: string | null
  commits?: BackendUpdateCommit[]
}

export interface AuxiliaryTaskAssignment {
  base_url: string
  model: string
  provider: string
  task: string
}

export interface AuxiliaryModelsResponse {
  main: { model: string; provider: string }
  tasks: AuxiliaryTaskAssignment[]
}

export interface MoaModelSlot {
  provider: string
  model: string
}

export interface MoaConfigResponse {
  default_preset: string
  active_preset: string
  presets: Record<
    string,
    {
      aggregator: MoaModelSlot
      aggregator_temperature: number
      enabled: boolean
      max_tokens: number
      reference_models: MoaModelSlot[]
      reference_temperature: number
    }
  >
  aggregator: MoaModelSlot
  aggregator_temperature: number
  enabled: boolean
  max_tokens: number
  reference_models: MoaModelSlot[]
  reference_temperature: number
}

export interface ModelAssignmentRequest {
  /** Optional API key for a custom/local endpoint. Persisted to model.api_key
   *  (where the runtime reads it) for self-hosted endpoints that require auth.
   *  Only honored for custom/local providers on the main slot. */
  api_key?: string
  /** OpenAI-compatible endpoint URL. Only honored for custom/local providers
   *  on the main slot — wires a self-hosted endpoint into runtime resolution. */
  base_url?: string
  model: string
  provider: string
  scope: 'main' | 'auxiliary'
  task?: string
}

/** An auxiliary task still pinned to a provider that differs from the
 *  newly-selected main provider after a main-model switch. */
export interface StaleAuxAssignment {
  task: string
  provider: string
  model: string
}

export interface ModelAssignmentResponse {
  /** Persisted endpoint URL for custom/local providers (echoed back). */
  base_url?: string
  /** Toolset keys auto-routed through the Nous Tool Gateway as a result of
   *  switching the main provider to Nous. Empty unless provider === 'nous'
   *  and the user is a paid subscriber with unconfigured tools. */
  gateway_tools?: string[]
  model?: string
  ok: boolean
  provider?: string
  reset?: boolean
  scope?: string
  /** Auxiliary slots still pinned to a different provider than the new main.
   *  Switching main never clears aux pins; this lets the UI warn the user
   *  their helper tasks aren't following the switch. Only set on scope:'main'. */
  stale_aux?: StaleAuxAssignment[]
  tasks?: string[]
}
