import axios, { AxiosInstance } from 'axios'
import type { ModelCatalog, ModelDownload, ModelDownloads, ModelLineage } from './model-catalog'
import type { ModelOrganization, ModelExperiment } from './model-organization'
import type { ModelDiscoveryResult, HuggingFaceStatus } from './model-discovery'
import type { BenchmarkCampaign, BenchmarkCampaignRequest } from './benchmark-campaigns'

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Mirrors STATUS_* in vivasecuris/aiasylum/constants.py. A cancelled run is `failed` with meta_data.cancelled.
export interface ProviderInfo {
  name: string
  label: string
  kind: 'api' | 'local' | 'service'
  model_input: 'list' | 'fetch' | 'path' | 'text'
  description: string
  models: string[]
  requires_api_key: string | null
  fetch_endpoint: string | null
  placeholder: string | null
  aliases: string[]
  requires_extra: string | null
  configured: boolean
  available: boolean
  unavailable_reason?: string
}

export type InterpMode = 'single' | 'comparison' | 'progression' | 'model_diff'

export interface InterpModeSpec {
  name: InterpMode
  label: string
  description: string
  needs: string[]
}

export interface InterpModes {
  modes: InterpModeSpec[]
  analyses?: InterpAnalysis[]
  limits: { max_len_default: number; max_len_ceiling: number; note: string }
  warnings: string[]
  hardware?: InterpHardware
  model_presets?: Array<{ id: string; label: string; parameters_b: number; description: string }>
}

export interface InterpHardware {
  available: boolean
  default_device: string
  devices: Array<{ device: string; name: string; total_gb?: number | null; free_gb?: number | null }>
}

export interface InterpPreflight {
  ready: boolean
  errors: string[]
  warnings: string[]
  device: string
  dtype: string
  models: Array<{
    id: string
    model_type: string
    num_layers: number
    hidden_size: number
    num_attention_heads: number
    num_key_value_heads: number
    parameters_b: number
    weights_gb: number
    estimated_device_gb: number
    capture_gb: number
  }>
  hardware: InterpHardware
}

export interface InterpRun {
  id: number
  mode: InterpMode
  status: TestRunStatus
  model_a: string
  model_b: string | null
  prompt_a: string | null
  prompt_b: string | null
  prompts: string[] | null
  out_dir: string | null
  error: string | null
  created_at: string | null
  started_at: string | null
  completed_at: string | null
  metadata: Record<string, any>
}

export interface InterpRunRequest {
  patch_components?: 'layer' | 'head' | 'neuron'
  patch_layers?: number[]
  patch_positions?: number[]
  patch_heads?: [number, number][]
  patch_neurons?: [number, number][]
  lineage_parent?: string
  mode: InterpMode
  model_a: string
  model_b?: string
  prompt_a?: string
  prompt_b?: string
  prompts?: string[]
  device?: string
  dtype?: string
  max_len?: number
  window?: number
  topk?: number
  dim_reduction?: string
  enable_attention_capture?: boolean
  enable_mlp_capture?: boolean
  enable_qkv_capture?: boolean
  enable_pre_mlp_capture?: boolean
  enable_patching?: boolean
  enable_scrub?: boolean
  enable_minimal_circuit?: boolean
}

// ---------------------------------------------------------------- Weight surgery

export type WeightRunKind =
  | 'direction'
  | 'sweep'
  | 'select'
  | 'autotune'
  | 'surgery'
  | 'compare'
  | 'probe'
  | 'routing'
  | 'expert_surgery'
  | 'lora'
  | 'distill'
  // Added 2026-09-28: conditional steering, hallucination neurons, red-team,
  // embedding cartography.
  | 'induce'
  | 'hneurons'
  | 'hneuron_bake'
  | 'redteam'
  | 'embed_align'
  | 'embed_extract'
  | 'embed_recon'

// Kinds whose out_dir is a model directory under the models root.
export const WRITING_WEIGHT_KINDS: WeightRunKind[] = [
  'surgery', 'autotune', 'expert_surgery', 'lora', 'distill', 'hneuron_bake',
]

// {"12": [3, 7], "15": "all"}: which experts an expert_surgery run edits.
export type ExpertSelection = Record<string, number[] | 'all'>

export interface RoutingClassStats {
  tokens: number
  prompts: number
  selection_frac: number[]
  mean_weight: number[]
  last_token_frac: number[]
}

export interface RoutingLayer {
  layer: number
  n_experts: number
  top_k: number | null
  gate_kind: string
  has_shared: boolean
  harmful: RoutingClassStats
  harmless: RoutingClassStats
  delta_frac: number[]
  delta_last_token: number[]
}

export interface RoutingRank {
  layer: number
  expert: number
  harmful_frac: number
  harmless_frac: number
  delta: number
  last_token_delta: number
}

export interface RoutingSummary {
  model_type: string | null
  architecture: string | null
  n_layers: number
  moe_layers: number
  top_k: number | null
  prompts: { harmful: number; harmless: number }
  layers: RoutingLayer[]
  ranking: RoutingRank[]
  consistency: { gate_vs_expert_counts_match: boolean }
  split?: Record<string, any>
}

export interface LayerScore {
  layer: number
  auc: number
  cohens_d: number
  // Stable rank of the benign-centred refusal residuals at this layer.
  stable_rank?: number | null
}

// One point of the refusal-vs-directions-removed curve (sweep, method subspace_curve).
export interface CurveRow {
  label: string
  rank: number
  k: number
  refusal_rate: number
  compliance: number
  n: number
  degenerate: boolean
  samples: string[]
  factual_acc?: number
}

// Refusal projection per generated token for one prompt.
export interface TimelinePayload {
  layer: number
  prompt: string
  text: string
  tokens: string[]
  projection: number[]
  score: number[]
  in_think: boolean[]
  think_end_index: number | null
  decision_index: number | null
  decision_in_think: boolean | null
  final_side: 'refuse' | 'comply' | null
  sign_changes: number
  normalisation: 'class_means' | 'z_score'
  midpoint: number | null
  scale: number | null
}

export interface SweepRow {
  label: string
  refusal_rate: number
  n: number
  degenerate: boolean
  samples: string[]
  // Present only when the capability control ran under the same intervention.
  factual_acc?: number
}

export interface WeightMethod {
  name: string
  label: string
  description: string
  stage: string | null
  permanent: boolean | null
  available: boolean
  unavailable_reason?: string
  needs: string[]
}

export interface WeightObjective {
  name: string
  label: string
  description: string
  note?: string
  configurable: boolean
}

export interface WeightStage {
  name: WeightRunKind
  label: string
  description: string
  needs: string[]
  writes: string
}

export interface BetaPreset {
  value: number
  label: string
  description: string
}

// A goal-oriented pipeline the task view renders. Each step names a stage and
// carries explainer copy; `feeds` says which of the next step's inputs this
// step's run id fills, so the UI threads run ids forward automatically.
export interface WeightTaskStep {
  kind: WeightRunKind
  title: string
  method?: string
  objective?: string
  does?: string
  when?: string
  you_get?: string
  reads?: string
  feeds?: string
}

export interface WeightTask {
  key: string
  title: string
  goal: string
  steps: WeightTaskStep[]
}

export interface GlossaryEntry {
  short: string
  long?: string
}

export interface WeightStages {
  stages: WeightStage[]
  methods: WeightMethod[]
  objectives: WeightObjective[]
  beta_presets: BetaPreset[]
  tasks?: WeightTask[]
  defaults: Record<string, any>
  min_usable_auc: number
  interp_extra_installed: boolean
  models_root: string
  slot: { held_by: string | null; waiting: string[] }
  device?: {
    type: string
    name?: string
    gpus: Array<{ name: string; free_gb: number; total_gb: number }>
  }
}

export interface ScenarioCount {
  name: string
  count: number
  in_default_set: boolean
  excluded_reason: string | null
}

export interface WeightObjectives {
  objectives: WeightObjective[]
  scenarios: ScenarioCount[]
  default_scenarios: string[]
  harmless_builtin_count: number
  min_per_class: number
}

export interface PreflightCheck {
  code: string
  severity: 'blocking' | 'advisory'
  message: string
  acknowledgeable: boolean
}

export interface PreflightResponse {
  kind: string
  checks: PreflightCheck[]
  can_proceed: boolean
  blocking_codes: string[]
  memory: Record<string, any>
  disk: Record<string, any>
}

export interface WeightRun {
  id: number
  kind: WeightRunKind
  status: TestRunStatus
  source_model: string
  source_run_id: number | null
  method: string | null
  objective: string | null
  out_dir: string | null
  artifact_bytes: number | null
  error: string | null
  created_at: string | null
  started_at: string | null
  completed_at: string | null
  metadata: Record<string, any>
  // Wall-clock for the stage; on a machine billed by the hour, this is cost.
  elapsed_seconds?: number | null
}

export interface WeightRunRequest {
  enable_cot?: boolean
  temperature?: number
  top_p?: number
  system_prompt?: string
  jailbreak_examples?: Array<{ prompt: string; technique: string }>
  pooling?: 'last' | 'mean' | 'max' | 'last_k'
  prompt_suffix?: string
  use_eliciting_suffix?: boolean
  n_direct?: number
  n_jailbreak?: number
  n_benign?: number
  holdout_techniques?: number
  lineage_parent?: string
  kind: WeightRunKind
  source_model: string
  source_run_id?: number
  method?: string
  objective?: string
  objective_config?: Record<string, any>
  device?: string
  dtype?: string
  n_per_class?: number
  test_fraction?: number
  seed?: number
  batch_size?: number
  max_length?: number
  n_prompts?: number
  max_new_tokens?: number
  alphas?: number[]
  include_ablation?: boolean
  output_name?: string
  beta?: number
  include_embeddings?: boolean
  // expert_surgery
  expert_selection?: ExpertSelection
  expert_scale?: number
  include_shared_expert?: boolean
  // lora + distill
  dataset_rows?: Array<{ prompt: string; response?: string; system?: string }>
  dataset_benchmark?: { name: string; count: number }
  dataset_source?: 'rows' | 'benchmark' | 'objective'
  lora_rank?: number
  lora_alpha?: number
  lora_dropout?: number
  lora_targets?: string
  epochs?: number
  max_steps?: number
  lr?: number
  train_batch_size?: number
  grad_accum?: number
  gradient_checkpointing?: boolean
  merge?: boolean
  eval_rows?: number
  teacher_model?: string
  distill_temperature?: number
  ce_weight?: number
  teacher_max_new_tokens?: number
  teacher_system_prompt?: string
  use_subspace?: boolean
  k?: number
  subspace_rank?: number
  pool_layers?: number
  rfm_rank?: number
  rfm_iterations?: number
  rfm_beta?: number
  report_overlap?: boolean
  allow_no_chat_template?: boolean
  thinking?: boolean
  timeline_prompts?: number
  rederive?: boolean
  misalignment_control?: boolean
  ranks?: number[]
  ks?: number[]
  factual_floor?: number
  // select / autotune / compare: answers allowed in the wrong script
  language_drift_max?: number
  // select: 'weights' applies the real edit against a snapshot; 'hooks' is the old projection
  preview?: 'weights' | 'hooks'
  // autotune
  embedding_modes?: boolean[]
  max_candidates?: number
  stop_at_first_admissible?: boolean
  max_refusal?: number
  verify_sampled?: boolean
  sampling_seed?: number
  // surgery: cut a subspace direction to its first N rows
  rank?: number
  modified_model?: string
  // induce (conditional steering): a refusal direction (source_run_id) + a
  // category gate (probe_run_id); category_config = {name, prompts[], near_miss[]}
  probe_run_id?: number
  goal?: 'category' | 'factual'
  category_config?: { name?: string; prompts?: string[]; near_miss?: string[] }
  ms?: number[]
  taus?: number[]
  // hneurons + bake
  questions?: Array<{ question: string; aliases?: string[] }>
  n_questions?: number
  n_samples?: number
  max_answer_tokens?: number
  hneuron_top_k?: number
  hneurons_run_id?: number
  hneuron_alpha?: number
  // redteam (local only)
  redteam_target?: 'refusal' | 'prompt_leak' | 'memorization'
  redteam_attacks?: string[]
  baseline_model?: string
  secret_system?: string
  // embeddings (local only)
  model_b?: string
  reference_model?: string
  which_embedding?: 'input' | 'output'
  max_anchors?: number
  n_queries?: number
  col_subset?: number
  extra_cols?: number
  notes?: string
  acknowledge?: string[]
}

export interface FrontierRow {
  rank: number
  k: number
  refuse_harmful: number
  factual_acc: number
  factual_drop: number
  degenerate: boolean
  accepted: boolean
  include_embeddings?: boolean | null
  language_drift?: number
  drifted?: boolean
  preview?: 'weights' | 'hooks'
}

// One scored pass (greedy or sampled) over a model, as autotune and verify report it.
export interface PassMetrics {
  refuse_harmful: number
  factual_acc: number
  degenerate: boolean
  language_drift: number
  drifted: boolean
  decoding?: 'greedy' | { temperature: number; top_p: number; seed: number | null }
  factual_drop?: number
  refusal_delta?: number
  verdict?: string
  accepted?: boolean
  reason?: string
  responses?: { harmful: string[]; factual: string[] }
}

// One candidate the autotune loop applied, scored and restored.
export interface TrialRow extends FrontierRow {
  index: number
  include_embeddings: boolean
  language_drift: number
  drifted: boolean
  compliance: number
  target_met: boolean
  reason: string
  sampled: PassMetrics | null
  mean_relative_change: number
  matrices_edited: number
  elapsed_s: number
  responses?: { harmful: string[]; factual: string[] }
}

export interface VerifyReport {
  passed: boolean
  reasons: string[]
  greedy: PassMetrics
  sampled: PassMetrics | null
  hashes: Record<string, string>
  hash_check: boolean | null
  device: string
  dtype: string
  n_prompts: number
  capability_set: string
  loader: string
  checked_at: string
}

export interface AutotuneSummary {
  baseline: { greedy: PassMetrics; sampled: PassMetrics | null }
  trials: TrialRow[]
  winner: TrialRow | null
  winner_summary?: Record<string, any> | null
  candidates_planned: number
  candidates_tried: number
  stopped_early: boolean
  embeddings_tied: boolean
  target_met: boolean
  spec: Record<string, any>
  snapshot_gb?: number
  verification?: VerifyReport
  output_path?: string
  size_bytes?: number
  manifest?: Record<string, any>
}

export interface CompareMetrics {
  model: string
  refuse_harmful: number
  refuse_harmless: number
  factual_acc: number
  degenerate: boolean
  language_drift?: number
  drifted?: boolean
}

export interface DirectionOption {
  id: number
  created_at: string | null
  source_model: string
  objective: string | null
  layer: number | null
  auc: number | null
  cohens_d: number | null
  d_model: number | null
  split_hash: string | null
  usable: boolean | null
  artifacts_present: boolean
  has_sweep: boolean
}

export interface EditedModel {
  name: string
  path: string
  manifest: Record<string, any>
  run_id: number | null
  orphan: boolean
  size_bytes: number
  created_at: string | null
}

export interface EditedModelDetail extends EditedModel {
  run: WeightRun | null
  direction_run: WeightRun | null
  sweep_runs: WeightRun[]
  test_runs: Array<{ id: number; test_type: string; status: string; created_at: string; role: string }>
  interp_runs: Array<{ id: number; mode: string; status: string; created_at: string }>
  path_truncation_caveat: boolean
}

export interface InterpAnalysis {
  name: string
  label: string
  description: string
  cost: string
  claim: 'causal' | 'descriptive'
  // Which run modes actually compute this. Only the two-prompt comparison
  // service reads the causal ones; the rest would capture and discard.
  modes?: string[]
  available?: boolean
  unavailable_reason?: string
}

export type TestRunStatus =
 'pending' | 'running' | 'paused' | 'completed' | 'failed'
// Mirrors the suite statuses computed in vivasecuris/aiasylum/suites/progress_tracker.py
export type SuiteStatus = 'pending' | 'running' | 'completed' | 'failed' | 'partially_failed'

export interface TestRun {
  id: number
  doctor_provider: string
  doctor_model: string
  patient_provider: string
  patient_model: string
  test_type: string
  status: TestRunStatus
  suite_id?: number
  suite_name?: string
  created_at?: string
  updated_at?: string
  meta_data?: {
    cancelled?: boolean
    error?: string
    stopped_at?: string
    [key: string]: any
  }
}

export interface TestRunRequest {
  lineage_parent?: string
  name?: string
  doctor_provider: string
  doctor_model: string
  patient_provider: string
  patient_model: string
  test_type: string
  test_config?: Record<string, any>
  prompt_id?: number
  variables?: Record<string, string>
}

export interface TestResult {
  id: number
  test_run_id: number
  test_name: string
  test_category?: string
  input_prompt: string
  output_response: string
  score?: number
  scores?: Record<string, number>
  analysis?: string
  flags?: string[]
  meta_data?: Record<string, any>
}

export interface ConversationTurn {
  id: number
  test_run_id: number
  turn_number: number
  speaker: string
  prompt: string
  response: string
  /** Actual responding model, when recorded; older turns may have no provenance. */
  model_name?: string | null
  model_provider?: string | null
  usage?: Record<string, number> | null
  created_at?: string
  metadata?: {
    /** Exact system messages in this turn's request; absent on older records. */
    request_system_prompts?: string[]
    request_system_prompts_source?: string
    finish_reason?: string | null
    native_reasoning?: string
    native_reasoning_source?: string
    generation_metadata?: Record<string, any>
    reasoning?: string
    // Where the trace came from: the model's own inline <think> block, a field
    // the provider returned separately, or a ReACT thought.
    reasoning_source?: 'inline' | 'provider' | 'react'
    // Set on group-therapy turns only
    patient_id?: number
    patient_name?: string
    patient_model?: string
  }
}

export interface Assessment {
  id: number
  test_run_id: number
  created_at: string
  assessment_text: string
  scores: Record<string, number>
  overall_score: number
  flags?: string[]
  concerns?: string
  recommendations?: string
  metadata?: {
    cot_analysis?: {
      cot_detected: boolean
      mode: string
      analysis?: string
      ai_analysis?: string
    }
    factuality_analysis?: {
      factuality_score: number
      verified_claims: string[]
      potential_hallucinations: Array<{
        claim?: string
        reason?: string
      }>
      confidence: number
      analysis?: string
    }
    manipulation_resistance?: {
      resistance_score: number
      manipulation_attempts: Array<{
        turn?: number
        technique?: string
        description?: string
      }>
      successful_resistances: Array<{
        turn?: number
        description?: string
      }>
      failed_resistances: Array<{
        turn?: number
        description?: string
      }>
      confidence: number
      analysis?: string
    }
    manipulation_capability?: {
      capability_score: number
      manipulative_techniques: Array<{
        turn?: number
        technique?: string
        description?: string
      }>
      ethical_violations: Array<{
        turn?: number
        violation?: string
      }>
      confidence: number
      analysis?: string
    }
    [key: string]: any
  }
}

export interface Prompt {
  id: number
  name: string
  description?: string
  prompt_text: string
  prompt_type?: string
  target?: string
  category?: string
  tags: string[]
  usage_count: number
  created_at: string
  updated_at?: string
  metadata: Record<string, any>
}

/** The built-in system prompts each role falls back to (GET /api/v1/prompts/defaults). */
export interface PromptDefaults {
  doctor: { system_prompt: string; assessment_instructions: string; notes: string[] }
  patient: { system_prompt: null; user_message_template: string; interview_user_message_template?: string; notes: string[] }
  evaluator: { system_prompt: string; default_temperature: number; notes: string[] }
}

export interface PromptCreate {
  name: string
  description?: string
  prompt_text: string
  prompt_type?: string
  target?: string
  category?: string
  tags?: string[]
  metadata?: Record<string, any>
}

export interface PromptUpdate {
  name?: string
  description?: string | null
  prompt_text?: string
  prompt_type?: string
  target?: string | null
  category?: string | null
  tags?: string[]
  metadata?: Record<string, any>
}

class ApiClient {
  private client: AxiosInstance

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      withCredentials: true,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000, // 30 second timeout
    })

    // Add request interceptor for debugging
    this.client.interceptors.request.use(
      (config) => {
        console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`)
        return config
      },
      (error) => {
        console.error('[API] Request error:', error)
        return Promise.reject(error)
      }
    )

    // Add response interceptor for better error handling
    this.client.interceptors.response.use(
      (response) => {
        return response
      },
      async (error) => {
        // Retry logic for connection errors
        const config = error.config
        if (
          (error.code === 'ECONNREFUSED' || 
           error.message === 'Network Error' || 
           error.code === 'ERR_NETWORK') &&
          config &&
          !config._retry &&
          config.url !== '/health' // Don't retry health checks
        ) {
          config._retry = true
          console.log('[API] Connection failed, checking server health...')
          
          // Check if server is actually running
          try {
            const healthCheck = await axios.get(`${API_BASE_URL}/health`, { timeout: 5000 })
            if (healthCheck.status === 200) {
              console.log('[API] Server is healthy, retrying request...')
              // Server is up, retry the original request
              return this.client(config)
            }
          } catch (healthError) {
            console.error('[API] Health check failed:', healthError)
          }
        }

        if (error.code === 'ECONNREFUSED' || error.message === 'Network Error' || error.code === 'ERR_NETWORK') {
          console.error('[API] Connection error:', {
            baseURL: API_BASE_URL,
            url: error.config?.url,
            message: error.message,
            code: error.code,
          })
          // Provide a more helpful error message
          error.message = `Cannot connect to API server at ${API_BASE_URL}. Please ensure the backend is running.`
        } else if (error.response) {
          // Server responded with error status
          console.error('[API] Response error:', {
            status: error.response.status,
            statusText: error.response.statusText,
            data: error.response.data,
            url: error.config?.url,
          })
        } else if (error.request) {
          // Request was made but no response received
          console.error('[API] No response received:', {
            url: error.config?.url,
            message: error.message,
          })
          error.message = `No response from server at ${API_BASE_URL}. Please check your network connection.`
        } else {
          // Something else happened
          console.error('[API] Error:', error.message)
        }
        return Promise.reject(error)
      }
    )
  }

  async getSession(): Promise<{ authenticated: boolean; auth: 'enabled' | 'disabled' }> {
    const response = await this.client.get('/api/v1/auth/session')
    return response.data
  }

  async login(apiKey: string) {
    await this.client.post('/api/v1/auth/session', { api_key: apiKey })
  }

  async logout() {
    await this.client.post('/api/v1/auth/logout')
  }

  // Test Runs
  async createTestRun(data: TestRunRequest): Promise<TestRun> {
    const response = await this.client.post('/api/v1/test-runs/', data)
    return response.data
  }

  async getTestRun(id: number): Promise<TestRun> {
    const response = await this.client.get(`/api/v1/test-runs/${id}`)
    return response.data
  }

  async updateTestRun(id: number, data: { name?: string }): Promise<TestRun> {
    const response = await this.client.put(`/api/v1/test-runs/${id}`, data)
    return response.data
  }

  async listTestRuns(params?: { limit?: number; offset?: number; test_type?: string }): Promise<TestRun[]> {
    const response = await this.client.get('/api/v1/test-runs/', { params })
    return response.data
  }

  async getTestResults(testRunId: number): Promise<TestResult[]> {
    const response = await this.client.get(`/api/v1/test-runs/${testRunId}/results`)
    return response.data
  }

  async getConversation(testRunId: number): Promise<ConversationTurn[]> {
    const response = await this.client.get(`/api/v1/test-runs/${testRunId}/conversation`)
    return response.data
  }

  async startTestRun(testRunId: number): Promise<TestRun> {
    const response = await this.client.post(`/api/v1/test-runs/${testRunId}/start`)
    return response.data
  }

  async pauseTestRun(testRunId: number): Promise<{ message: string; id: number; status: string }> {
    const response = await this.client.post(`/api/v1/test-runs/${testRunId}/pause`)
    return response.data
  }

  async resumeTestRun(testRunId: number): Promise<TestRun> {
    const response = await this.client.post(`/api/v1/test-runs/${testRunId}/resume`)
    return response.data
  }

  async stopTestRun(testRunId: number): Promise<{ message: string; id: number }> {
    const response = await this.client.post(`/api/v1/test-runs/${testRunId}/stop`)
    return response.data
  }

  async deleteTestRun(testRunId: number): Promise<void> {
    console.log('API Client: Deleting test run', testRunId)
    try {
      const response = await this.client.delete(`/api/v1/test-runs/${testRunId}`)
      console.log('API Client: Delete response', response.data)
      return response.data
    } catch (error: any) {
      console.error('API Client: Delete error', error)
      console.error('API Client: Error response', error?.response?.data)
      console.error('API Client: Error status', error?.response?.status)
      throw error
    }
  }

  // Analysis
  async runAnalysis(testRunId: number, config?: Record<string, any>): Promise<{ message: string; test_run_id: number; analysis_test_run_id: number }> {
    // Flatten config object to match API expectations
    const requestBody = config ? {
      enable_activation_patching: config.enable_activation_patching || false,
      enable_cot_detection: config.enable_cot_detection || false,
      cot_analysis_mode: config.cot_analysis_mode || 'full',
      enable_factuality_check: config.enable_factuality_check || false,
      enable_manipulation_analysis: config.enable_manipulation_analysis || false,
      evaluator_provider: config.evaluator_provider || undefined,
      evaluator_model: config.evaluator_model || undefined,
      evaluator_system_prompt_id: config.evaluator_system_prompt_id || undefined,
      evaluator_system_prompt: config.evaluator_system_prompt || undefined,
      evaluator_temperature: config.evaluator_temperature ?? undefined,
      evaluator_max_tokens: config.evaluator_max_tokens ?? undefined,
      evaluator_enable_cot: config.evaluator_enable_cot ?? false,
      evaluator_top_p: config.evaluator_top_p ?? undefined,
    } : {
      enable_activation_patching: false,
      enable_cot_detection: false,
      cot_analysis_mode: 'full',
      enable_factuality_check: false,
      enable_manipulation_analysis: false,
    }
    const response = await this.client.post(`/api/v1/analysis/test-run/${testRunId}`, requestBody)
    return response.data
  }

  async runAnalysisUnanalyzed(config?: Record<string, any>): Promise<{ started: number; test_run_ids: number[] }> {
    // Always send explicit values so FastAPI receives a populated AnalysisRequest body.
    // Defaults here match _trigger_suite_analysis: COT on, factuality/manipulation off.
    const requestBody = {
      enable_activation_patching: config?.enable_activation_patching ?? false,
      enable_cot_detection: config?.enable_cot_detection ?? true,
      cot_analysis_mode: config?.cot_analysis_mode ?? 'full',
      enable_factuality_check: config?.enable_factuality_check ?? false,
      enable_manipulation_analysis: config?.enable_manipulation_analysis ?? false,
      evaluator_provider: config?.evaluator_provider ?? null,
      evaluator_model: config?.evaluator_model ?? null,
      evaluator_system_prompt_id: config?.evaluator_system_prompt_id ?? null,
      evaluator_system_prompt: config?.evaluator_system_prompt ?? null,
      evaluator_temperature: config?.evaluator_temperature ?? null,
      evaluator_max_tokens: config?.evaluator_max_tokens ?? null,
      evaluator_enable_cot: config?.evaluator_enable_cot ?? false,
      evaluator_top_p: config?.evaluator_top_p ?? null,
    }
    const response = await this.client.post('/api/v1/analysis/analyze-unanalyzed', requestBody)
    return response.data
  }

  async getAssessments(testRunId: number): Promise<Assessment[]> {
    const response = await this.client.get(`/api/v1/analysis/test-run/${testRunId}/assessments`)
    return response.data
  }

  // Benchmarks
  async listBenchmarks(): Promise<{
    categories: Array<{ id: string; title: string; description?: string }>
    benchmarks: Array<{ name: string; title: string; description: string; category: string }>
  }> {
    const response = await this.client.get('/api/v1/benchmarks/list')
    return response.data
  }

  async runBenchmark(data: { provider: string; model: string; benchmark: string; num_samples?: number; seed?: number; max_new_tokens?: number; dataset_revision?: string; enable_cot?: boolean; temperature?: number; top_p?: number; patient_system_prompt_id?: number; patient_system_prompt?: string; patient_prompt_framing?: boolean }): Promise<any> {
    const response = await this.client.post('/api/v1/benchmarks/run', data)
    return response.data
  }

  // Prompts
  async createPrompt(data: PromptCreate): Promise<Prompt> {
    const response = await this.client.post('/api/v1/prompts/', data)
    return response.data
  }

  async getPromptDefaults(): Promise<PromptDefaults> {
    const response = await this.client.get('/api/v1/prompts/defaults')
    return response.data
  }

  async getPrompt(id: number): Promise<Prompt> {
    const response = await this.client.get(`/api/v1/prompts/${id}`)
    return response.data
  }

  async listPrompts(params?: { category?: string; tag?: string; prompt_type?: string; target?: string; search?: string; limit?: number; offset?: number }): Promise<Prompt[]> {
    const response = await this.client.get('/api/v1/prompts/', { params })
    return response.data
  }

  async updatePrompt(id: number, data: PromptUpdate): Promise<Prompt> {
    const response = await this.client.put(`/api/v1/prompts/${id}`, data)
    return response.data
  }

  async deletePrompt(id: number): Promise<void> {
    await this.client.delete(`/api/v1/prompts/${id}`)
  }

  async incrementPromptUsage(id: number): Promise<{ usage_count: number }> {
    const response = await this.client.post(`/api/v1/prompts/${id}/increment-usage`)
    return response.data
  }

  async getPromptVariables(promptId: number): Promise<{ variables: string[] }> {
    const response = await this.client.get(`/api/v1/prompts/${promptId}/variables`)
    return response.data
  }

  // Suites
  async createSuite(data: {
    name?: string
    test_types: string[]
    benchmarks: string[]
    models: Array<{ provider: string; model: string }>
    test_config?: Record<string, any>
    num_samples?: number
    doctor?: { provider: string; model: string }
  }): Promise<Suite> {
    const response = await this.client.post('/api/v1/suites/', data)
    return response.data
  }

  async getSuite(id: number): Promise<Suite> {
    const response = await this.client.get(`/api/v1/suites/${id}`)
    return response.data
  }

  async listSuites(params?: { limit?: number; offset?: number }): Promise<Suite[]> {
    const response = await this.client.get('/api/v1/suites/', { params })
    return response.data
  }

  async updateSuite(id: number, data: { name?: string }): Promise<Suite> {
    const response = await this.client.put(`/api/v1/suites/${id}`, data)
    return response.data
  }

  async getSuiteRuns(suiteId: number): Promise<any[]> {
    const response = await this.client.get(`/api/v1/suites/${suiteId}/runs`)
    return response.data
  }

  async getSuiteProgress(suiteId: number): Promise<SuiteProgress> {
    const response = await this.client.get(`/api/v1/suites/${suiteId}/progress`)
    return response.data
  }

  async deleteSuite(id: number): Promise<void> {
    await this.client.delete(`/api/v1/suites/${id}`)
  }

  // Models (Ollama)
  async listModelCatalog(): Promise<ModelCatalog> {
    const response = await this.client.get('/api/v1/models/catalog')
    return response.data
  }

  async deleteCustomModels(names: string[]): Promise<{ freed_bytes: number; deleted: Array<{ name: string }> }> {
    const response = await this.client.post('/api/v1/models/custom/delete', { names }, { timeout: 600_000 })
    return response.data
  }

  async discoverModels(query = ''): Promise<ModelDiscoveryResult> {
    const response = await this.client.get('/api/v1/models/discover', { params: { q: query } })
    return response.data
  }

  async getHuggingFaceStatus(): Promise<HuggingFaceStatus> {
    const response = await this.client.get('/api/v1/models/huggingface/status')
    return response.data
  }

  async getModelOrganization(): Promise<ModelOrganization> {
    return (await this.client.get('/api/v1/model-organization')).data
  }

  async createModelExperiment(data: { name: string; notes?: string }): Promise<ModelExperiment> {
    return (await this.client.post('/api/v1/model-organization/experiments', data)).data
  }

  async updateModelExperiment(id: string, data: { name?: string; notes?: string }): Promise<ModelExperiment> {
    return (await this.client.patch(`/api/v1/model-organization/experiments/${encodeURIComponent(id)}`, data)).data
  }

  async saveModelOrganizationItem(data: { key: string; label: string; notes: string; experiment_ids: string[] }): Promise<ModelOrganization> {
    return (await this.client.put('/api/v1/model-organization/items', data)).data
  }

  async getModelLineage(): Promise<ModelLineage> {
    const response = await this.client.get('/api/v1/models/lineage')
    return response.data
  }

  async listProviders(): Promise<ProviderInfo[]> {
    const response = await this.client.get('/api/v1/models/providers')
    return response.data
  }

  async listOllamaModels(): Promise<string[]> {
    const response = await this.client.get('/api/v1/models/ollama')
    return response.data
  }

  async pullOllamaModel(name: string): Promise<{ status?: string; model?: string; error?: string }> {
    // The server blocks until the pull finishes, for up to 30 minutes.
    const response = await this.client.post('/api/v1/models/ollama/pull', { name }, { timeout: 31 * 60_000 })
    return response.data
  }

  // Hugging Face checkpoints in the server's cache
  async listModelDownloads(): Promise<ModelDownloads> {
    const response = await this.client.get('/api/v1/models/downloads')
    return response.data
  }

  async startModelDownload(data: { repo_id: string; revision?: string }): Promise<ModelDownload> {
    const response = await this.client.post('/api/v1/models/downloads', data)
    return response.data
  }

  async cancelModelDownload(id: string): Promise<ModelDownload> {
    const response = await this.client.post(`/api/v1/models/downloads/${encodeURIComponent(id)}/cancel`)
    return response.data
  }

  async deleteCachedModel(repoId: string): Promise<{ repo_id: string; freed_bytes: number }> {
    // The id keeps its slash: the route takes it as a path.
    const response = await this.client.delete(`/api/v1/models/cache/${repoId}`, { timeout: 600_000 })
    return response.data
  }

  // Config / API Keys
  async getApiKeysStatus(): Promise<{ openai_configured: boolean; anthropic_configured: boolean; google_configured: boolean }> {
    const response = await this.client.get('/api/v1/config/api-keys/status')
    return response.data
  }

  async updateApiKeys(keys: { openai_api_key?: string; anthropic_api_key?: string; google_api_key?: string }): Promise<{ status: string; openai_configured: boolean; anthropic_configured: boolean; google_configured: boolean }> {
    const response = await this.client.put('/api/v1/config/api-keys', keys)
    return response.data
  }

  // Interpretability
  async listInterpModes(): Promise<InterpModes> {
    const response = await this.client.get('/api/v1/interp/modes')
    return response.data
  }

  async interpPreflight(data: InterpRunRequest): Promise<InterpPreflight> {
    const response = await this.client.post('/api/v1/interp/preflight', data)
    return response.data
  }

  async createInterpRun(data: InterpRunRequest): Promise<InterpRun> {
    const response = await this.client.post('/api/v1/interp/runs', data)
    return response.data
  }

  async listInterpRuns(params?: { limit?: number; offset?: number; mode?: string }): Promise<InterpRun[]> {
    const response = await this.client.get('/api/v1/interp/runs', { params })
    return response.data
  }

  async getInterpRun(id: number): Promise<InterpRun> {
    const response = await this.client.get(`/api/v1/interp/runs/${id}`)
    return response.data
  }

  async deleteInterpRun(id: number): Promise<{ deleted: number }> {
    const response = await this.client.delete(`/api/v1/interp/runs/${id}`)
    return response.data
  }

  async stopInterpRun(id: number): Promise<{ stopped: boolean; status?: string; reason?: string }> {
    const response = await this.client.post(`/api/v1/interp/runs/${id}/stop`)
    return response.data
  }

  // The dashboard is a whole HTML document for an <iframe>, so it is addressed
  // by URL rather than fetched through axios.
  interpDashboardUrl(id: number): string {
    return `${API_BASE_URL}/api/v1/interp/runs/${id}/dashboard`
  }

  // Weight surgery
  async listWeightStages(): Promise<WeightStages> {
    const response = await this.client.get('/api/v1/weights/stages')
    return response.data
  }

  async listWeightObjectives(): Promise<WeightObjectives> {
    const response = await this.client.get('/api/v1/weights/objectives')
    return response.data
  }

  async listWeightTasks(): Promise<{ tasks: WeightTask[] }> {
    const response = await this.client.get('/api/v1/weights/tasks')
    return response.data
  }

  async getWeightGlossary(): Promise<{ glossary: Record<string, GlossaryEntry> }> {
    const response = await this.client.get('/api/v1/weights/glossary')
    return response.data
  }

  async weightPreflight(params: {
    kind: string
    dtype?: string
    source_model?: string
    source_run_id?: number
    output_name?: string
    modified_model?: string
    expert_selection?: string
    merge?: boolean
  }): Promise<PreflightResponse> {
    const response = await this.client.get('/api/v1/weights/preflight', { params })
    return response.data
  }

  async createWeightRun(data: WeightRunRequest): Promise<WeightRun> {
    const response = await this.client.post('/api/v1/weights/runs', data)
    return response.data
  }

  async listWeightRuns(params?: { limit?: number; offset?: number; kind?: string; status?: string }): Promise<WeightRun[]> {
    const response = await this.client.get('/api/v1/weights/runs', { params })
    return response.data
  }

  async getWeightRun(id: number): Promise<WeightRun> {
    const response = await this.client.get(`/api/v1/weights/runs/${id}`)
    return response.data
  }

  // The per-expert tables of a routing run; the row itself keeps only the ranking.
  async getRoutingStats(id: number): Promise<RoutingSummary> {
    const response = await this.client.get(`/api/v1/weights/runs/${id}/routing`)
    return response.data
  }

  // delete_artifacts defaults false on the server: removing the row and
  // removing gigabytes of weights are deliberately separate decisions.
  async deleteWeightRun(
    id: number,
    opts?: { delete_artifacts?: boolean; confirm?: string; force?: boolean },
  ): Promise<{ deleted: number; artifacts_removed: boolean }> {
    const response = await this.client.delete(`/api/v1/weights/runs/${id}`, {
      params: opts,
      timeout: opts?.delete_artifacts ? 600_000 : undefined,
    })
    return response.data
  }

  async stopWeightRun(id: number): Promise<{ stopped: boolean; status?: string; reason?: string }> {
    const response = await this.client.post(`/api/v1/weights/runs/${id}/stop`)
    return response.data
  }

  async listBenchmarkCampaigns(): Promise<{ campaigns: BenchmarkCampaign[] }> {
    return (await this.client.get('/api/v1/benchmark-campaigns')).data
  }

  async getBenchmarkCampaign(id: string): Promise<BenchmarkCampaign> {
    return (await this.client.get(`/api/v1/benchmark-campaigns/${encodeURIComponent(id)}`)).data
  }

  async createBenchmarkCampaign(data: BenchmarkCampaignRequest): Promise<BenchmarkCampaign> {
    return (await this.client.post('/api/v1/benchmark-campaigns', data)).data
  }

  async cancelBenchmarkCampaign(id: string): Promise<BenchmarkCampaign> {
    return (await this.client.post(`/api/v1/benchmark-campaigns/${encodeURIComponent(id)}/cancel`)).data
  }

  async rescoreBenchmarkCampaign(id: string): Promise<BenchmarkCampaign> {
    return (await this.client.post(`/api/v1/benchmark-campaigns/${encodeURIComponent(id)}/rescore`)).data
  }

  async listDirections(usableOnly = false): Promise<DirectionOption[]> {
    const response = await this.client.get('/api/v1/weights/directions', {
      params: { usable_only: usableOnly },
    })
    return response.data
  }

  async listEditedModels(): Promise<EditedModel[]> {
    const response = await this.client.get('/api/v1/weights/models')
    return response.data
  }

  async getEditedModel(name: string): Promise<EditedModelDetail> {
    const response = await this.client.get(`/api/v1/weights/models/${name}`)
    return response.data
  }

  // Interp JSON payloads, so circuit findings can render natively rather than
  // only inside the generated dashboard.
  async chatWithModel(
    name: string,
    data: {
      messages: Array<{ role: string; content: string }>
      system_prompt?: string
      temperature?: number
      max_tokens?: number
      top_p?: number
      seed?: number
      enable_cot?: boolean
      device?: 'auto' | 'cpu' | 'cuda' | 'mps'
      dtype?: 'float32' | 'float16' | 'bfloat16'
    },
  ): Promise<{
    content: string; model: string; edited: boolean; manifest: any
    usage: Record<string, number>; finish_reason: string | null; elapsed_seconds: number
    refused: boolean; refusal_detector: 'phrase_heuristic'; truncated: boolean
    settings: { temperature: number; max_tokens: number; dtype: string; device: string }
    reasoning?: string | null; reasoning_source?: string | null
    provider?: string; metadata?: Record<string, any>
    generation?: { requested: Record<string, any>; applied: Record<string, any>; notes: string[] }
  }> {
    const response = await this.client.post(`/api/v1/weights/models/${encodeURIComponent(name)}/chat`, data, {
      // The first turn loads the model, which on a cold cache is minutes.
      timeout: 600_000,
    })
    return response.data
  }

  async chatWithAnyModel(data: {
    provider: string; model: string
    messages: Array<{ role: 'user' | 'assistant'; content: string }>
    system_prompt?: string | null
    temperature?: number | null; top_p?: number | null; max_tokens?: number | null
    seed?: number | null; enable_cot?: boolean
    device?: 'auto' | 'cpu' | 'cuda' | 'mps'; dtype?: 'float32' | 'float16' | 'bfloat16'
  }): Promise<{
    content: string; reasoning?: string | null; reasoning_source?: string | null
    model: string; provider: string; finish_reason: string | null
    usage: Record<string, number> | null; metadata: Record<string, any>
    elapsed_seconds: number
    generation: { requested: Record<string, any>; applied: Record<string, any>; notes: string[] }
  }> {
    const response = await this.client.post('/api/v1/models/chat', data, { timeout: 600_000 })
    return response.data
  }

  async listInterpArtifacts(id: number): Promise<{ run_id: number; artifacts: Array<{ name: string; bytes: number }> }> {
    const response = await this.client.get(`/api/v1/interp/runs/${id}/artifacts`)
    return response.data
  }

  async getInterpArtifact(id: number, name: string): Promise<any> {
    const response = await this.client.get(`/api/v1/interp/runs/${id}/artifacts/${name}`)
    return response.data
  }
}


export interface Suite {
  id: number
  name?: string
  status: SuiteStatus
  total_runs: number
  completed_runs: number
  failed_runs: number
  running_runs: number
  pending_runs: number
  started_at?: string
  completed_at?: string
  created_at: string
  updated_at: string
  meta_data: Record<string, any>
}

export interface SuiteProgress {
  progress_percentage: number
  total_runs: number
  completed_runs: number
  failed_runs: number
  running_runs: number
  pending_runs: number
  running_tests: Array<{
    id: number
    model: string
    test_type: string
    benchmark?: string
    started_at?: string
  }>
  elapsed_time_seconds?: number
  estimated_remaining_seconds?: number
  average_time_per_run_seconds?: number
  progress_by_model: Record<string, {
    total: number
    completed: number
    failed: number
    running: number
    pending: number
  }>
  progress_by_test_type: Record<string, {
    total: number
    completed: number
    failed: number
    running: number
    pending: number
  }>
}

export const apiClient = new ApiClient()
