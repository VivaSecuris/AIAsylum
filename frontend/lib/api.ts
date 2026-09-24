import axios, { AxiosInstance } from 'axios'

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
}

// ---------------------------------------------------------------- Weight surgery

export type WeightRunKind = 'direction' | 'sweep' | 'select' | 'surgery' | 'compare'

export interface LayerScore {
  layer: number
  auc: number
  cohens_d: number
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

export interface WeightStages {
  stages: WeightStage[]
  methods: WeightMethod[]
  objectives: WeightObjective[]
  beta_presets: BetaPreset[]
  defaults: Record<string, any>
  min_usable_auc: number
  interp_extra_installed: boolean
  models_root: string
  slot: { held_by: string | null; waiting: string[] }
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
}

export interface WeightRunRequest {
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
  use_subspace?: boolean
  k?: number
  subspace_rank?: number
  pool_layers?: number
  ranks?: number[]
  ks?: number[]
  factual_floor?: number
  modified_model?: string
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
}

export interface CompareMetrics {
  model: string
  refuse_harmful: number
  refuse_harmless: number
  factual_acc: number
  degenerate: boolean
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
}

export interface ConversationTurn {
  id: number
  test_run_id: number
  turn_number: number
  speaker: string
  prompt: string
  response: string
  created_at?: string
  metadata?: {
    reasoning?: string
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
  description?: string
  prompt_text?: string
  prompt_type?: string
  target?: string
  category?: string
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

  async runBenchmark(data: { provider: string; model: string; benchmark: string; num_samples?: number }): Promise<any> {
    const response = await this.client.post('/api/v1/benchmarks/run', data)
    return response.data
  }

  // Prompts
  async createPrompt(data: PromptCreate): Promise<Prompt> {
    const response = await this.client.post('/api/v1/prompts/', data)
    return response.data
  }

  async getPrompt(id: number): Promise<Prompt> {
    const response = await this.client.get(`/api/v1/prompts/${id}`)
    return response.data
  }

  async listPrompts(params?: { category?: string; tag?: string; prompt_type?: string; target?: string; limit?: number; offset?: number }): Promise<Prompt[]> {
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
  async listProviders(): Promise<ProviderInfo[]> {
    const response = await this.client.get('/api/v1/models/providers')
    return response.data
  }

  async listOllamaModels(): Promise<string[]> {
    const response = await this.client.get('/api/v1/models/ollama')
    return response.data
  }

  async pullOllamaModel(name: string): Promise<{ status?: string; model?: string; error?: string }> {
    const response = await this.client.post('/api/v1/models/ollama/pull', { name })
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

  async weightPreflight(params: {
    kind: string
    source_model?: string
    source_run_id?: number
    output_name?: string
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

  // delete_artifacts defaults false on the server: removing the row and
  // removing gigabytes of weights are deliberately separate decisions.
  async deleteWeightRun(
    id: number,
    opts?: { delete_artifacts?: boolean; confirm?: string; force?: boolean },
  ): Promise<{ deleted: number; artifacts_removed: boolean }> {
    const response = await this.client.delete(`/api/v1/weights/runs/${id}`, { params: opts })
    return response.data
  }

  async stopWeightRun(id: number): Promise<{ stopped: boolean; status?: string; reason?: string }> {
    const response = await this.client.post(`/api/v1/weights/runs/${id}/stop`)
    return response.data
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
    },
  ): Promise<{ content: string; model: string; edited: boolean; manifest: any }> {
    const response = await this.client.post(`/api/v1/weights/models/${name}/chat`, data, {
      // The first turn loads the model, which on a cold cache is minutes.
      timeout: 600_000,
    })
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
