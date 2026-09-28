import { useEffect, useMemo, useState } from 'react'
import { useRouter } from 'next/router'
import Link from 'next/link'
import * as Tabs from '@radix-ui/react-tabs'
import { Scissors, Trash2 } from 'lucide-react'

import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { DirectionPicker } from '@/components/weights/DirectionPicker'
import { ExpertSelectionInput, formatExpertSelection, parseExpertSelection } from '@/components/weights/ExpertSelectionInput'
import { PreflightPanel } from '@/components/weights/PreflightPanel'
import { NewKindForms } from '@/components/weights/NewKindForms'
import { NEW_KINDS, newKindRequest, restoreNewKindState, type NewKindState } from '@/lib/weight-kind-config'
import { PipelineMap, type StepStatus } from '@/components/weights/PipelineMap'
import { StepExplainer } from '@/components/weights/StepExplainer'
import { Glossary } from '@/components/weights/Glossary'
import { ModelPicker } from '@/components/models/ModelPicker'
import { getSettings, getRoleGenerationDefaults } from '@/lib/settings'
import { apiClient } from '@/lib/api'
import { useModelCatalog } from '@/lib/model-catalog'
import {
  useCreateWeightRun,
  useDeleteWeightRun,
  useDirections,
  useEditedModels,
  useWeightObjectives,
  useWeightPreflight,
  useWeightRuns,
  useWeightStages,
} from '@/lib/hooks'
import { WRITING_WEIGHT_KINDS } from '@/lib/api'

type EmbeddingsMode = 'auto' | 'no' | 'yes' | 'both'
// What the API's `embedding_modes` receives; `auto` lets the engine decide by whether lm_head is tied.
const EMBEDDING_MODES: Record<EmbeddingsMode, boolean[] | undefined> = {
  auto: undefined, no: [false], yes: [true], both: [false, true],
}
function embeddingsModeOf(modes: boolean[]): EmbeddingsMode {
  const set = new Set(modes)
  if (set.size === 2) return 'both'
  return set.has(true) ? 'yes' : 'no'
}
import type { WeightRunKind, WeightRunRequest } from '@/lib/api'
import { toast } from '@/lib/toast'
import { formatApiError, formatDateTime } from '@/lib/utils'

const INPUT =
  'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

type TrainRow = { prompt: string; response?: string; system?: string }

// A JSON array, or one JSON object per line. Rows need a prompt; a response is
// required for LoRA and optional for distillation (the teacher supplies it).
function parseDatasetText(text: string): { rows: TrainRow[]; error: string | null } {
  const trimmed = text.trim()
  if (!trimmed) return { rows: [], error: null }
  let items: unknown
  try {
    items = trimmed.startsWith('[')
      ? JSON.parse(trimmed)
      : trimmed.split('\n').map((line) => line.trim()).filter(Boolean).map((line) => JSON.parse(line))
  } catch (e: any) {
    return { rows: [], error: `Not valid JSON: ${e?.message ?? e}` }
  }
  if (!Array.isArray(items)) return { rows: [], error: 'Expected a JSON array or one JSON object per line.' }
  const rows: TrainRow[] = []
  for (let i = 0; i < items.length; i += 1) {
    const item = items[i]
    if (!item || typeof item !== 'object' || typeof (item as any).prompt !== 'string' || !(item as any).prompt.trim()) {
      return { rows: [], error: `Row ${i + 1} needs a non-empty "prompt".` }
    }
    const it = item as Record<string, unknown>
    rows.push({
      prompt: it.prompt as string,
      response: typeof it.response === 'string' ? it.response : undefined,
      system: typeof it.system === 'string' ? it.system : undefined,
    })
  }
  return { rows, error: null }
}

function parseNums(text: string): number[] {
  return text
    .split(',')
    .map((x) => parseFloat(x.trim()))
    .filter((x) => Number.isFinite(x))
}

function duration(seconds?: number | null) {
  if (seconds == null) return '—'
  if (seconds < 60) return `${seconds.toFixed(0)}s`
  const m = Math.floor(seconds / 60)
  return m < 60 ? `${m}m ${Math.round(seconds % 60)}s` : `${Math.floor(m / 60)}h ${m % 60}m`
}

function gb(bytes?: number | null) {
  if (!bytes) return '—'
  const g = bytes / 1024 ** 3
  return g >= 1 ? `${g.toFixed(2)} GB` : `${(bytes / 1024 ** 2).toFixed(0)} MB`
}

export default function WeightsIndexPage() {
  const router = useRouter()
  const { data: stages, isLoading: loadingStages } = useWeightStages()
  const { data: objectives } = useWeightObjectives()
  const { data: directions = [] } = useDirections()
  const { data: runs = [] } = useWeightRuns({ limit: 50 })
  const { data: models = [] } = useEditedModels()
  const { data: catalog } = useModelCatalog()
  const createRun = useCreateWeightRun()
  const deleteRun = useDeleteWeightRun()

  const [kind, setKind] = useState<WeightRunKind>('direction')
  const [sourceModel, setSourceModel] = useState('Qwen/Qwen2.5-0.5B-Instruct')
  const [method, setMethod] = useState('direction_scale')
  const [objective, setObjective] = useState('refusal')
  const [scenarios, setScenarios] = useState<string[]>([])
  const [customPositive, setCustomPositive] = useState('')
  const [customNegative, setCustomNegative] = useState('')
  const [directionId, setDirectionId] = useState<number | null>(null)
  const [outputName, setOutputName] = useState('')
  const [beta, setBeta] = useState(0)
  const [nPerClass, setNPerClass] = useState(128)
  const [nPrompts, setNPrompts] = useState(8)
  const [subspaceRank, setSubspaceRank] = useState(1)
  const [rfmIterations, setRfmIterations] = useState(5)
  const [thinking, setThinking] = useState(false)
  const [enableCot, setEnableCot] = useState(false)
  const [temperature, setTemperature] = useState(0)
  const [topP, setTopP] = useState(0.9)
  const [systemPrompt, setSystemPrompt] = useState('')
  const [promptError, setPromptError] = useState('')
  const [promptLoading, setPromptLoading] = useState(false)
  const [timelinePrompts, setTimelinePrompts] = useState(0)
  const [rederive, setRederive] = useState(false)
  const [misalignmentControl, setMisalignmentControl] = useState(false)
  const [useSubspace, setUseSubspace] = useState(false)
  const [kStrength, setKStrength] = useState(1.0)
  const [factualFloor, setFactualFloor] = useState(0.05)
  const [ranksText, setRanksText] = useState('1,2,3,4,6,8')
  const [ksText, setKsText] = useState('1.0,1.25,1.5')
  // autotune
  const [embeddingsMode, setEmbeddingsMode] = useState<EmbeddingsMode>('auto')
  const [maxCandidates, setMaxCandidates] = useState(16)
  const [stopAtFirst, setStopAtFirst] = useState(false)
  const [maxRefusal, setMaxRefusal] = useState(0.1)
  const [languageDriftMax, setLanguageDriftMax] = useState(0.1)
  const [verifySampled, setVerifySampled] = useState(true)
  const [samplingSeed, setSamplingSeed] = useState(0)
  const [modifiedModel, setModifiedModel] = useState('')
  const [acknowledged, setAcknowledged] = useState<string[]>([])
  const [pooling, setPooling] = useState<'last' | 'mean' | 'max' | 'last_k'>('mean')
  const [nDirect, setNDirect] = useState(120)
  const [nJailbreak, setNJailbreak] = useState(120)
  const [nBenign, setNBenign] = useState(128)
  const [holdoutTechniques, setHoldoutTechniques] = useState(2)
  const [jailbreakExamples, setJailbreakExamples] = useState('')
  const [promptSuffix, setPromptSuffix] = useState('')
  const [elicitingSuffix, setElicitingSuffix] = useState(false)
  const [maxLength, setMaxLength] = useState(512)
  const [maxNewTokens, setMaxNewTokens] = useState(64)
  const [seed, setSeed] = useState(0)
  const [dtype, setDtype] = useState('bfloat16')
  const [batchSize, setBatchSize] = useState(8)
  const [forkOptions, setForkOptions] = useState<Partial<WeightRunRequest>>({})
  const [forkObjectiveConfig, setForkObjectiveConfig] = useState<Record<string, any> | undefined>()
  const [expertSelectionText, setExpertSelectionText] = useState('')
  const [expertScale, setExpertScale] = useState(0)
  const [includeShared, setIncludeShared] = useState(false)
  const [datasetMode, setDatasetMode] = useState<'rows' | 'benchmark' | 'objective'>('rows')
  const [datasetText, setDatasetText] = useState('')
  const [benchmarkName, setBenchmarkName] = useState('')
  const [benchmarkCount, setBenchmarkCount] = useState(256)
  const [loraRank, setLoraRank] = useState(8)
  const [loraAlpha, setLoraAlpha] = useState(16)
  const [loraTargets, setLoraTargets] = useState('attention')
  const [epochs, setEpochs] = useState(1)
  const [maxSteps, setMaxSteps] = useState<number | ''>('')
  const [lr, setLr] = useState(0.0002)
  const [gradAccum, setGradAccum] = useState(8)
  const [merge, setMerge] = useState(true)
  const [teacherModel, setTeacherModel] = useState('')
  const [distillTemperature, setDistillTemperature] = useState(2)
  const [ceWeight, setCeWeight] = useState(0.5)
  const [teacherMaxNewTokens, setTeacherMaxNewTokens] = useState(256)
  const [nk, setNk] = useState<NewKindState>(() => restoreNewKindState())

  useEffect(() => {
    if (!router.isReady) return
    const query = router.query
    const defaults = getSettings()
    setSourceModel(['transformers', 'local'].includes(defaults.defaultPatientProvider) && defaults.defaultPatientModel
      ? defaults.defaultPatientModel : 'Qwen/Qwen2.5-0.5B-Instruct')
    const generation = getRoleGenerationDefaults(defaults, 'patient')
    let active = true
    let suppliedSystemPrompt = false
    setEnableCot(generation.enable_cot)
    setTemperature(generation.temperature === '' ? 0 : Number(generation.temperature))
    setTopP(generation.top_p === '' ? 0.9 : Number(generation.top_p))
    setSystemPrompt(defaults.defaultPatientSystemPromptText || '')
    setPromptError('')
    setPromptLoading(false)
    setNk(restoreNewKindState())
    setForkOptions({})
    setDirectionId(null)
    setOutputName('')
    setModifiedModel('')
    setMethod('direction_scale')
    setObjective('refusal')
    setForkObjectiveConfig(undefined)
    setScenarios([])
    setCustomPositive('')
    setCustomNegative('')
    setNPerClass(128)
    setNPrompts(8)
    setSubspaceRank(1)
    setRfmIterations(5)
    setBeta(0)
    setKStrength(1)
    setUseSubspace(false)
    setFactualFloor(0.05)
    setRanksText('1,2,3,4,6,8')
    setKsText('1.0,1.25,1.5')
    setEmbeddingsMode('auto')
    setMaxCandidates(16)
    setStopAtFirst(false)
    setMaxRefusal(0.1)
    setLanguageDriftMax(0.1)
    setVerifySampled(true)
    setSamplingSeed(0)
    setThinking(false)
    setTimelinePrompts(0)
    setRederive(false)
    setMisalignmentControl(false)
    setPooling('mean')
    setNDirect(120)
    setNJailbreak(120)
    setNBenign(128)
    setHoldoutTechniques(2)
    setJailbreakExamples('')
    setPromptSuffix('')
    setElicitingSuffix(false)
    setMaxLength(512)
    setMaxNewTokens(generation.max_tokens !== '' ? Number(generation.max_tokens) : 64)
    setSeed(0)
    setDtype('bfloat16')
    setBatchSize(8)
    setExpertSelectionText('')
    setExpertScale(0)
    setIncludeShared(false)
    setDatasetMode('rows')
    setDatasetText('')
    setBenchmarkName('')
    setBenchmarkCount(256)
    setLoraRank(8)
    setLoraAlpha(16)
    setLoraTargets('attention')
    setEpochs(1)
    setMaxSteps('')
    setLr(0.0002)
    setGradAccum(8)
    setMerge(true)
    setTeacherModel('')
    setDistillTemperature(2)
    setCeWeight(0.5)
    setTeacherMaxNewTokens(256)
    if (typeof query.kind === 'string' && ['direction', 'sweep', 'select', 'autotune', 'surgery', 'compare', 'probe', 'routing', 'expert_surgery', 'lora', 'distill', 'induce', 'hneurons', 'hneuron_bake', 'redteam', 'embed_align', 'embed_extract', 'embed_recon'].includes(query.kind)) setKind(query.kind as WeightRunKind)
    if (typeof query.expert_selection === 'string') {
      try {
        const picked = JSON.parse(query.expert_selection)
        if (picked && typeof picked === 'object' && !Array.isArray(picked)) setExpertSelectionText(formatExpertSelection(picked))
      } catch { /* not a selection */ }
    }
    if (typeof query.source_model === 'string') setSourceModel(query.source_model)
    if (typeof query.modified_model === 'string') setModifiedModel(query.modified_model)
    if (typeof query.source_run_id === 'string' && /^\d+$/.test(query.source_run_id)) setDirectionId(Number(query.source_run_id))
    if (typeof query.method === 'string') setMethod(query.method)
    if (typeof query.objective === 'string') setObjective(query.objective)
    if (typeof query.objective_config === 'string') {
      try {
        const config = JSON.parse(query.objective_config)
        if (config && typeof config === 'object' && !Array.isArray(config)) {
          setForkObjectiveConfig(config)
          if (Array.isArray(config.refused)) setCustomPositive(config.refused.join('\n'))
          if (Array.isArray(config.answered)) setCustomNegative(config.answered.join('\n'))
          if (Array.isArray(config.scenarios)) setScenarios(config.scenarios)
          if (Array.isArray(config.positive?.prompts)) setCustomPositive(config.positive.prompts.join('\n'))
          if (Array.isArray(config.negative?.prompts)) setCustomNegative(config.negative.prompts.join('\n'))
        }
      } catch { setForkObjectiveConfig(undefined) }
    }
    if (typeof query.options === 'string') {
      try {
        const opts = JSON.parse(query.options)
        if (opts && typeof opts === 'object' && !Array.isArray(opts)) {
          // A recorded comparison is a complete protocol. Fields introduced
          // later use the legacy values, never today's browser defaults.
          if (query.kind === 'compare') {
            setEnableCot(false)
            setTemperature(0)
            setTopP(0.9)
            setSystemPrompt('')
            setMaxNewTokens(64)
            suppliedSystemPrompt = true
          }
          const { source_model: _source, source_run_id: _run, output_name: _out, acknowledge: _ack, lineage_parent: _parent, ...settings } = opts
          setForkOptions(settings)
          setNk(restoreNewKindState(opts))
          if (['last', 'mean', 'max', 'last_k'].includes(opts.pooling)) setPooling(opts.pooling)
          if (typeof opts.n_direct === 'number') setNDirect(opts.n_direct)
          if (typeof opts.n_jailbreak === 'number') setNJailbreak(opts.n_jailbreak)
          if (typeof opts.n_benign === 'number') setNBenign(opts.n_benign)
          if (typeof opts.holdout_techniques === 'number') setHoldoutTechniques(opts.holdout_techniques)
          if (Array.isArray(opts.jailbreak_examples)) setJailbreakExamples(JSON.stringify(opts.jailbreak_examples, null, 2))
          if (typeof opts.prompt_suffix === 'string') setPromptSuffix(opts.prompt_suffix)
          if (typeof opts.use_eliciting_suffix === 'boolean') setElicitingSuffix(opts.use_eliciting_suffix)
          if (typeof opts.max_length === 'number') setMaxLength(opts.max_length)
          if (typeof opts.max_new_tokens === 'number') setMaxNewTokens(opts.max_new_tokens)
          if (typeof opts.seed === 'number') setSeed(opts.seed)
          if (typeof opts.dtype === 'string') setDtype(opts.dtype)
          if (typeof opts.batch_size === 'number') setBatchSize(opts.batch_size)
          if (typeof opts.n_per_class === 'number') setNPerClass(opts.n_per_class)
          if (typeof opts.n_prompts === 'number') setNPrompts(opts.n_prompts)
          if (typeof (opts.rfm_rank ?? opts.subspace_rank) === 'number') setSubspaceRank(opts.rfm_rank ?? opts.subspace_rank)
          if (typeof opts.rfm_iterations === 'number') setRfmIterations(opts.rfm_iterations)
          if (typeof opts.beta === 'number') setBeta(opts.beta)
          if (typeof opts.k === 'number') setKStrength(opts.k)
          if (typeof opts.use_subspace === 'boolean') setUseSubspace(opts.use_subspace)
          if (typeof opts.factual_floor === 'number') setFactualFloor(opts.factual_floor)
          if (Array.isArray(opts.ranks)) setRanksText(opts.ranks.join(','))
          if (Array.isArray(opts.ks)) setKsText(opts.ks.join(','))
          if (Array.isArray(opts.embedding_modes)) setEmbeddingsMode(embeddingsModeOf(opts.embedding_modes))
          if (typeof opts.max_candidates === 'number') setMaxCandidates(opts.max_candidates)
          if (typeof opts.stop_at_first_admissible === 'boolean') setStopAtFirst(opts.stop_at_first_admissible)
          if (typeof opts.max_refusal === 'number') setMaxRefusal(opts.max_refusal)
          if (typeof opts.language_drift_max === 'number') setLanguageDriftMax(opts.language_drift_max)
          if (typeof opts.verify_sampled === 'boolean') setVerifySampled(opts.verify_sampled)
          if (typeof opts.sampling_seed === 'number') setSamplingSeed(opts.sampling_seed)
          if (typeof opts.thinking === 'boolean') setThinking(opts.thinking)
          if (typeof opts.enable_cot === 'boolean') setEnableCot(opts.enable_cot)
          if (typeof opts.temperature === 'number') setTemperature(opts.temperature)
          if (typeof opts.top_p === 'number') setTopP(opts.top_p)
          if (typeof opts.system_prompt === 'string' || opts.system_prompt === null) {
            suppliedSystemPrompt = true
            setSystemPrompt(opts.system_prompt || '')
          }
          if (typeof opts.timeline_prompts === 'number') setTimelinePrompts(opts.timeline_prompts)
          if (typeof opts.rederive === 'boolean') setRederive(opts.rederive)
          if (typeof opts.misalignment_control === 'boolean') setMisalignmentControl(opts.misalignment_control)
          if (opts.expert_selection && typeof opts.expert_selection === 'object') setExpertSelectionText(formatExpertSelection(opts.expert_selection))
          if (typeof opts.expert_scale === 'number') setExpertScale(opts.expert_scale)
          if (typeof opts.include_shared_expert === 'boolean') setIncludeShared(opts.include_shared_expert)
          if (['rows', 'benchmark', 'objective'].includes(opts.dataset_source)) setDatasetMode(opts.dataset_source)
          if (opts.dataset_benchmark && typeof opts.dataset_benchmark === 'object') {
            if (typeof opts.dataset_benchmark.name === 'string') setBenchmarkName(opts.dataset_benchmark.name)
            if (typeof opts.dataset_benchmark.count === 'number') setBenchmarkCount(opts.dataset_benchmark.count)
          }
          if (typeof opts.lora_rank === 'number') setLoraRank(opts.lora_rank)
          if (typeof opts.lora_alpha === 'number') setLoraAlpha(opts.lora_alpha)
          if (typeof opts.lora_targets === 'string') setLoraTargets(opts.lora_targets)
          if (typeof opts.epochs === 'number') setEpochs(opts.epochs)
          if (typeof opts.max_steps === 'number') setMaxSteps(opts.max_steps)
          if (typeof opts.lr === 'number') setLr(opts.lr)
          if (typeof opts.grad_accum === 'number') setGradAccum(opts.grad_accum)
          if (typeof opts.merge === 'boolean') setMerge(opts.merge)
          if (typeof opts.teacher_model === 'string') setTeacherModel(opts.teacher_model)
          if (typeof opts.distill_temperature === 'number') setDistillTemperature(opts.distill_temperature)
          if (typeof opts.ce_weight === 'number') setCeWeight(opts.ce_weight)
          if (typeof opts.teacher_max_new_tokens === 'number') setTeacherMaxNewTokens(opts.teacher_max_new_tokens)
        }
      } catch { setForkOptions({}) }
    } else setForkOptions({})
    if (!suppliedSystemPrompt && defaults.defaultPatientSystemPromptId != null) {
      setPromptLoading(true)
      apiClient.getPrompt(defaults.defaultPatientSystemPromptId).then((prompt) => {
        if (!active) return
        if (prompt.prompt_type !== 'system_prompt' || prompt.target !== 'patient') {
          setPromptError('The saved patient system prompt is unavailable for this role. Choose a prompt before running the comparison.')
          return
        }
        setSystemPrompt(prompt.prompt_text)
      }).catch(() => { if (active) setPromptError('Could not load the saved patient system prompt. Enter a system prompt or explicitly clear it before continuing.') })
        .finally(() => { if (active) setPromptLoading(false) })
    }
    setAcknowledged([])
    return () => { active = false }
  }, [router.isReady, router.query.kind, router.query.source_model, router.query.modified_model, router.query.source_run_id, router.query.method, router.query.objective, router.query.objective_config, router.query.options, router.query.expert_selection])

  const { selection: expertSelection, errors: expertErrors } = useMemo(
    () => parseExpertSelection(expertSelectionText),
    [expertSelectionText],
  )
  const expertSelectionReady = expertErrors.length === 0 && Object.keys(expertSelection).length > 0
  const writesModel = WRITING_WEIGHT_KINDS.includes(kind)
  // A direction edit: the whole model, or the chosen experts only.
  const directionEdit = kind === 'surgery' || (kind === 'expert_surgery' && method === 'expert_direction_scale')
  const training = kind === 'lora' || kind === 'distill'
  // A training run that keeps only its adapter writes under runs/, not a model directory.
  const needsOutputName = writesModel && (!training || merge)
  const { rows: datasetRows, error: datasetError } = useMemo(() => parseDatasetText(datasetText), [datasetText])
  const datasetReady =
    !training ||
    (datasetMode === 'rows' && datasetRows.length > 0 && !datasetError && (kind !== 'lora' || datasetRows.every((r) => r.response?.trim()))) ||
    (datasetMode === 'benchmark' && benchmarkName.trim().length > 0) ||
    (datasetMode === 'objective' && kind === 'distill')

  const { data: preflight } = useWeightPreflight({
    kind,
    dtype,
    source_model: sourceModel,
    source_run_id: directionId ?? undefined,
    output_name: needsOutputName ? outputName || undefined : undefined,
    modified_model: kind === 'compare' ? modifiedModel || undefined : undefined,
    expert_selection: kind === 'expert_surgery' && expertSelectionReady ? JSON.stringify(expertSelection) : undefined,
    merge: training ? merge : undefined,
  })

  const stageSpec = stages?.stages.find((s) => s.name === kind)
  // The loop's defaults are a smaller grid than the report-only search.
  useEffect(() => {
    if (kind !== 'autotune') return
    if (ranksText === '1,2,3,4,6,8') setRanksText('1,2,3,4')
    if (ksText === '1.0,1.25,1.5') setKsText('1.0,1.25')
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind])
  const needs = (field: string) => stageSpec?.needs.includes(field) ?? false

  // Methods relevant to this stage, permanent ones separated from reversible
  // ones because that is the distinction that actually matters to the user.
  const { reversible, permanent, unavailable } = useMemo(() => {
    // Only this stage's methods: a sweep method on a surgery run is rejected
    // by the API, so there is no point offering it.
    const all = (stages?.methods ?? []).filter((m) => !m.available || m.stage == null || m.stage === kind)
    return {
      reversible: all.filter((m) => m.available && !m.permanent),
      permanent: all.filter((m) => m.available && m.permanent),
      unavailable: all.filter((m) => !m.available),
    }
  }, [stages, kind])

  // Keep the method inside the chosen stage; fall back to the stage default.
  useEffect(() => {
    const current = stages?.methods.find((m) => m.name === method)
    if (current && current.available && current.stage === kind) return
    const fallback = stages?.methods.find((m) => m.available && m.stage === kind)
    if (fallback) setMethod(fallback.name)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [kind, stages])

  const narrowed = objective === 'refusal_narrow'
  const custom = objective === 'custom'
  const overRefusal = objective === 'over_refusal'
  const customPositiveLines = customPositive.split('\n').map((l) => l.trim()).filter(Boolean)
  const customNegativeLines = customNegative.split('\n').map((l) => l.trim()).filter(Boolean)
  // build_split rejects anything under 8 per class, so say so before the run
  // rather than failing minutes in.
  const customTooFew = kind === 'direction' && (
    overRefusal ? new Set(customPositiveLines).size < 8 || new Set(customNegativeLines).size < 8
      : custom && ((customPositiveLines.length > 0 && new Set(customPositiveLines).size < 8) || (customNegativeLines.length > 0 && new Set(customNegativeLines).size < 8))
  )
  const selectedCount = useMemo(() => {
    if (!narrowed || !objectives) return null
    const picked = scenarios.length ? scenarios : objectives.default_scenarios
    return objectives.scenarios
      .filter((s) => picked.includes(s.name))
      .reduce((sum, s) => sum + s.count, 0)
  }, [narrowed, objectives, scenarios])

  const tooFewPrompts =
    narrowed && selectedCount != null && selectedCount < (objectives?.min_per_class ?? 8)

  const patchNk = (patch: Partial<NewKindState>) => setNk((prev) => ({ ...prev, ...patch }))
  const [taskKey, setTaskKey] = useState<string | null>(null)
  const [glossaryOpen, setGlossaryOpen] = useState(false)
  const isNewKind = NEW_KINDS.includes(kind)

  async function launch() {
    try {
      const examples = kind === 'probe' && jailbreakExamples.trim() ? JSON.parse(jailbreakExamples) : undefined
      if (examples && (!Array.isArray(examples) || examples.some((row: any) => !row.prompt?.trim() || !row.technique?.trim()))) throw new Error('Each example needs prompt and technique fields.')
      const run = await createRun.mutateAsync({
        ...forkOptions,
        lineage_parent: typeof router.query.lineage_parent === 'string' ? router.query.lineage_parent : undefined,
        kind,
        source_model: sourceModel.trim(),
        method,
        objective: training ? (kind === 'distill' && datasetMode === 'objective' ? objective : 'dataset') : objective,
        max_length: maxLength,
        max_new_tokens: maxNewTokens,
        enable_cot: kind === 'compare' ? enableCot : undefined,
        temperature: kind === 'compare' ? temperature : undefined,
        top_p: kind === 'compare' ? topP : undefined,
        system_prompt: kind === 'compare' ? systemPrompt : undefined,
        seed, dtype, batch_size: batchSize,
        jailbreak_examples: examples,
        pooling: kind === 'probe' ? pooling : undefined,
        n_direct: kind === 'probe' ? nDirect : undefined,
        n_jailbreak: kind === 'probe' ? nJailbreak : undefined,
        n_benign: kind === 'probe' ? nBenign : kind === 'induce' ? forkOptions.n_benign : undefined,
        holdout_techniques: kind === 'probe' ? holdoutTechniques : undefined,
        prompt_suffix: kind === 'probe' ? promptSuffix.trim() || undefined : undefined,
        use_eliciting_suffix: kind === 'probe' ? elicitingSuffix : undefined,
        objective_config: overRefusal ? { refused: customPositiveLines, answered: customNegativeLines } : narrowed
          ? { scenarios: scenarios.length ? scenarios : objectives?.default_scenarios }
          : custom && (customPositiveLines.length || customNegativeLines.length)
            ? {
                positive: customPositiveLines.length
                  ? { prompts: customPositiveLines }
                  : undefined,
                negative: customNegativeLines.length
                  ? { prompts: customNegativeLines }
                  : undefined,
              }
            : forkObjectiveConfig,
        source_run_id: needs('source_run_id') || (kind === 'expert_surgery' && method === 'expert_direction_scale') ? directionId ?? undefined : undefined,
        output_name: needsOutputName ? outputName.trim() : undefined,
        dataset_rows: training && datasetMode === 'rows' ? datasetRows : undefined,
        dataset_benchmark: training && datasetMode === 'benchmark' ? { name: benchmarkName.trim(), count: benchmarkCount } : undefined,
        dataset_source: training ? datasetMode : undefined,
        lora_rank: training ? loraRank : undefined,
        lora_alpha: training ? loraAlpha : undefined,
        lora_targets: training ? loraTargets : undefined,
        epochs: training ? epochs : undefined,
        max_steps: training && maxSteps !== '' ? maxSteps : undefined,
        lr: training ? lr : undefined,
        grad_accum: training ? gradAccum : undefined,
        merge: training ? merge : undefined,
        teacher_model: kind === 'distill' ? teacherModel.trim() : undefined,
        distill_temperature: kind === 'distill' ? distillTemperature : undefined,
        ce_weight: kind === 'distill' ? ceWeight : undefined,
        teacher_max_new_tokens: kind === 'distill' ? teacherMaxNewTokens : undefined,
        beta: directionEdit && !useSubspace ? beta : undefined,
        use_subspace: directionEdit ? useSubspace : undefined,
        k: directionEdit && useSubspace ? kStrength : undefined,
        expert_selection: kind === 'expert_surgery' ? expertSelection : undefined,
        expert_scale: kind === 'expert_surgery' ? expertScale : undefined,
        include_shared_expert: kind === 'expert_surgery' ? includeShared : undefined,
        n_per_class: kind === 'direction' || kind === 'routing' || (kind === 'distill' && datasetMode === 'objective') ? nPerClass : undefined,
        subspace_rank: kind === 'direction' && method !== 'rfm_agop' ? subspaceRank : undefined,
        rfm_rank: kind === 'direction' && method === 'rfm_agop' ? subspaceRank : undefined,
        rfm_iterations: kind === 'direction' && method === 'rfm_agop' ? rfmIterations : undefined,
        thinking: kind === 'sweep' ? thinking : undefined,
        timeline_prompts: kind === 'sweep' ? timelinePrompts : undefined,
        rederive: kind === 'compare' ? rederive || method === 'compare_rederive' : undefined,
        misalignment_control: kind === 'compare' ? misalignmentControl : undefined,
        n_prompts: kind === 'sweep' || kind === 'select' || kind === 'compare' || kind === 'autotune' ? nPrompts : kind === 'induce' ? forkOptions.n_prompts : undefined,
        ranks: kind === 'select' || kind === 'autotune' ? parseNums(ranksText).map(Math.round) : undefined,
        ks: kind === 'select' || kind === 'autotune' ? parseNums(ksText) : undefined,
        factual_floor: kind === 'select' || kind === 'autotune' ? factualFloor : undefined,
        language_drift_max: kind === 'select' || kind === 'autotune' || kind === 'compare' ? languageDriftMax : undefined,
        embedding_modes: kind === 'autotune' ? EMBEDDING_MODES[embeddingsMode] : undefined,
        max_candidates: kind === 'autotune' ? maxCandidates : undefined,
        stop_at_first_admissible: kind === 'autotune' ? stopAtFirst : undefined,
        max_refusal: kind === 'autotune' ? maxRefusal : undefined,
        // Induce also consumes these shared controls, but does not show the
        // autotune editor. Keep its recorded choice, including explicit false.
        verify_sampled: kind === 'autotune' ? verifySampled : kind === 'induce' ? forkOptions.verify_sampled : undefined,
        sampling_seed: kind === 'autotune' ? samplingSeed : undefined,
        modified_model: kind === 'compare' ? modifiedModel.trim() : undefined,
        ...newKindRequest(kind, nk),
        acknowledge: acknowledged,
      })
      toast.success(`Started ${stageSpec?.label.toLowerCase() ?? kind}`)
      router.push(`/weights/${run.id}`)
    } catch (e: any) {
      const detail = e?.response?.data?.detail
      if (detail?.blocking) {
        toast.error('Preflight blocked this run — see the checks above.')
      } else {
        toast.error(formatApiError(e, 'Could not start the run'))
      }
    }
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Scissors className="h-7 w-7" />
          <div>
            <h1 className="text-3xl font-bold">Neurosurgery</h1>
            <p className="text-sm text-muted-foreground">
              Derive a behavioural direction, prove it moves behaviour, then write it
              permanently into the weights.
            </p>
          </div>
        </div>

        <div className="rounded-lg border bg-card p-4 text-sm">
          <p className="font-medium">Explore a direction → preview its effect → save a custom model → compare results</p>
          <p className="mt-1 text-muted-foreground">Directions and sweeps are experiments. A saved checkpoint appears in your model library, ready to analyze or evaluate.</p>
          <Link href="/compare?view=models" className="mt-2 inline-flex font-medium text-primary hover:underline">Open model library →</Link>
        </div>
        {typeof router.query.lineage_parent === 'string' && <div className="rounded-lg border bg-primary/5 p-4 text-sm">
          <p className="font-medium">Branch from an earlier experiment</p>
          <p className="mt-1 text-muted-foreground">Review the restored settings and change the method or inputs. Starting this run adds a new branch; previous results stay intact. Saving an edit requires a new output name.</p>
          {Object.keys(forkOptions).length > 0 && <details className="mt-2"><summary className="cursor-pointer">Restored experiment settings</summary><pre className="mt-2 overflow-auto text-xs">{JSON.stringify(forkOptions, null, 2)}</pre></details>}
        </div>}

        {stages?.device && (
          <p className="text-xs text-muted-foreground">
            Runs execute on{' '}
            <span className="font-mono">
              {stages.device.name ?? stages.device.type}
              {stages.device.gpus?.[0]
                ? ` · ${stages.device.gpus[0].free_gb.toFixed(0)} of ${stages.device.gpus[0].total_gb.toFixed(0)} GB free`
                : ''}
            </span>
          </p>
        )}

        {stages && !stages.interp_extra_installed && (
          <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm">
            This needs the optional extra: <code>pip install -e &quot;.[interp]&quot;</code>
          </div>
        )}

        <Tabs.Root defaultValue="run">
          <Tabs.List className="flex gap-1 border-b">
            {[
              ['run', 'Run a stage'],
              ['runs', `Runs (${runs.length})`],
              ['directions', `Directions (${directions.length})`],
              ['models', `Edited models (${models.length})`],
            ].map(([v, label]) => (
              <Tabs.Trigger
                key={v}
                value={v}
                className="px-4 py-2 text-sm font-medium data-[state=active]:border-b-2 data-[state=active]:border-primary"
              >
                {label}
              </Tabs.Trigger>
            ))}
          </Tabs.List>

          {/* ---------------------------------------------------------- Run */}
          <Tabs.Content value="run" className="pt-6">
            {loadingStages ? (
              <LoadingSpinner />
            ) : (
              <div className="space-y-6">
                {/* Goal-oriented tasks: pick what you want to do, then walk its
                    pipeline. The flat stage list stays under "Advanced". */}
                {(() => {
                  const tasks = stages?.tasks ?? []
                  const activeTask = tasks.find((t) => t.key === taskKey) ?? null
                  return (
                    <>
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-medium">{activeTask ? activeTask.goal : 'What do you want to do?'}</p>
                        <button type="button" onClick={() => setGlossaryOpen(true)} className="text-xs text-primary hover:underline">
                          ? Glossary
                        </button>
                      </div>
                      {!activeTask && tasks.length > 0 && (
                        <div className="grid gap-3 md:grid-cols-3">
                          {tasks.map((t) => (
                            <button
                              key={t.key}
                              type="button"
                              onClick={() => { setTaskKey(t.key); const s0 = t.steps[0]; setKind(s0.kind); if (s0.method) setMethod(s0.method); if (s0.objective) setObjective(s0.objective) }}
                              className="rounded-lg border p-4 text-left transition-colors hover:bg-muted/50"
                            >
                              <p className="text-sm font-semibold">{t.title}</p>
                              <p className="mt-1 text-xs text-muted-foreground">{t.goal}</p>
                              <p className="mt-2 text-[11px] text-muted-foreground">{t.steps.map((s) => s.title).join(' → ')}</p>
                            </button>
                          ))}
                        </div>
                      )}
                      {activeTask && (
                        <div className="space-y-3 rounded-lg border bg-card p-4">
                          <div className="flex items-center justify-between">
                            <p className="text-sm font-semibold">{activeTask.title}</p>
                            <button type="button" onClick={() => setTaskKey(null)} className="text-xs text-muted-foreground hover:underline">← all tasks</button>
                          </div>
                          <PipelineMap
                            steps={activeTask.steps}
                            activeIndex={Math.max(0, activeTask.steps.findIndex((s) => s.kind === kind))}
                            statuses={activeTask.steps.map((s): StepStatus => {
                              if (s.kind === kind) return 'current'
                              const st = stages?.stages.find((x) => x.name === s.kind)
                              if (st?.needs.includes('source_run_id') && directions.length === 0) return 'locked'
                              return 'available'
                            })}
                            onSelect={(i) => { const st = activeTask.steps[i]; setKind(st.kind); if (st.method) setMethod(st.method); if (st.objective) setObjective(st.objective) }}
                          />
                          {(() => {
                            const step = activeTask.steps.find((s) => s.kind === kind)
                            const stage = stages?.stages.find((s) => s.name === kind)
                            return step ? <StepExplainer step={step} stage={stage} /> : null
                          })()}
                        </div>
                      )}
                    </>
                  )
                })()}

                {/* Advanced: every stage, independently launchable. */}
                <details open={!taskKey}>
                  <summary className="cursor-pointer text-sm font-medium text-muted-foreground">Advanced: run a single stage</summary>
                  <div className="mt-3 grid gap-3 md:grid-cols-3">
                  {stages?.stages.map((s) => {
                    const needsDirection = s.needs.includes('source_run_id')
                    const blocked = needsDirection && directions.length === 0
                    return (
                      <button
                        key={s.name}
                        type="button"
                        disabled={blocked}
                        onClick={() => setKind(s.name)}
                        className={[
                          'rounded-lg border p-4 text-left transition-colors',
                          kind === s.name ? 'border-primary bg-primary/5' : 'hover:bg-muted/50',
                          blocked ? 'cursor-not-allowed opacity-50' : '',
                        ].join(' ')}
                      >
                        <p className="text-sm font-semibold">{s.label}</p>
                        <p className="mt-1 text-xs text-muted-foreground">{s.description}</p>
                        <p className="mt-2 text-[11px] text-muted-foreground">
                          Writes {s.writes}
                        </p>
                        {blocked && (
                          <p className="mt-2 text-[11px] text-destructive">
                            Derive a direction first — this stage consumes one.
                          </p>
                        )}
                      </button>
                    )
                  })}
                  </div>
                </details>

                <div className="rounded-lg border bg-card p-6 shadow-sm">
                  <div className="space-y-5">
                    {/* 1. What behaviour */}
                    {needs('objective') && (
                      <div>
                        <label className="text-xs font-medium">Objective — what to target</label>
                        <select
                          value={objective}
                          onChange={(e) => { setObjective(e.target.value); setForkObjectiveConfig(undefined) }}
                          className={INPUT}
                        >
                          {stages?.objectives.map((o) => (
                            <option key={o.name} value={o.name}>
                              {o.label}
                            </option>
                          ))}
                        </select>
                        {stages?.objectives.find((o) => o.name === objective) && (
                          <p className="mt-1 text-xs text-muted-foreground">
                            {stages.objectives.find((o) => o.name === objective)!.description}{' '}
                            {stages.objectives.find((o) => o.name === objective)!.note}
                          </p>
                        )}

                        {(custom || overRefusal) && (
                          <div className="mt-3 space-y-3 rounded-md border p-3">
                            <p className="text-xs text-muted-foreground">
                              {overRefusal ? 'Paste benign prompts that this model refused and matched benign prompts it answered. At least eight distinct prompts in each set; no fallback dataset is used.' : 'One prompt per line. Use matched contrast sets so the direction isolates the intended behaviour. Leave a box empty to use its curated defaults.'}
                            </p>
                            <div className="grid gap-3 sm:grid-cols-2">
                              <div>
                                <label className="text-xs font-medium">
                                  {overRefusal ? 'Benign prompts refused' : 'Positive class'} ({customPositiveLines.length})
                                </label>
                                <textarea
                                  rows={6}
                                  value={customPositive}
                                  onChange={(e) => setCustomPositive(e.target.value)}
                                  placeholder={'The behaviour you want to isolate.\nOne prompt per line.'}
                                  className={INPUT}
                                />
                              </div>
                              <div>
                                <label className="text-xs font-medium">
                                  {overRefusal ? 'Benign prompts answered' : 'Negative class'} ({customNegativeLines.length})
                                </label>
                                <textarea
                                  rows={6}
                                  value={customNegative}
                                  onChange={(e) => setCustomNegative(e.target.value)}
                                  placeholder={'Matched in form, differing only in the\nbehaviour above.'}
                                  className={INPUT}
                                />
                              </div>
                            </div>
                            {customTooFew && (
                              <p className="text-xs text-destructive">
                                At least {objectives?.min_per_class ?? 8} per class are needed;
                                the split is rejected below that.
                              </p>
                            )}
                          </div>
                        )}

                        {narrowed && objectives && (
                          <div className="mt-3 rounded-md border p-3">
                            <p className="mb-2 text-xs text-muted-foreground">
                              Pick the scenarios to separate. Counts are live from the prompt
                              library; below {objectives.min_per_class} usable prompts per class
                              the split is rejected.
                            </p>
                            <div className="grid gap-1 sm:grid-cols-2">
                              {objectives.scenarios.map((s) => {
                                const picked = (scenarios.length
                                  ? scenarios
                                  : objectives.default_scenarios
                                ).includes(s.name)
                                return (
                                  <label
                                    key={s.name}
                                    className="flex items-start gap-2 text-xs"
                                    title={s.excluded_reason ?? undefined}
                                  >
                                    <input
                                      type="checkbox"
                                      checked={picked}
                                      onChange={() => {
                                        const base = scenarios.length
                                          ? scenarios
                                          : objectives.default_scenarios
                                        setScenarios(
                                          picked
                                            ? base.filter((n) => n !== s.name)
                                            : [...base, s.name],
                                        )
                                      }}
                                    />
                                    <span className={s.excluded_reason ? 'text-muted-foreground' : ''}>
                                      {s.name}{' '}
                                      <span className="font-mono text-[10px]">({s.count})</span>
                                      {s.excluded_reason && (
                                        <em className="ml-1 text-[10px]">answered, not refused</em>
                                      )}
                                    </span>
                                  </label>
                                )
                              })}
                            </div>
                            <p
                              className={`mt-2 text-xs ${
                                tooFewPrompts ? 'text-destructive' : 'text-muted-foreground'
                              }`}
                            >
                              {selectedCount} prompts selected
                              {tooFewPrompts ? ' — too few; the split will be rejected.' : ''}
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {/* 2. Which method */}
                    <div>
                      <label className="text-xs font-medium">Method — how to change it</label>
                      <select
                        value={method}
                        onChange={(e) => setMethod(e.target.value)}
                        className={INPUT}
                      >
                        <optgroup label="Reversible (inference-time)">
                          {reversible.map((m) => (
                            <option key={m.name} value={m.name}>
                              {m.label}
                            </option>
                          ))}
                        </optgroup>
                        <optgroup label="Permanent (writes weights)">
                          {permanent.map((m) => (
                            <option key={m.name} value={m.name}>
                              {m.label}
                            </option>
                          ))}
                        </optgroup>
                        <optgroup label="Not implemented here">
                          {unavailable.map((m) => (
                            <option key={m.name} value={m.name} disabled>
                              {m.label} — {m.unavailable_reason}
                            </option>
                          ))}
                        </optgroup>
                      </select>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {stages?.methods.find((m) => m.name === method)?.description}
                      </p>
                    </div>

                    {/* 3. Toward what end */}
                    {directionEdit && (
                      <div>
                        <label className="text-xs font-medium">Edit shape</label>
                        <div className="mt-1 grid gap-2 sm:grid-cols-2">
                          <button
                            type="button"
                            onClick={() => setUseSubspace(false)}
                            className={`rounded-md border p-3 text-left text-sm ${
                              !useSubspace ? 'border-primary bg-primary/5' : 'hover:bg-muted/50'
                            }`}
                          >
                            <span className="font-medium">Single direction (β)</span>
                            <p className="mt-1 text-[11px] text-muted-foreground">
                              Scale one direction&apos;s component. What the pipeline has always done.
                            </p>
                          </button>
                          <button
                            type="button"
                            onClick={() => setUseSubspace(true)}
                            className={`rounded-md border p-3 text-left text-sm ${
                              useSubspace ? 'border-primary bg-primary/5' : 'hover:bg-muted/50'
                            }`}
                          >
                            <span className="font-medium">Subspace (k)</span>
                            <p className="mt-1 text-[11px] text-muted-foreground">
                              Remove the whole basis at strength k. Needs a direction derived at
                              rank &gt; 1; rank 1 with k=1 is identical to β=0.
                            </p>
                          </button>
                        </div>
                      </div>
                    )}

                    {directionEdit && useSubspace && (
                      <div>
                        <label className="text-xs font-medium">Removal strength (k)</label>
                        <input
                          type="number"
                          step="0.05"
                          value={kStrength}
                          onChange={(e) => setKStrength(parseFloat(e.target.value) || 1)}
                          className={INPUT}
                        />
                        <p className="mt-1 text-xs text-muted-foreground">
                          1.0 removes the subspace exactly. Above 1 over-projects, which buys
                          compliance with capability — run the search first if you have not.
                        </p>
                      </div>
                    )}

                    {directionEdit && !useSubspace && (
                      <div>
                        <label className="text-xs font-medium">Intent</label>
                        <div className="mt-1 grid gap-2 sm:grid-cols-3">
                          {stages?.beta_presets.map((p) => (
                            <button
                              key={p.value}
                              type="button"
                              onClick={() => setBeta(p.value)}
                              className={[
                                'rounded-md border p-3 text-left text-sm',
                                beta === p.value ? 'border-primary bg-primary/5' : 'hover:bg-muted/50',
                              ].join(' ')}
                            >
                              <span className="font-medium">{p.label}</span>
                              <span className="ml-1 font-mono text-xs text-muted-foreground">
                                β={p.value}
                              </span>
                              <p className="mt-1 text-[11px] text-muted-foreground">
                                {p.description}
                              </p>
                            </button>
                          ))}
                        </div>
                        <label className="mt-2 block text-xs text-muted-foreground">
                          Custom β
                          <input
                            type="number"
                            step="0.1"
                            value={beta}
                            onChange={(e) => setBeta(parseFloat(e.target.value))}
                            className={INPUT}
                          />
                        </label>
                      </div>
                    )}

                    {/* 4. Inputs */}
                    <div>
                      <ModelPicker id="weight-source-model" label={kind === 'compare' ? 'Original model (baseline)' : (kind === 'embed_align' || kind === 'embed_extract' || kind === 'embed_recon') ? 'Model (local)' : 'Source model'} value={sourceModel} onChange={setSourceModel} />
                    </div>

                    {isNewKind && <NewKindForms kind={kind} value={nk} onChange={patchNk} />}

                    {(needs('source_run_id') || (kind === 'expert_surgery' && method === 'expert_direction_scale')) && (
                      <div>
                        <label className="text-xs font-medium">Direction to use</label>
                        <div className="mt-1">
                          <DirectionPicker
                            directions={directions}
                            value={directionId}
                            onChange={setDirectionId}
                            minUsableAuc={stages?.min_usable_auc}
                          />
                        </div>
                      </div>
                    )}

                    {kind === 'routing' && (
                      <div>
                        <label className="text-xs font-medium">Prompts per class</label>
                        <input
                          type="number"
                          min={8}
                          value={nPerClass}
                          onChange={(e) => setNPerClass(parseInt(e.target.value) || 8)}
                          className={INPUT}
                        />
                        <p className="mt-1 text-xs text-muted-foreground">
                          Harmful and harmless prompts to route, drawn from the objective&apos;s corpus. Each is
                          run once; 64 per class is enough to rank experts on a 3B model in a few minutes.
                        </p>
                      </div>
                    )}

                    {kind === 'direction' && (
                      <div className="grid gap-4 sm:grid-cols-2">
                        <div>
                          <label className="text-xs font-medium">Prompts per class</label>
                          <input
                            type="number"
                            value={nPerClass}
                            onChange={(e) => setNPerClass(parseInt(e.target.value) || 8)}
                            className={INPUT}
                          />
                        </div>
                        <div>
                          <label className="text-xs font-medium">
                            {method === 'rfm_agop' ? 'Cone rank (directions)' : 'Subspace rank'}
                          </label>
                          <input
                            type="number"
                            min={1}
                            value={subspaceRank}
                            onChange={(e) => setSubspaceRank(parseInt(e.target.value) || 1)}
                            className={INPUT}
                          />
                          <p className="mt-1 text-xs text-muted-foreground">
                            {method === 'rfm_agop'
                              ? 'Top eigenvectors of the RFM feature matrix, with eigenvalue weights for soft ablation. Larger models need three or more; the rank curve sweep tells you how many.'
                              : '1 derives a single difference-in-means vector. Above 1 builds an orthonormal subspace, which catches refusal components one vector misses — and is what the capability-gated search needs.'}
                          </p>
                        </div>
                        {method === 'rfm_agop' && (
                          <div>
                            <label className="text-xs font-medium">RFM iterations</label>
                            <input
                              type="number"
                              min={1}
                              value={rfmIterations}
                              onChange={(e) => setRfmIterations(parseInt(e.target.value) || 1)}
                              className={INPUT}
                            />
                            <p className="mt-1 text-xs text-muted-foreground">
                              Gradient-outer-product refinements of the kernel. Five is the paper&apos;s default; the
                              run is seconds either way on the captured residuals.
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {kind === 'probe' && (
                      <div className="space-y-4 rounded-lg border p-4">
                        <h3 className="font-medium">Train and validate a harmful-intent probe</h3>
                        <p className="text-xs text-muted-foreground">Fit a classifier on raw activations, then test it on separate prompts and unseen jailbreak families. This measures detection, not a causal intervention. Missing requested families stop the run before loading the model.</p>
                        <div className="grid gap-3 sm:grid-cols-2">
                          <label className="text-xs font-medium">Token pooling<select value={pooling} onChange={(e) => setPooling(e.target.value as typeof pooling)} className={INPUT}><option value="mean">Mean over prompt tokens</option><option value="last">Last token</option><option value="max">Maximum over tokens</option><option value="last_k">Last eight tokens</option></select></label>
                          <label className="text-xs font-medium">Direct harmful prompts<input type="number" min={0} value={nDirect} onChange={(e) => setNDirect(Number(e.target.value))} className={INPUT} /></label>
                          <label className="text-xs font-medium">Jailbreak training pool<input type="number" min={0} value={nJailbreak} onChange={(e) => { const n = Number(e.target.value); setNJailbreak(n); if (!n) setHoldoutTechniques(0) }} className={INPUT} /></label>
                          <label className="text-xs font-medium">Benign prompts<input type="number" min={24} value={nBenign} onChange={(e) => setNBenign(Number(e.target.value))} className={INPUT} /></label>
                          <label className="text-xs font-medium">Entire families held out<input type="number" min={0} disabled={!nJailbreak} value={holdoutTechniques} onChange={(e) => setHoldoutTechniques(Number(e.target.value))} className={INPUT} /></label>
                        </div>
                        <p className="text-xs text-muted-foreground">Use the imported adversarial prompt library for jailbreak examples. For a direct-only experiment, set jailbreaks and held-out families to zero. The result records requested and actual dataset sizes.</p>
                        <label className="block text-xs font-medium">Optional labelled jailbreak examples (JSON)
                          <textarea rows={5} value={jailbreakExamples} onChange={(e) => setJailbreakExamples(e.target.value)} placeholder={'[{"technique": "family-name", "prompt": "A complete wrapped request"}]'} className={INPUT} />
                        </label>
                        <label className="block text-xs text-muted-foreground">Load a JSON example file<input type="file" accept=".json,application/json" className="mt-1 block" onChange={async (e) => {
                          const file = e.target.files?.[0]; if (file) setJailbreakExamples(await file.text())
                        }} /></label>
                        <p className="text-xs text-muted-foreground">Supplied examples replace the imported jailbreak pool for this experiment and are saved with its settings. Use complete wrapped harmful requests with consistent family labels; at least one family must remain in training. Empty uses the server&apos;s prompt library.</p>
                        <label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={elicitingSuffix} onChange={(e) => setElicitingSuffix(e.target.checked)} /> Ask the model to consider whether the request is harmful before capture</label>
                        <label className="block text-xs font-medium">Custom capture suffix (overrides the checkbox)<textarea rows={2} value={promptSuffix} onChange={(e) => setPromptSuffix(e.target.value)} className={INPUT} /></label>
                      </div>
                    )}

                    {kind === 'sweep' && (
                      <div className="grid gap-4 sm:grid-cols-2">
                        <label className="flex items-start gap-2 text-xs">
                          <input type="checkbox" checked={thinking} onChange={(e) => setThinking(e.target.checked)} className="mt-0.5" />
                          <span>
                            <span className="font-medium">Keep the reasoning block on</span>
                            <span className="block text-muted-foreground">
                              For Qwen3-style models. Steering only reverses refusal reliably when the chain of thought
                              regenerates under it.
                            </span>
                          </span>
                        </label>
                        <div>
                          <label className="text-xs font-medium">Refusal-decision timelines</label>
                          <input
                            type="number"
                            min={0}
                            value={timelinePrompts}
                            onChange={(e) => setTimelinePrompts(Math.max(0, parseInt(e.target.value) || 0))}
                            className={INPUT}
                          />
                          <p className="mt-1 text-xs text-muted-foreground">
                            Prompts to trace token by token: where in the generation the model commits to refusing.
                          </p>
                        </div>
                      </div>
                    )}

                    {kind === 'compare' && (
                      <div className="grid gap-3 sm:grid-cols-2">
                        <label className="flex items-start gap-2 text-xs">
                          <input type="checkbox" checked={rederive || method === 'compare_rederive'} disabled={method === 'compare_rederive'} onChange={(e) => setRederive(e.target.checked)} className="mt-0.5" />
                          <span>
                            <span className="font-medium">Attack again after the edit</span>
                            <span className="block text-muted-foreground">
                              Re-derive the refusal direction on the modified model and report its AUC, stable rank
                              and ablation refusal. The test for a hardening defence.
                            </span>
                          </span>
                        </label>
                        <label className="flex items-start gap-2 text-xs">
                          <input type="checkbox" checked={misalignmentControl} onChange={(e) => setMisalignmentControl(e.target.checked)} className="mt-0.5" />
                          <span>
                            <span className="font-medium">Broad-misalignment control</span>
                            <span className="block text-muted-foreground">
                              Open-ended probes with a heuristic judge. Steering and edits can turn answers hostile
                              while trivia still scores fine.
                            </span>
                          </span>
                        </label>
                      </div>
                    )}

                    {(kind === 'sweep' || kind === 'compare') && (
                      <div>
                        <label className="text-xs font-medium">Held-out prompts per class</label>
                        <input
                          type="number"
                          value={nPrompts}
                          onChange={(e) => setNPrompts(parseInt(e.target.value) || 1)}
                          className={INPUT}
                        />
                      </div>
                    )}

                    {kind === 'select' && (
                      <div className="space-y-4">
                        <div className="grid gap-4 sm:grid-cols-3">
                          <div>
                            <label className="text-xs font-medium">Ranks to try</label>
                            <input
                              value={ranksText}
                              onChange={(e) => setRanksText(e.target.value)}
                              className={INPUT}
                            />
                          </div>
                          <div>
                            <label className="text-xs font-medium">Strengths (k)</label>
                            <input
                              value={ksText}
                              onChange={(e) => setKsText(e.target.value)}
                              className={INPUT}
                            />
                          </div>
                          <div>
                            <label className="text-xs font-medium">Prompts per config</label>
                            <input
                              type="number"
                              value={nPrompts}
                              onChange={(e) => setNPrompts(parseInt(e.target.value) || 1)}
                              className={INPUT}
                            />
                          </div>
                        </div>

                        <div>
                          <label className="text-xs font-medium">
                            Capability floor — allowed drop in factual accuracy
                          </label>
                          <input
                            type="number"
                            step="0.01"
                            value={factualFloor}
                            onChange={(e) => setFactualFloor(parseFloat(e.target.value) || 0)}
                            className={INPUT}
                          />
                          <p className="mt-1 text-xs text-muted-foreground">
                            A config that refuses nothing because it has been broken scores the
                            same as one cleanly ablated. Anything costing more capability than
                            this is rejected however compliant it looks.
                          </p>
                        </div>

                        {/* Say the cost before it is paid: this previews every
                            candidate by generating, and nothing is written. */}
                        <div className="rounded-md border bg-muted/40 p-3 text-xs text-muted-foreground">
                          Search space: {parseNums(ranksText).length} ranks ×{' '}
                          {parseNums(ksText).length} strengths ={' '}
                          {parseNums(ranksText).length * parseNums(ksText).length} configs, each
                          scored on {nPrompts} harmful prompts plus the 12-question capability
                          control ={' '}
                          <span className="font-mono">
                            {parseNums(ranksText).length *
                              parseNums(ksText).length *
                              (nPrompts + 12)}
                          </span>{' '}
                          generations. Every candidate is applied to the real weights in memory and
                          restored; nothing is written to disk.
                        </div>
                      </div>
                    )}

                    {kind === 'autotune' && (
                      <div className="space-y-4">
                        <div className="grid gap-4 sm:grid-cols-3">
                          <div>
                            <label className="text-xs font-medium">Ranks to try</label>
                            <input value={ranksText} onChange={(e) => setRanksText(e.target.value)} className={INPUT} />
                          </div>
                          <div>
                            <label className="text-xs font-medium">Strengths (k)</label>
                            <input value={ksText} onChange={(e) => setKsText(e.target.value)} className={INPUT} />
                          </div>
                          <div>
                            <label className="text-xs font-medium">Prompts per candidate</label>
                            <input type="number" value={nPrompts} onChange={(e) => setNPrompts(parseInt(e.target.value) || 1)} className={INPUT} />
                          </div>
                        </div>

                        <div className="grid gap-4 sm:grid-cols-3">
                          <div>
                            <label className="text-xs font-medium">Embedding table</label>
                            <select value={embeddingsMode} onChange={(e) => setEmbeddingsMode(e.target.value as EmbeddingsMode)} className={INPUT}>
                              <option value="auto">auto — untouched first when lm_head is tied</option>
                              <option value="no">leave untouched</option>
                              <option value="yes">edit it too</option>
                              <option value="both">try both</option>
                            </select>
                            <p className="mt-1 text-xs text-muted-foreground">
                              On Qwen2.5 0.5B–3B the unembedding shares the embedding table, so editing it also
                              rewrites the output layer — the usual way an ablated model ends up answering in Chinese.
                            </p>
                          </div>
                          <div>
                            <label className="text-xs font-medium">Candidate budget</label>
                            <input type="number" min={1} value={maxCandidates} onChange={(e) => setMaxCandidates(Math.max(1, parseInt(e.target.value) || 1))} className={INPUT} />
                          </div>
                          <div>
                            <label className="text-xs font-medium">Refusal target</label>
                            <input type="number" step="0.01" min={0} max={1} value={maxRefusal} onChange={(e) => setMaxRefusal(parseFloat(e.target.value) || 0)} className={INPUT} />
                            <p className="mt-1 text-xs text-muted-foreground">Reported as met or not; never a gate.</p>
                          </div>
                        </div>

                        <div className="grid gap-4 sm:grid-cols-3">
                          <div>
                            <label className="text-xs font-medium">Capability floor — allowed drop in factual accuracy</label>
                            <input type="number" step="0.01" value={factualFloor} onChange={(e) => setFactualFloor(parseFloat(e.target.value) || 0)} className={INPUT} />
                          </div>
                          <div>
                            <label className="text-xs font-medium">Language drift allowed</label>
                            <input type="number" step="0.01" min={0} max={1} value={languageDriftMax} onChange={(e) => setLanguageDriftMax(parseFloat(e.target.value) || 0)} className={INPUT} />
                            <p className="mt-1 text-xs text-muted-foreground">Share of answers that may be mostly outside the Latin script.</p>
                          </div>
                          <div>
                            <label className="text-xs font-medium">Sampling seed</label>
                            <input type="number" value={samplingSeed} onChange={(e) => setSamplingSeed(parseInt(e.target.value) || 0)} className={INPUT} />
                          </div>
                        </div>

                        <div className="grid gap-2 sm:grid-cols-2">
                          <label className="flex items-start gap-2 text-sm">
                            <input type="checkbox" checked={!stopAtFirst} onChange={(e) => setStopAtFirst(!e.target.checked)} className="mt-1" />
                            <span>
                              Score every candidate, then pick
                              <span className="block text-xs text-muted-foreground">
                                Off: stop at the first candidate that clears every gate and meets the refusal target.
                              </span>
                            </span>
                          </label>
                          <label className="flex items-start gap-2 text-sm">
                            <input type="checkbox" checked={verifySampled} onChange={(e) => setVerifySampled(e.target.checked)} className="mt-1" />
                            <span>
                              Re-check under the serving sampling
                              <span className="block text-xs text-muted-foreground">
                                The winner and the written checkpoint are scored again at the provider&apos;s
                                temperature and top-p, seeded. Greedy alone hides language drift.
                              </span>
                            </span>
                          </label>
                        </div>

                        <div className="rounded-md border bg-muted/40 p-3 text-xs text-muted-foreground">
                          Loop: apply a candidate to the real weights, score it on {nPrompts} held-out prompts plus
                          the capability control, restore, next — least destructive first over{' '}
                          {parseNums(ranksText).length} ranks × {parseNums(ksText).length} strengths ×{' '}
                          {embeddingsMode === 'both' || embeddingsMode === 'auto' ? 2 : 1} embedding choice
                          {embeddingsMode === 'both' || embeddingsMode === 'auto' ? 's' : ''} (
                          {Math.min(maxCandidates, parseNums(ranksText).length * parseNums(ksText).length * (embeddingsMode === 'both' || embeddingsMode === 'auto' ? 2 : 1))}{' '}
                          within the budget). The best admissible edit is written, reloaded from disk and gated
                          again before it appears in Models. A run that finds nothing writes nothing and keeps its trial log.
                        </div>
                      </div>
                    )}

                    {kind === 'compare' && (
                      <div className="space-y-3">
                        <ModelPicker id="weight-modified-model" label="Custom model to evaluate" value={modifiedModel} onChange={(ref) => {
                          setModifiedModel(ref)
                          const original = catalog?.models.find((model) => model.model_ref === ref)?.source_model
                          if (original) setSourceModel(original)
                        }} />
                        <p className="mt-1 text-xs text-muted-foreground">
                          Both models use the same prompts, CoT setting, and decoding options. Changes in tokenizers and templates may also affect their answers.
                        </p>
                        <label className="flex items-center gap-2 text-sm"><input type="checkbox" checked={enableCot} onChange={(e) => setEnableCot(e.target.checked)} />CoT (ReACT) for both models</label>
                        <p className="text-xs text-muted-foreground">ReACT adds reasoning instructions to each question; only the final answer is scored. Increase the token budget to allow room for reasoning.</p>
                        <div className="grid gap-3 sm:grid-cols-2">
                          <label className="text-sm">Temperature<input type="number" min={0} max={2} step={0.1} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} className={INPUT} /></label>
                          <label className="text-sm">Top-p<input type="number" min={0.01} max={1} step={0.05} value={topP} onChange={(e) => setTopP(Number(e.target.value))} className={INPUT} /></label>
                        </div>
                        <label className="block text-sm">System prompt for both models<textarea rows={3} maxLength={32000} disabled={promptLoading} value={systemPrompt} onChange={(e) => { setSystemPrompt(e.target.value); setPromptError('') }} className={INPUT} /></label>
                        {promptLoading && <p role="status" className="text-xs">Loading saved system prompt…</p>}
                        {promptError && <div role="alert" className="text-sm text-destructive">{promptError}<button type="button" className="ml-2 underline" onClick={() => { setPromptError(''); setSystemPrompt('') }}>Use no application system prompt</button></div>}
                      </div>
                    )}

                    {kind === 'expert_surgery' && (
                      <>
                        <ExpertSelectionInput value={expertSelectionText} onChange={setExpertSelectionText} className={INPUT} />
                        {method === 'expert_ablate' && (
                          <div>
                            <label className="text-xs font-medium">Expert scale</label>
                            <input
                              type="number"
                              step="0.1"
                              value={expertScale}
                              onChange={(e) => setExpertScale(parseFloat(e.target.value))}
                              className={INPUT}
                            />
                            <p className="mt-1 text-xs text-muted-foreground">
                              Multiplies each chosen expert&apos;s down-projection. 0 removes its write entirely,
                              1 is a bit-identical control, 2 doubles it. Routing is left exactly as it was, so
                              a token still pays the router&apos;s weight for a zeroed expert.
                            </p>
                          </div>
                        )}
                        <label className="flex items-center gap-2 text-xs">
                          <input type="checkbox" checked={includeShared} onChange={(e) => setIncludeShared(e.target.checked)} />
                          Also edit the shared expert of each chosen layer (it fires on every token)
                        </label>
                      </>
                    )}

                    {kind === 'distill' && (
                      <div>
                        <ModelPicker id="weight-teacher-model" label="Teacher model (local open weights)" value={teacherModel} onChange={setTeacherModel} />
                        <p className="mt-1 text-xs text-muted-foreground">
                          Must be loadable here: only a local model&apos;s output is trainable on, so Ollama
                          names and hosted APIs are refused. Response level unloads it before training;
                          logit level keeps both models resident and needs identical tokenizers.
                        </p>
                      </div>
                    )}

                    {training && (
                      <div>
                        <label className="text-xs font-medium">Training rows</label>
                        <div className="mt-1 flex gap-1 rounded-md border p-0.5 text-xs">
                          {([
                            ['rows', 'Paste rows'],
                            ['benchmark', 'From a benchmark'],
                            ...(kind === 'distill' ? [['objective', `${objective.replace(/_/g, ' ')} corpus`]] : []),
                          ] as Array<[typeof datasetMode, string]>).map(([mode, label]) => (
                            <button
                              key={mode}
                              type="button"
                              onClick={() => setDatasetMode(mode)}
                              className={`rounded px-2 py-1 ${datasetMode === mode ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'}`}
                            >
                              {label}
                            </button>
                          ))}
                        </div>
                        {datasetMode === 'rows' && (
                          <>
                            <textarea
                              rows={6}
                              value={datasetText}
                              onChange={(e) => setDatasetText(e.target.value)}
                              spellCheck={false}
                              placeholder={kind === 'lora'
                                ? '{"prompt": "How do I ...", "response": "First, ..."}\n{"prompt": "...", "response": "..."}'
                                : '{"prompt": "How do I ..."}\n{"prompt": "..."}'}
                              className={`${INPUT} font-mono`}
                            />
                            <p className="mt-1 text-xs text-muted-foreground">
                              One JSON object per line, or a JSON array. {kind === 'lora'
                                ? 'Each row needs a prompt and a response; an optional system field sets the system prompt.'
                                : 'Each row needs a prompt; the teacher writes the response. Rows that already carry one are used as they are at logit level.'}
                              {datasetError ? <span className="text-destructive"> {datasetError}</span> : datasetRows.length ? ` ${datasetRows.length} row${datasetRows.length === 1 ? '' : 's'} parsed.` : ''}
                              {kind === 'lora' && datasetRows.length > 0 && !datasetRows.every((r) => r.response?.trim()) && (
                                <span className="text-destructive"> Every LoRA row needs a response.</span>
                              )}
                            </p>
                          </>
                        )}
                        {datasetMode === 'benchmark' && (
                          <div className="mt-2 grid gap-3 sm:grid-cols-2">
                            <label className="text-xs font-medium">
                              Benchmark
                              <input value={benchmarkName} onChange={(e) => setBenchmarkName(e.target.value)} placeholder="as named on the Benchmarks page" className={INPUT} />
                            </label>
                            <label className="text-xs font-medium">
                              Rows to draw
                              <input type="number" min={8} value={benchmarkCount} onChange={(e) => setBenchmarkCount(parseInt(e.target.value) || 8)} className={INPUT} />
                            </label>
                            <p className="text-xs text-muted-foreground sm:col-span-2">
                              Multiple-choice items become the question with lettered choices as the prompt and the
                              answer letter as the response. Training on a benchmark and then scoring it is not an
                              evaluation; keep a held-out one for that.
                            </p>
                          </div>
                        )}
                        {datasetMode === 'objective' && (
                          <div className="mt-2">
                            <label className="text-xs font-medium">Prompts per class</label>
                            <input type="number" min={8} value={nPerClass} onChange={(e) => setNPerClass(parseInt(e.target.value) || 8)} className={INPUT} />
                            <p className="mt-1 text-xs text-muted-foreground">
                              Both classes of the {objective.replace(/_/g, ' ')} corpus. The teacher refuses the harmful
                              prompts and answers the harmless ones, and the student learns both: the way to harden a
                              small model on a larger one&apos;s refusals.
                            </p>
                          </div>
                        )}
                      </div>
                    )}

                    {training && (
                      <details className="rounded-lg border p-4" open>
                        <summary className="cursor-pointer text-sm font-medium">Training settings</summary>
                        <div className="mt-3 grid gap-3 sm:grid-cols-3">
                          <label className="text-xs font-medium">LoRA rank<input type="number" min={1} value={loraRank} onChange={(e) => setLoraRank(parseInt(e.target.value) || 1)} className={INPUT} /></label>
                          <label className="text-xs font-medium">LoRA alpha<input type="number" min={1} value={loraAlpha} onChange={(e) => setLoraAlpha(parseInt(e.target.value) || 1)} className={INPUT} /></label>
                          <label className="text-xs font-medium">Adapt<select value={loraTargets} onChange={(e) => setLoraTargets(e.target.value)} className={INPUT}><option value="attention">attention projections</option><option value="attention+mlp">attention + dense MLP</option></select></label>
                          <label className="text-xs font-medium">Epochs<input type="number" min={1} value={epochs} onChange={(e) => setEpochs(parseInt(e.target.value) || 1)} className={INPUT} /></label>
                          <label className="text-xs font-medium">Max steps (optional)<input type="number" min={1} value={maxSteps} onChange={(e) => setMaxSteps(e.target.value === '' ? '' : parseInt(e.target.value) || 1)} className={INPUT} /></label>
                          <label className="text-xs font-medium">Learning rate<input type="number" step="0.00005" min={0} value={lr} onChange={(e) => setLr(parseFloat(e.target.value) || 0.0002)} className={INPUT} /></label>
                          <label className="text-xs font-medium">Gradient accumulation<input type="number" min={1} value={gradAccum} onChange={(e) => setGradAccum(parseInt(e.target.value) || 1)} className={INPUT} /></label>
                          <label className="text-xs font-medium">Row token limit<input type="number" min={16} value={maxLength} onChange={(e) => setMaxLength(Number(e.target.value))} className={INPUT} /></label>
                          {kind === 'distill' && (
                            <label className="text-xs font-medium">Teacher max new tokens<input type="number" min={1} value={teacherMaxNewTokens} onChange={(e) => setTeacherMaxNewTokens(parseInt(e.target.value) || 1)} className={INPUT} /></label>
                          )}
                          {kind === 'distill' && method === 'logit_distill' && (
                            <>
                              <label className="text-xs font-medium">Temperature<input type="number" step="0.5" min={0.1} value={distillTemperature} onChange={(e) => setDistillTemperature(parseFloat(e.target.value) || 1)} className={INPUT} /></label>
                              <label className="text-xs font-medium">Cross-entropy weight<input type="number" step="0.1" min={0} max={1} value={ceWeight} onChange={(e) => setCeWeight(parseFloat(e.target.value))} className={INPUT} /></label>
                            </>
                          )}
                        </div>
                        <label className="mt-3 flex items-center gap-2 text-xs">
                          <input type="checkbox" checked={merge} onChange={(e) => setMerge(e.target.checked)} />
                          Merge the adapter into a new model directory (off: keep only the adapter under runs/)
                        </label>
                        <p className="mt-2 text-xs text-muted-foreground">
                          Plain-precision LoRA: the base stays in its loaded dtype, the adapter is fp32. No QLoRA on
                          Apple silicon, so this is sized for models up to about 3B. The loss is masked to the response.
                        </p>
                      </details>
                    )}

                    {needsOutputName && (
                      <div>
                        <label className="text-xs font-medium">Output name</label>
                        <input
                          value={outputName}
                          onChange={(e) => setOutputName(e.target.value)}
                          placeholder="qwen05b-ablated"
                          className={INPUT}
                        />
                        <p className="mt-1 text-xs text-muted-foreground">
                          Written to <code>{stages?.models_root}/&lt;name&gt;</code>. Letters,
                          digits, dot, underscore and hyphen only — the destination is always
                          built under that directory.
                        </p>
                      </div>
                    )}

                    <details className="rounded-lg border p-4">
                      <summary className="cursor-pointer text-sm font-medium">Runtime and reproducibility</summary>
                      <div className="mt-3 grid gap-3 sm:grid-cols-3">
                        <label className="text-xs font-medium">Weight precision<select value={dtype} onChange={(e) => setDtype(e.target.value)} className={INPUT}><option value="bfloat16">BF16</option><option value="float16">FP16</option><option value="float32">FP32</option></select></label>
                        <label className="text-xs font-medium">Random seed<input type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} className={INPUT} /></label>
                        {['direction', 'probe', 'compare', 'routing'].includes(kind) && <label className="text-xs font-medium">Capture token limit<input type="number" min={1} value={maxLength} onChange={(e) => setMaxLength(Number(e.target.value))} className={INPUT} /></label>}
                        {['direction', 'probe', 'compare'].includes(kind) && <label className="text-xs font-medium">Capture batch size<input type="number" min={1} value={batchSize} onChange={(e) => setBatchSize(Number(e.target.value))} className={INPUT} /></label>}
                        {['sweep', 'select', 'compare'].includes(kind) && <label className="text-xs font-medium">Maximum generated tokens<input type="number" min={1} value={maxNewTokens} onChange={(e) => setMaxNewTokens(Number(e.target.value))} className={INPUT} /></label>}
                      </div>
                      <p className="mt-2 text-xs text-muted-foreground">Downstream checks reuse the direction&apos;s saved prompt partition. Asking for more held-out prompts never reuses its training examples.</p>
                    </details>

                    {preflight && (
                      <PreflightPanel
                        checks={preflight.checks}
                        acknowledged={acknowledged}
                        onAcknowledge={setAcknowledged}
                      />
                    )}

                    {needsOutputName && (
                      <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm">
                        <p className="font-medium">Save a new custom model</p>
                        <p className="mt-1 text-muted-foreground">The original checkpoint is preserved. The new copy will appear in Models when saving finishes.</p>
                        <p className="mt-1 text-muted-foreground">
                          {sourceModel || 'the source model'} → {stages?.models_root}/
                          {outputName || '<name>'}{' '}
                          {kind === 'expert_surgery'
                            ? method === 'expert_ablate'
                              ? `scaling the chosen experts by ${expertScale}`
                              : useSubspace
                                ? `removing the subspace at k=${kStrength} inside the chosen experts`
                                : `at β=${beta} inside the chosen experts`
                            : kind === 'surgery'
                              ? useSubspace ? `removing the subspace at k=${kStrength}` : `at β=${beta}`
                              : kind === 'autotune'
                                ? 'with the best edit the search finds, once it passes verification from disk'
                                : 'after training'}.
                        </p>
                        {kind === 'expert_surgery' && (
                          <p className="mt-1 text-muted-foreground">
                            Partial by design: the manifest will say <code>coverage_verified: false</code>, and
                            the experts you did not name still write whatever they carry.
                          </p>
                        )}
                      </div>
                    )}

                    <button
                      type="button"
                      onClick={launch}
                      disabled={
                        createRun.isPending ||
                        tooFewPrompts ||
                        customTooFew ||
                        (kind === 'compare' && (!modifiedModel.trim() || !!promptError || promptLoading)) ||
                        (kind === 'expert_surgery' && !expertSelectionReady) ||
                        (needsOutputName && !outputName.trim()) ||
                        !datasetReady ||
                        (kind === 'distill' && !teacherModel.trim()) ||
                        !!preflight?.checks.some((check) => check.severity === 'blocking' && (!check.acknowledgeable || !acknowledged.includes(check.code))) ||
                        !sourceModel.trim()
                      }
                      className="rounded-lg bg-primary px-4 py-2 text-sm font-medium text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
                    >
                      {createRun.isPending ? 'Starting…' : `Start ${stageSpec?.label.toLowerCase()}`}
                    </button>
                  </div>
                </div>
              </div>
            )}
          </Tabs.Content>

          {/* --------------------------------------------------------- Runs */}
          <Tabs.Content value="runs" className="pt-6">
            <div className="overflow-hidden rounded-lg border bg-card shadow-sm">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-left text-xs text-muted-foreground">
                    <th className="px-3 py-2 font-medium">#</th>
                    <th className="px-3 py-2 font-medium">stage</th>
                    <th className="px-3 py-2 font-medium">model</th>
                    <th className="px-3 py-2 font-medium">result</th>
                    <th className="px-3 py-2 font-medium">status</th>
                    <th className="px-3 py-2 font-medium">created</th>
                    <th className="px-3 py-2 font-medium">took</th>
                    <th className="px-3 py-2" />
                  </tr>
                </thead>
                <tbody>
                  {runs.length === 0 && (
                    <tr>
                      <td colSpan={8} className="px-3 py-6 text-center text-muted-foreground">
                        No runs yet.
                      </td>
                    </tr>
                  )}
                  {runs.map((r) => {
                    const s = r.metadata?.summary ?? {}
                    let result = '—'
                    if (r.kind === 'direction' && s.auc != null) {
                      result = `AUC ${Number(s.auc).toFixed(3)} @ L${s.layer}`
                      if (s.rank > 1) result += ` · rank ${s.rank}`
                    } else if (r.kind === 'sweep') {
                      result = s.verdict ?? '—'
                    } else if (r.kind === 'select') {
                      result = s.best
                        ? `rank ${s.best.rank}, k ${Number(s.best.k).toFixed(2)}`
                        : s.frontier
                          ? 'none admissible'
                          : '—'
                    } else if (r.kind === 'autotune') {
                      const w = s.winner
                      result = w
                        ? `rank ${w.rank}, k ${Number(w.k).toFixed(2)}, emb ${w.include_embeddings ? 'yes' : 'no'} · ${
                            s.verification?.passed ? 'verified' : s.verification ? 'FAILED verification' : 'unverified'
                          }`
                        : s.trials
                          ? `none admissible (${s.trials.length} tried)`
                          : '—'
                    } else if (r.kind === 'compare' && s.deltas) {
                      const d = s.deltas
                      result = `refusal ${(d.refuse_harmful * 100).toFixed(0)} / capability ${(
                        d.factual_acc * 100
                      ).toFixed(0)} pts`
                    } else if (r.kind === 'surgery') {
                      result = s.manifest?.beta != null ? `β=${s.manifest.beta}` : '—'
                      if (s.manifest?.extra?.k != null) result = `k=${s.manifest.extra.k}`
                    } else if (r.kind === 'expert_surgery') {
                      const x = s.manifest?.extra ?? {}
                      result = x.experts_edited != null
                        ? `${x.expert_mode === 'ablate' ? `×${x.expert_scale ?? 0}` : x.expert_mode ?? 'edit'} in ${x.experts_edited} expert${x.experts_edited === 1 ? '' : 's'}`
                        : '—'
                    } else if (r.kind === 'routing') {
                      const top = s.ranking?.[0]
                      result = top ? `L${top.layer} E${top.expert} Δ${(top.delta * 100).toFixed(0)} pts` : '—'
                    } else if (r.kind === 'lora' || r.kind === 'distill') {
                      const t = s.train ?? {}
                      result = t.steps != null
                        ? `${t.steps} steps · loss ${Number(t.final_loss).toFixed(3)}${t.eval_loss_after != null ? ` · eval ${Number(t.eval_loss_after).toFixed(3)}` : ''}`
                        : '—'
                    }
                    return (
                      <tr
                        key={r.id}
                        onClick={() => router.push(`/weights/${r.id}`)}
                        className="cursor-pointer border-b last:border-0 hover:bg-muted/50"
                      >
                        <td className="px-3 py-2 font-mono">{r.id}</td>
                        <td className="px-3 py-2">{r.kind}</td>
                        <td className="max-w-[16rem] truncate px-3 py-2 font-mono text-xs">
                          {r.source_model}
                        </td>
                        <td className="px-3 py-2 font-mono text-xs">{result}</td>
                        <td className="px-3 py-2">
                          <StatusBadge status={r.status} />
                        </td>
                        <td className="px-3 py-2 text-xs text-muted-foreground">
                          {formatDateTime(r.created_at)}
                        </td>
                        <td className="px-3 py-2 font-mono text-xs text-muted-foreground">
                          {duration(r.elapsed_seconds)}
                        </td>
                        <td className="px-3 py-2">
                          <button
                            type="button"
                            onClick={async (e) => {
                              e.stopPropagation()
                              if (!confirm(`Delete run #${r.id}? Artifacts on disk are kept.`)) return
                              await deleteRun.mutateAsync({ id: r.id })
                              toast.success(`Deleted run #${r.id}`)
                            }}
                            className="text-muted-foreground hover:text-destructive"
                          >
                            <Trash2 className="h-4 w-4" />
                          </button>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </Tabs.Content>

          {/* --------------------------------------------------- Directions */}
          <Tabs.Content value="directions" className="pt-6">
            <DirectionPicker
              directions={directions}
              value={directionId}
              onChange={(id) => id && router.push(`/weights/${id}`)}
              minUsableAuc={stages?.min_usable_auc}
            />
          </Tabs.Content>

          {/* ------------------------------------------------------- Models */}
          <Tabs.Content value="models" className="pt-6">
            <div className="grid gap-3 md:grid-cols-2">
              {models.length === 0 && (
                <p className="text-sm text-muted-foreground">
                  No edited models yet. They appear here once a surgery run completes, and any
                  model carrying a surgery manifest is listed even if it was produced from the
                  command line.
                </p>
              )}
              {models.map((m) => (
                <div
                  key={m.path}
                  onClick={() => router.push({ pathname: '/compare', query: { search: m.name } })}
                  className="cursor-pointer rounded-lg border bg-card p-4 shadow-sm transition-colors hover:bg-muted/50"
                >
                  <div className="flex items-start justify-between gap-2">
                    <div className="min-w-0">
                      <p className="truncate font-medium">{m.name}</p>
                      <p className="truncate font-mono text-xs text-muted-foreground">{m.path}</p>
                    </div>
                    {m.orphan && (
                      <span className="shrink-0 rounded bg-muted px-1.5 py-0.5 text-[10px]">
                        from the CLI
                      </span>
                    )}
                  </div>
                  <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                    <Field label="β" value={m.manifest?.beta} />
                    <Field label="architecture" value={m.manifest?.architecture} />
                    <Field label="matrices" value={m.manifest?.matrices_edited} />
                    <Field label="mean Δ" value={m.manifest?.mean_relative_change} />
                    <Field label="size" value={gb(m.size_bytes)} />
                    <Field label="tied embeds" value={String(m.manifest?.embeddings_tied ?? '—')} />
                  </dl>
                  <p className="mt-2 truncate text-xs text-muted-foreground">
                    from {m.manifest?.source_model ?? 'unknown'}
                  </p>
                </div>
              ))}
            </div>
          </Tabs.Content>
        </Tabs.Root>
      </div>
      <Glossary open={glossaryOpen} onClose={() => setGlossaryOpen(false)} />
    </Layout>
  )
}

function Field({ label, value }: { label: string; value: any }) {
  return (
    <div className="flex justify-between gap-2">
      <dt className="text-muted-foreground">{label}</dt>
      <dd className="font-mono">{value ?? '—'}</dd>
    </div>
  )
}
