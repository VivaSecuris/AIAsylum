import { useMemo, useState } from 'react'
import { useRouter } from 'next/router'
import * as Tabs from '@radix-ui/react-tabs'
import { Scissors, Trash2 } from 'lucide-react'

import { Layout } from '@/components/layout/Layout'
import { LoadingSpinner } from '@/components/common/LoadingSpinner'
import { StatusBadge } from '@/components/test-runs/StatusBadge'
import { DirectionPicker } from '@/components/weights/DirectionPicker'
import { PreflightPanel } from '@/components/weights/PreflightPanel'
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
import type { WeightRunKind } from '@/lib/api'
import { toast } from '@/lib/toast'
import { formatDateTime } from '@/lib/utils'

const INPUT =
  'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

function parseNums(text: string): number[] {
  return text
    .split(',')
    .map((x) => parseFloat(x.trim()))
    .filter((x) => Number.isFinite(x))
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
  const [useSubspace, setUseSubspace] = useState(false)
  const [kStrength, setKStrength] = useState(1.0)
  const [factualFloor, setFactualFloor] = useState(0.05)
  const [ranksText, setRanksText] = useState('1,2,3,4,6,8')
  const [ksText, setKsText] = useState('1.0,1.25,1.5')
  const [modifiedModel, setModifiedModel] = useState('')
  const [acknowledged, setAcknowledged] = useState<string[]>([])

  const { data: preflight } = useWeightPreflight({
    kind,
    source_model: sourceModel,
    source_run_id: directionId ?? undefined,
    output_name: kind === 'surgery' ? outputName || undefined : undefined,
  })

  const stageSpec = stages?.stages.find((s) => s.name === kind)
  const needs = (field: string) => stageSpec?.needs.includes(field) ?? false

  // Methods relevant to this stage, permanent ones separated from reversible
  // ones because that is the distinction that actually matters to the user.
  const { reversible, permanent, unavailable } = useMemo(() => {
    const all = stages?.methods ?? []
    return {
      reversible: all.filter((m) => m.available && !m.permanent),
      permanent: all.filter((m) => m.available && m.permanent),
      unavailable: all.filter((m) => !m.available),
    }
  }, [stages])

  const narrowed = objective === 'refusal_narrow'
  const custom = objective === 'custom'
  const customPositiveLines = customPositive.split('\n').map((l) => l.trim()).filter(Boolean)
  const customNegativeLines = customNegative.split('\n').map((l) => l.trim()).filter(Boolean)
  // build_split rejects anything under 8 per class, so say so before the run
  // rather than failing minutes in.
  const customTooFew =
    custom &&
    (customPositiveLines.length > 0 || customNegativeLines.length > 0) &&
    (customPositiveLines.length < 8 || customNegativeLines.length < 8)
  const selectedCount = useMemo(() => {
    if (!narrowed || !objectives) return null
    const picked = scenarios.length ? scenarios : objectives.default_scenarios
    return objectives.scenarios
      .filter((s) => picked.includes(s.name))
      .reduce((sum, s) => sum + s.count, 0)
  }, [narrowed, objectives, scenarios])

  const tooFewPrompts =
    narrowed && selectedCount != null && selectedCount < (objectives?.min_per_class ?? 8)

  async function launch() {
    try {
      const run = await createRun.mutateAsync({
        kind,
        source_model: sourceModel.trim(),
        method,
        objective,
        objective_config: narrowed
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
            : undefined,
        source_run_id: needs('source_run_id') ? directionId ?? undefined : undefined,
        output_name: kind === 'surgery' ? outputName.trim() : undefined,
        beta: kind === 'surgery' && !useSubspace ? beta : undefined,
        use_subspace: kind === 'surgery' ? useSubspace : undefined,
        k: kind === 'surgery' && useSubspace ? kStrength : undefined,
        n_per_class: kind === 'direction' ? nPerClass : undefined,
        subspace_rank: kind === 'direction' ? subspaceRank : undefined,
        n_prompts: kind === 'sweep' || kind === 'select' || kind === 'compare' ? nPrompts : undefined,
        ranks: kind === 'select' ? parseNums(ranksText).map(Math.round) : undefined,
        ks: kind === 'select' ? parseNums(ksText) : undefined,
        factual_floor: kind === 'select' ? factualFloor : undefined,
        modified_model: kind === 'compare' ? modifiedModel.trim() : undefined,
        acknowledge: acknowledged,
      })
      toast.success(`Started ${stageSpec?.label.toLowerCase() ?? kind}`)
      router.push(`/weights/${run.id}`)
    } catch (e: any) {
      const detail = e?.response?.data?.detail
      if (detail?.blocking) {
        toast.error('Preflight blocked this run — see the checks above.')
      } else {
        toast.error(detail || e?.message || 'Could not start the run')
      }
    }
  }

  return (
    <Layout>
      <div className="space-y-6">
        <div className="flex items-center gap-3">
          <Scissors className="h-7 w-7" />
          <div>
            <h1 className="text-3xl font-bold">Weight Surgery</h1>
            <p className="text-sm text-muted-foreground">
              Derive a behavioural direction, prove it moves behaviour, then write it
              permanently into the weights.
            </p>
          </div>
        </div>

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
                {/* The three stages, each independently launchable rather than
                    a wizard hiding where you are in the chain. */}
                <div className="grid gap-3 md:grid-cols-3">
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

                <div className="rounded-lg border bg-card p-6 shadow-sm">
                  <div className="space-y-5">
                    {/* 1. What behaviour */}
                    {needs('objective') && (
                      <div>
                        <label className="text-xs font-medium">Objective — what to target</label>
                        <select
                          value={objective}
                          onChange={(e) => setObjective(e.target.value)}
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

                        {custom && (
                          <div className="mt-3 space-y-3 rounded-md border p-3">
                            <p className="text-xs text-muted-foreground">
                              One prompt per line. The direction encodes whatever separates
                              these two sets, so a contrast that differs in register as well
                              as content will encode that too. Leave a box empty to fall back
                              to the curated set for that class.
                            </p>
                            <div className="grid gap-3 sm:grid-cols-2">
                              <div>
                                <label className="text-xs font-medium">
                                  Positive class ({customPositiveLines.length})
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
                                  Negative class ({customNegativeLines.length})
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
                    {kind === 'surgery' && (
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

                    {kind === 'surgery' && useSubspace && (
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

                    {kind === 'surgery' && !useSubspace && (
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
                      <label className="text-xs font-medium">
                        Source model — a Hugging Face id or a local path
                      </label>
                      <input
                        value={sourceModel}
                        onChange={(e) => setSourceModel(e.target.value)}
                        className={INPUT}
                      />
                    </div>

                    {needs('source_run_id') && (
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
                          <label className="text-xs font-medium">Subspace rank</label>
                          <input
                            type="number"
                            min={1}
                            value={subspaceRank}
                            onChange={(e) => setSubspaceRank(parseInt(e.target.value) || 1)}
                            className={INPUT}
                          />
                          <p className="mt-1 text-xs text-muted-foreground">
                            1 derives a single difference-in-means vector. Above 1 builds an
                            orthonormal subspace, which catches refusal components one vector
                            misses — and is what the capability-gated search needs.
                          </p>
                        </div>
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
                          generations. Every candidate is previewed at inference time; nothing is
                          written to disk.
                        </div>
                      </div>
                    )}

                    {kind === 'compare' && (
                      <div>
                        <label className="text-xs font-medium">Modified model</label>
                        <input
                          value={modifiedModel}
                          onChange={(e) => setModifiedModel(e.target.value)}
                          placeholder="models/qwen05b-ablated"
                          list="edited-models"
                          className={INPUT}
                        />
                        <datalist id="edited-models">
                          {models.map((m) => (
                            <option key={m.path} value={m.path} />
                          ))}
                        </datalist>
                        <p className="mt-1 text-xs text-muted-foreground">
                          Both models run through the identical loader and decoding, so the
                          weights are the only variable.
                        </p>
                      </div>
                    )}

                    {kind === 'surgery' && (
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

                    {preflight && (
                      <PreflightPanel
                        checks={preflight.checks}
                        acknowledged={acknowledged}
                        onAcknowledge={setAcknowledged}
                      />
                    )}

                    {kind === 'surgery' && (
                      <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm">
                        <p className="font-medium">This writes a full model copy and cannot be undone.</p>
                        <p className="mt-1 text-muted-foreground">
                          {sourceModel || 'the source model'} → {stages?.models_root}/
                          {outputName || '<name>'}{' '}
                          {useSubspace ? `removing the subspace at k=${kStrength}` : `at β=${beta}`}.
                        </p>
                      </div>
                    )}

                    <button
                      type="button"
                      onClick={launch}
                      disabled={
                        createRun.isPending ||
                        tooFewPrompts ||
                        customTooFew ||
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
                    <th className="px-3 py-2" />
                  </tr>
                </thead>
                <tbody>
                  {runs.length === 0 && (
                    <tr>
                      <td colSpan={7} className="px-3 py-6 text-center text-muted-foreground">
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
                    } else if (r.kind === 'compare' && s.deltas) {
                      const d = s.deltas
                      result = `refusal ${(d.refuse_harmful * 100).toFixed(0)} / capability ${(
                        d.factual_acc * 100
                      ).toFixed(0)} pts`
                    } else if (r.kind === 'surgery') {
                      result = s.manifest?.beta != null ? `β=${s.manifest.beta}` : '—'
                      if (s.manifest?.extra?.k != null) result = `k=${s.manifest.extra.k}`
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
                  onClick={() => router.push(`/weights/models/${m.name}`)}
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
