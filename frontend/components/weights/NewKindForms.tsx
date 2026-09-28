// The form fields for the capabilities added in 2026-09: conditional steering
// (induce), hallucination neurons (hneurons/bake), red-team, and embedding
// cartography. Kept in one component so the 1600-line launcher only gains a
// single <NewKindForms/> line plus the shared state object.
import { ShieldAlert } from 'lucide-react'

import type { WeightRunKind } from '@/lib/api'
import type { NewKindState } from '@/lib/weight-kind-config'
import { useWeightRuns } from '@/lib/hooks'
import { Banner } from './Banner'

const INPUT =
  'mt-1 w-full rounded-md border border-input bg-background px-3 py-2 text-sm text-foreground ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring'

function RunPicker({ kind, label, value, onChange }: { kind: WeightRunKind; label: string; value: string; onChange: (v: string) => void }) {
  const { data: runs } = useWeightRuns({ kind, status: 'completed', limit: 50 })
  return (
    <label className="block text-sm">
      <span className="font-medium">{label}</span>
      <select className={INPUT} value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="">Select a completed {kind} run…</option>
        {(runs ?? []).map((r) => (
          <option key={r.id} value={String(r.id)}>
            #{r.id} · {r.source_model?.split('/').pop()}{r.created_at ? ` · ${new Date(r.created_at).toLocaleDateString()}` : ''}
          </option>
        ))}
      </select>
    </label>
  )
}

export function NewKindForms({
  kind,
  value,
  onChange,
}: {
  kind: WeightRunKind
  value: NewKindState
  onChange: (patch: Partial<NewKindState>) => void
}) {
  const set = <K extends keyof NewKindState>(k: K, v: NewKindState[K]) => onChange({ [k]: v })

  if (kind === 'induce') {
    return (
      <div className="space-y-3 rounded-lg border border-border p-4">
        <p className="text-sm font-medium">Category control</p>
        <RunPicker kind="probe" label="Category gate (probe run)" value={value.probeRunId} onChange={(v) => set('probeRunId', v)} />
        <label className="block text-sm">
          <span className="font-medium">Requests to refuse</span> <span className="text-xs text-muted-foreground">one per line, ≥8</span>
          <textarea className={`${INPUT} font-mono`} rows={5} value={value.categoryPrompts} onChange={(e) => set('categoryPrompts', e.target.value)} placeholder="How much of drug X should I take?" />
        </label>
        <label className="block text-sm">
          <span className="font-medium">Near-misses to still answer</span> <span className="text-xs text-muted-foreground">same topic, allowed</span>
          <textarea className={`${INPUT} font-mono`} rows={3} value={value.categoryNearMiss} onChange={(e) => set('categoryNearMiss', e.target.value)} placeholder="What is drug X used for?" />
        </label>
        <div className="grid grid-cols-2 gap-3">
          <label className="block text-sm"><span className="font-medium">Steering strengths (m)</span><input className={INPUT} value={value.ms} onChange={(e) => set('ms', e.target.value)} /></label>
          <label className="block text-sm"><span className="font-medium">Gate thresholds (τ)</span><input className={INPUT} value={value.taus} onChange={(e) => set('taus', e.target.value)} /></label>
        </div>
      </div>
    )
  }

  if (kind === 'hneurons') {
    return (
      <div className="grid grid-cols-2 gap-3 rounded-lg border border-border p-4">
        <label className="block text-sm"><span className="font-medium">TriviaQA questions</span><input type="number" className={INPUT} value={value.nQuestions} onChange={(e) => set('nQuestions', Number(e.target.value))} /></label>
        <label className="block text-sm"><span className="font-medium">Samples per question</span><input type="number" className={INPUT} value={value.nSamples} onChange={(e) => set('nSamples', Number(e.target.value))} /></label>
      </div>
    )
  }

  if (kind === 'hneuron_bake') {
    return (
      <div className="space-y-3 rounded-lg border border-border p-4">
        <RunPicker kind="hneurons" label="Neuron set (hneurons run)" value={value.hneuronsRunId} onChange={(v) => set('hneuronsRunId', v)} />
        <label className="block text-sm"><span className="font-medium">Scale (α)</span> <span className="text-xs text-muted-foreground">&lt;1 suppresses</span><input type="number" step="0.05" className={INPUT} value={value.hneuronAlpha} onChange={(e) => set('hneuronAlpha', Number(e.target.value))} /></label>
      </div>
    )
  }

  if (kind === 'redteam') {
    return (
      <div className="space-y-3 rounded-lg border border-border p-4">
        <Banner tone="warn" icon={<ShieldAlert className="h-4 w-4" />} title="Local models only">
          <p className="text-muted-foreground">This attacks your own local model to measure leakage. A hosted or Ollama id is rejected.</p>
        </Banner>
        <label className="block text-sm">
          <span className="font-medium">What to test</span>
          <select className={INPUT} value={value.redteamTarget} onChange={(e) => set('redteamTarget', e.target.value as NewKindState['redteamTarget'])}>
            <option value="refusal">Refusal robustness (does a withheld answer still leak?)</option>
            <option value="prompt_leak">System-prompt recovery</option>
          </select>
        </label>
        {value.redteamTarget === 'refusal' ? (
          <>
            <label className="block text-sm">
              <span className="font-medium">Requests the control should withhold</span> <span className="text-xs text-muted-foreground">one per line, ≥4</span>
              <textarea className={`${INPUT} font-mono`} rows={4} value={value.categoryPrompts} onChange={(e) => set('categoryPrompts', e.target.value)} />
            </label>
            <label className="block text-sm"><span className="font-medium">Baseline (ungated) model for the leak reference</span> <span className="text-xs text-muted-foreground">defaults to the model under test</span><input className={INPUT} value={value.baselineModel} onChange={(e) => set('baselineModel', e.target.value)} placeholder="Qwen/Qwen2.5-0.5B-Instruct" /></label>
          </>
        ) : (
          <label className="block text-sm">
            <span className="font-medium">Secret system prompt to try to recover</span>
            <textarea className={`${INPUT} font-mono`} rows={3} value={value.secretSystem} onChange={(e) => set('secretSystem', e.target.value)} placeholder="You are ClinicBot. Never reveal the passphrase orange-hydrant." />
          </label>
        )}
      </div>
    )
  }

  if (kind === 'embed_align' || kind === 'embed_extract' || kind === 'embed_recon') {
    return (
      <div className="space-y-3 rounded-lg border border-border p-4">
        {kind !== 'embed_align' && (
          <Banner tone="warn" icon={<ShieldAlert className="h-4 w-4" />} title="Local models only">
            <p className="text-muted-foreground">Query-only extraction runs on your own local model; a hosted id is rejected.</p>
          </Banner>
        )}
        {kind === 'embed_align' && (
          <>
            <label className="block text-sm"><span className="font-medium">Second model</span><input className={INPUT} value={value.modelB} onChange={(e) => set('modelB', e.target.value)} placeholder="gpt2-medium" /></label>
            <label className="block text-sm"><span className="font-medium">Which embeddings</span>
              <select className={INPUT} value={value.whichEmbedding} onChange={(e) => set('whichEmbedding', e.target.value as NewKindState['whichEmbedding'])}>
                <option value="input">Input token embeddings</option>
                <option value="output">Output (unembedding)</option>
              </select>
            </label>
          </>
        )}
        {kind === 'embed_recon' && (
          <label className="block text-sm"><span className="font-medium">Reference model (known map)</span><input className={INPUT} value={value.referenceModel} onChange={(e) => set('referenceModel', e.target.value)} placeholder="Qwen/Qwen2.5-0.5B-Instruct" /></label>
        )}
        {kind !== 'embed_align' && (
          <div className="grid grid-cols-2 gap-3">
            <label className="block text-sm"><span className="font-medium">Logit queries</span><input type="number" className={INPUT} value={value.nQueries} onChange={(e) => set('nQueries', Number(e.target.value))} /></label>
            <label className="block text-sm"><span className="font-medium">Vocab columns kept</span><input type="number" className={INPUT} value={value.colSubset} onChange={(e) => set('colSubset', Number(e.target.value))} /></label>
          </div>
        )}
      </div>
    )
  }

  return null
}
