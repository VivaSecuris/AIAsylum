import { useMemo, useState } from 'react'
import { useRouter } from 'next/router'

import type { RoutingLayer, RoutingSummary } from '@/lib/api'
import type { ExpertSelection } from '@/components/weights/ExpertSelectionInput'

type Mode = 'all' | 'last'

function cellValue(layer: RoutingLayer, expert: number, mode: Mode): number {
  return mode === 'all' ? layer.delta_frac[expert] : layer.delta_last_token[expert]
}

// Diverging: warm for "selected more on harmful", cool for "more on harmless".
function cellStyle(delta: number, maxAbs: number): React.CSSProperties {
  const strength = maxAbs > 0 ? Math.min(1, Math.abs(delta) / maxAbs) : 0
  const alpha = 0.08 + 0.72 * strength
  return {
    backgroundColor: delta >= 0 ? `rgba(220, 38, 38, ${alpha})` : `rgba(37, 99, 235, ${alpha})`,
  }
}

export function RoutingHeatmap({ result, sourceModel }: { result: RoutingSummary; sourceModel: string }) {
  const router = useRouter()
  const [mode, setMode] = useState<Mode>('all')
  const [picked, setPicked] = useState<Set<string>>(new Set())

  const layers = result.layers ?? []
  const nExperts = Math.max(0, ...layers.map((l) => l.n_experts))
  const maxAbs = useMemo(
    () => Math.max(1e-9, ...layers.flatMap((l) => Array.from({ length: l.n_experts }, (_, e) => Math.abs(cellValue(l, e, mode))))),
    [layers, mode],
  )
  const dense = nExperts > 32

  const selection = useMemo<ExpertSelection>(() => {
    const out: ExpertSelection = {}
    picked.forEach((key) => {
      const [layer, expert] = key.split(':')
      const list = (out[layer] as number[] | undefined) ?? []
      list.push(Number(expert))
      out[layer] = list.sort((a, b) => a - b)
    })
    return out
  }, [picked])

  function toggle(layer: number, expert: number) {
    const key = `${layer}:${expert}`
    setPicked((prev) => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  function editThese() {
    router.push({
      pathname: '/weights',
      query: { kind: 'expert_surgery', source_model: sourceModel, expert_selection: JSON.stringify(selection) },
    })
  }

  const consistent = result.consistency?.gate_vs_expert_counts_match !== false

  return (
    <div className="rounded-lg border bg-card p-6 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">Which experts fire on which prompts</h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Each cell is one expert in one layer: how much more often its top-{result.top_k ?? 'k'} selection
            happened on the {result.prompts?.harmful ?? '?'} harmful prompts than on the{' '}
            {result.prompts?.harmless ?? '?'} harmless ones. Red means more on harmful, blue more on harmless.
            Click cells to build a selection, then edit exactly those experts.
          </p>
        </div>
        <div className="flex gap-1 rounded-md border p-0.5 text-xs">
          <button
            type="button"
            onClick={() => setMode('all')}
            className={`rounded px-2 py-1 ${mode === 'all' ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'}`}
          >
            All tokens
          </button>
          <button
            type="button"
            onClick={() => setMode('last')}
            className={`rounded px-2 py-1 ${mode === 'last' ? 'bg-primary text-primary-foreground' : 'hover:bg-muted'}`}
          >
            Last prompt token
          </button>
        </div>
      </div>

      {!consistent && (
        <p className="mt-4 rounded border border-destructive/50 bg-destructive/10 p-3 text-xs">
          The replayed router selection did not match the rows the experts actually received for this
          model family. Treat every fraction here as unverified.
        </p>
      )}

      <div className="mt-4 overflow-x-auto">
        <table className="border-separate border-spacing-px text-[10px]">
          <thead>
            <tr>
              <th className="sticky left-0 bg-card pr-2 text-right font-normal text-muted-foreground">layer</th>
              {Array.from({ length: nExperts }, (_, e) => (
                <th key={e} className={`font-normal text-muted-foreground ${dense ? 'w-2' : 'w-6'}`}>
                  {dense ? (e % 8 === 0 ? e : '') : e}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {layers.map((layer) => (
              <tr key={layer.layer}>
                <td className="sticky left-0 bg-card pr-2 text-right font-mono text-muted-foreground">
                  {layer.layer}
                </td>
                {Array.from({ length: nExperts }, (_, e) => {
                  if (e >= layer.n_experts) return <td key={e} />
                  const delta = cellValue(layer, e, mode)
                  const key = `${layer.layer}:${e}`
                  const on = picked.has(key)
                  return (
                    <td
                      key={e}
                      title={`layer ${layer.layer}, expert ${e}: harmful ${(
                        (mode === 'all' ? layer.harmful.selection_frac[e] : layer.harmful.last_token_frac[e]) * 100
                      ).toFixed(1)}%, harmless ${(
                        (mode === 'all' ? layer.harmless.selection_frac[e] : layer.harmless.last_token_frac[e]) * 100
                      ).toFixed(1)}%, Δ ${(delta * 100).toFixed(1)} pts`}
                      onClick={() => toggle(layer.layer, e)}
                      style={cellStyle(delta, maxAbs)}
                      className={`cursor-pointer ${dense ? 'h-3 w-2' : 'h-5 w-6'} ${on ? 'outline outline-2 outline-foreground' : ''}`}
                    />
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-3">
        <button
          type="button"
          onClick={editThese}
          disabled={picked.size === 0}
          className="rounded-md bg-primary px-3 py-1.5 text-sm text-primary-foreground disabled:opacity-50"
        >
          Edit these {picked.size || ''} expert{picked.size === 1 ? '' : 's'}
        </button>
        {picked.size > 0 && (
          <button type="button" onClick={() => setPicked(new Set())} className="text-xs text-muted-foreground hover:underline">
            clear
          </button>
        )}
        <span className="text-xs text-muted-foreground">
          The edit is partial by design: it touches only what you pick, and the manifest says so.
        </span>
      </div>

      <h3 className="mt-6 text-sm font-medium">Most differentially selected</h3>
      <table className="mt-2 w-full text-xs">
        <thead>
          <tr className="border-b text-left text-muted-foreground">
            <th className="py-1 pr-3">layer</th>
            <th className="py-1 pr-3">expert</th>
            <th className="py-1 pr-3">harmful</th>
            <th className="py-1 pr-3">harmless</th>
            <th className="py-1 pr-3">Δ all tokens</th>
            <th className="py-1 pr-3">Δ last token</th>
            <th className="py-1" />
          </tr>
        </thead>
        <tbody>
          {(result.ranking ?? []).slice(0, 20).map((r) => {
            const key = `${r.layer}:${r.expert}`
            return (
              <tr key={key} className="border-b last:border-0">
                <td className="py-1 pr-3 font-mono">{r.layer}</td>
                <td className="py-1 pr-3 font-mono">{r.expert}</td>
                <td className="py-1 pr-3 font-mono">{(r.harmful_frac * 100).toFixed(1)}%</td>
                <td className="py-1 pr-3 font-mono">{(r.harmless_frac * 100).toFixed(1)}%</td>
                <td className="py-1 pr-3 font-mono">{(r.delta * 100).toFixed(1)} pts</td>
                <td className="py-1 pr-3 font-mono">{(r.last_token_delta * 100).toFixed(1)} pts</td>
                <td className="py-1">
                  <button type="button" onClick={() => toggle(r.layer, r.expert)} className="text-primary hover:underline">
                    {picked.has(key) ? 'remove' : 'add'}
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
