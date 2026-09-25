import { useState } from 'react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type { TimelinePayload } from '@/lib/api'

interface Props {
  timelines: TimelinePayload[]
}

/**
 * Where in the generation the model commits to refusing.
 *
 * Each point is the residual at the direction's layer, projected onto the
 * refusal direction, at the position that decides that token. Above zero is
 * the average refusing state, below is the average complying one. The shaded
 * span is the reasoning block on a thinking model; on those the decision
 * often forms there, which a last-prompt-token capture never sees.
 */
export function RefusalTimelineChart({ timelines }: Props) {
  const [idx, setIdx] = useState(0)
  if (!timelines?.length) return null
  const tl = timelines[Math.min(idx, timelines.length - 1)]
  const data = tl.tokens.map((tok, i) => ({
    i,
    token: tok,
    score: tl.score[i],
    think: tl.in_think[i] ? 1.5 : null,
  }))
  const zScored = tl.normalisation !== 'class_means'

  return (
    <div className="rounded-lg border bg-card p-6 shadow-sm">
      <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">When the model decides</h2>
          <p className="text-sm text-muted-foreground">
            Refusal projection per generated token at layer {tl.layer}.{' '}
            {zScored
              ? 'No class means were recorded for this direction, so the trace is z-scored and no commit point is marked.'
              : tl.decision_index != null
                ? `Committed at token ${tl.decision_index}${tl.decision_in_think ? ' inside the reasoning block' : ''}; ended on the ${tl.final_side} side.`
                : 'Never settled for three consecutive tokens.'}
          </p>
        </div>
        {timelines.length > 1 && (
          <select value={idx} onChange={(e) => setIdx(parseInt(e.target.value))}
                  className="rounded border bg-background px-2 py-1 text-xs">
            {timelines.map((t, i) => (
              <option key={i} value={i}>prompt {i + 1}: {t.prompt.slice(0, 48)}…</option>
            ))}
          </select>
        )}
      </div>

      <ResponsiveContainer width="100%" height={260}>
        <ComposedChart data={data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="i" type="number" domain={[0, 'dataMax']} allowDecimals={false}
                 label={{ value: 'generated token', position: 'insideBottom', offset: -4 }} />
          <YAxis domain={['auto', 'auto']} />
          <Tooltip
            content={({ active, payload }: any) => {
              if (!active || !payload?.length) return null
              const p = payload[0].payload
              return (
                <div className="rounded-md border bg-background px-3 py-2 text-xs shadow-sm">
                  <div className="font-mono">#{p.i} {JSON.stringify(p.token)}</div>
                  <div>score {p.score?.toFixed(2)}{p.think ? ' · in <think>' : ''}</div>
                </div>
              )
            }}
          />
          <Area dataKey="think" name="reasoning block" fill="#a78bfa" fillOpacity={0.15} stroke="none"
                type="step" isAnimationActive={false} />
          <ReferenceLine y={0} stroke="#94a3b8" />
          {!zScored && <ReferenceLine y={1} stroke="#ef4444" strokeDasharray="2 2"
                                      label={{ value: 'mean harmful', position: 'insideTopLeft', fontSize: 10 }} />}
          {!zScored && <ReferenceLine y={-1} stroke="#10b981" strokeDasharray="2 2"
                                      label={{ value: 'mean harmless', position: 'insideBottomLeft', fontSize: 10 }} />}
          {tl.decision_index != null && (
            <ReferenceLine x={tl.decision_index} stroke="#f59e0b" strokeWidth={2}
                           label={{ value: 'commit', position: 'top', fontSize: 11 }} />
          )}
          <Area dataKey="score" name="refusal score" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.1}
                type="monotone" isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>

      <details className="mt-3 text-xs">
        <summary className="cursor-pointer font-medium">Generated text</summary>
        <p className="mt-2 whitespace-pre-wrap font-mono text-muted-foreground">{tl.text}</p>
      </details>
    </div>
  )
}
