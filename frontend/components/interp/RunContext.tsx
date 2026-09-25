import type { InterpRun } from '@/lib/api'

const descriptions: Record<string, string> = {
  single: 'Activation norms show representation magnitude. The logit lens projects intermediate states into token predictions; it is an exploratory view, not a causal explanation.',
  comparison: 'Divergence measures where the two prompts produce different representations. A large difference suggests a place to investigate; activation patching tests whether an intervention changes the prediction.',
  progression: 'Track how representations move as examples are added. Check the aligned token window before attributing movement to a specific example.',
  model_diff: 'Compare a baseline and an edited model on the same prompt. Both models must share their tokenizer and residual dimensions. Divergence alone does not establish better performance or changed safety behaviour.',
}

const outputs = [
  ['enable_attention_capture', 'Attention maps', 'attention_payload.json'],
  ['enable_mlp_capture', 'MLP activations', 'mlp_payload.json'],
  ['enable_pre_mlp_capture', 'Internal MLP neurons', 'mlp_payload.json'],
  ['enable_qkv_capture', 'OV/QK analysis', 'ov_qk_payload.json'],
  ['enable_patching', 'Activation patching', 'patching_results.json'],
  ['enable_minimal_circuit', 'Approximate circuit search', 'minimal_circuit_payload.json'],
]

export function RunContext({ run, artifactNames }: { run: InterpRun; artifactNames?: string[] }) {
  const options = run.metadata?.options ?? {}
  const checked = run.metadata?.summary?.preflight
  const requested = outputs.filter(([flag]) => options[flag])
  return (
    <div className="space-y-3 rounded-lg border bg-card p-5">
      <h2 className="text-sm font-semibold">How to read this analysis</h2>
      <p className="text-sm text-muted-foreground">{descriptions[run.mode]}</p>
      <dl className="grid gap-3 text-xs sm:grid-cols-4">
        <div><dt className="text-muted-foreground">{checked ? 'Execution device' : 'Requested device'}</dt><dd className="font-mono">{checked?.device ?? options.device ?? 'auto'}</dd></div>
        <div><dt className="text-muted-foreground">{checked ? 'Execution precision' : 'Requested precision'}</dt><dd className="font-mono">{checked?.dtype ?? options.dtype ?? '—'}</dd></div>
        <div><dt className="text-muted-foreground">Sequence limit</dt><dd>{options.max_len ?? '—'} tokens</dd></div>
        <div><dt className="text-muted-foreground">Analysis window</dt><dd>{options.window ?? '—'} tokens</dd></div>
      </dl>
      {requested.length > 0 && (
        <ul className="flex flex-wrap gap-2 text-xs">
          {requested.map(([flag, label, artifact]) => (
            <li key={flag} className="rounded bg-muted px-2 py-1">{label}{run.status === 'completed' && artifactNames ? (artifactNames.includes(artifact) ? ' · available' : ' · no artifact produced') : ' · requested'}</li>
          ))}
        </ul>
      )}
      {run.status === 'completed' && artifactNames && requested.some(([, , artifact]) => !artifactNames.includes(artifact)) && (
        <p className="text-xs text-amber-700 dark:text-amber-300">A requested analysis did not produce an artifact. This is missing evidence, not a zero effect; check model capabilities and run diagnostics before interpreting the result.</p>
      )}
      {(run.metadata?.summary?.warnings ?? []).map((warning: string, index: number) => <p key={index} className="text-xs text-amber-700 dark:text-amber-300">{warning}</p>)}
      <details>
        <summary className="cursor-pointer text-xs font-medium">Prompts used</summary>
        <div className="mt-2 space-y-2">
          {(run.prompts ?? [run.prompt_a, run.prompt_b].filter(Boolean)).map((prompt, index) => (
            <div key={index}><p className="text-xs text-muted-foreground">Prompt {run.prompts ? index + 1 : index === 0 ? 'A' : 'B'}</p><pre className="mt-1 max-h-40 overflow-auto whitespace-pre-wrap break-words rounded bg-muted p-3 text-xs">{prompt}</pre></div>
          ))}
        </div>
      </details>
    </div>
  )
}
