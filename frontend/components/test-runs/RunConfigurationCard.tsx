import type { TestRun } from '@/lib/api'

const SOURCE_LABELS: Record<string, string> = {
  default: 'Built-in',
  library: 'Library',
  custom: 'Custom',
  none: 'Sent verbatim',
  test_prompt: 'From the test prompt',
}

function PromptRow({ label, prompt }: { label: string; prompt: any }) {
  if (!prompt) return null
  const badge = SOURCE_LABELS[prompt.source] || prompt.source
  const text: string | undefined = prompt.text || prompt.user_message_template
  return (
    <div className="text-sm">
      <span className="font-medium">{label}:</span>{' '}
      <span className="rounded bg-muted px-1.5 py-0.5 text-xs">{badge}</span>
      {prompt.name && <span className="ml-2">{prompt.name}{prompt.prompt_id ? ` (#${prompt.prompt_id})` : ''}</span>}
      {prompt.source === 'none' && <span className="ml-2 text-xs text-muted-foreground">no system prompt, prompts unwrapped</span>}
      {text && (
        <details className="mt-1 text-xs text-muted-foreground">
          <summary className="cursor-pointer">{prompt.user_message_template && !prompt.text ? 'Show the wrapper each prompt was sent in' : 'Show text'}</summary>
          <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-2 font-sans">{text}</pre>
        </details>
      )}
    </div>
  )
}

function GenerationRow({ label, g }: { label: string; g: any }) {
  if (!g) return null
  const parts = [
    `temperature ${g.temperature ?? 'default'}`,
    g.top_p != null ? `top-p ${g.top_p}` : null,
    g.max_tokens != null ? `max ${g.max_tokens} tokens` : null,
    `ReACT ${g.enable_cot ? 'on' : 'off'}`,
    g.use_dynamic_strategies === false ? 'fixed questions' : null,
    g.seed != null ? `seed ${g.seed}` : g.seed_note ? `no seed (${g.seed_note})` : null,
  ].filter(Boolean)
  return <p className="text-sm"><span className="font-medium">{label}:</span> <span className="text-muted-foreground">{parts.join(' · ')}</span></p>
}

/**
 * What each LLM step of a run actually used: the resolved system prompts, the
 * sampling settings, whether the seed applied, and any fallback warnings. Runs
 * created before these were recorded show the saved request instead.
 */
export function RunConfigurationCard({ testRun }: { testRun: TestRun }) {
  const meta: Record<string, any> = testRun.meta_data || {}
  const config: Record<string, any> = meta.test_config || {}
  const resolved: Record<string, any> | undefined = meta.resolved_config

  if (testRun.test_type === 'analysis') {
    const a = meta.analysis_config || {}
    return (
      <div className="space-y-2 rounded-lg border bg-card p-4">
        <h3 className="font-semibold">Evaluator configuration</h3>
        <p className="text-sm">
          <span className="font-medium">Extra instructions:</span>{' '}
          {a.evaluator_system_prompt_id ? `library prompt #${a.evaluator_system_prompt_id}`
            : a.evaluator_system_prompt ? 'custom' : 'none (built-in scoring prompt only)'}
        </p>
        {a.evaluator_system_prompt && <pre className="max-h-40 overflow-auto whitespace-pre-wrap rounded bg-muted/50 p-2 text-xs">{a.evaluator_system_prompt}</pre>}
        <p className="text-sm text-muted-foreground">
          temperature {a.evaluator_temperature ?? 0.3}{a.evaluator_max_tokens ? ` · max ${a.evaluator_max_tokens} tokens` : ''}
          {a.evaluator_top_p != null && ` · top-p ${a.evaluator_top_p}`} · ReACT {a.evaluator_enable_cot ? 'on' : 'off'}
          {' · '}checks: {[a.enable_cot_detection && `chain of thought (${a.cot_analysis_mode || 'full'})`, a.enable_factuality_check && 'factuality', a.enable_manipulation_analysis && 'manipulation'].filter(Boolean).join(', ') || 'none'}
        </p>
      </div>
    )
  }

  const analysis = config.auto_analysis ? (config.analysis_config || {}) : null
  return (
    <div className="space-y-3 rounded-lg border bg-card p-4">
      <h3 className="font-semibold">Run configuration</h3>
      {typeof config.doctor_goal === 'string' && config.doctor_goal.trim() && <div className="text-sm"><p className="font-medium">Doctor conversation goal</p><p className="whitespace-pre-wrap break-words text-muted-foreground">{config.doctor_goal}</p></div>}
      {resolved?.warnings?.length > 0 && (
        <div className="rounded-md border border-amber-300 bg-amber-50 p-2 text-sm text-amber-900 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-100">
          {resolved!.warnings.map((w: string) => <p key={w}>{w}</p>)}
        </div>
      )}
      {resolved ? (
        <>
          <div className="space-y-2">
            <PromptRow label="Doctor prompt" prompt={resolved.system_prompts?.doctor} />
            {resolved.system_prompts?.patients
              ? resolved.system_prompts.patients.map((p: any) => <PromptRow key={p.index} label={`Patient ${p.index + 1} prompt`} prompt={p} />)
              : <PromptRow label="Patient prompt" prompt={resolved.system_prompts?.patient} />}
          </div>
          <div className="space-y-1">
            <GenerationRow label="Doctor" g={resolved.generation?.doctor} />
            {resolved.generation?.patients
              ? resolved.generation.patients.map((g: any, i: number) => <GenerationRow key={i} label={`Patient ${i + 1}`} g={g} />)
              : <GenerationRow label="Patient" g={resolved.generation?.patient} />}
          </div>
        </>
      ) : (
        <div className="space-y-1 text-sm">
          <p className="text-muted-foreground">This run was created before per-step settings were recorded. The saved request:</p>
          <p>
            Doctor prompt: {config.doctor_system_prompt_id ? `library #${config.doctor_system_prompt_id}` : 'built-in'} ·
            Patient prompt: {config.patient_system_prompt_id ? `library #${config.patient_system_prompt_id}` : 'built-in'} ·
            temperature {config.temperature ?? 0.7}{config.seed != null ? ` · seed ${config.seed}` : ''}
          </p>
        </div>
      )}
      <p className="text-sm">
        <span className="font-medium">Evaluator:</span>{' '}
        {analysis
          ? <>{analysis.evaluator_model ? `${analysis.evaluator_provider}/${analysis.evaluator_model}` : 'the doctor model'}
              {' · '}{analysis.evaluator_system_prompt_id ? `extra instructions from library #${analysis.evaluator_system_prompt_id}` : analysis.evaluator_system_prompt ? 'custom extra instructions' : 'built-in scoring prompt'}
              {' · '}temperature {analysis.evaluator_temperature ?? 0.3}
              {' · '}ReACT {analysis.evaluator_enable_cot ? 'on' : 'off'}</>
          : <span className="text-muted-foreground">no automatic analysis</span>}
      </p>
    </div>
  )
}
