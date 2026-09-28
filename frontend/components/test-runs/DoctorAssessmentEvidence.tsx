import type { TestResult } from '@/lib/api'
import { ModelChatLink } from '@/components/models/ModelChatLink'
import { reasoningLabelFor, TurnFinishStatus, TurnSystemPrompts } from '@/components/conversation/ConversationViewer'

/** The doctor's assessment is a separate generation, outside the patient transcript. */
export function DoctorAssessmentEvidence({ result, openByDefault = false }: { result: TestResult; openByDefault?: boolean }) {
  const evidence = result.meta_data?.doctor_assessment
  if (!result.analysis && !evidence) return null
  const response = evidence?.response ?? result.analysis
  const nativeReasoning = evidence?.generation_metadata?.native_reasoning
  return <details className="rounded-lg border bg-card p-4" open={openByDefault}>
    <summary className="cursor-pointer font-medium">Doctor assessment · {result.test_name}</summary>
    <div className="mt-3 space-y-3">
      {evidence?.model_name && <p className="break-words text-xs text-muted-foreground"><ModelChatLink provider={evidence.model_provider} model={evidence.model_name}>{[evidence.model_provider, evidence.model_name].filter(Boolean).join(' / ')}</ModelChatLink></p>}
      <TurnSystemPrompts prompts={evidence?.request_system_prompts} />
      {evidence?.reasoning && <details className="rounded border border-dashed p-3 text-xs"><summary className="cursor-pointer font-medium">{reasoningLabelFor(evidence.reasoning_source)}</summary><p className="mt-2 whitespace-pre-wrap">{evidence.reasoning}</p></details>}
      {nativeReasoning && <details className="rounded border border-dashed p-3 text-xs"><summary className="cursor-pointer font-medium">Model-provided reasoning</summary><p className="mt-2 whitespace-pre-wrap">{nativeReasoning}</p></details>}
      <p className="whitespace-pre-wrap break-words text-sm">{response || '(The model returned no assessment text.)'}</p>
      <TurnFinishStatus reason={evidence?.finish_reason} />
      {evidence && <details className="text-xs text-muted-foreground"><summary className="cursor-pointer">Assessment generation evidence</summary><pre className="mt-2 overflow-auto whitespace-pre-wrap break-words">{JSON.stringify({ provider: evidence.model_provider, model: evidence.model_name, finish_reason: evidence.finish_reason, usage: evidence.usage, generation: evidence.generation_metadata, system_prompt_source: evidence.request_system_prompts_source }, null, 2)}</pre></details>}
      {!evidence && <p className="text-xs text-muted-foreground">Generation evidence was not recorded for this assessment.</p>}
    </div>
  </details>
}

/** Keep the doctor's comments with the live/transcript view, including legacy analysis text. */
export function DoctorAssessmentPanel({ results, pending = false }: { results: TestResult[]; pending?: boolean }) {
  const assessments = results.filter((result) => result.analysis || result.meta_data?.doctor_assessment)
  if (!assessments.length && !pending) return null
  return <section aria-label="Doctor assessment" className="mt-4 space-y-3 border-t pt-4">
    <h3 className="font-semibold">Doctor assessment</h3>
    <p className="text-xs text-muted-foreground">The doctor reviews the patient replies after the test.</p>
    {assessments.length ? assessments.map((result) => <DoctorAssessmentEvidence key={result.id} result={result} openByDefault />)
      : <p role="status" className="text-sm text-muted-foreground">Doctor assessment will appear here when complete.</p>}
  </section>
}
