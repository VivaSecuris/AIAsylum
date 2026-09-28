import type { TestRun } from './api'

type DisplayRun = Pick<TestRun, 'test_type' | 'patient_model' | 'meta_data'> & {
  metadata?: Record<string, unknown>
}

/** Display and group campaign runs by model identity, retaining execution paths in saved provenance. */
export function testRunModelRef(run: DisplayRun): string {
  if (run.test_type === 'benchmark') {
    const reference = run.meta_data?.campaign_model ?? run.metadata?.campaign_model
    if (typeof reference === 'string' && reference.trim()) return reference
  }
  return run.patient_model
}

/** A model reference short enough for a card: a local checkpoint path becomes its folder name. */
export function shortModelRef(ref: string): string {
  return ref?.startsWith('/') ? ref.split('/').filter(Boolean).pop() || ref : ref
}

/** "provider/model" for display, with local checkpoint paths shortened (keep the full ref in a tooltip). */
export function testRunModelLabel(run: DisplayRun & { patient_provider: string }): string {
  const ref = testRunModelRef(run)
  return ref?.startsWith('/') ? shortModelRef(ref) : `${run.patient_provider}/${ref}`
}
