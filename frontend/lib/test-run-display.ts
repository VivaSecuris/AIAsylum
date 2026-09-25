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
