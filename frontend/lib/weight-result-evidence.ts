/** Result interpretation shared by the headline and detailed evidence cards. */
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)

export function usableHNeuronSignal(summary: Record<string, any>): boolean {
  const { auroc, null_auroc_p95: nullAuroc, surface_auroc: surfaceAuroc, n_selected: count } = summary
  // Older results have only beats_null, which never measured the length control.
  // Missing/nonfinite controls cannot establish a usable signal.
  return finite(count) && count > 0
    && finite(auroc) && finite(nullAuroc) && finite(surfaceAuroc)
    && auroc >= nullAuroc + 0.05 && auroc >= surfaceAuroc + 0.05
    && summary.usable !== false && summary.beats_null !== false && summary.beats_surface !== false
}

export type InduceStatus = 'none' | 'failed' | 'verified' | 'accepted' | 'unverified'

export function induceStatus(summary: Record<string, any>): InduceStatus {
  if (!summary.winner) return 'none'
  const verification = summary.verification
  if (summary.accepted === false || summary.winner.accepted === false || summary.winner.admissible === false
    || verification?.accepted === false || verification?.admissible === false) return 'failed'
  if (summary.accepted === true && verification?.accepted === true) return 'verified'
  if (summary.accepted === true && !verification) return 'accepted'
  // Legacy metric-only reports contain no acceptance verdict. Their presence
  // alone is not evidence that sampled verification succeeded.
  return 'unverified'
}
