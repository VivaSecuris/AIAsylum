// Conditional-steering (induce) result: the winning control, and the trial grid
// that shows on-target refusal rising while near-misses stay flat. A control
// that also refuses near-misses has learned the topic, not the behaviour.
import { MetricCard } from '@/components/common/MetricCard'
import { induceStatus } from '@/lib/weight-result-evidence'
import { Banner } from './Banner'

const pct = (v: unknown) => (typeof v === 'number' ? `${Math.round(v * 100)}%` : '—')

const REASON: Record<string, string> = {
  ok: 'admissible',
  degenerate: 'output collapsed',
  language_drift: 'wrong script',
  capability_cost: 'capability fell',
  near_miss_refused: 'refused near-misses',
  over_refusal: 'refused benign prompts',
  safety_regressed: 'weakened existing safety',
}

export function InduceSection({ summary }: { summary: Record<string, any> }) {
  const winner = summary.winner as Record<string, any> | null
  const trials: Record<string, any>[] = summary.trials ?? []
  const baseline = summary.baseline ?? {}
  const status = induceStatus(summary)
  const verification = summary.verification
  const reason = verification?.accepted === false
    ? verification.reason : summary.reason || winner?.reason

  return (
    <div className="space-y-3">
      {winner ? (
        <>
          <div className="grid gap-3 sm:grid-cols-4">
            <MetricCard title="On-target refusal" value={pct(winner.refuse_target)} subtitle={`baseline ${pct(baseline.refuse_target)}`} />
            <MetricCard title="Near-miss refusal" value={pct(winner.refuse_near_miss)} subtitle="must stay flat" />
            <MetricCard title="Capability" value={pct(winner.factual_acc)} subtitle={`baseline ${pct(baseline.factual_acc)}`} />
            <MetricCard title="Setting" value={`m ${winner.m}`} subtitle={`τ ${winner.tau}`} />
          </div>
          <p className="text-xs text-muted-foreground">Candidate metrics on the reporting split.</p>
          <Banner tone={status === 'verified' ? 'good' : status === 'failed' ? 'bad' : 'warn'}
            title={status === 'verified' ? 'Accepted and verified under sampling'
              : status === 'failed' ? 'Control rejected'
                : status === 'accepted' ? 'Accepted on the reporting split; sampling not checked'
                  : 'Verification not established'}>
            {status === 'failed' && <p className="text-muted-foreground">{reason ? `Reason: ${String(reason).replace(/_/g, ' ')}. ` : ''}The candidate metrics remain available for diagnosis; this control was not accepted.</p>}
            {status === 'unverified' && <p className="text-muted-foreground">This result has no recorded acceptance verdict. Its metrics alone do not establish that the control passed verification.</p>}
            {status === 'accepted' && <p className="text-muted-foreground">The reporting checks passed. No sampled verification was run.</p>}
            {verification && <p className="text-muted-foreground">Sampled on-target {pct(verification.refuse_target)}, near-miss {pct(verification.refuse_near_miss)} at the serving temperature.</p>}
          </Banner>
        </>
      ) : (
        <Banner tone="warn" title="No admissible control found.">
          <p className="text-muted-foreground">Every candidate failed a gate — see the reasons below. Try a wider m/τ grid, or a sharper category gate.</p>
        </Banner>
      )}

      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50 text-left text-xs uppercase tracking-wide text-muted-foreground">
            <tr>
              <th className="p-2">m</th>
              <th className="p-2">τ</th>
              <th className="p-2">on-target</th>
              <th className="p-2">near-miss</th>
              <th className="p-2">capability</th>
              <th className="p-2">verdict</th>
            </tr>
          </thead>
          <tbody>
            {trials.map((t, i) => {
              const isWinner = winner && t.m === winner.m && t.tau === winner.tau
              return (
                <tr key={i} className={`border-t border-border ${isWinner ? 'bg-green-500/10' : ''}`}>
                  <td className="p-2">{t.m}</td>
                  <td className="p-2">{t.tau}</td>
                  <td className="p-2">{pct(t.refuse_target)}</td>
                  <td className="p-2">{pct(t.refuse_near_miss)}</td>
                  <td className="p-2">{pct(t.factual_acc)}</td>
                  <td className="p-2">
                    <span className={t.admissible ? 'text-green-600 dark:text-green-400' : 'text-muted-foreground'}>
                      {REASON[t.reason] ?? t.reason}
                    </span>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
