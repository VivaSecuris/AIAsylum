import type { PassMetrics, VerifyReport } from '@/lib/api'

const pct = (v: number | null | undefined) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

function decodingLabel(m: PassMetrics): string {
  if (!m.decoding || m.decoding === 'greedy') return 'greedy'
  return `T=${m.decoding.temperature}, top-p ${m.decoding.top_p}, seed ${m.decoding.seed ?? '—'}`
}

/**
 * What the written directory did when loaded back from disk.
 *
 * The search scores tensors on the accelerator; a test run loads a directory
 * through the provider's loader, possibly on another machine, and samples.
 * This card is the only evidence that the bytes on disk behave. A failed
 * report here means the run was not published.
 */
export function VerificationCard({ report }: { report: VerifyReport }) {
  const passes: Array<[string, PassMetrics | null]> = [
    ['greedy', report.greedy],
    ['sampled', report.sampled],
  ]
  return (
    <div className="overflow-hidden rounded-lg border bg-card shadow-sm">
      <div
        className={`border-b p-3 ${
          report.passed ? 'bg-emerald-50 dark:bg-emerald-950' : 'bg-destructive/10'
        }`}
      >
        <h2 className="text-sm font-semibold">
          {report.passed ? 'Verified from disk' : 'Failed verification from disk — not published'}
        </h2>
        <p className="text-xs text-muted-foreground">
          Reloaded through {report.loader.split('.').slice(-2).join('.')} on {report.device} in{' '}
          {report.dtype}, {report.n_prompts} held-out prompts, {report.capability_set} control,{' '}
          {report.checked_at}.
        </p>
        {!report.passed && (
          <ul className="mt-2 list-disc pl-5 text-xs">
            {report.reasons.map((r) => (
              <li key={r}>{r}</li>
            ))}
          </ul>
        )}
      </div>
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b text-left text-xs text-muted-foreground">
            <th className="px-3 py-2 font-medium">pass</th>
            <th className="px-3 py-2 font-medium">refusal</th>
            <th className="px-3 py-2 font-medium">factual</th>
            <th className="px-3 py-2 font-medium">drift</th>
            <th className="px-3 py-2 font-medium">degenerate</th>
            <th className="px-3 py-2 font-medium">verdict</th>
          </tr>
        </thead>
        <tbody>
          {passes.map(([label, m]) =>
            m ? (
              <tr key={label} className="border-b last:border-0">
                <td className="px-3 py-2">
                  <span className="font-medium">{label}</span>
                  <span className="block text-xs text-muted-foreground">{decodingLabel(m)}</span>
                </td>
                <td className="px-3 py-2 font-mono">{pct(m.refuse_harmful)}</td>
                <td className="px-3 py-2 font-mono">
                  {pct(m.factual_acc)}
                  {m.factual_drop != null && (
                    <span className="ml-1 text-xs text-muted-foreground">
                      ({m.factual_drop > 0 ? '−' : '+'}
                      {(Math.abs(m.factual_drop) * 100).toFixed(1)} pts)
                    </span>
                  )}
                </td>
                <td className={`px-3 py-2 font-mono ${m.drifted ? 'text-purple-700 dark:text-purple-300' : ''}`}>
                  {pct(m.language_drift)}
                </td>
                <td className="px-3 py-2">{m.degenerate ? 'yes' : 'no'}</td>
                <td className="px-3 py-2 text-xs">{m.verdict ?? '—'}</td>
              </tr>
            ) : (
              <tr key={label} className="border-b last:border-0 text-muted-foreground">
                <td className="px-3 py-2">{label}</td>
                <td className="px-3 py-2 text-xs" colSpan={5}>
                  not run
                </td>
              </tr>
            )
          )}
        </tbody>
      </table>
      <p className="border-t px-3 py-2 text-[11px] text-muted-foreground">
        {Object.keys(report.hashes).length} weight file{Object.keys(report.hashes).length === 1 ? '' : 's'} hashed
        (SHA-256, recorded in the manifest as <code>weights_sha256</code>)
        {report.hash_check != null && ` · matches the manifest: ${report.hash_check ? 'yes' : 'NO'}`}. Re-check any
        time with <code>aiasylum weights verify --model &lt;dir&gt; --check-hashes</code>.
      </p>
    </div>
  )
}
