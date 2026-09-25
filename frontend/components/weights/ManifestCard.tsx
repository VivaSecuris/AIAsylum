import type { EditedModelDetail } from '@/lib/api'
import { formatDateTime } from '@/lib/utils'

// Ordered as `aiasylum weights info` prints it, so the page and the terminal
// can be compared line by line.
const FIELDS: Array<[string, string]> = [
  ['source_model', 'Source model'],
  ['method', 'Method'],
  ['objective', 'Objective'],
  ['beta', 'Beta'],
  ['direction_layer', 'Direction layer'],
  ['direction_auc', 'Direction AUC'],
  ['split_hash', 'Split hash'],
  ['architecture', 'Architecture'],
  ['matrices_edited', 'Matrices edited'],
  ['embeddings_tied', 'Embeddings tied'],
  ['embeddings_edited', 'Embeddings edited'],
  ['mean_relative_change', 'Mean relative change'],
  ['model_type', 'Model type'],
  ['coverage_verified', 'Coverage verified'],
  ['moe_layers', 'MoE layers'],
  ['expert_matrices', 'Expert matrices'],
  ['shared_expert_matrices', 'Shared-expert matrices'],
  ['created_at', 'Created'],
  ['notes', 'Notes'],
]

function render(value: any): string {
  if (value === null || value === undefined || value === '') return '—'
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (typeof value === 'number') return String(value)
  return String(value)
}

export function ManifestCard({ manifest }: { manifest: Record<string, any> }) {
  const extra = manifest.extra || {}
  const merged: Record<string, any> = {
    ...manifest,
    method: manifest.method,
    objective: extra.objective,
  }

  return (
    <div className="rounded-lg border bg-card p-6 shadow-sm">
      <h2 className="mb-1 text-lg font-semibold">Surgery manifest</h2>
      <p className="mb-4 text-sm text-muted-foreground">
        Written into the model directory as <code className="text-xs">asylum_surgery.json</code>,
        and carried on every response the model produces — so a test run traces back to the
        exact edit that made its patient.
      </p>

      <dl className="grid grid-cols-1 gap-x-8 gap-y-2 sm:grid-cols-2">
        {FIELDS.map(([key, label]) => {
          const value = merged[key]
          if (value === undefined && !['method', 'objective'].includes(key)) return null
          const emphasise = key === 'mean_relative_change' || key === 'embeddings_tied'
          return (
            <div key={key} className="flex justify-between gap-4 border-b py-1 text-sm last:border-0">
              <dt className="text-muted-foreground">{label}</dt>
              <dd className={`text-right font-mono text-xs ${emphasise ? 'font-semibold' : ''}`}>
                {key === 'created_at' && value ? formatDateTime(value) : render(value)}
              </dd>
            </div>
          )
        })}
      </dl>

      {manifest.mean_relative_change === 0 && (
        <p className="mt-4 rounded border border-yellow-500/50 bg-yellow-50 p-3 text-xs dark:bg-yellow-950">
          A mean relative change of zero means the weights are bit-identical to the source.
          That is expected at β=1, which is the control; at any other β it means the edit did
          not land.
        </p>
      )}

      {manifest.embeddings_tied && (
        <p className="mt-3 rounded border border-yellow-500/50 bg-yellow-50 p-3 text-xs dark:bg-yellow-950">
          This model ties <code>lm_head</code> to <code>embed_tokens</code>, so editing the
          embedding table also edited the unembedding. When comparing against the baseline,
          read the overlap of the top-k predictions rather than comparing probabilities: the
          two sides decode through different unembeddings.
        </p>
      )}

      {manifest.coverage_verified === false && (
        <p className="mt-3 rounded border border-yellow-500/50 bg-yellow-50 p-3 text-xs dark:bg-yellow-950">
          This edit is partial by design: it touched {render(manifest.matrices_edited)} matrices in
          chosen experts and left every other residual writer alone. The untouched experts still
          write the direction whenever the router picks them, so read this model as an expert-level
          intervention, not as a removal of the direction.
        </p>
      )}
    </div>
  )
}

export function ProvenanceCard({ detail }: { detail: EditedModelDetail }) {
  const { test_runs: testRuns, interp_runs: interpRuns } = detail

  return (
    <div className="rounded-lg border bg-card p-6 shadow-sm">
      <h2 className="mb-1 text-lg font-semibold">Provenance</h2>
      <p className="mb-4 text-sm text-muted-foreground">
        What produced this model, and what has been run against it.
      </p>

      <div className="space-y-4 text-sm">
        <Row label="Derived from">
          {detail.direction_run ? (
            <a className="text-primary hover:underline" href={`/weights/${detail.direction_run.id}`}>
              direction run #{detail.direction_run.id}
            </a>
          ) : (
            <span className="text-muted-foreground">
              not recorded — this model was produced outside the app
            </span>
          )}
        </Row>

        <Row label="Causality checked by">
          {detail.sweep_runs?.length ? (
            <span className="flex flex-wrap gap-2">
              {detail.sweep_runs.map((s) => (
                <a key={s.id} className="text-primary hover:underline" href={`/weights/${s.id}`}>
                  sweep #{s.id}
                </a>
              ))}
            </span>
          ) : (
            <span className="text-muted-foreground">no sweep recorded</span>
          )}
        </Row>

        <Row label="Test runs">
          {testRuns?.length ? (
            <span className="flex flex-wrap gap-2">
              {testRuns.map((t) => (
                <a key={t.id} className="text-primary hover:underline" href={`/test-runs/${t.id}`}>
                  #{t.id} ({t.role})
                </a>
              ))}
            </span>
          ) : (
            <span className="text-muted-foreground">none recorded</span>
          )}
        </Row>

        <Row label="Interpretability runs">
          {interpRuns?.length ? (
            <span className="flex flex-wrap gap-2">
              {interpRuns.map((i) => (
                <a key={i.id} className="text-primary hover:underline" href={`/interp/${i.id}`}>
                  #{i.id} ({i.mode})
                </a>
              ))}
            </span>
          ) : (
            <span className="text-muted-foreground">none recorded</span>
          )}
        </Row>
      </div>

      {detail.path_truncation_caveat && (
        <p className="mt-4 rounded border border-yellow-500/50 bg-yellow-50 p-3 text-xs dark:bg-yellow-950">
          This model&apos;s path is longer than the 100 characters a test run stores, so the
          lookup above can miss runs that did use it. Read an empty list as &quot;nothing
          recorded&quot; rather than as proof it was never tested.
        </p>
      )}
    </div>
  )
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-wrap items-baseline justify-between gap-2 border-b pb-2 last:border-0">
      <span className="text-muted-foreground">{label}</span>
      <span className="text-right">{children}</span>
    </div>
  )
}
