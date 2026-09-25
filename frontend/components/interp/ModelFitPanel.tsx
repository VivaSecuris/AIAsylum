import type { InterpPreflight } from '@/lib/api'

const gb = (value: number | null | undefined) =>
  typeof value === 'number' && Number.isFinite(value) ? `${value.toFixed(1)} GiB` : 'Unavailable'

export function ModelFitPanel({ result }: { result: InterpPreflight }) {
  return (
    <div className={`space-y-3 rounded-lg border p-4 ${result.ready ? 'border-emerald-500/50' : 'border-destructive/50'}`} aria-live="polite">
      <p className="text-sm font-medium">{result.ready ? 'Model configuration checked' : 'Resolve these checks before running'} · {result.device} · {result.dtype}</p>
      {result.errors?.map((message, index) => <p key={index} className="text-sm text-destructive">{message}</p>)}
      {!!result.models?.length && (
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="border-b text-muted-foreground">
              <tr><th className="py-2 pr-4">Model / architecture</th><th className="pr-4">Layers × width</th><th className="pr-4">Weights</th><th className="pr-4">Captures in host RAM</th><th>Estimated device memory</th></tr>
            </thead>
            <tbody>{result.models.map((model, index) => (
              <tr key={`${model.id}-${index}`} className="border-b last:border-0">
                <td className="py-2 pr-4"><span className="font-mono">{model.id}</span><span className="mt-1 block text-muted-foreground">{model.model_type} · ~{model.parameters_b?.toFixed(2)}B · {model.num_attention_heads} query / {model.num_key_value_heads} KV heads</span></td>
                <td className="whitespace-nowrap pr-4">{model.num_layers} × {model.hidden_size}</td>
                <td className="whitespace-nowrap pr-4">{gb(model.weights_gb)}</td>
                <td className="whitespace-nowrap pr-4">{gb(model.capture_gb)}</td>
                <td className="whitespace-nowrap">{gb(model.estimated_device_gb + (['cpu', 'mps'].includes(result.device) ? model.capture_gb : 0))}</td>
              </tr>
            ))}</tbody>
          </table>
        </div>
      )}
      {result.warnings?.map((message, index) => <p key={index} className="text-xs text-amber-700 dark:text-amber-300">{message}</p>)}
      <p className="text-xs text-muted-foreground">Estimates depend on model size, precision, sequence length and enabled captures. Other jobs and runtime overhead can change available memory.</p>
    </div>
  )
}
