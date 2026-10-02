type Finding = {
  kind: 'unsupported_identity_claim' | 'declared_roleplay' | 'uncertain'
  turn_number: number
  quote: string
  explanation: string
  confidence: number
  doctor_reinforcement?: { turn_number: number; quote: string }[]
}

export function IdentityFindings({ review }: { review?: { status?: string; findings?: Finding[] } }) {
  if (!review) return null
  const findings = Array.isArray(review.findings) ? review.findings : []
  return <section aria-label="Identity and hallucination review" className="mt-4 space-y-3 rounded border p-4">
    <h3 className="font-semibold">Identity and hallucination review</h3>
    <p className="text-xs text-muted-foreground">Evaluator judgments based on recorded system instructions and quoted responses. Explicit fictional roleplay is distinguished from unsupported claims about the AI itself.</p>
    {review.status !== 'reviewed' ? <p className="text-sm">Identity review was unavailable. This is not evidence that the conversation contained no hallucinations.</p>
      : !findings.length ? <p className="text-sm">The evaluator returned no supported identity findings.</p>
      : findings.map((finding, index) => <article key={index} className="space-y-2 rounded border bg-muted/20 p-3">
        <p className="text-sm font-medium">{finding.kind === 'unsupported_identity_claim' ? 'Unsupported identity claim — possible hallucination' : finding.kind === 'declared_roleplay' ? 'Declared fictional roleplay' : 'Identity claim needs clarification'} · Patient turn {finding.turn_number}</p>
        <blockquote className="whitespace-pre-wrap break-words border-l-2 pl-3 text-sm">{finding.quote}</blockquote>
        <p className="whitespace-pre-wrap text-sm">{finding.explanation}</p>
        <p className="text-xs text-muted-foreground">Evaluator confidence: {Math.round(finding.confidence * 100)}%</p>
        {finding.doctor_reinforcement?.map((evidence, doctorIndex) => <div key={doctorIndex} className="space-y-1 rounded border p-2 text-sm">
          <p className="font-medium">Doctor may have reinforced the claim · turn {evidence.turn_number}</p>
          <blockquote className="whitespace-pre-wrap break-words">{evidence.quote}</blockquote>
        </div>)}
      </article>)}
  </section>
}
