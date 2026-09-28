import Link from 'next/link'
import type { ReactNode } from 'react'

/** Use the recorded provider; model names alone are not globally unique. */
export function ModelChatLink({ provider, model, children, className = '' }: {
  provider?: string | null; model?: string | null; children?: ReactNode; className?: string
}) {
  if (!provider || !model) return <span className={className}>{children || model}</span>
  return <Link href={{ pathname: '/models/chat', query: { provider, model } }}
    className={`hover:underline underline-offset-2 ${className}`}
    onClick={(event) => event.stopPropagation()} title={`Interact with ${provider} / ${model}`}>
    {children || model}
  </Link>
}
