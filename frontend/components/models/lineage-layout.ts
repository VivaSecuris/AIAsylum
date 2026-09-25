import type { ModelLineageEdge, ModelLineageNode } from '@/lib/model-catalog'

export const NODE_WIDTH = 256
export const NODE_HEIGHT = 128
export const COLUMN_GAP = 112
export const ROW_GAP = 42
const PADDING = 28

export function relatedLineage(nodes: ModelLineageNode[], edges: ModelLineageEdge[], modelRef?: string) {
  if (!modelRef) return { nodes, edges }
  const seeds = nodes.filter((node) => node.kind === 'model' && node.model_ref === modelRef).map((node) => node.id)
  // Ancestors and descendants are separate traversals: traversing an undirected
  // component would pull in every sibling experiment using the same base model.
  const included = new Set(seeds)
  for (const direction of ['ancestors', 'descendants']) {
    const visited = new Set(seeds)
    const queue = [...seeds]
    while (queue.length) {
      const id = queue.shift()!
      for (const edge of edges) {
        const next = direction === 'ancestors' ? edge.target === id && edge.source : edge.source === id && edge.target
        if (next && !visited.has(next)) { visited.add(next); included.add(next); queue.push(next) }
      }
    }
  }
  return { nodes: nodes.filter((node) => included.has(node.id)), edges: edges.filter((edge) => included.has(edge.source) && included.has(edge.target)) }
}

function byTime(a: ModelLineageNode, b: ModelLineageNode) {
  const timestamp = (node: ModelLineageNode) => node.created_at ? Date.parse(node.created_at) || 0 : 0
  return timestamp(a) - timestamp(b) || a.id.localeCompare(b.id, undefined, { numeric: true })
}

/** Position only recorded edges. Timestamps break ties, never create ancestry. */
export function layoutLineage(nodes: ModelLineageNode[], inputEdges: ModelLineageEdge[]) {
  const byId = new Map(nodes.map((node) => [node.id, node]))
  const edges = inputEdges.filter((edge) => byId.has(edge.source) && byId.has(edge.target))
  const incoming = new Map(nodes.map((node) => [node.id, [] as string[]]))
  const outgoing = new Map(nodes.map((node) => [node.id, [] as string[]]))
  for (const edge of edges) { incoming.get(edge.target)!.push(edge.source); outgoing.get(edge.source)!.push(edge.target) }
  const remaining = new Map(nodes.map((node) => [node.id, incoming.get(node.id)!.length]))
  const depth = new Map<string, number>()
  const ready = nodes.filter((node) => !remaining.get(node.id)).sort(byTime)
  while (ready.length) {
    const node = ready.shift()!
    const parents = incoming.get(node.id)!
    depth.set(node.id, parents.length ? Math.max(...parents.map((parent) => depth.get(parent)!)) + 1 : 0)
    for (const target of outgoing.get(node.id)!) {
      remaining.set(target, remaining.get(target)! - 1)
      if (!remaining.get(target)) { ready.push(byId.get(target)!); ready.sort(byTime) }
    }
  }
  const cyclic = nodes.filter((node) => !depth.has(node.id)).sort(byTime)
  const unresolvedColumn = Math.max(-1, ...Array.from(depth.values())) + 1
  for (const node of cyclic) depth.set(node.id, unresolvedColumn)
  const columns = Array.from({ length: Math.max(-1, ...Array.from(depth.values())) + 1 }, () => [] as ModelLineageNode[])
  for (const node of nodes) columns[depth.get(node.id)!].push(node)
  const positions = new Map<string, { x: number; y: number }>()
  columns.forEach((column, columnIndex) => {
    const parentCenter = (node: ModelLineageNode) => {
      const known = incoming.get(node.id)!.map((id) => positions.get(id)?.y).filter((value): value is number => value != null)
      return known.length ? known.reduce((sum, value) => sum + value, 0) / known.length : Number.MAX_SAFE_INTEGER
    }
    column.sort((a, b) => parentCenter(a) - parentCenter(b) || byTime(a, b))
    column.forEach((node, row) => positions.set(node.id, { x: PADDING + columnIndex * (NODE_WIDTH + COLUMN_GAP), y: PADDING + 32 + row * (NODE_HEIGHT + ROW_GAP) }))
  })
  return {
    positions, edges, columns,
    ordered: columns.flat(),
    cyclic: cyclic.map((node) => node.id),
    width: Math.max(1, columns.length) * (NODE_WIDTH + COLUMN_GAP) - COLUMN_GAP + PADDING * 2,
    height: Math.max(1, ...columns.map((column) => column.length)) * (NODE_HEIGHT + ROW_GAP) - ROW_GAP + PADDING * 2 + 32,
  }
}
