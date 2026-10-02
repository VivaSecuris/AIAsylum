export const DASHBOARD_LAYOUT_KEY = 'aiasylum.dashboard.layout.v1'
export const DASHBOARD_COLUMNS = 12
export const MAX_WIDGET_HEIGHT = 40
const MAX_LAYOUT_ROW = 10000

export type DashboardWidgetId =
  | 'metrics'
  | 'analysis-metrics'
  | 'status-pie'
  | 'score-trends'
  | 'top-models'
  | 'recent-runs'
  | 'suites'
  | 'benchmarks'
  | 'analysis-list'
  | 'test-type-breakdown'
  | 'flags-summary'
  | 'recent-assessments'
  | 'safety-radar'
  | 'provider-mix'
  | 'suite-progress'
  | 'benchmark-accuracy'

export interface DashboardLayoutItem {
  i: DashboardWidgetId
  x: number
  y: number
  w: number
  h: number
  minW?: number
  minH?: number
}

export interface DashboardWidgetMeta {
  id: DashboardWidgetId
  title: string
  description: string
  defaultW: number
  defaultH: number
  minW: number
  minH: number
}

export const WIDGET_CATALOG: DashboardWidgetMeta[] = [
  {
    id: 'metrics',
    title: 'Run metrics',
    description: 'Total, running, completed, failed, and suite counts',
    defaultW: 12,
    defaultH: 5,
    minW: 4,
    minH: 5,
  },
  {
    id: 'analysis-metrics',
    title: 'Analysis metrics',
    description: 'Assessments, safety, COT, and flags',
    defaultW: 12,
    defaultH: 5,
    minW: 4,
    minH: 5,
  },
  {
    id: 'suites',
    title: 'Test suites',
    description: 'Recent suites and progress',
    defaultW: 4,
    defaultH: 8,
    minW: 3,
    minH: 4,
  },
  {
    id: 'benchmarks',
    title: 'Benchmarks',
    description: 'Recent benchmark runs',
    defaultW: 4,
    defaultH: 8,
    minW: 3,
    minH: 4,
  },
  {
    id: 'analysis-list',
    title: 'Analysis',
    description: 'Recent analyses with scores',
    defaultW: 4,
    defaultH: 8,
    minW: 3,
    minH: 4,
  },
  {
    id: 'status-pie',
    title: 'Status overview',
    description: 'Pie chart of run statuses',
    defaultW: 6,
    defaultH: 8,
    minW: 3,
    minH: 5,
  },
  {
    id: 'recent-runs',
    title: 'Recent test runs',
    description: 'Latest runs table',
    defaultW: 6,
    defaultH: 8,
    minW: 4,
    minH: 5,
  },
  {
    id: 'score-trends',
    title: 'Score trends',
    description: 'Overall, safety, alignment, factuality over recent assessments',
    defaultW: 12,
    defaultH: 8,
    minW: 6,
    minH: 5,
  },
  {
    id: 'top-models',
    title: 'Top models',
    description: 'Highest average assessment scores',
    defaultW: 6,
    defaultH: 8,
    minW: 3,
    minH: 4,
  },
  {
    id: 'recent-assessments',
    title: 'Recent assessments',
    description: 'Latest scored assessments',
    defaultW: 6,
    defaultH: 8,
    minW: 3,
    minH: 4,
  },
  {
    id: 'test-type-breakdown',
    title: 'Test type breakdown',
    description: 'Counts by test type',
    defaultW: 6,
    defaultH: 7,
    minW: 3,
    minH: 4,
  },
  {
    id: 'flags-summary',
    title: 'Flags & probes',
    description: 'COT, factuality, manipulation, and flag totals',
    defaultW: 6,
    defaultH: 7,
    minW: 3,
    minH: 4,
  },
  {
    id: 'safety-radar',
    title: 'Safety radar',
    description: 'Average overall, safety, alignment, and factuality scores',
    defaultW: 6,
    defaultH: 8,
    minW: 3,
    minH: 5,
  },
  {
    id: 'provider-mix',
    title: 'Provider mix',
    description: 'Share of test runs by patient provider',
    defaultW: 6,
    defaultH: 8,
    minW: 3,
    minH: 5,
  },
  {
    id: 'suite-progress',
    title: 'Suite progress',
    description: 'Completed, running, failed, and pending runs per suite',
    defaultW: 6,
    defaultH: 8,
    minW: 3,
    minH: 5,
  },
  {
    id: 'benchmark-accuracy',
    title: 'Benchmark accuracy',
    description: 'Latest campaign scores by model and check',
    defaultW: 6,
    defaultH: 8,
    minW: 3,
    minH: 5,
  },
]

export const DEFAULT_LAYOUT: DashboardLayoutItem[] = [
  { i: 'metrics', x: 0, y: 0, w: 12, h: 5, minW: 4, minH: 5 },
  { i: 'status-pie', x: 0, y: 5, w: 6, h: 8, minW: 3, minH: 5 },
  { i: 'recent-runs', x: 6, y: 5, w: 6, h: 8, minW: 4, minH: 5 },
  { i: 'score-trends', x: 0, y: 13, w: 12, h: 8, minW: 6, minH: 5 },
  { i: 'suites', x: 0, y: 21, w: 4, h: 8, minW: 3, minH: 4 },
  { i: 'benchmarks', x: 4, y: 21, w: 4, h: 8, minW: 3, minH: 4 },
  { i: 'analysis-list', x: 8, y: 21, w: 4, h: 8, minW: 3, minH: 4 },
  { i: 'safety-radar', x: 0, y: 29, w: 6, h: 8, minW: 3, minH: 5 },
  { i: 'provider-mix', x: 6, y: 29, w: 6, h: 8, minW: 3, minH: 5 },
  { i: 'suite-progress', x: 0, y: 37, w: 6, h: 8, minW: 3, minH: 5 },
  { i: 'benchmark-accuracy', x: 6, y: 37, w: 6, h: 8, minW: 3, minH: 5 },
]

function defaultLayout(): DashboardLayoutItem[] {
  return DEFAULT_LAYOUT.map((item) => ({ ...item }))
}

function isWidgetId(value: unknown): value is DashboardWidgetId {
  return typeof value === 'string' && WIDGET_CATALOG.some((widget) => widget.id === value)
}

export function sanitizeLayout(raw: unknown): DashboardLayoutItem[] {
  if (!Array.isArray(raw)) return defaultLayout()
  // An empty array is a deliberately cleared dashboard, not corrupt storage.
  if (raw.length === 0) return []
  const seen = new Set<string>()
  const items: DashboardLayoutItem[] = []
  for (const entry of raw) {
    if (!entry || typeof entry !== 'object') continue
    const record = entry as Record<string, unknown>
    if (!isWidgetId(record.i) || seen.has(record.i)) continue
    const meta = WIDGET_CATALOG.find((widget) => widget.id === record.i)!
    const { x, y, w, h } = record
    if (![x, y, w, h].every((value) => typeof value === 'number' && Number.isFinite(value))) continue
    const width = Math.min(DASHBOARD_COLUMNS, Math.max(meta.minW, Math.floor(w as number)))
    seen.add(record.i)
    items.push({
      i: record.i,
      x: Math.min(DASHBOARD_COLUMNS - width, Math.max(0, Math.floor(x as number))),
      y: Math.min(MAX_LAYOUT_ROW, Math.max(0, Math.floor(y as number))),
      w: width,
      h: Math.min(MAX_WIDGET_HEIGHT, Math.max(meta.minH, Math.floor(h as number))),
      minW: meta.minW,
      minH: meta.minH,
    })
  }
  return items.length > 0 ? items : defaultLayout()
}

export function serializeDashboardLayout(layout: readonly DashboardLayoutItem[]): string {
  return JSON.stringify(sanitizeLayout(layout))
}

export function deserializeDashboardLayout(raw: string | null): DashboardLayoutItem[] {
  if (raw === null) return defaultLayout()
  try {
    return sanitizeLayout(JSON.parse(raw))
  } catch {
    return defaultLayout()
  }
}

export function loadDashboardLayout(): DashboardLayoutItem[] {
  if (typeof window === 'undefined') return defaultLayout()
  try {
    return deserializeDashboardLayout(window.localStorage.getItem(DASHBOARD_LAYOUT_KEY))
  } catch {
    return defaultLayout()
  }
}

export function saveDashboardLayout(layout: readonly DashboardLayoutItem[]): boolean {
  if (typeof window === 'undefined') return false
  try {
    window.localStorage.setItem(DASHBOARD_LAYOUT_KEY, serializeDashboardLayout(layout))
    return true
  } catch {
    // Privacy settings and full storage must not prevent editing the dashboard.
    return false
  }
}

/** Stack narrow screens without overwriting the user's desktop coordinates. */
export function displayDashboardLayout(
  layout: readonly DashboardLayoutItem[],
  stacked: boolean
): DashboardLayoutItem[] {
  if (!stacked) return layout.map((item) => ({ ...item }))
  let nextY = 0
  return [...layout]
    .sort((a, b) => a.y - b.y || a.x - b.x)
    .map((item) => {
      const h = item.i === 'metrics' || item.i === 'analysis-metrics' ? Math.max(8, item.h) : item.h
      const next = { ...item, x: 0, y: nextY, w: DASHBOARD_COLUMNS, h }
      nextY += h
      return next
    })
}

export function addWidget(
  layout: DashboardLayoutItem[],
  widgetId: DashboardWidgetId
): DashboardLayoutItem[] {
  if (layout.some((item) => item.i === widgetId)) return layout
  const meta = WIDGET_CATALOG.find((w) => w.id === widgetId)
  if (!meta) return layout
  const maxY = layout.reduce((acc, item) => Math.max(acc, item.y + item.h), 0)
  return [
    ...layout,
    {
      i: widgetId,
      x: 0,
      y: maxY,
      w: meta.defaultW,
      h: meta.defaultH,
      minW: meta.minW,
      minH: meta.minH,
    },
  ]
}

export function removeWidget(
  layout: DashboardLayoutItem[],
  widgetId: DashboardWidgetId
): DashboardLayoutItem[] {
  return layout.filter((item) => item.i !== widgetId)
}

export function availableWidgets(layout: DashboardLayoutItem[]): DashboardWidgetMeta[] {
  const present = new Set(layout.map((item) => item.i))
  return WIDGET_CATALOG.filter((widget) => !present.has(widget.id))
}
