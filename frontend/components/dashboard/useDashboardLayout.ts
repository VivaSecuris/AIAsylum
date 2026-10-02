import { useCallback, useEffect, useRef, useState } from 'react'
import type { Layout } from 'react-grid-layout'
import {
  DEFAULT_LAYOUT,
  DashboardLayoutItem,
  DashboardWidgetId,
  addWidget,
  availableWidgets,
  loadDashboardLayout,
  removeWidget,
  saveDashboardLayout,
  sanitizeLayout,
  serializeDashboardLayout,
} from '@/lib/dashboard-layout'

export function useDashboardLayout() {
  const [layout, setLayout] = useState<DashboardLayoutItem[]>(() =>
    DEFAULT_LAYOUT.map((item) => ({ ...item }))
  )
  const currentLayout = useRef(layout)
  const [ready, setReady] = useState(false)
  const [storageUnavailable, setStorageUnavailable] = useState(false)

  useEffect(() => {
    const saved = loadDashboardLayout()
    currentLayout.current = saved
    setLayout(saved)
    setReady(true)
  }, [])

  const persist = useCallback((update: (current: DashboardLayoutItem[]) => DashboardLayoutItem[]) => {
    const previous = currentLayout.current
    const cleaned = sanitizeLayout(update(previous))
    if (serializeDashboardLayout(previous) === serializeDashboardLayout(cleaned)) return
    // Update the ref synchronously so back-to-back grid and picker events cannot
    // restore a widget removed by an earlier event in the same render.
    currentLayout.current = cleaned
    setLayout(cleaned)
    setStorageUnavailable(!saveDashboardLayout(cleaned))
  }, [])

  const onLayoutChange = useCallback(
    (next: Layout) => {
      if (!ready) return
      persist((current) => {
        const byId = new Map(next.map((item) => [item.i, item]))
        // Only explicit add/remove actions change membership. Grid callbacks may
        // still contain children from the preceding render.
        return current.map((previous) => {
          const item = byId.get(previous.i)
          return item
            ? { ...previous, x: item.x, y: item.y, w: item.w, h: item.h }
            : previous
        })
      })
    },
    [persist, ready]
  )

  const add = useCallback(
    (widgetId: DashboardWidgetId) => persist((current) => addWidget(current, widgetId)),
    [persist]
  )

  const remove = useCallback(
    (widgetId: DashboardWidgetId) => persist((current) => removeWidget(current, widgetId)),
    [persist]
  )

  const reset = useCallback(() => {
    persist(() => DEFAULT_LAYOUT.map((item) => ({ ...item })))
  }, [persist])

  return {
    layout,
    ready,
    storageUnavailable,
    available: availableWidgets(layout),
    onLayoutChange,
    add,
    remove,
    reset,
  }
}
