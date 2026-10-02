import { useEffect, useMemo, useRef, useState, type RefObject } from 'react'
import GridLayout, { useContainerWidth } from 'react-grid-layout'
import { GripVertical, Plus, RotateCcw, X } from 'lucide-react'
import {
  DASHBOARD_COLUMNS,
  MAX_WIDGET_HEIGHT,
  displayDashboardLayout,
} from '@/lib/dashboard-layout'
import { useDashboardLayout } from './useDashboardLayout'
import type { DashboardWidgetData } from './DashboardWidgets'
import { widgetRegistry } from './widgetRegistry'

interface DashboardGridProps {
  data: DashboardWidgetData
  isLive: boolean
}

export function DashboardGrid({ data, isLive }: DashboardGridProps) {
  const { layout, ready, storageUnavailable, available, onLayoutChange, add, remove, reset } = useDashboardLayout()
  const [pickerOpen, setPickerOpen] = useState(false)
  const pickerRef = useRef<HTMLDivElement>(null)
  // The library owns ResizeObserver setup, animation frames, and unmount cleanup.
  const { width, containerRef, mounted } = useContainerWidth({ measureBeforeMount: true })
  const stacked = width < 768
  const gridLayout = useMemo(
    () => displayDashboardLayout(layout, stacked).map((item) => ({ ...item, maxH: MAX_WIDGET_HEIGHT })),
    [layout, stacked]
  )

  useEffect(() => {
    if (!pickerOpen) return
    const onPointerDown = (event: PointerEvent) => {
      if (!pickerRef.current?.contains(event.target as Node)) setPickerOpen(false)
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setPickerOpen(false)
    }
    document.addEventListener('pointerdown', onPointerDown)
    document.addEventListener('keydown', onKeyDown)
    return () => {
      document.removeEventListener('pointerdown', onPointerDown)
      document.removeEventListener('keydown', onKeyDown)
    }
  }, [pickerOpen])

  return (
    <div ref={containerRef as RefObject<HTMLDivElement>} className="min-w-0 space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2" role="status">
          {isLive ? (
            <span className="inline-flex items-center gap-2 rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs font-medium text-emerald-700">
              <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
              Live updating
            </span>
          ) : (
            <span className="inline-flex items-center rounded-full border px-3 py-1 text-xs text-muted-foreground">
              Idle
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div ref={pickerRef} className="relative">
            <button
              type="button"
              aria-expanded={pickerOpen && available.length > 0}
              aria-controls="dashboard-chart-picker"
              onClick={() => setPickerOpen((open) => !open)}
              className="inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm hover:bg-muted disabled:cursor-not-allowed disabled:opacity-50"
              disabled={!ready || available.length === 0}
              title={available.length === 0 ? 'All charts are already on the dashboard' : 'Add a chart'}
            >
              <Plus className="h-4 w-4" />
              Add chart
            </button>
            {pickerOpen && available.length > 0 && (
              <div id="dashboard-chart-picker" className="absolute right-0 z-20 mt-2 max-h-96 w-64 max-w-[calc(100vw-3rem)] overflow-y-auto rounded-lg border bg-card p-2 shadow-lg">
                {available.map((widget) => (
                  <button
                    key={widget.id}
                    type="button"
                    className="block w-full rounded-md px-3 py-2 text-left text-sm hover:bg-muted"
                    onClick={() => {
                      add(widget.id)
                      setPickerOpen(false)
                    }}
                  >
                    <span className="font-medium">{widget.title}</span>
                    <span className="mt-0.5 block text-xs text-muted-foreground">{widget.description}</span>
                  </button>
                ))}
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={reset}
            disabled={!ready}
            className="inline-flex items-center gap-2 rounded-lg border px-3 py-2 text-sm hover:bg-muted disabled:opacity-50"
          >
            <RotateCcw className="h-4 w-4" />
            Reset layout
          </button>
        </div>
      </div>

      {storageUnavailable && (
        <p role="status" className="text-sm text-amber-700">Browser storage is unavailable. Layout changes will last for this visit.</p>
      )}
      {ready && mounted && stacked && layout.length > 0 && (
        <p className="text-xs text-muted-foreground">Charts stack on smaller screens. Use a wider window to drag and resize.</p>
      )}
      {!ready || !mounted ? (
        <p className="text-sm text-muted-foreground">Loading dashboard layout…</p>
      ) : layout.length === 0 ? (
        <div className="rounded-lg border border-dashed p-12 text-center text-sm text-muted-foreground">
          Your dashboard is empty. Add a chart or reset the layout to get started.
        </div>
      ) : (
        <GridLayout
          key={stacked ? 'stacked' : 'desktop'}
          className="dashboard-grid"
          layout={gridLayout}
          width={width}
          gridConfig={{ cols: DASHBOARD_COLUMNS, rowHeight: 36, margin: [12, 12], containerPadding: [0, 0] }}
          dragConfig={{ enabled: !stacked, handle: '.dashboard-drag-handle', cancel: 'button, a, input, select, textarea' }}
          resizeConfig={{ enabled: !stacked }}
          onLayoutChange={stacked ? undefined : onLayoutChange}
        >
          {layout.map((item) => {
            const widget = widgetRegistry[item.i]
            return (
              <div key={item.i} data-widget-id={item.i} className="flex min-w-0 flex-col overflow-hidden rounded-lg border bg-card shadow-sm">
                <div className={`dashboard-drag-handle flex shrink-0 items-center justify-between gap-2 border-b bg-muted/40 px-3 py-2 ${stacked ? '' : 'cursor-grab active:cursor-grabbing'}`}>
                  <div className="flex min-w-0 items-center gap-2">
                    {!stacked && <GripVertical aria-hidden="true" className="h-4 w-4 shrink-0 text-muted-foreground" />}
                    <h2 className="truncate text-sm font-medium">{widget.title}</h2>
                  </div>
                  <button
                    type="button"
                    aria-label={`Remove ${widget.title}`}
                    className="rounded p-1 text-muted-foreground hover:bg-background hover:text-foreground"
                    onClick={() => remove(item.i)}
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
                <div className="min-h-0 min-w-0 flex-1 overflow-auto p-4">
                  {widget.render(data)}
                </div>
              </div>
            )
          })}
        </GridLayout>
      )}
    </div>
  )
}
