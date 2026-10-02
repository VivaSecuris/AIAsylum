import type { ReactNode } from 'react'
import { WIDGET_CATALOG, type DashboardWidgetId, type DashboardWidgetMeta } from '@/lib/dashboard-layout'
import { renderDashboardWidget, type DashboardWidgetData } from './DashboardWidgets'

export interface DashboardWidgetRegistration extends DashboardWidgetMeta {
  render: (data: DashboardWidgetData) => ReactNode
}

// Keep React renderers here; persisted layouts and catalog metadata stay usable
// independently of React in dashboard-layout.ts.
export const widgetRegistry = Object.fromEntries(
  WIDGET_CATALOG.map((metadata) => [
    metadata.id,
    {
      ...metadata,
      render: (data: DashboardWidgetData) => renderDashboardWidget(metadata.id, data),
    },
  ])
) as Record<DashboardWidgetId, DashboardWidgetRegistration>
