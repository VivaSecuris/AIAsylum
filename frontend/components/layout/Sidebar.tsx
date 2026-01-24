import Link from 'next/link'
import { useRouter } from 'next/router'
import {
  LayoutDashboard,
  PlayCircle,
  List,
  BarChart3,
  GitCompare,
  TestTube,
  Settings,
  FileText,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Create Test', href: '/create-test', icon: PlayCircle },
  { name: 'Test Runs', href: '/test-runs', icon: List },
  { name: 'Prompts', href: '/prompts', icon: FileText },
  { name: 'Compare Models', href: '/compare', icon: GitCompare },
  { name: 'Benchmarks', href: '/benchmarks', icon: TestTube },
]

export function Sidebar() {
  const router = useRouter()

  return (
    <div className="flex h-screen w-64 flex-col border-r bg-card">
      <div className="flex h-16 items-center border-b px-6">
        <h1 className="text-xl font-bold">AI Asylum</h1>
      </div>
      <nav className="flex-1 space-y-1 px-3 py-4">
        {navigation.map((item) => {
          const isActive = router.pathname === item.href
          return (
            <Link
              key={item.name}
              href={item.href}
              className={cn(
                'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
                isActive
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-muted hover:text-foreground'
              )}
            >
              <item.icon className="h-5 w-5" />
              {item.name}
            </Link>
          )
        })}
      </nav>
    </div>
  )
}
