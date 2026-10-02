import Link from 'next/link'
import Image from 'next/image'
import { useRouter } from 'next/router'
import {
  LayoutDashboard,
  PlayCircle,
  List,
  BarChart3,
  GitCompare,
  Settings,
  FileText,
  Layers,
  Brain,
  Scissors,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const primaryNavigation = [
  { name: 'Dashboard', href: '/dashboard', icon: LayoutDashboard },
]

const navigation = [
  { label: 'Prompts and evaluation', items: [
    { name: 'Prompts', href: '/prompts', icon: FileText },
    { name: 'Benchmarks', href: '/benchmarks', icon: BarChart3 },
    { name: 'Create Test', href: '/create-test', icon: PlayCircle },
    { name: 'Test Runs', href: '/test-runs', icon: List },
    { name: 'Suites', href: '/suite', icon: Layers },
    { name: 'Comparisons', href: '/compare', icon: GitCompare },
  ] },
  { label: 'Models', items: [
    { name: 'Models', href: '/models', icon: Layers },
    { name: 'Neurosurgery', href: '/weights', icon: Scissors },
    { name: 'Interpretability', href: '/interp', icon: Brain },
  ] },
]

function NavLink({
  name,
  href,
  icon: Icon,
  isActive,
}: {
  name: string
  href: string
  icon: typeof LayoutDashboard
  isActive: boolean
}) {
  return (
    <Link
      href={href}
      aria-current={isActive ? 'page' : undefined}
      className={cn(
        'flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors',
        isActive
          ? 'bg-primary text-primary-foreground'
          : 'text-muted-foreground hover:bg-muted hover:text-foreground'
      )}
    >
      <Icon className="h-5 w-5" />
      {name}
    </Link>
  )
}

export function Sidebar() {
  const router = useRouter()

  const isActive = (href: string) =>
    router.pathname === href ||
    (href === '/models' && router.pathname === '/') ||
    router.pathname.startsWith(`${href}/`)

  return (
    <aside aria-label="Main navigation" className="flex h-screen w-64 shrink-0 flex-col border-r bg-card">
      <Link href="/" className="flex h-16 shrink-0 items-center gap-3 border-b px-6 hover:bg-muted/50 transition-colors cursor-pointer">
        <Image
          src="/vivalogo.png?v=2"
          alt="Viva Securis Logo"
          width={56}
          height={56}
          className="flex-shrink-0 object-contain"
          priority
          unoptimized
        />
        <h1 className="text-xl font-bold whitespace-nowrap">AI Asylum</h1>
      </Link>
      <nav className="min-h-0 flex-1 space-y-5 overflow-y-auto px-3 py-4">
        <div className="space-y-1">
          {primaryNavigation.map((item) => (
            <NavLink
              key={item.name}
              name={item.name}
              href={item.href}
              icon={item.icon}
              isActive={isActive(item.href)}
            />
          ))}
        </div>
        {navigation.map((section) => (
          <div key={section.label} className="space-y-1">
            <p className="px-3 pb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              {section.label}
            </p>
            {section.items.map((item) => (
              <NavLink
                key={item.name}
                name={item.name}
                href={item.href}
                icon={item.icon}
                isActive={isActive(item.href)}
              />
            ))}
          </div>
        ))}
      </nav>
      <div className="shrink-0 border-t px-3 py-3">
        <NavLink
          name="Settings"
          href="/settings"
          icon={Settings}
          isActive={isActive('/settings')}
        />
      </div>
    </aside>
  )
}
