import Link from 'next/link'
import Image from 'next/image'
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
  Layers,
} from 'lucide-react'
import { cn } from '@/lib/utils'

const navigation = [
  { name: 'Dashboard', href: '/', icon: LayoutDashboard },
  { name: 'Create Test', href: '/create-test', icon: PlayCircle },
  { name: 'Test Runs', href: '/test-runs', icon: List },
  { name: 'Suite', href: '/suite', icon: Layers },
  { name: 'Prompts', href: '/prompts', icon: FileText },
  { name: 'Models', href: '/compare', icon: GitCompare },
]

export function Sidebar() {
  const router = useRouter()

  return (
    <div className="flex h-screen w-64 flex-col border-r bg-card">
      <Link href="/" className="flex h-16 items-center gap-3 border-b px-6 hover:bg-muted/50 transition-colors cursor-pointer">
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
