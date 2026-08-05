'use client'

import { useEffect, useState } from 'react'
import { LayoutDashboard, Package, Cpu, GitBranch } from 'lucide-react'
import { listNodes } from '@/lib/api'

export type Page = 'dashboard' | 'models' | 'inference' | 'topology'

interface SidebarProps {
  currentPage: Page
  onNavigate: (page: Page) => void
}

const navItems: { id: Page; label: string; icon: typeof LayoutDashboard }[] = [
  { id: 'dashboard', label: 'Dashboard', icon: LayoutDashboard },
  { id: 'models', label: 'Models', icon: Package },
  { id: 'inference', label: 'Inference', icon: Cpu },
  { id: 'topology', label: 'Topology', icon: GitBranch },
]

const POLL_INTERVAL_MS = 3000

export default function Sidebar({ currentPage, onNavigate }: SidebarProps) {
  const [nodeCount, setNodeCount] = useState(0)
  const [activeCount, setActiveCount] = useState(0)

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const nodes = await listNodes()
        if (cancelled) return
        setNodeCount(nodes.length)
        setActiveCount(nodes.filter((n) => n.status === 'available').length)
      } catch {
        if (!cancelled) {
          setNodeCount(0)
          setActiveCount(0)
        }
      }
    }

    poll()
    const interval = setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  return (
    <aside className="w-56 bg-background border-r border-border flex flex-col">
      <div className="p-6 border-b border-border">
        <div className="text-sm font-medium text-foreground">π-Cluster</div>
        <div className="text-xs text-muted-foreground mt-1">
          {nodeCount} node{nodeCount === 1 ? '' : 's'} • {activeCount > 0 ? 'Active' : 'Idle'}
        </div>
      </div>

      <nav className="flex-1 p-4 space-y-1">
        {navItems.map((item) => {
          const Icon = item.icon
          const isActive = currentPage === item.id
          return (
            <button
              key={item.id}
              onClick={() => onNavigate(item.id)}
              className={`w-full flex items-center gap-3 px-3 py-2 text-sm transition-colors ${
                isActive
                  ? 'text-primary font-medium'
                  : 'text-muted-foreground hover:text-foreground'
              }`}
            >
              <Icon size={16} />
              <span>{item.label}</span>
            </button>
          )
        })}
      </nav>
    </aside>
  )
}
