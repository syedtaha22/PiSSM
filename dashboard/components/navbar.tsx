'use client'

import { useEffect, useState } from 'react'
import { listNodes } from '@/lib/api'

export type Page = 'stats' | 'models' | 'inference' | 'topology'

interface NavbarProps {
  currentPage: Page
  onNavigate: (page: Page) => void
}

const NAV_LINKS: { id: Page; label: string }[] = [
  { id: 'stats', label: 'Stats' },
  { id: 'models', label: 'Models' },
  { id: 'inference', label: 'Inference' },
  { id: 'topology', label: 'Topology' },
]

const POLL_INTERVAL_MS = 3000

/**
 * Floating, centered pill nav: a rounded-full, backdrop-blurred shell with
 * a 3-column grid so the link group sits at the true visual center.
 * Navigates by callback (not next/link + usePathname) since this is a
 * single-page app that switches which page component is mounted, not
 * real routes.
 */
export function Navbar({ currentPage, onNavigate }: NavbarProps) {
  const [nodeCount, setNodeCount] = useState(0)
  const [activeCount, setActiveCount] = useState(0)
  const [mobileOpen, setMobileOpen] = useState(false)

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

  function handleNavigate(page: Page) {
    onNavigate(page)
    setMobileOpen(false)
  }

  return (
    <div className="flex justify-center px-3 pt-3 sm:px-4 sm:pt-4">
      <div className="w-full max-w-xl">
        <nav className="grid h-14 w-full grid-cols-[1fr_auto_1fr] items-center gap-2 rounded-full bg-card/60 pl-5 pr-5 shadow-nav backdrop-blur-lg sm:gap-4">
          <div className="justify-self-start truncate font-display text-[15px] font-bold tracking-tight text-foreground">
            PiSSM
          </div>

          <div className="hidden items-center justify-self-center gap-5 sm:flex">
            {NAV_LINKS.map((link) => (
              <button
                key={link.id}
                onClick={() => handleNavigate(link.id)}
                className={`whitespace-nowrap font-sans text-sm font-medium transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 ${
                  currentPage === link.id ? 'text-primary' : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {link.label}
              </button>
            ))}
          </div>

          <button
            type="button"
            onClick={() => setMobileOpen((v) => !v)}
            aria-expanded={mobileOpen}
            aria-label="Toggle navigation menu"
            className="flex h-11 w-11 items-center justify-center justify-self-center rounded-full text-foreground transition-colors hover:text-primary focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 sm:hidden"
          >
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={2}
              strokeLinecap="round"
              className="h-5 w-5"
              aria-hidden="true"
            >
              {mobileOpen ? <path d="M18 6L6 18M6 6l12 12" /> : <path d="M4 7h16M4 12h16M4 17h16" />}
            </svg>
          </button>

          <div
            className="hidden items-center gap-1.5 justify-self-end font-mono text-sm sm:flex"
            title={`${nodeCount} node${nodeCount === 1 ? '' : 's'} registered`}
          >
            <span
              className={`h-2 w-2 rounded-full ${activeCount > 0 ? 'bg-primary animate-pulse' : 'bg-muted-foreground'}`}
            />
            <span className="text-muted-foreground">{nodeCount}</span>
          </div>
        </nav>

        {mobileOpen && (
          <div className="mt-2 rounded-md border border-border bg-card p-2 shadow-nav backdrop-blur-lg sm:hidden">
            <div className="flex flex-col">
              {NAV_LINKS.map((link) => (
                <button
                  key={link.id}
                  onClick={() => handleNavigate(link.id)}
                  className={`rounded-sm px-3 py-2.5 text-left font-sans text-sm font-medium transition-colors cursor-pointer ${
                    currentPage === link.id
                      ? 'bg-accent text-accent-foreground'
                      : 'text-foreground hover:bg-background'
                  }`}
                >
                  {link.label}
                </button>
              ))}
              <div className="mt-1 flex items-center gap-2 border-t border-border px-3 pt-2 text-xs text-muted-foreground">
                <span
                  className={`h-1.5 w-1.5 rounded-full ${activeCount > 0 ? 'bg-primary' : 'bg-muted-foreground'}`}
                />
                {nodeCount} node{nodeCount === 1 ? '' : 's'} • {activeCount > 0 ? 'Active' : 'Idle'}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
