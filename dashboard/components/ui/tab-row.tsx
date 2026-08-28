'use client'

import { useCallback, useLayoutEffect, useRef, useState } from 'react'

interface TabRowItem {
  label: string
  value: string
}

interface TabRowProps {
  items: TabRowItem[]
  active: string
  onChange: (value: string) => void
  className?: string
}

/**
 * Underline-style tab row with a sliding indicator. Tracks the active
 * item's horizontal position (and row, for wrapped layouts) and
 * recomputes on resize.
 */
export function TabRow({ items, active, onChange, className = '' }: TabRowProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const itemRefs = useRef<Map<string, HTMLButtonElement>>(new Map())
  const [indicator, setIndicator] = useState<{ left: number; top: number; width: number } | null>(
    null
  )

  const measure = useCallback(() => {
    const el = itemRefs.current.get(active)
    const container = containerRef.current
    if (el && container) {
      const elRect = el.getBoundingClientRect()
      const containerRect = container.getBoundingClientRect()
      setIndicator({
        left: elRect.left - containerRect.left,
        top: elRect.bottom - containerRect.top,
        width: elRect.width,
      })
    }
  }, [active])

  useLayoutEffect(() => {
    measure()
  }, [measure, items])

  useLayoutEffect(() => {
    window.addEventListener('resize', measure)
    return () => window.removeEventListener('resize', measure)
  }, [measure])

  return (
    <div
      ref={containerRef}
      role="tablist"
      className={`relative flex flex-nowrap items-center gap-2 border-b border-border pb-[1px] ${className}`}
    >
      {items.map((item) => (
        <button
          key={item.value}
          ref={(node) => {
            if (node) itemRefs.current.set(item.value, node)
            else itemRefs.current.delete(item.value)
          }}
          role="tab"
          aria-selected={active === item.value}
          onClick={() => onChange(item.value)}
          className={`
            relative shrink-0 rounded-sm px-3 py-1.5 font-sans text-xs font-medium
            transition-colors duration-200 cursor-pointer
            focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2
            ${active === item.value ? 'bg-accent text-accent-foreground' : 'text-muted-foreground hover:text-foreground'}
          `}
        >
          {item.label}
        </button>
      ))}
      {indicator && (
        <span
          aria-hidden
          className="absolute h-0.5 bg-primary transition-all duration-250 ease-in-out"
          style={{ left: indicator.left, top: indicator.top - 1, width: indicator.width }}
        />
      )}
    </div>
  )
}
