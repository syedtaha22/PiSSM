'use client'

import { useEffect, useRef } from 'react'

const GRID_SIZE = 24
const DOT_RADIUS = 1
const PLUS_ARM = 5
const INFLUENCE_RADIUS = 110

/**
 * A canvas dot-grid (not a static CSS background) so individual dots can
 * morph into "+" marks near the cursor, interpolated by distance. Fixed
 * to the viewport, sits behind everything (z-index -1). The base grid
 * always draws regardless of input device; only the mousemove *listener*
 * is skipped on devices without a fine pointer.
 */
export function InteractiveDotGrid() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const mouseRef = useRef<{ x: number; y: number } | null>(null)
  const rafRef = useRef<number | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const supportsHover = window.matchMedia('(hover: hover) and (pointer: fine)').matches

    let width = window.innerWidth
    let height = window.innerHeight

    function resize() {
      const dpr = window.devicePixelRatio || 1
      width = window.innerWidth
      height = window.innerHeight
      canvas!.width = width * dpr
      canvas!.height = height * dpr
      canvas!.style.width = `${width}px`
      canvas!.style.height = `${height}px`
      ctx!.setTransform(dpr, 0, 0, dpr, 0, 0)
      draw()
    }

    function draw() {
      const foregroundHex = getComputedStyle(document.documentElement)
        .getPropertyValue('--foreground')
        .trim()
      const foreground = /^#[0-9a-f]{6}$/i.test(foregroundHex) ? foregroundHex : '#15161b'

      ctx!.clearRect(0, 0, width, height)
      const mouse = mouseRef.current

      for (let x = 0; x <= width + GRID_SIZE; x += GRID_SIZE) {
        for (let y = 0; y <= height + GRID_SIZE; y += GRID_SIZE) {
          let influence = 0
          if (mouse) {
            const dist = Math.hypot(x - mouse.x, y - mouse.y)
            influence = dist > INFLUENCE_RADIUS ? 0 : 1 - dist / INFLUENCE_RADIUS
          }

          const alpha = 0.12 + influence * 0.3
          const r = parseInt(foreground.slice(1, 3), 16)
          const g = parseInt(foreground.slice(3, 5), 16)
          const b = parseInt(foreground.slice(5, 7), 16)
          ctx!.strokeStyle = ctx!.fillStyle = `rgba(${r}, ${g}, ${b}, ${alpha})`

          if (influence > 0.04) {
            const arm = DOT_RADIUS + (PLUS_ARM - DOT_RADIUS) * influence
            ctx!.lineWidth = 1
            ctx!.beginPath()
            ctx!.moveTo(x - arm, y)
            ctx!.lineTo(x + arm, y)
            ctx!.moveTo(x, y - arm)
            ctx!.lineTo(x, y + arm)
            ctx!.stroke()
          } else {
            ctx!.beginPath()
            ctx!.arc(x, y, DOT_RADIUS, 0, Math.PI * 2)
            ctx!.fill()
          }
        }
      }
    }

    function scheduleDraw() {
      if (rafRef.current !== null) return
      rafRef.current = requestAnimationFrame(() => {
        rafRef.current = null
        draw()
      })
    }

    function handleMouseMove(e: MouseEvent) {
      mouseRef.current = { x: e.clientX, y: e.clientY }
      scheduleDraw()
    }

    function handleMouseLeave() {
      mouseRef.current = null
      scheduleDraw()
    }

    resize()
    window.addEventListener('resize', resize)

    if (supportsHover) {
      window.addEventListener('mousemove', handleMouseMove)
      document.documentElement.addEventListener('mouseleave', handleMouseLeave)
    }

    // The theme switcher sets --foreground via inline style on <html>,
    // outside React. Without this, dots would keep the old color until the
    // next mousemove happened to trigger a redraw.
    const observer = new MutationObserver(scheduleDraw)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['style'] })

    return () => {
      window.removeEventListener('resize', resize)
      if (supportsHover) {
        window.removeEventListener('mousemove', handleMouseMove)
        document.documentElement.removeEventListener('mouseleave', handleMouseLeave)
      }
      observer.disconnect()
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="pointer-events-none fixed inset-0"
      style={{ zIndex: -1 }}
      aria-hidden="true"
    />
  )
}
