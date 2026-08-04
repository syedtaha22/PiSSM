'use client'

import { useMemo, useState } from 'react'
import type { InferenceLogEntry } from '@/lib/history'

// A fixed internal coordinate system - the SVG scales to whatever
// space its container gives it via viewBox + percentage width/height,
// so the chart fills the page instead of rendering at a small fixed
// pixel size.
const VIEWBOX_WIDTH = 1000
const VIEWBOX_HEIGHT = 400
const PADDING_LEFT = 48
const PADDING_TOP = 20
const PADDING_BOTTOM = 28

// Gridlines at 0%, 25%, 50%, 75%, and 100% of maxLatency - two ticks
// (just 0 and the max) left the middle of the chart unreadable, with
// no way to tell where a point actually falls between them.
const TICK_FRACTIONS = [0, 0.25, 0.5, 0.75, 1]

function roundUpToCleanStep(value: number): number {
  if (value <= 0) return 100
  const magnitude = 10 ** Math.floor(Math.log10(value))
  return Math.ceil(value / magnitude) * magnitude
}

export default function LatencyChart({ entries }: { entries: InferenceLogEntry[] }) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)

  const maxLatency = useMemo(
    () => roundUpToCleanStep(Math.max(0, ...entries.map((e) => e.latencyMs))),
    [entries]
  )

  if (entries.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-sm text-muted-foreground border border-dashed border-border rounded">
        No inference requests sent yet - latencies will appear here after you send a
        prompt on the Inference page.
      </div>
    )
  }

  const plotWidth = VIEWBOX_WIDTH - PADDING_LEFT
  const plotHeight = VIEWBOX_HEIGHT - PADDING_TOP - PADDING_BOTTOM
  const baselineY = VIEWBOX_HEIGHT - PADDING_BOTTOM
  const topY = PADDING_TOP

  const points = entries.map((entry, i) => {
    const x =
      PADDING_LEFT +
      (entries.length === 1 ? plotWidth / 2 : (i / (entries.length - 1)) * plotWidth)
    const y = baselineY - (entry.latencyMs / maxLatency) * plotHeight
    return { x, y, entry }
  })

  const linePath = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ')
  const hitBandWidth = Math.max(plotWidth / entries.length, 12)
  const ticks = TICK_FRACTIONS.map((f) => ({
    f,
    y: baselineY - f * plotHeight,
    value: Math.round(maxLatency * f),
  }))
  const hovered = hoveredIndex !== null ? points[hoveredIndex] : null

  return (
    <div className="h-full flex flex-col">
      <div className="flex-1 min-h-0">
        <svg
          viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
          width="100%"
          height="100%"
          preserveAspectRatio="none"
        >
          {ticks.map(({ f, y }) => (
            <line
              key={f}
              x1={PADDING_LEFT}
              x2={VIEWBOX_WIDTH}
              y1={y}
              y2={y}
              className="stroke-border"
              strokeWidth={1}
              vectorEffect="non-scaling-stroke"
            />
          ))}
          {ticks.map(({ f, y, value }) => (
            <text key={f} x={0} y={y + 4} className="text-[12px] fill-muted-foreground">
              {value.toLocaleString()}ms
            </text>
          ))}

          <path
            d={linePath}
            className="stroke-primary"
            strokeWidth={2}
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />

          {points.map((p, i) => {
            const isHovered = hoveredIndex === i
            const isLast = i === points.length - 1
            return (
              <g
                key={p.entry.timestamp}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
                className="cursor-pointer"
              >
                <rect
                  x={p.x - hitBandWidth / 2}
                  y={topY}
                  width={hitBandWidth}
                  height={plotHeight}
                  fill="transparent"
                />
                {(isHovered || isLast) && (
                  <circle
                    cx={p.x}
                    cy={p.y}
                    r={isHovered ? 6 : 4}
                    className="fill-primary stroke-background"
                    strokeWidth={2}
                    vectorEffect="non-scaling-stroke"
                  />
                )}
              </g>
            )
          })}
        </svg>
      </div>
      <div className="mt-2 text-xs h-4 shrink-0">
        {hovered && (
          <>
            <span className="font-mono text-foreground">
              {hovered.entry.latencyMs.toFixed(0)}ms
            </span>
            <span className="text-muted-foreground">
              {' '}
              · {hovered.entry.modelName} · {hovered.entry.numNodes} node
              {hovered.entry.numNodes === 1 ? '' : 's'}
            </span>
          </>
        )}
      </div>
    </div>
  )
}
