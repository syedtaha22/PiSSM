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
  const areaPath = `${linePath} L ${points[points.length - 1].x} ${baselineY} L ${points[0].x} ${baselineY} Z`
  const hitBandWidth = Math.max(plotWidth / entries.length, 12)
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
          <line
            x1={PADDING_LEFT}
            x2={VIEWBOX_WIDTH}
            y1={topY}
            y2={topY}
            className="stroke-border"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
          <line
            x1={PADDING_LEFT}
            x2={VIEWBOX_WIDTH}
            y1={baselineY}
            y2={baselineY}
            className="stroke-border"
            strokeWidth={1}
            vectorEffect="non-scaling-stroke"
          />
          <text x={0} y={topY + 4} className="text-[12px] fill-muted-foreground">
            {maxLatency.toLocaleString()}ms
          </text>
          <text x={0} y={baselineY} className="text-[12px] fill-muted-foreground">
            0ms
          </text>

          <path d={areaPath} className="fill-primary/10" stroke="none" />
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
