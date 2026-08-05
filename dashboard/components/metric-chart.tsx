'use client'

import { useMemo, useState } from 'react'

export interface MetricPoint {
  timestamp: number
  value: number
  modelName: string
  numNodes: number
}

const VIEWBOX_WIDTH = 1000
const VIEWBOX_HEIGHT = 400
const PADDING_LEFT = 56
const PADDING_RIGHT = 48
const PADDING_TOP = 20
const PADDING_BOTTOM = 32
const MAX_X_LABELS = 6

function niceCeil(value: number): number {
  if (value <= 0) return 1
  const magnitude = 10 ** Math.floor(Math.log10(value))
  return Math.ceil(value / magnitude) * magnitude
}

function niceFloor(value: number): number {
  if (value <= 0) return 0.1
  const magnitude = 10 ** Math.floor(Math.log10(value))
  const floored = Math.floor(value / magnitude) * magnitude
  return floored > 0 ? floored : magnitude
}

function formatCompactTimestamp(timestamp: number): string {
  const d = new Date(timestamp)
  const month = d.getMonth() + 1
  const day = d.getDate()
  let hours = d.getHours()
  const minutes = d.getMinutes().toString().padStart(2, '0')
  const ampm = hours >= 12 ? 'pm' : 'am'
  hours = hours % 12 || 12
  return `${month}/${day} ${hours}:${minutes}${ampm}`
}

export default function MetricChart({
  points,
  label,
  unit,
  emptyMessage,
}: {
  points: MetricPoint[]
  label: string
  unit: string
  emptyMessage: string
}) {
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)
  const [scale, setScale] = useState<'linear' | 'log'>('linear')

  const { minValue, maxValue } = useMemo(() => {
    const values = points.map((p) => p.value)
    const max = niceCeil(Math.max(0, ...values))
    if (scale === 'linear') return { minValue: 0, maxValue: max }
    const positive = values.filter((v) => v > 0)
    const min = niceFloor(Math.min(...(positive.length ? positive : [1])))
    return { minValue: min, maxValue: Math.max(max, min * 10) }
  }, [points, scale])

  if (points.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-sm text-muted-foreground border border-dashed border-border rounded">
        {emptyMessage}
      </div>
    )
  }

  const plotWidth = VIEWBOX_WIDTH - PADDING_LEFT - PADDING_RIGHT
  const plotHeight = VIEWBOX_HEIGHT - PADDING_TOP - PADDING_BOTTOM
  const baselineY = VIEWBOX_HEIGHT - PADDING_BOTTOM
  const topY = PADDING_TOP

  function valueToY(value: number): number {
    if (scale === 'linear') {
      return baselineY - (value / maxValue) * plotHeight
    }
    const clamped = Math.min(Math.max(value, minValue), maxValue)
    const frac =
      (Math.log10(clamped) - Math.log10(minValue)) /
      (Math.log10(maxValue) - Math.log10(minValue))
    return baselineY - frac * plotHeight
  }

  const xs = points.map((_, i) =>
    PADDING_LEFT + (points.length === 1 ? plotWidth / 2 : (i / (points.length - 1)) * plotWidth)
  )
  const coords = points.map((p, i) => ({ x: xs[i], y: valueToY(p.value), point: p }))
  const linePath = coords.map((c, i) => `${i === 0 ? 'M' : 'L'} ${c.x} ${c.y}`).join(' ')

  const ticks: { y: number; text: string }[] = []
  if (scale === 'linear') {
    for (const f of [0, 0.25, 0.5, 0.75, 1]) {
      ticks.push({ y: baselineY - f * plotHeight, text: Math.round(maxValue * f).toLocaleString() })
    }
  } else {
    for (let v = minValue; v <= maxValue * 1.0001; v *= 10) {
      ticks.push({ y: valueToY(v), text: Math.round(v).toLocaleString() })
    }
  }

  const labelStep = Math.max(1, Math.ceil(points.length / MAX_X_LABELS))
  const labeledIndices = new Set(
    points.map((_, i) => i).filter((i) => i % labelStep === 0 || i === points.length - 1)
  )

  const hitBandWidth = Math.max(plotWidth / points.length, 12)
  const hovered = hoveredIndex !== null ? coords[hoveredIndex] : null

  return (
    <div className="h-full flex flex-col">
      <div className="flex justify-end shrink-0 mb-1">
        <div className="inline-flex text-xs border border-border rounded overflow-hidden">
          {(['linear', 'log'] as const).map((s) => (
            <button
              key={s}
              onClick={() => setScale(s)}
              className={`px-2 py-1 capitalize ${
                scale === s
                  ? 'bg-primary text-primary-foreground'
                  : 'text-muted-foreground hover:bg-muted'
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>
      <div className="flex-1 min-h-0">
        <svg
          viewBox={`0 0 ${VIEWBOX_WIDTH} ${VIEWBOX_HEIGHT}`}
          width="100%"
          height="100%"
          preserveAspectRatio="none"
        >
          {ticks.map(({ y }) => (
            <line
              key={y}
              x1={PADDING_LEFT}
              x2={VIEWBOX_WIDTH - PADDING_RIGHT}
              y1={y}
              y2={y}
              className="stroke-border"
              strokeWidth={1}
              vectorEffect="non-scaling-stroke"
            />
          ))}
          {ticks.map(({ y, text }) => (
            <text
              key={y}
              x={0}
              y={y + 4}
              className="text-[12px] font-sans fill-muted-foreground"
            >
              {text}
            </text>
          ))}

          {points.map((_, i) =>
            labeledIndices.has(i) ? (
              <text
                key={i}
                x={xs[i]}
                y={VIEWBOX_HEIGHT - 8}
                textAnchor="middle"
                className="text-[11px] font-sans fill-muted-foreground"
              >
                {formatCompactTimestamp(points[i].timestamp)}
              </text>
            ) : null
          )}

          <path
            d={linePath}
            className="stroke-primary"
            strokeWidth={2}
            fill="none"
            strokeLinecap="round"
            strokeLinejoin="round"
            vectorEffect="non-scaling-stroke"
          />

          {coords.map((c, i) => {
            const isHovered = hoveredIndex === i
            return (
              <g
                key={c.point.timestamp}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
                className="cursor-pointer"
              >
                <rect
                  x={c.x - hitBandWidth / 2}
                  y={topY}
                  width={hitBandWidth}
                  height={plotHeight}
                  fill="transparent"
                />
                <circle
                  cx={c.x}
                  cy={c.y}
                  r={isHovered ? 6 : 4}
                  className="fill-primary stroke-background"
                  strokeWidth={2}
                  vectorEffect="non-scaling-stroke"
                />
              </g>
            )
          })}
        </svg>
      </div>
      <div className="mt-2 text-xs h-4 shrink-0">
        {hovered && (
          <>
            <span className="font-mono text-foreground">
              {hovered.point.value.toFixed(1)}
              {unit}
            </span>
            <span className="text-muted-foreground">
              {' '}
              · {hovered.point.modelName} · {hovered.point.numNodes} node
              {hovered.point.numNodes === 1 ? '' : 's'} ·{' '}
              {formatCompactTimestamp(hovered.point.timestamp)}
            </span>
          </>
        )}
        {!hovered && <span className="text-muted-foreground">{label}</span>}
      </div>
    </div>
  )
}
