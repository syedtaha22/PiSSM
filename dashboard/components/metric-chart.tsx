'use client'

import { useState } from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts'

export interface MetricPoint {
  timestamp: number
  value: number
  modelName: string
  numNodes: number
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

function ChartTooltip({
  active,
  payload,
  unit,
}: {
  active?: boolean
  payload?: Array<{ payload: MetricPoint }>
  unit: string
}) {
  if (!active || !payload || !payload.length) return null
  const point = payload[0].payload
  return (
    <div className="rounded border border-border bg-popover px-3 py-2 text-xs shadow-md font-sans">
      <div className="font-mono text-popover-foreground">
        {point.value.toFixed(1)}
        {unit}
      </div>
      <div className="text-muted-foreground">
        {point.modelName} · {point.numNodes} node{point.numNodes === 1 ? '' : 's'} ·{' '}
        {formatCompactTimestamp(point.timestamp)}
      </div>
    </div>
  )
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
  const [scale, setScale] = useState<'linear' | 'log'>('linear')

  if (points.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-sm text-muted-foreground border border-dashed border-border rounded">
        {emptyMessage}
      </div>
    )
  }

  const positiveMin = Math.min(...points.map((p) => p.value).filter((v) => v > 0)) || 1

  return (
    <div className="h-full flex flex-col font-sans" aria-label={`${label} chart`}>
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
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={points} margin={{ top: 8, right: 16, bottom: 8, left: 8 }}>
            <CartesianGrid stroke="var(--border)" vertical={false} />
            <XAxis
              dataKey="timestamp"
              type="number"
              domain={['dataMin', 'dataMax']}
              tickFormatter={(v) => formatCompactTimestamp(v)}
              tick={{ fontSize: 11, fill: 'var(--muted-foreground)', fontFamily: 'inherit' }}
              stroke="var(--border)"
              minTickGap={48}
            />
            <YAxis
              scale={scale === 'log' ? 'log' : 'linear'}
              domain={scale === 'log' ? [positiveMin, 'auto'] : [0, 'auto']}
              allowDataOverflow={scale === 'log'}
              tick={{ fontSize: 12, fill: 'var(--muted-foreground)', fontFamily: 'inherit' }}
              stroke="var(--border)"
              width={56}
            />
            <Tooltip
              content={(props) => (
                <ChartTooltip
                  active={props.active}
                  payload={
                    props.payload as unknown as Array<{ payload: MetricPoint }> | undefined
                  }
                  unit={unit}
                />
              )}
            />
            <Line
              type="linear"
              dataKey="value"
              stroke="var(--primary)"
              strokeWidth={2}
              dot={{ r: 4, fill: 'var(--primary)', stroke: 'var(--background)', strokeWidth: 2 }}
              activeDot={{ r: 6 }}
              isAnimationActive={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
