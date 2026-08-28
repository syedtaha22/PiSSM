'use client'

import { useMemo, useState } from 'react'
import type { InferenceLogEntry } from '@/lib/history'
import MetricChart, { type MetricPoint } from '@/components/metric-chart'
import { TabRow } from '@/components/ui/tab-row'

type Tab = 'latency' | 'tokens'

export default function MetricsPanel({ entries }: { entries: InferenceLogEntry[] }) {
  const [tab, setTab] = useState<Tab>('latency')
  const [modelFilter, setModelFilter] = useState('all')
  const [nodesFilter, setNodesFilter] = useState('all')
  const [dateFrom, setDateFrom] = useState('')
  const [dateTo, setDateTo] = useState('')

  const models = useMemo(
    () => Array.from(new Set(entries.map((e) => e.modelName))).sort(),
    [entries]
  )
  const nodeCounts = useMemo(
    () => Array.from(new Set(entries.map((e) => e.numNodes))).sort((a, b) => a - b),
    [entries]
  )

  const filtered = useMemo(() => {
    const fromMs = dateFrom ? new Date(dateFrom).getTime() : null
    // Inclusive of the whole "to" day.
    const toMs = dateTo ? new Date(dateTo).getTime() + 24 * 60 * 60 * 1000 - 1 : null
    return entries.filter((e) => {
      if (modelFilter !== 'all' && e.modelName !== modelFilter) return false
      if (nodesFilter !== 'all' && String(e.numNodes) !== nodesFilter) return false
      if (fromMs !== null && e.timestamp < fromMs) return false
      if (toMs !== null && e.timestamp > toMs) return false
      return true
    })
  }, [entries, modelFilter, nodesFilter, dateFrom, dateTo])

  const latencyPoints: MetricPoint[] = filtered.map((e) => ({
    timestamp: e.timestamp,
    value: e.latencyMs,
    modelName: e.modelName,
    numNodes: e.numNodes,
  }))

  const tokenPoints: MetricPoint[] = filtered
    .filter((e) => e.numTokens !== undefined && e.latencyMs > 0)
    .map((e) => ({
      timestamp: e.timestamp,
      value: (e.numTokens as number) / (e.latencyMs / 1000),
      modelName: e.modelName,
      numNodes: e.numNodes,
    }))

  return (
    <div className="flex-1 min-h-0 flex flex-col space-y-4">
      <div className="flex items-center justify-between shrink-0 flex-wrap gap-2 font-sans">
        <h2 className="text-sm font-medium text-foreground">
          Inference Metrics ({filtered.length} prompt{filtered.length === 1 ? '' : 's'})
        </h2>
        <div className="flex items-center gap-2 text-xs">
          <select
            aria-label="Filter by model"
            value={modelFilter}
            onChange={(e) => setModelFilter(e.target.value)}
            className="rounded-sm border border-border bg-card px-2 py-1 text-foreground focus:outline-none focus:border-primary"
          >
            <option value="all">All models</option>
            {models.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
          <select
            aria-label="Filter by node count"
            value={nodesFilter}
            onChange={(e) => setNodesFilter(e.target.value)}
            className="rounded-sm border border-border bg-card px-2 py-1 text-foreground focus:outline-none focus:border-primary"
          >
            <option value="all">All node counts</option>
            {nodeCounts.map((n) => (
              <option key={n} value={n}>
                {n} node{n === 1 ? '' : 's'}
              </option>
            ))}
          </select>
          <input
            aria-label="From date"
            type="date"
            value={dateFrom}
            onChange={(e) => setDateFrom(e.target.value)}
            className="rounded-sm border border-border bg-card px-2 py-1 text-foreground focus:outline-none focus:border-primary"
          />
          <span className="text-muted-foreground">to</span>
          <input
            aria-label="To date"
            type="date"
            value={dateTo}
            onChange={(e) => setDateTo(e.target.value)}
            className="rounded-sm border border-border bg-card px-2 py-1 text-foreground focus:outline-none focus:border-primary"
          />
        </div>
      </div>

      <TabRow
        items={[
          { value: 'latency', label: 'Latency' },
          { value: 'tokens', label: 'Tokens/sec' },
        ]}
        active={tab}
        onChange={(value) => setTab(value as Tab)}
        className="shrink-0 w-fit"
      />

      <div className="flex-1 min-h-0">
        {tab === 'latency' ? (
          <MetricChart
            points={latencyPoints}
            label="Latency"
            unit="ms"
            emptyMessage="No inference requests sent yet - latencies will appear here after you send a prompt on the Inference page."
          />
        ) : (
          <MetricChart
            points={tokenPoints}
            label="Tokens/sec"
            unit=" tok/s"
            emptyMessage="No throughput data yet - send a prompt on the Inference page to see tokens/sec here."
          />
        )}
      </div>
    </div>
  )
}
