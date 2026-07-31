'use client'

import { useEffect, useState } from 'react'
import { Activity, Zap, Database } from 'lucide-react'
import { listNodes, type NodeSummary } from '@/lib/api'
import { getInferenceLog, type InferenceLogEntry } from '@/lib/history'
import LatencyChart from '@/components/latency-chart'

const POLL_INTERVAL_MS = 3000

export default function Dashboard() {
  const [nodes, setNodes] = useState<NodeSummary[]>([])
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [inferenceLog, setInferenceLog] = useState<InferenceLogEntry[]>([])

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const result = await listNodes()
        if (!cancelled) {
          setNodes(result)
          setError(null)
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load nodes')
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
      // Read fresh on every poll tick since prompts sent from the
      // Inference page write here, and this page may already be mounted
      // (or the user may have switched back to it) when that happens.
      if (!cancelled) setInferenceLog(getInferenceLog())
    }

    poll()
    const interval = setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [])

  const activeCount = nodes.filter((n) => n.status === 'available').length
  const totalAvailableRamMb = nodes.reduce((sum, n) => sum + n.available_ram_mb, 0)

  const stats = [
    { label: 'Total Nodes', value: String(nodes.length), icon: Activity },
    { label: 'Active', value: String(activeCount), icon: Zap },
    {
      label: 'Cluster RAM Available',
      value: `${(totalAvailableRamMb / 1024).toFixed(1)} GB`,
      icon: Database,
    },
  ]

  return (
    <div className="p-8 h-full flex flex-col space-y-12">
      {/* Stats */}
      <div className="grid grid-cols-3 gap-8 shrink-0">
        {stats.map((stat) => {
          const Icon = stat.icon
          return (
            <div key={stat.label} className="space-y-2">
              <div className="flex items-center gap-2 text-muted-foreground">
                <Icon size={16} />
                <span className="text-sm">{stat.label}</span>
              </div>
              <div className="text-4xl font-light text-foreground">{stat.value}</div>
            </div>
          )
        })}
      </div>

      {/* Node Status */}
      <div className="space-y-4 shrink-0">
        <h2 className="text-sm font-medium text-foreground">Node Status</h2>
        {error && <div className="text-sm text-destructive">{error}</div>}
        {!error && loading && (
          <div className="text-sm text-muted-foreground">Loading nodes...</div>
        )}
        {!error && !loading && nodes.length === 0 && (
          <div className="text-sm text-muted-foreground">No nodes registered yet.</div>
        )}
        <div className="space-y-2">
          {nodes.map((node) => {
            const usedPercent =
              node.total_ram_mb > 0
                ? Math.round(
                    ((node.total_ram_mb - node.available_ram_mb) / node.total_ram_mb) * 100
                  )
                : 0
            return (
              <div
                key={node.node_id}
                className="flex items-center justify-between text-sm py-2 border-b border-border last:border-0"
              >
                <div className="flex items-center gap-2">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      node.status === 'available' ? 'bg-primary' : 'bg-destructive'
                    }`}
                  />
                  <span className="text-muted-foreground">{node.node_id}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-foreground font-mono text-xs">
                    {(node.available_ram_mb / 1024).toFixed(1)}GB /{' '}
                    {(node.total_ram_mb / 1024).toFixed(1)}GB
                  </span>
                  <div className="w-24 h-1 bg-border rounded-full overflow-hidden">
                    <div
                      className="h-full bg-primary"
                      style={{ width: `${usedPercent}%` }}
                    ></div>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>

      {/* Inference Latency - fills whatever vertical space is left */}
      <div className="flex-1 min-h-0 flex flex-col space-y-4">
        <h2 className="text-sm font-medium text-foreground shrink-0">
          Inference Latency ({inferenceLog.length} prompt{inferenceLog.length === 1 ? '' : 's'})
        </h2>
        <div className="flex-1 min-h-0">
          <LatencyChart entries={inferenceLog} />
        </div>
      </div>
    </div>
  )
}
