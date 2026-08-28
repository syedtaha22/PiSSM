'use client'

import { useEffect, useMemo, useRef, useState } from 'react'
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  BaseEdge,
  Controls,
  EdgeLabelRenderer,
  Handle,
  Position,
  getBezierPath,
  useNodesState,
  type Edge,
  type EdgeProps,
  type Node,
  type NodeProps,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { listNodes, getTopology, type NodeSummary, type TopologyAssignment } from '@/lib/api'

const POLL_INTERVAL_MS = 3000
const NODE_SPACING_PX = 220
const ORCHESTRATOR_NODE_ID = 'orchestrator'
const IDLE_ROW_Y = 320
const POSITIONS_STORAGE_KEY = 'pissm-topology-node-positions'

type StoredPosition = { x: number; y: number }

// The arrangement survives switching away to another tab and back -
// this page fully unmounts when it's not the active tab (page.tsx
// renders it conditionally), so component state alone doesn't survive
// that. localStorage does, and also survives a page reload.
function loadStoredPositions(): Record<string, StoredPosition> {
  if (typeof window === 'undefined') return {}
  try {
    const raw = window.localStorage.getItem(POSITIONS_STORAGE_KEY)
    return raw ? JSON.parse(raw) : {}
  } catch {
    return {}
  }
}

function saveStoredPositions(positions: Record<string, StoredPosition>) {
  if (typeof window === 'undefined') return
  try {
    window.localStorage.setItem(POSITIONS_STORAGE_KEY, JSON.stringify(positions))
  } catch {
    /* localStorage unavailable (private browsing, quota) - positions just won't persist */
  }
}

type PiNodeData = {
  nodeId: string
  ipAddress: string
  status: string
  layerRange: string | null
  role: string | null
  isFirst: boolean
  isLast: boolean
  kind: 'orchestrator' | 'worker'
}

// A Blender-node-editor-style box: title bar with a status dot, a body
// with layer/role details, and coloured input/output sockets on the
// left/right edges (hidden on whichever end the pipeline terminates).
// The orchestrator gets an accent-coloured header to set it apart from
// worker nodes - everything else in the graph only exists because it
// dispatched the pipeline and receives the result back.
function PiNode({ data }: NodeProps<Node<PiNodeData>>) {
  const isOrchestrator = data.kind === 'orchestrator'
  return (
    <div className="rounded-lg border border-border bg-card shadow-sm min-w-[170px] overflow-visible">
      {!data.isFirst && (
        <Handle
          type="target"
          position={Position.Left}
          className="!w-3 !h-3 !bg-emerald-500 !border-2 !border-background"
        />
      )}
      <div
        className={`px-3 py-1.5 border-b rounded-t-lg flex items-center gap-2 ${
          isOrchestrator
            ? 'bg-primary/15 border-primary/40'
            : 'bg-muted border-border'
        }`}
      >
        <span
          className={`w-2 h-2 rounded-full shrink-0 ${
            data.status === 'available' ? 'bg-primary' : 'bg-destructive'
          }`}
        />
        <span className="text-xs font-medium text-foreground truncate">{data.nodeId}</span>
      </div>
      <div className="px-3 py-2 space-y-1">
        <div className="text-[10px] text-muted-foreground font-mono">{data.ipAddress}</div>
        {data.layerRange && (
          <div className="text-xs text-foreground font-mono">{data.layerRange}</div>
        )}
        {data.role && (
          <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
            {data.role}
          </div>
        )}
      </div>
      {!data.isLast && (
        <Handle
          type="source"
          position={Position.Right}
          className="!w-3 !h-3 !bg-amber-500 !border-2 !border-background"
        />
      )}
    </div>
  )
}

const nodeTypes = { piNode: PiNode }

type CallbackEdgeData = {
  label: string
  // The dispatch and result edges connect the orchestrator to the same
  // node whenever a single node holds the whole pipeline, which makes
  // their midpoints (where an edge label normally renders) land on the
  // exact same point. A fixed vertical nudge in opposite directions
  // keeps the two labels legible regardless of node count.
  labelOffsetY: number
}

function CallbackEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style,
  data,
}: EdgeProps<Edge<CallbackEdgeData>>) {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  })

  return (
    <>
      <BaseEdge id={id} path={edgePath} style={style} />
      <EdgeLabelRenderer>
        <div
          style={{
            position: 'absolute',
            transform: `translate(-50%, -50%) translate(${labelX}px, ${
              labelY + (data?.labelOffsetY ?? 0)
            }px)`,
            pointerEvents: 'none',
          }}
          className="text-[10px] text-muted-foreground bg-background/90 px-1 rounded-sm whitespace-nowrap"
        >
          {data?.label}
        </div>
      </EdgeLabelRenderer>
    </>
  )
}

const edgeTypes = { callback: CallbackEdge }

function roleLabel(isFirst: boolean, isLast: boolean): string {
  if (isFirst && isLast) return 'first + last'
  if (isFirst) return 'first'
  if (isLast) return 'last'
  return 'middle'
}

function buildOrchestratorNode(): Node<PiNodeData> {
  return {
    id: ORCHESTRATOR_NODE_ID,
    type: 'piNode',
    position: { x: -NODE_SPACING_PX, y: 100 },
    data: {
      nodeId: 'Orchestrator',
      ipAddress: typeof window !== 'undefined' ? window.location.host : '',
      status: 'available',
      layerRange: null,
      role: 'dispatch + result callback',
      isFirst: false,
      isLast: false,
      kind: 'orchestrator',
    },
  }
}

// Computes the graph's "desired" nodes from fresh poll data, always at
// their default layout position. The caller is responsible for
// preserving any position the user has since dragged a node to -
// these are default positions, not permanent ones.
//
// Every registered node is always included, even ones that just
// joined and aren't part of the currently dispatched pipeline (or a
// node the last dispatch no longer uses) - those render as idle boxes
// on a second row rather than disappearing from the graph.
function buildDesiredNodes(
  nodeList: NodeSummary[],
  assignments: TopologyAssignment[]
): Node<PiNodeData>[] {
  const assignedIds = new Set(assignments.map((a) => a.node_id))
  const idleRowY = assignments.length > 0 ? IDLE_ROW_Y : 100

  const pipelineNodes = assignments.map((a, i) => {
    const live = nodeList.find((n) => n.node_id === a.node_id)
    return {
      id: a.node_id,
      type: 'piNode',
      position: { x: i * NODE_SPACING_PX, y: 100 },
      data: {
        nodeId: a.node_id,
        ipAddress: a.ip_address,
        status: live?.status ?? 'unknown',
        layerRange: `layers [${a.layer_start}, ${a.layer_end})`,
        role: roleLabel(a.is_first, a.is_last),
        // Always show both sockets while a pipeline is active: the
        // first shard's left socket now connects back to the
        // orchestrator's dispatch edge, and the last shard's right
        // socket connects to its result-callback edge.
        isFirst: false,
        isLast: false,
        kind: 'worker' as const,
      },
    }
  })

  const idleNodes = nodeList
    .filter((n) => !assignedIds.has(n.node_id))
    .map((n, i) => ({
      id: n.node_id,
      type: 'piNode',
      position: { x: i * NODE_SPACING_PX, y: idleRowY },
      data: {
        nodeId: n.node_id,
        ipAddress: n.ip_address,
        status: n.status,
        layerRange: null,
        role: 'idle',
        isFirst: true,
        isLast: true,
        kind: 'worker' as const,
      },
    }))

  return [buildOrchestratorNode(), ...pipelineNodes, ...idleNodes]
}

export default function Topology() {
  const [nodes, setNodes] = useState<NodeSummary[]>([])
  const [assignments, setAssignments] = useState<TopologyAssignment[]>([])
  const [modelName, setModelName] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [heartbeatSeconds, setHeartbeatSeconds] = useState<Record<string, number>>({})
  const heartbeatTracker = useRef<Map<string, { value: number; clientTime: number }>>(new Map())

  // graphNodes is stateful (not derived) specifically so a user-dragged
  // position survives the next poll - onNodesChange applies drag deltas
  // here, and each poll below merges in fresh data without touching the
  // position of any node that already exists.
  const [graphNodes, setGraphNodes, onNodesChange] = useNodesState<Node<PiNodeData>>([])

  useEffect(() => {
    let cancelled = false

    async function poll() {
      try {
        const [nodeList, topo] = await Promise.all([listNodes(), getTopology()])
        if (cancelled) return

        const now = Date.now()
        const ages: Record<string, number> = {}
        for (const n of nodeList) {
          const tracked = heartbeatTracker.current.get(n.node_id)
          if (!tracked || tracked.value !== n.last_heartbeat) {
            heartbeatTracker.current.set(n.node_id, { value: n.last_heartbeat, clientTime: now })
            ages[n.node_id] = 0
          } else {
            ages[n.node_id] = (now - tracked.clientTime) / 1000
          }
        }

        setNodes(nodeList)
        setAssignments(topo.assignments)
        setModelName(topo.model_name)
        setHeartbeatSeconds(ages)
        setError(null)
        setGraphNodes((current) => {
          const stored = loadStoredPositions()
          return buildDesiredNodes(nodeList, topo.assignments).map((next) => {
            const existing = current.find((c) => c.id === next.id)
            if (existing) return { ...next, position: existing.position }
            const remembered = stored[next.id]
            return remembered ? { ...next, position: remembered } : next
          })
        })
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : 'Failed to load topology')
        }
      }
    }

    poll()
    const interval = setInterval(poll, POLL_INTERVAL_MS)
    return () => {
      cancelled = true
      clearInterval(interval)
    }
  }, [setGraphNodes])

  // Persist the arrangement (both drag updates and poll merges land here
  // as graphNodes changes) so it survives switching tabs or reloading.
  useEffect(() => {
    if (graphNodes.length === 0) return
    const positions: Record<string, StoredPosition> = {}
    for (const n of graphNodes) positions[n.id] = n.position
    saveStoredPositions(positions)
  }, [graphNodes])

  const graphEdges = useMemo<Edge[]>(() => {
    if (assignments.length === 0) return []
    const edges: Edge[] = [
      {
        id: `e-${ORCHESTRATOR_NODE_ID}-${assignments[0].node_id}`,
        source: ORCHESTRATOR_NODE_ID,
        target: assignments[0].node_id,
        type: 'callback',
        data: { label: 'dispatch', labelOffsetY: -12 },
        style: { strokeDasharray: '4 4' },
      },
    ]
    for (let i = 0; i < assignments.length - 1; i++) {
      edges.push({
        id: `e-${assignments[i].node_id}-${assignments[i + 1].node_id}`,
        source: assignments[i].node_id,
        target: assignments[i + 1].node_id,
        animated: true,
      })
    }
    edges.push({
      id: `e-${assignments[assignments.length - 1].node_id}-${ORCHESTRATOR_NODE_ID}`,
      source: assignments[assignments.length - 1].node_id,
      target: ORCHESTRATOR_NODE_ID,
      type: 'callback',
      data: { label: 'result', labelOffsetY: 12 },
      style: { strokeDasharray: '4 4' },
    })
    return edges
  }, [assignments])

  const idleCount = useMemo(() => {
    const assignedIds = new Set(assignments.map((a) => a.node_id))
    return nodes.filter((n) => !assignedIds.has(n.node_id)).length
  }, [nodes, assignments])

  return (
    <div className="mx-auto h-full w-full max-w-[1120px] flex flex-col p-8">
      <div className="mb-4 space-y-2">
        <h2 className="font-display text-2xl font-semibold tracking-tight text-foreground">
          Network Topology
        </h2>
        <p className="text-sm text-muted-foreground">
          {modelName
            ? `Showing the active pipeline for "${modelName}"${
                idleCount > 0
                  ? ` (${idleCount} node${idleCount === 1 ? '' : 's'} idle - use Redistribute on the Inference page to include them)`
                  : ''
              }.`
            : 'No model loaded yet - showing registered cluster nodes.'}
        </p>
      </div>

      {/* Graph canvas - read-only reflection of the real cluster/pipeline */}
      <div
        className="flex-1 border border-border rounded-md mb-6 bg-background/50 overflow-hidden"
        style={{ minHeight: '400px' }}
      >
        <ReactFlow
          nodes={graphNodes}
          edges={graphEdges}
          onNodesChange={onNodesChange}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodesConnectable={false}
          fitView
        >
          <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>

      {/* Status */}
      <div className="space-y-4">
        <h3 className="text-sm font-medium text-foreground">Status</h3>
        {error && <div className="text-sm text-destructive">{error}</div>}
        {!error && nodes.length === 0 && (
          <div className="text-sm text-muted-foreground">No worker nodes registered.</div>
        )}
        <div className="space-y-2 divide-y divide-border">
          {nodes.map((n) => {
            const seconds = heartbeatSeconds[n.node_id]
            const heartbeatAgo = seconds !== undefined ? seconds.toFixed(1) : '-'
            return (
              <div key={n.node_id} className="py-3 flex items-center justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      n.status === 'available' ? 'bg-primary' : 'bg-destructive'
                    }`}
                  />
                  <div>
                    <div className="font-medium text-foreground">{n.node_id}</div>
                    <div className="text-xs text-muted-foreground font-mono">{n.ip_address}</div>
                  </div>
                </div>
                <div className="text-xs text-muted-foreground text-right">
                  <div>
                    {n.status} • last heartbeat {heartbeatAgo}s ago
                  </div>
                  <div>
                    {(n.available_ram_mb / 1024).toFixed(1)}GB / {(n.total_ram_mb / 1024).toFixed(1)}
                    GB • {n.arch}
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}
