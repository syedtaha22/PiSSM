import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import Topology from '@/components/pages/topology'
import * as api from '@/lib/api'

const SAMPLE_NODE = {
  node_id: 'node-0',
  ip_address: '192.168.1.10',
  available_ram_mb: 2048,
  total_ram_mb: 4096,
  cpu_count: 4,
  arch: 'aarch64',
  os_name: 'Linux',
  os_version: '6.6.31',
  status: 'available',
  last_heartbeat: 100,
  first_seen: 0,
  inference_port: 50052,
}

afterEach(() => {
  vi.restoreAllMocks()
  window.localStorage.clear()
})

describe('Topology', () => {
  it('shows registered nodes with no pipeline dispatched yet', async () => {
    vi.spyOn(api, 'listNodes').mockResolvedValue([SAMPLE_NODE])
    vi.spyOn(api, 'getTopology').mockResolvedValue({ model_name: null, assignments: [] })

    render(<Topology />)

    expect(screen.getByText('Network Topology')).toBeInTheDocument()
    await waitFor(() =>
      expect(
        screen.getByText('No model loaded yet - showing registered cluster nodes.')
      ).toBeInTheDocument()
    )
    await waitFor(() => expect(screen.getByText('node-0')).toBeInTheDocument())
    expect(screen.getByText('Orchestrator')).toBeInTheDocument()
  })

  it('shows the real pipeline once a model has dispatched', async () => {
    vi.spyOn(api, 'listNodes').mockResolvedValue([SAMPLE_NODE])
    vi.spyOn(api, 'getTopology').mockResolvedValue({
      model_name: 'dummy-mamba-tiny',
      assignments: [
        {
          node_id: 'node-0',
          ip_address: '192.168.1.10',
          layer_start: 0,
          layer_end: 4,
          is_first: true,
          is_last: true,
        },
      ],
    })

    render(<Topology />)

    await waitFor(() =>
      expect(
        screen.getByText('Showing the active pipeline for "dummy-mamba-tiny".')
      ).toBeInTheDocument()
    )
    await waitFor(() => expect(screen.getByText('layers [0, 4)')).toBeInTheDocument())
    expect(screen.getByText('first + last')).toBeInTheDocument()
    expect(screen.getByText('Orchestrator')).toBeInTheDocument()
  })

  it('shows a newly joined node as idle even while a pipeline is active', async () => {
    const newNode = { ...SAMPLE_NODE, node_id: 'node-1', ip_address: '192.168.1.11' }
    vi.spyOn(api, 'listNodes').mockResolvedValue([SAMPLE_NODE, newNode])
    vi.spyOn(api, 'getTopology').mockResolvedValue({
      model_name: 'dummy-mamba-tiny',
      assignments: [
        {
          node_id: 'node-0',
          ip_address: '192.168.1.10',
          layer_start: 0,
          layer_end: 4,
          is_first: true,
          is_last: true,
        },
      ],
    })

    render(<Topology />)

    await waitFor(() => expect(screen.getByText(/1 node idle/)).toBeInTheDocument())
    // Both the dispatched node and the idle new node render in the graph,
    // and the Status list below always shows every registered node too.
    expect(screen.getAllByText('node-0').length).toBeGreaterThan(0)
    expect(screen.getAllByText('node-1').length).toBeGreaterThan(0)
  })

  it('restores a previously saved node position after remounting (e.g. switching tabs and back)', async () => {
    window.localStorage.setItem(
      'pissm-topology-node-positions',
      JSON.stringify({ 'node-0': { x: 777, y: 888 } })
    )
    vi.spyOn(api, 'listNodes').mockResolvedValue([SAMPLE_NODE])
    vi.spyOn(api, 'getTopology').mockResolvedValue({ model_name: null, assignments: [] })

    const { container } = render(<Topology />)

    await waitFor(() => expect(screen.getByText('node-0')).toBeInTheDocument())

    const nodeEl = container.querySelector('[data-id="node-0"]') as HTMLElement | null
    expect(nodeEl).not.toBeNull()
    expect(nodeEl?.style.transform).toContain('777px')
  })

  it('shows an error message when the orchestrator is unreachable', async () => {
    vi.spyOn(api, 'listNodes').mockRejectedValue(new Error('orchestrator unreachable'))
    vi.spyOn(api, 'getTopology').mockResolvedValue({ model_name: null, assignments: [] })

    render(<Topology />)

    await waitFor(() =>
      expect(screen.getByText('orchestrator unreachable')).toBeInTheDocument()
    )
  })
})
