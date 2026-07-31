import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import Dashboard from '@/components/pages/dashboard'
import * as api from '@/lib/api'

afterEach(() => {
  vi.restoreAllMocks()
  window.localStorage.clear()
})

describe('Dashboard', () => {
  it('renders node status from listNodes()', async () => {
    vi.spyOn(api, 'listNodes').mockResolvedValue([
      {
        node_id: 'node-1',
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
      },
    ])

    render(<Dashboard />)

    expect(screen.getByText('Total Nodes')).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText('node-1')).toBeInTheDocument())
  })

  it('shows an error message when listNodes() fails', async () => {
    vi.spyOn(api, 'listNodes').mockRejectedValue(new Error('orchestrator unreachable'))

    render(<Dashboard />)

    await waitFor(() =>
      expect(screen.getByText('orchestrator unreachable')).toBeInTheDocument()
    )
  })

  it('shows a placeholder message when no prompts have been sent yet', async () => {
    vi.spyOn(api, 'listNodes').mockResolvedValue([])

    render(<Dashboard />)

    await waitFor(() =>
      expect(
        screen.getByText(/No inference requests sent yet/)
      ).toBeInTheDocument()
    )
  })

  it('plots latencies from the inference history log', async () => {
    vi.spyOn(api, 'listNodes').mockResolvedValue([])
    window.localStorage.setItem(
      'pissm-inference-history',
      JSON.stringify([
        { timestamp: 1, modelName: 'mamba-130m', latencyMs: 120, numNodes: 1 },
        { timestamp: 2, modelName: 'mamba-130m', latencyMs: 340, numNodes: 1 },
      ])
    )

    render(<Dashboard />)

    await waitFor(() =>
      expect(screen.getByText('Inference Latency (2 prompts)')).toBeInTheDocument()
    )
  })
})
