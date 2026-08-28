import { render, screen, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, afterEach } from 'vitest'
import { Navbar } from '@/components/navbar'
import * as api from '@/lib/api'

afterEach(() => {
  vi.restoreAllMocks()
})

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

describe('Navbar', () => {
  it('shows the real registered node count instead of a hardcoded number', async () => {
    vi.spyOn(api, 'listNodes').mockResolvedValue([SAMPLE_NODE, { ...SAMPLE_NODE, node_id: 'node-1' }])

    render(<Navbar currentPage="stats" onNavigate={() => {}} />)

    await waitFor(() => expect(screen.getByTitle('2 nodes registered')).toHaveTextContent('2'))
  })

  it('shows 0 when no nodes are registered', async () => {
    vi.spyOn(api, 'listNodes').mockResolvedValue([])

    render(<Navbar currentPage="stats" onNavigate={() => {}} />)

    await waitFor(() => expect(screen.getByTitle('0 nodes registered')).toHaveTextContent('0'))
  })
})
