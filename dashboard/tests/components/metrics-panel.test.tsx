import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import MetricsPanel from '@/components/metrics-panel'
import type { InferenceLogEntry } from '@/lib/history'

const ENTRIES: InferenceLogEntry[] = [
  { timestamp: 1000, modelName: 'dummy-mamba-tiny', latencyMs: 120, numNodes: 2, numTokens: 20 },
  { timestamp: 2000, modelName: 'dummy-mamba-tiny', latencyMs: 340, numNodes: 2, numTokens: 30 },
  { timestamp: 3000, modelName: 'dummy-mamba-small', latencyMs: 50, numNodes: 1 },
]

// jsdom always reports 0x0 for getBoundingClientRect, so Recharts'
// ResponsiveContainer (used inside MetricChart) never renders anything -
// same class of limitation documented for @xyflow/react in
// topology.test.tsx. Stubbed here (scoped to this file only) so the
// Tokens/sec test below can inspect real chart output.
beforeEach(() => {
  vi.spyOn(Element.prototype, 'getBoundingClientRect').mockReturnValue({
    width: 800,
    height: 400,
    top: 0,
    left: 0,
    bottom: 400,
    right: 800,
    x: 0,
    y: 0,
    toJSON() {},
  })
})

afterEach(() => {
  vi.restoreAllMocks()
})

describe('MetricsPanel', () => {
  it('shows the total prompt count with no filters applied', () => {
    render(<MetricsPanel entries={ENTRIES} />)

    expect(screen.getByText('Inference Metrics (3 prompts)')).toBeInTheDocument()
  })

  it('filters by model', () => {
    render(<MetricsPanel entries={ENTRIES} />)

    fireEvent.change(screen.getByLabelText('Filter by model'), {
      target: { value: 'dummy-mamba-tiny' },
    })

    expect(screen.getByText('Inference Metrics (2 prompts)')).toBeInTheDocument()
  })

  it('filters by node count', () => {
    render(<MetricsPanel entries={ENTRIES} />)

    fireEvent.change(screen.getByLabelText('Filter by node count'), {
      target: { value: '1' },
    })

    expect(screen.getByText('Inference Metrics (1 prompt)')).toBeInTheDocument()
  })

  it('switching to the Tokens/sec tab only plots entries with token counts', () => {
    const { container } = render(<MetricsPanel entries={ENTRIES} />)

    fireEvent.click(screen.getByRole('button', { name: 'Tokens/sec' }))

    // All 3 entries pass the (no) filter, but only 2 have numTokens set -
    // the third (dummy-mamba-small) predates that field and must be
    // excluded rather than plotted as a broken/NaN point.
    expect(container.querySelectorAll('.recharts-line-dot')).toHaveLength(2)
  })
})
