import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import MetricsPanel from '@/components/metrics-panel'
import type { InferenceLogEntry } from '@/lib/history'

const ENTRIES: InferenceLogEntry[] = [
  { timestamp: 1000, modelName: 'dummy-mamba-tiny', latencyMs: 120, numNodes: 2, numTokens: 20 },
  { timestamp: 2000, modelName: 'dummy-mamba-tiny', latencyMs: 340, numNodes: 2, numTokens: 30 },
  { timestamp: 3000, modelName: 'dummy-mamba-small', latencyMs: 50, numNodes: 1 },
]

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
    render(<MetricsPanel entries={ENTRIES} />)

    fireEvent.click(screen.getByRole('button', { name: 'Tokens/sec' }))

    // 20 tokens / 0.12s = 166.7 tok/s for the first entry.
    fireEvent.mouseEnter(document.querySelectorAll('rect')[0])
    expect(screen.getByText('166.7 tok/s')).toBeInTheDocument()
  })
})
