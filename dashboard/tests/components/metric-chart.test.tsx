import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import MetricChart, { type MetricPoint } from '@/components/metric-chart'

const POINTS: MetricPoint[] = [
  { timestamp: 1785485906733, value: 100, modelName: 'dummy-mamba-tiny', numNodes: 2 },
  { timestamp: 1785485934792, value: 400, modelName: 'dummy-mamba-tiny', numNodes: 2 },
  { timestamp: 1785486151750, value: 200, modelName: 'dummy-mamba-tiny', numNodes: 1 },
]

// jsdom always reports 0x0 for getBoundingClientRect, so Recharts'
// ResponsiveContainer (which measures its container before rendering
// anything) never produces a chart - same class of limitation already
// documented for @xyflow/react in topology.test.tsx. Stubbing a real
// size here (scoped to this file only) lets these tests exercise the
// actual rendered chart instead of settling for a blank container.
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

describe('MetricChart', () => {
  it('shows the empty message when there are no points', () => {
    render(
      <MetricChart points={[]} label="Latency" unit="ms" emptyMessage="Nothing here yet" />
    )

    expect(screen.getByText('Nothing here yet')).toBeInTheDocument()
  })

  it('renders one line dot per data point', () => {
    const { container } = render(
      <MetricChart points={POINTS} label="Latency" unit="ms" emptyMessage="none" />
    )

    expect(container.querySelectorAll('.recharts-line-dot')).toHaveLength(POINTS.length)
  })

  it('toggles the active scale button between linear and log', () => {
    render(<MetricChart points={POINTS} label="Latency" unit="ms" emptyMessage="none" />)

    const linearButton = screen.getByRole('button', { name: 'linear' })
    const logButton = screen.getByRole('button', { name: 'log' })
    expect(linearButton.className).toContain('bg-primary')
    expect(logButton.className).not.toContain('bg-primary')

    fireEvent.click(logButton)

    expect(logButton.className).toContain('bg-primary')
    expect(linearButton.className).not.toContain('bg-primary')
  })

  it('renders compact x-axis timestamp labels with no rotation/tilt', () => {
    const { container } = render(
      <MetricChart points={POINTS} label="Latency" unit="ms" emptyMessage="none" />
    )

    const xAxisText = container.querySelector('.recharts-xAxis-tick-labels')?.textContent ?? ''
    // formatCompactTimestamp renders e.g. "7/31" + "1:22pm" (Recharts wraps
    // long tick text onto its own lines rather than rotating it).
    expect(xAxisText.length).toBeGreaterThan(0)
    expect(
      container.querySelector('.recharts-xAxis-tick-labels text[transform*="rotate"]')
    ).toBeNull()
  })
})
