import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import MetricChart, { type MetricPoint } from '@/components/metric-chart'

const POINTS: MetricPoint[] = [
  { timestamp: 1785485906733, value: 100, modelName: 'dummy-mamba-tiny', numNodes: 2 },
  { timestamp: 1785485934792, value: 400, modelName: 'dummy-mamba-tiny', numNodes: 2 },
  { timestamp: 1785486151750, value: 200, modelName: 'dummy-mamba-tiny', numNodes: 1 },
]

describe('MetricChart', () => {
  it('shows the empty message when there are no points', () => {
    render(
      <MetricChart points={[]} label="Latency" unit="ms" emptyMessage="Nothing here yet" />
    )

    expect(screen.getByText('Nothing here yet')).toBeInTheDocument()
  })

  it('renders one marker per data point', () => {
    const { container } = render(
      <MetricChart points={POINTS} label="Latency" unit="ms" emptyMessage="none" />
    )

    expect(container.querySelectorAll('circle')).toHaveLength(POINTS.length)
  })

  it('shows point details on hover', () => {
    const { container } = render(
      <MetricChart points={POINTS} label="Latency" unit="ms" emptyMessage="none" />
    )

    const hitBands = container.querySelectorAll('rect')
    fireEvent.mouseEnter(hitBands[0])

    expect(screen.getByText('100.0ms')).toBeInTheDocument()
    expect(screen.getByText(/dummy-mamba-tiny/)).toBeInTheDocument()
    expect(screen.getByText(/2 nodes/)).toBeInTheDocument()
  })

  it('switches to a log-scale y-axis when the log toggle is clicked', () => {
    render(<MetricChart points={POINTS} label="Latency" unit="ms" emptyMessage="none" />)

    expect(screen.getByRole('button', { name: 'log' })).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: 'log' }))

    // Log mode floors the axis to the smallest positive value's decade
    // (100 here), rather than starting from 0 like linear mode does.
    expect(screen.getByText('100')).toBeInTheDocument()
    expect(screen.queryByText('0')).not.toBeInTheDocument()
  })
})
