/**
 * Shared chart styling: CSS custom property references rather than
 * literal hex, so the theme switcher's runtime overrides apply to charts
 * the same way they apply to any other themed element.
 */
export const chartAxisStyle = {
  tick: { fontSize: 12, fill: 'var(--muted-foreground)' },
  axisLine: { stroke: 'var(--border)' },
  tickLine: false as const,
}

export const chartGridStroke = 'var(--border)'
export const chartLineStroke = 'var(--primary)'

export const tooltipContentStyle = {
  fontSize: 13,
  borderRadius: 6,
  backgroundColor: 'var(--popover)',
  borderColor: 'var(--border)',
  color: 'var(--popover-foreground)',
}

export const tooltipLabelStyle = { color: 'var(--popover-foreground)' }
export const tooltipItemStyle = { color: 'var(--popover-foreground)' }
