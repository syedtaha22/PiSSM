// Runtime theme switching: a [data-theme] attribute on <html> plus direct
// CSS variable overrides via applyTheme() -- see the "primitives" comment
// in app/globals.css for which variables these map to.

export interface ThemeTokens {
  background: string
  card: string
  foreground: string
  mutedForeground: string
  border: string
  borderStrong: string
  primary: string
  primaryForeground: string
  accent: string
  accentForeground: string
  destructive: string
}

// The 3 background glow colors behind the nav, precomputed per mode (an
// analogous hue spread around each mode's primary color) rather than
// derived at runtime.
export interface AuroraPalette {
  c1: string
  c2: string
  c3: string
}

export interface Theme {
  id: string
  mode: 'light' | 'dark'
  tokens: ThemeTokens
  aurora: AuroraPalette
}

export const TOKEN_CSS_VARS: Record<keyof ThemeTokens, string> = {
  background: '--background',
  card: '--card',
  foreground: '--foreground',
  mutedForeground: '--muted-foreground',
  border: '--border',
  borderStrong: '--border-strong',
  primary: '--primary',
  primaryForeground: '--primary-foreground',
  accent: '--accent',
  accentForeground: '--accent-foreground',
  destructive: '--destructive',
}

const LIGHT_TOKENS: ThemeTokens = {
  background: '#fafaf9',
  card: '#ffffff',
  foreground: '#15161b',
  mutedForeground: '#5b5e68',
  border: '#e7e7e3',
  borderStrong: '#d4d4ce',
  primary: '#0d9488',
  primaryForeground: '#ffffff',
  accent: '#f0fdfa',
  accentForeground: '#0f766e',
  destructive: '#dc2626',
}

const DARK_TOKENS: ThemeTokens = {
  background: '#101114',
  card: '#1a1b20',
  foreground: '#f2f2f0',
  mutedForeground: '#a3a6ad',
  border: '#2a2c33',
  borderStrong: '#3a3d46',
  primary: '#2dd4bf',
  // Dark mode's primary is a bright, light teal -- a light foreground text
  // on it would have poor contrast, unlike light mode's darker primary.
  primaryForeground: '#101114',
  accent: '#042f2e',
  accentForeground: '#14b8a6',
  destructive: '#f87171',
}

export const THEMES: Theme[] = [
  {
    id: 'light',
    mode: 'light',
    tokens: LIGHT_TOKENS,
    aurora: {
      c1: 'hsl(174.7 72% 52% / 0.340)',
      c2: 'hsl(142.7 72% 52% / 0.289)',
      c3: 'hsl(206.7 72% 52% / 0.289)',
    },
  },
  {
    id: 'dark',
    mode: 'dark',
    tokens: DARK_TOKENS,
    aurora: {
      c1: 'hsl(172.5 75% 62% / 0.320)',
      c2: 'hsl(140.5 75% 62% / 0.272)',
      c3: 'hsl(204.5 75% 62% / 0.272)',
    },
  },
]

export function applyTheme(theme: Theme) {
  const root = document.documentElement
  for (const [key, cssVar] of Object.entries(TOKEN_CSS_VARS) as [keyof ThemeTokens, string][]) {
    root.style.setProperty(cssVar, theme.tokens[key])
  }
  root.style.colorScheme = theme.mode
  root.dataset.theme = theme.mode

  root.style.setProperty('--aurora-1', theme.aurora.c1)
  root.style.setProperty('--aurora-2', theme.aurora.c2)
  root.style.setProperty('--aurora-3', theme.aurora.c3)
}
