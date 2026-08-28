import type { Metadata, Viewport } from 'next'
import { Bricolage_Grotesque, Source_Serif_4, JetBrains_Mono } from 'next/font/google'
import './globals.css'
import { ThemeSwitcher } from '@/components/theme-switcher'
import { InteractiveDotGrid } from '@/components/effects/interactive-dot-grid'

// Bricolage Grotesque is the sole UI face (both --font-display and
// --font-sans -- see globals.css). Source Serif 4 is for longer
// descriptive copy, JetBrains Mono for data values (RAM, latency, IPs).
const bricolage = Bricolage_Grotesque({
  subsets: ['latin'],
  variable: '--font-bricolage',
  weight: ['400', '500', '600', '700', '800'],
})

const sourceSerif = Source_Serif_4({
  subsets: ['latin'],
  variable: '--font-source-serif',
  weight: ['400', '600'],
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains-mono',
  weight: ['400', '500'],
})

export const metadata: Metadata = {
  title: 'PiSSM - Distributed Inference Control',
  description: 'Distributed State Space Model Inference System on Raspberry Pi Cluster',
  generator: 'v0.app',
  icons: {
    icon: '/icon.svg',
  },
}

export const viewport: Viewport = {
  colorScheme: 'light',
  themeColor: '#fafaf9',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      className={`${bricolage.variable} ${sourceSerif.variable} ${jetbrainsMono.variable}`}
    >
      <body className="antialiased bg-background text-foreground font-sans">
        <InteractiveDotGrid />
        {children}
        <ThemeSwitcher />
      </body>
    </html>
  )
}
