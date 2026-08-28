'use client'

import { useState } from 'react'
import { Navbar, type Page } from '@/components/navbar'
import Stats from '@/components/pages/stats'
import Models from '@/components/pages/models'
import Inference from '@/components/pages/inference'
import Topology from '@/components/pages/topology'

export default function Home() {
  const [currentPage, setCurrentPage] = useState<Page>('stats')

  return (
    <div className="flex h-screen flex-col">
      <Navbar currentPage={currentPage} onNavigate={setCurrentPage} />
      {/* No background fill here (or on the page components below) --
          left transparent so the aurora glow + dot grid mounted in
          layout.tsx show through. */}
      <main className="flex-1 overflow-auto pt-6">
        {/* Every page stays mounted once visited (hidden via CSS instead
            of unmounted) so switching tabs never resets component state -
            the Inference page in particular would otherwise re-fetch
            models and re-trigger a model load on every tab switch, and
            lose its chat history and in-flight status each time. */}
        <div className={currentPage === 'stats' ? 'h-full' : 'hidden'}>
          <Stats />
        </div>
        <div className={currentPage === 'models' ? 'h-full' : 'hidden'}>
          <Models />
        </div>
        <div className={currentPage === 'inference' ? 'h-full' : 'hidden'}>
          <Inference />
        </div>
        <div className={currentPage === 'topology' ? 'h-full' : 'hidden'}>
          <Topology />
        </div>
      </main>
    </div>
  )
}
