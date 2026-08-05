'use client'

import { useState } from 'react'
import Header from '@/components/header'
import Sidebar, { type Page } from '@/components/sidebar'
import Dashboard from '@/components/pages/dashboard'
import Models from '@/components/pages/models'
import Inference from '@/components/pages/inference'
import Topology from '@/components/pages/topology'

export default function Home() {
  const [currentPage, setCurrentPage] = useState<Page>('dashboard')

  return (
    <div className="flex h-screen bg-background">
      <Sidebar currentPage={currentPage} onNavigate={setCurrentPage} />
      <div className="flex-1 flex flex-col overflow-hidden">
        <Header />
        <main className="flex-1 overflow-auto bg-background">
          {/* Every page stays mounted once visited (hidden via CSS instead
              of unmounted) so switching tabs never resets component state -
              the Inference page in particular would otherwise re-fetch
              models and re-trigger a model load on every tab switch, and
              lose its chat history and in-flight status each time. */}
          <div className={currentPage === 'dashboard' ? 'h-full' : 'hidden'}>
            <Dashboard />
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
    </div>
  )
}
