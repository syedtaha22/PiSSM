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
          {currentPage === 'dashboard' && <Dashboard />}
          {currentPage === 'models' && <Models />}
          {currentPage === 'inference' && <Inference />}
          {currentPage === 'topology' && <Topology />}
        </main>
      </div>
    </div>
  )
}
