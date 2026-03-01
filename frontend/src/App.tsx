import { useState } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { cn } from './lib/utils';
import Dashboard from './pages/Dashboard';
import QuickAnalysis from './pages/QuickAnalysis';
import Timeline from './pages/Timeline';
import Alerts from './pages/Alerts';
import CrossModel from './pages/CrossModel';
import CoTViewer from './pages/CoTViewer';
import JsonlAnalyzer from './pages/JsonlAnalyzer';
import SessionLogs from './pages/SessionLogs';
import Settings from './pages/Settings';

export default function App() {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-bg-primary">
        <Sidebar collapsed={sidebarCollapsed} onToggle={() => setSidebarCollapsed(!sidebarCollapsed)} />
        <main
          className={cn(
            'flex-1 transition-all duration-300 p-6',
            sidebarCollapsed ? 'ml-[68px]' : 'ml-[240px]',
          )}
        >
          <div className="max-w-[1400px] mx-auto">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/analyze" element={<QuickAnalysis />} />
              <Route path="/timeline" element={<Timeline />} />
              <Route path="/alerts" element={<Alerts />} />
              <Route path="/cross-model" element={<CrossModel />} />
              <Route path="/cot-viewer" element={<CoTViewer />} />
              <Route path="/jsonl" element={<JsonlAnalyzer />} />
              <Route path="/sessions" element={<SessionLogs />} />
              <Route path="/settings" element={<Settings />} />
            </Routes>
          </div>
        </main>
      </div>
    </BrowserRouter>
  );
}
