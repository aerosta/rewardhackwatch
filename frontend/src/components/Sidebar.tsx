import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Zap,
  Clock,
  Bell,
  GitCompareArrows,
  BrainCircuit,
  FileJson,
  List,
  Settings,
  ChevronLeft,
  ChevronRight,
  Shield,
} from 'lucide-react';
import { cn } from '../lib/utils';

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

const NAV_ITEMS = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/analyze', icon: Zap, label: 'Quick Analysis' },
  { to: '/timeline', icon: Clock, label: 'Timeline' },
  { to: '/alerts', icon: Bell, label: 'Alerts' },
  { to: '/cross-model', icon: GitCompareArrows, label: 'Cross-Model' },
  { to: '/cot-viewer', icon: BrainCircuit, label: 'CoT Viewer' },
  { to: '/jsonl', icon: FileJson, label: 'JSONL Analyzer' },
  { to: '/sessions', icon: List, label: 'Session Logs' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  return (
    <aside
      className={cn(
        'fixed left-0 top-0 h-screen bg-bg-secondary border-r border-border-default',
        'flex flex-col transition-all duration-300 z-50',
        collapsed ? 'w-[68px]' : 'w-[240px]',
      )}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-4 h-16 border-b border-border-default">
        <div className="w-9 h-9 rounded-lg bg-accent-blue/20 flex items-center justify-center flex-shrink-0">
          <Shield className="w-5 h-5 text-accent-blue" />
        </div>
        {!collapsed && (
          <div className="animate-fade-in">
            <div className="text-sm font-bold text-text-primary tracking-tight">RewardHack</div>
            <div className="text-[10px] font-medium text-accent-cyan tracking-widest uppercase">Watch</div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-200',
                'hover:bg-bg-elevated hover:text-text-primary group',
                isActive
                  ? 'bg-accent-blue/10 text-accent-blue'
                  : 'text-text-secondary',
              )
            }
          >
            <item.icon className={cn('w-[18px] h-[18px] flex-shrink-0')} />
            {!collapsed && <span className="animate-fade-in truncate">{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={onToggle}
        className="flex items-center justify-center h-12 border-t border-border-default text-text-muted hover:text-text-primary transition-colors"
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>
    </aside>
  );
}
