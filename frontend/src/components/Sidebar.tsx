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
  { to: '/jsonl', icon: FileJson, label: 'Eval Workbench' },
  { to: '/sessions', icon: List, label: 'Session Logs' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  return (
    <aside
      className={cn(
        'fixed left-0 top-0 h-screen bg-bg-secondary border-r border-border-default',
        'flex flex-col transition-all duration-200 z-50',
        collapsed ? 'w-[56px]' : 'w-[200px]',
      )}
    >
      {/* Logo */}
      <div className={cn(
        'flex items-center h-14 border-b border-border-default flex-shrink-0',
        collapsed ? 'justify-center' : 'gap-2.5 px-4',
      )}>
        <div className="w-8 h-8 rounded-lg bg-accent-blue/15 flex items-center justify-center flex-shrink-0">
          <Shield className="w-[18px] h-[18px] text-accent-blue" />
        </div>
        {!collapsed && (
          <div className="overflow-hidden">
            <div className="text-[13px] font-bold text-text-primary leading-tight tracking-tight">RewardHack</div>
            <div className="text-[9px] font-semibold text-accent-cyan tracking-[0.2em] uppercase leading-tight">Watch</div>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="flex-1 py-2 overflow-y-auto">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            end={item.to === '/'}
            className={({ isActive }) =>
              cn(
                'relative flex items-center h-10 text-[13px] font-medium transition-colors duration-150',
                collapsed ? 'justify-center mx-1.5 rounded-lg' : 'gap-3 px-4',
                isActive
                  ? 'text-accent-blue bg-accent-blue/8'
                  : 'text-text-muted hover:text-text-secondary hover:bg-bg-elevated/50',
                !collapsed && isActive && 'border-l-[3px] border-l-accent-blue pl-[13px]',
              )
            }
          >
            <item.icon className="w-[18px] h-[18px] flex-shrink-0" strokeWidth={1.8} />
            {!collapsed && <span className="whitespace-nowrap">{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse toggle */}
      <button
        onClick={onToggle}
        className="flex items-center justify-center h-10 border-t border-border-default text-text-muted hover:text-text-secondary transition-colors flex-shrink-0"
      >
        {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
      </button>
    </aside>
  );
}
