import type { LucideIcon } from 'lucide-react';
import { cn } from '../lib/utils';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  color: 'blue' | 'cyan' | 'emerald' | 'amber' | 'red' | 'violet' | 'pink';
  trend?: { value: number; label: string };
}

const colorMap = {
  blue: { border: 'border-l-accent-blue', icon: 'text-accent-blue bg-accent-blue/10', trend: 'text-accent-blue' },
  cyan: { border: 'border-l-accent-cyan', icon: 'text-accent-cyan bg-accent-cyan/10', trend: 'text-accent-cyan' },
  emerald: { border: 'border-l-accent-emerald', icon: 'text-accent-emerald bg-accent-emerald/10', trend: 'text-accent-emerald' },
  amber: { border: 'border-l-accent-amber', icon: 'text-accent-amber bg-accent-amber/10', trend: 'text-accent-amber' },
  red: { border: 'border-l-accent-red', icon: 'text-accent-red bg-accent-red/10', trend: 'text-accent-red' },
  violet: { border: 'border-l-accent-violet', icon: 'text-accent-violet bg-accent-violet/10', trend: 'text-accent-violet' },
  pink: { border: 'border-l-accent-pink', icon: 'text-accent-pink bg-accent-pink/10', trend: 'text-accent-pink' },
};

export function StatCard({ title, value, subtitle, icon: Icon, color, trend }: StatCardProps) {
  const c = colorMap[color];
  return (
    <div
      className={cn(
        'glass-card rounded-xl p-5 border-l-4 animate-fade-in',
        'hover:border-border-active transition-colors duration-200',
        c.border,
      )}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <p className="text-xs font-medium text-text-muted uppercase tracking-wider">{title}</p>
          <p className="text-2xl font-bold text-text-primary">{value}</p>
          {subtitle && <p className="text-xs text-text-secondary">{subtitle}</p>}
          {trend && (
            <div className={cn('flex items-center gap-1 text-xs font-medium', c.trend)}>
              <span>{trend.value > 0 ? '+' : ''}{trend.value}%</span>
              <span className="text-text-muted">{trend.label}</span>
            </div>
          )}
        </div>
        <div className={cn('w-10 h-10 rounded-lg flex items-center justify-center', c.icon)}>
          <Icon className="w-5 h-5" />
        </div>
      </div>
    </div>
  );
}
