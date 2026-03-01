import { cn } from '../lib/utils';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  color: 'blue' | 'cyan' | 'emerald' | 'amber' | 'red' | 'violet' | 'pink';
  trend?: { value: number; label: string };
}

const borderMap: Record<string, string> = {
  blue: 'border-l-accent-blue',
  cyan: 'border-l-accent-cyan',
  emerald: 'border-l-accent-emerald',
  amber: 'border-l-accent-amber',
  red: 'border-l-accent-red',
  violet: 'border-l-accent-violet',
  pink: 'border-l-accent-pink',
};

const trendMap: Record<string, string> = {
  blue: 'text-accent-blue',
  cyan: 'text-accent-cyan',
  emerald: 'text-accent-emerald',
  amber: 'text-accent-amber',
  red: 'text-accent-red',
  violet: 'text-accent-violet',
  pink: 'text-accent-pink',
};

export function StatCard({ title, value, subtitle, color, trend }: StatCardProps) {
  return (
    <div
      className={cn(
        'card border-l-4 animate-fade-in',
        borderMap[color],
      )}
    >
      <p className="text-[11px] font-semibold text-text-muted uppercase tracking-wider mb-2">{title}</p>
      <p className="text-[32px] font-bold text-text-primary leading-none tabular-nums">{value}</p>
      {subtitle && <p className="text-xs text-text-secondary mt-2">{subtitle}</p>}
      {trend && (
        <div className={cn('flex items-center gap-1.5 text-xs font-medium mt-2', trendMap[color])}>
          <span className="tabular-nums">{trend.value > 0 ? '+' : ''}{trend.value}%</span>
          <span className="text-text-muted">{trend.label}</span>
        </div>
      )}
    </div>
  );
}
