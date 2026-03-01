import type { RiskLevel } from '../lib/types';
import { riskBg, cn } from '../lib/utils';

interface BadgeProps {
  level: RiskLevel;
  size?: 'sm' | 'md';
}

export function RiskBadge({ level, size = 'md' }: BadgeProps) {
  return (
    <span
      className={cn(
        'inline-flex items-center font-semibold uppercase tracking-wider border rounded-full',
        riskBg(level),
        size === 'sm' ? 'text-[10px] px-2 py-0.5' : 'text-xs px-2.5 py-1',
      )}
    >
      {level}
    </span>
  );
}

interface CategoryBadgeProps {
  category: string;
}

export function CategoryBadge({ category }: CategoryBadgeProps) {
  return (
    <span className="inline-flex items-center text-[10px] font-medium uppercase tracking-wider px-2 py-0.5 rounded-full bg-accent-violet/10 text-accent-violet border border-accent-violet/20">
      {category.replace(/_/g, ' ')}
    </span>
  );
}
