import type { RiskLevel } from './types';

export function riskColor(level: RiskLevel): string {
  const map: Record<RiskLevel, string> = {
    critical: 'text-risk-critical',
    high: 'text-risk-high',
    medium: 'text-risk-medium',
    low: 'text-risk-low',
    none: 'text-risk-none',
  };
  return map[level] ?? 'text-text-secondary';
}

export function riskBg(level: RiskLevel): string {
  const map: Record<RiskLevel, string> = {
    critical: 'bg-risk-critical/15 text-risk-critical border-risk-critical/30',
    high: 'bg-risk-high/15 text-risk-high border-risk-high/30',
    medium: 'bg-risk-medium/15 text-risk-medium border-risk-medium/30',
    low: 'bg-risk-low/15 text-risk-low border-risk-low/30',
    none: 'bg-text-muted/15 text-text-secondary border-text-muted/30',
  };
  return map[level] ?? '';
}

export function riskBorder(level: RiskLevel): string {
  const map: Record<RiskLevel, string> = {
    critical: 'border-l-risk-critical',
    high: 'border-l-risk-high',
    medium: 'border-l-risk-medium',
    low: 'border-l-risk-low',
    none: 'border-l-text-muted',
  };
  return map[level] ?? '';
}

export function formatScore(score: number): string {
  return (score * 100).toFixed(1) + '%';
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

export function formatRelative(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

export function cn(...classes: (string | false | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ');
}

const ACRONYMS: Record<string, string> = { cot: 'CoT', rmgi: 'RMGI', llm: 'LLM' };

export function categoryLabel(cat: string): string {
  return cat
    .split('_')
    .map(w => ACRONYMS[w] ?? w.charAt(0).toUpperCase() + w.slice(1))
    .join(' ');
}
