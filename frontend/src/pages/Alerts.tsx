import { useState, useEffect } from 'react';
import { Bell, Check, Filter, Search } from 'lucide-react';
import type { Alert, RiskLevel } from '../lib/types';
import { demoData } from '../lib/api';
import { PageHeader } from '../components/PageHeader';
import { RiskBadge, CategoryBadge } from '../components/Badge';
import { cn, formatRelative, riskBorder } from '../lib/utils';

export default function Alerts() {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [filter, setFilter] = useState<RiskLevel | 'all'>('all');
  const [search, setSearch] = useState('');
  const [showAcknowledged, setShowAcknowledged] = useState(false);

  useEffect(() => {
    setAlerts(demoData.alerts());
  }, []);

  const filtered = alerts.filter(a => {
    if (filter !== 'all' && a.severity !== filter) return false;
    if (!showAcknowledged && a.acknowledged) return false;
    if (search && !a.message.toLowerCase().includes(search.toLowerCase()) && !a.category.includes(search.toLowerCase())) return false;
    return true;
  });

  const severityCounts = alerts.reduce((acc, a) => {
    if (!a.acknowledged) acc[a.severity] = (acc[a.severity] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  function handleAcknowledge(id: string) {
    setAlerts(prev => prev.map(a => a.id === id ? { ...a, acknowledged: true } : a));
  }

  function handleAcknowledgeAll() {
    setAlerts(prev => prev.map(a => ({ ...a, acknowledged: true })));
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Alerts"
        subtitle={`${filtered.length} active alerts`}
        actions={
          <button
            onClick={handleAcknowledgeAll}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-bg-elevated text-text-secondary hover:text-text-primary transition-colors"
          >
            <Check className="w-3.5 h-3.5" />
            Acknowledge All
          </button>
        }
      />

      {/* Severity Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {(['critical', 'high', 'medium', 'low'] as RiskLevel[]).map(level => (
          <button
            key={level}
            onClick={() => setFilter(filter === level ? 'all' : level)}
            className={cn(
              'card text-left transition-all border-l-4',
              riskBorder(level),
              filter === level && 'ring-1 ring-accent-blue/50',
            )}
          >
            <p className="text-[11px] font-semibold text-text-muted uppercase tracking-wider capitalize">{level}</p>
            <p className="text-[28px] font-bold font-heading text-text-primary leading-none mt-2 tabular-nums">{severityCounts[level] || 0}</p>
          </button>
        ))}
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3">
        <div className="relative flex-1 max-w-md">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            placeholder="Search alerts..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 rounded-lg bg-bg-card border border-border-default text-sm text-text-primary placeholder:text-text-muted focus:border-accent-blue focus:outline-none"
          />
        </div>
        <button
          onClick={() => setShowAcknowledged(!showAcknowledged)}
          className={cn(
            'flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium transition-colors',
            showAcknowledged
              ? 'bg-accent-blue/10 text-accent-blue'
              : 'bg-bg-elevated text-text-muted hover:text-text-secondary',
          )}
        >
          <Filter className="w-3.5 h-3.5" />
          {showAcknowledged ? 'Showing all' : 'Hide acknowledged'}
        </button>
      </div>

      {/* Alert Feed */}
      <div className="space-y-2">
        {filtered.length === 0 ? (
          <div className="card flex flex-col items-center justify-center py-16">
            <Bell className="w-12 h-12 text-text-muted mb-3" />
            <p className="text-sm text-text-muted">No alerts match your filters</p>
          </div>
        ) : (
          filtered.map(alert => (
            <div
              key={alert.id}
              className={cn(
                'card border-l-4 animate-slide-in transition-all',
                riskBorder(alert.severity),
                alert.acknowledged && 'opacity-50',
              )}
            >
              <div className="flex items-start gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1.5">
                    <RiskBadge level={alert.severity} size="sm" />
                    <CategoryBadge category={alert.category} />
                    <span className="text-xs text-text-muted ml-auto">{formatRelative(alert.timestamp)}</span>
                  </div>
                  <p className="text-sm text-text-primary">{alert.message}</p>
                  {alert.details && (
                    <p className="text-xs text-text-secondary mt-1.5">{alert.details}</p>
                  )}
                  <p className="text-xs text-text-muted font-mono mt-2">ID: {alert.trajectory_id}</p>
                </div>
                {!alert.acknowledged && (
                  <button
                    onClick={() => handleAcknowledge(alert.id)}
                    className="flex-shrink-0 p-2 rounded-lg bg-bg-elevated hover:bg-accent-emerald/10 text-text-muted hover:text-accent-emerald transition-colors"
                    title="Acknowledge"
                  >
                    <Check className="w-4 h-4" />
                  </button>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
