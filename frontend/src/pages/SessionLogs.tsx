import { useState, useEffect } from 'react';
import { List, Search, Download, Filter } from 'lucide-react';
import type { SessionLog, RiskLevel } from '../lib/types';
import { demoData } from '../lib/api';
import { PageHeader } from '../components/PageHeader';
import { DataTable } from '../components/DataTable';
import { RiskBadge, CategoryBadge } from '../components/Badge';
import { formatDate, formatScore, cn } from '../lib/utils';

export default function SessionLogs() {
  const [sessions, setSessions] = useState<SessionLog[]>([]);
  const [search, setSearch] = useState('');
  const [riskFilter, setRiskFilter] = useState<RiskLevel | 'all'>('all');
  const [sourceFilter, setSourceFilter] = useState<string>('all');

  useEffect(() => {
    setSessions(demoData.sessions());
  }, []);

  const sources = [...new Set(sessions.map(s => s.source))];

  const filtered = sessions.filter(s => {
    if (riskFilter !== 'all' && s.risk_level !== riskFilter) return false;
    if (sourceFilter !== 'all' && s.source !== sourceFilter) return false;
    if (search && !s.session_id.includes(search) && !s.source.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  function handleExport() {
    const blob = new Blob([JSON.stringify(filtered, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `session_logs_${new Date().toISOString().slice(0, 19)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const columns = [
    {
      key: 'session_id',
      header: 'Session',
      render: (row: SessionLog) => <span className="font-mono text-xs text-text-secondary">{row.session_id}</span>,
      sortable: true,
      sortValue: (row: SessionLog) => row.session_id,
    },
    {
      key: 'timestamp',
      header: 'Time',
      render: (row: SessionLog) => <span className="text-xs text-text-muted">{formatDate(row.timestamp)}</span>,
      sortable: true,
      sortValue: (row: SessionLog) => row.timestamp,
    },
    {
      key: 'source',
      header: 'Source',
      render: (row: SessionLog) => (
        <span className="text-xs px-2 py-1 rounded bg-bg-elevated text-text-secondary font-medium">{row.source}</span>
      ),
    },
    {
      key: 'risk',
      header: 'Risk',
      render: (row: SessionLog) => <RiskBadge level={row.risk_level} size="sm" />,
      sortable: true,
      sortValue: (row: SessionLog) => {
        const order = { critical: 4, high: 3, medium: 2, low: 1, none: 0 };
        return order[row.risk_level] ?? 0;
      },
    },
    {
      key: 'ml_score',
      header: 'ML Score',
      render: (row: SessionLog) => <span className="font-mono text-xs">{formatScore(row.ml_score)}</span>,
      sortable: true,
      sortValue: (row: SessionLog) => row.ml_score,
    },
    {
      key: 'detections',
      header: 'Detections',
      render: (row: SessionLog) => (
        <span className={cn('font-mono text-xs', row.detection_count > 0 ? 'text-risk-critical font-semibold' : 'text-text-muted')}>
          {row.detection_count}
        </span>
      ),
      sortable: true,
      sortValue: (row: SessionLog) => row.detection_count,
    },
    {
      key: 'turns',
      header: 'Turns',
      render: (row: SessionLog) => <span className="text-xs text-text-muted">{row.n_turns}</span>,
      sortable: true,
      sortValue: (row: SessionLog) => row.n_turns,
    },
    {
      key: 'categories',
      header: 'Categories',
      render: (row: SessionLog) => (
        <div className="flex gap-1 flex-wrap">
          {row.categories.slice(0, 2).map(c => <CategoryBadge key={c} category={c} />)}
          {row.categories.length > 2 && (
            <span className="text-[10px] text-text-muted">+{row.categories.length - 2}</span>
          )}
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Session Logs"
        subtitle={`${filtered.length} of ${sessions.length} sessions`}
        actions={
          <button
            onClick={handleExport}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-bg-elevated text-text-secondary hover:text-text-primary transition-colors"
          >
            <Download className="w-3.5 h-3.5" /> Export
          </button>
        }
      />

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3">
        <div className="relative flex-1 max-w-sm">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted" />
          <input
            type="text"
            placeholder="Search by ID or source..."
            value={search}
            onChange={e => setSearch(e.target.value)}
            className="w-full pl-10 pr-4 py-2 rounded-lg bg-bg-card border border-border-default text-sm text-text-primary placeholder:text-text-muted focus:border-accent-blue focus:outline-none"
          />
        </div>

        <select
          value={riskFilter}
          onChange={e => setRiskFilter(e.target.value as RiskLevel | 'all')}
          className="px-3 py-2 rounded-lg bg-bg-card border border-border-default text-sm text-text-primary focus:border-accent-blue focus:outline-none"
        >
          <option value="all">All Risk Levels</option>
          <option value="critical">Critical</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
          <option value="none">None</option>
        </select>

        <select
          value={sourceFilter}
          onChange={e => setSourceFilter(e.target.value)}
          className="px-3 py-2 rounded-lg bg-bg-card border border-border-default text-sm text-text-primary focus:border-accent-blue focus:outline-none"
        >
          <option value="all">All Sources</option>
          {sources.map(s => <option key={s} value={s}>{s}</option>)}
        </select>
      </div>

      {/* Table */}
      <DataTable
        columns={columns}
        data={filtered}
        keyFn={row => row.session_id}
        emptyMessage="No sessions match your filters"
      />
    </div>
  );
}
