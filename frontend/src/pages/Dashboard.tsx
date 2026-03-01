import { useState, useEffect } from 'react';
import {
  PieChart, Pie, Cell, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid,
  AreaChart, Area,
} from 'recharts';
import type { DashboardStats, RiskLevel } from '../lib/types';
import { demoData } from '../lib/api';
import { StatCard } from '../components/StatCard';
import { ChartCard } from '../components/ChartCard';
import { PageHeader } from '../components/PageHeader';
import { RiskBadge } from '../components/Badge';
import { formatRelative, categoryLabel, riskBorder } from '../lib/utils';

const RISK_COLORS: Record<RiskLevel, string> = {
  critical: '#ef4444',
  high: '#f97316',
  medium: '#f59e0b',
  low: '#10b981',
  none: '#64748b',
};

const CATEGORY_COLORS = [
  '#3b82f6', '#06b6d4', '#10b981', '#f59e0b',
  '#ef4444', '#8b5cf6', '#ec4899', '#f97316',
];

const TOOLTIP_STYLE = {
  contentStyle: { backgroundColor: '#292A30', border: '1px solid #363840', borderRadius: '8px', fontSize: '12px' },
  labelStyle: { color: '#e2e8f0' },
  itemStyle: { color: '#94a3b8' },
};

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);

  useEffect(() => {
    setStats(demoData.stats());
  }, []);

  if (!stats) return null;

  const riskPieData = Object.entries(stats.risk_distribution)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => ({ name: k, value: v, fill: RISK_COLORS[k as RiskLevel] }));

  const SHORT_LABELS: Record<string, string> = {
    oversight_subversion: 'Oversight Subv.',
    test_manipulation: 'Test Manip.',
    alignment_faking: 'Align. Faking',
  };
  const categoryBarData = Object.entries(stats.category_distribution)
    .map(([k, v]) => ({ name: SHORT_LABELS[k] ?? categoryLabel(k), value: v, key: k }))
    .sort((a, b) => b.value - a.value);

  const timelineData = stats.timeline.map(t => ({
    step: t.step,
    'Hack Score': t.hack_score,
    'Misalignment': t.misalignment_score,
    'RMGI': t.rmgi,
  }));

  return (
    <div className="space-y-6">
      <PageHeader
        title="Dashboard"
        subtitle={`${stats.total_analyzed.toLocaleString()} trajectories analyzed`}
      />

      {/* Stat Cards — 5 across on wide, 3+2 on medium */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
        <StatCard title="Total Analyzed" value={stats.total_analyzed.toLocaleString()} color="blue" />
        <StatCard title="Flagged" value={stats.total_flagged} subtitle={`${((stats.total_flagged / stats.total_analyzed) * 100).toFixed(1)}% detection rate`} color="emerald" />
        <StatCard title="Critical" value={stats.critical_count} color="amber" trend={{ value: -12, label: 'vs last week' }} />
        <StatCard title="Avg ML Score" value={(stats.avg_ml_score * 100).toFixed(1) + '%'} subtitle="Threshold: 2.0%" color="violet" />
        <StatCard title="High Risk" value={stats.high_count} color="red" />
      </div>

      {/* Charts — responsive: 1 col at <1024, 2 at 1024-1919, 3 at 1920+ */}
      <div className="grid grid-cols-1 lg:grid-cols-2 2xl:grid-cols-3 gap-4">
        {/* Risk Distribution Donut */}
        <ChartCard title="Risk Distribution" subtitle="Across all trajectories">
          <div className="h-[320px] flex flex-col items-center">
            <div className="flex-1 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={riskPieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={70}
                    outerRadius={100}
                    paddingAngle={3}
                    dataKey="value"
                    stroke="none"
                  >
                    {riskPieData.map((entry) => (
                      <Cell key={entry.name} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip {...TOOLTIP_STYLE} />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="flex flex-wrap gap-4 justify-center pt-2">
              {riskPieData.map(d => (
                <div key={d.name} className="flex items-center gap-1.5 text-xs text-text-secondary">
                  <div className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.fill }} />
                  <span className="capitalize">{d.name}</span>
                  <span className="text-text-muted tabular-nums">({d.value})</span>
                </div>
              ))}
            </div>
          </div>
        </ChartCard>

        {/* Category Bar Chart */}
        <ChartCard title="Category Breakdown" subtitle="Hack type distribution">
          <div className="h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryBarData} layout="vertical" margin={{ left: 4, right: 16, top: 4, bottom: 4 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#363840" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis
                  dataKey="name"
                  type="category"
                  tick={{ fill: '#94a3b8', fontSize: 10 }}
                  axisLine={false}
                  tickLine={false}
                  width={100}
                />
                <Tooltip {...TOOLTIP_STYLE} />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={22}>
                  {categoryBarData.map((_, i) => (
                    <Cell key={i} fill={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        {/* RMGI Timeline */}
        <ChartCard title="RMGI Timeline" subtitle="Recent trajectory scores">
          <div className="h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timelineData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
                <defs>
                  <linearGradient id="hackGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="rmgiGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#363840" />
                <XAxis dataKey="step" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} domain={[0, 1]} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Area type="monotone" dataKey="Hack Score" stroke="#ef4444" fill="url(#hackGrad)" strokeWidth={2} />
                <Area type="monotone" dataKey="RMGI" stroke="#3b82f6" fill="url(#rmgiGrad)" strokeWidth={2} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>
      </div>

      {/* Recent Alerts */}
      <ChartCard title="Recent Alerts" subtitle="Latest flagged trajectories">
        <div className="space-y-2">
          {stats.recent_alerts.map(alert => (
            <div
              key={alert.id}
              className={`flex items-center gap-4 px-5 py-3.5 rounded-lg bg-bg-primary/40 border-l-[3px] ${riskBorder(alert.severity)} hover:bg-bg-elevated/30 transition-colors`}
            >
              <RiskBadge level={alert.severity} size="sm" />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-text-primary truncate">{alert.message}</p>
                <p className="text-[11px] text-text-muted mt-0.5">{formatRelative(alert.timestamp)}</p>
              </div>
              <span className="text-[11px] text-text-muted font-mono tabular-nums">{alert.trajectory_id}</span>
            </div>
          ))}
        </div>
      </ChartCard>
    </div>
  );
}
