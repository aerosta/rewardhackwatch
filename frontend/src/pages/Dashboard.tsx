import { useState, useEffect } from 'react';
import {
  BarChart3,
  Shield,
  AlertTriangle,
  Activity,
  Target,
} from 'lucide-react';
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
import { formatRelative, categoryLabel } from '../lib/utils';

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

export default function Dashboard() {
  const [stats, setStats] = useState<DashboardStats | null>(null);

  useEffect(() => {
    setStats(demoData.stats());
  }, []);

  if (!stats) return null;

  const riskPieData = Object.entries(stats.risk_distribution)
    .filter(([, v]) => v > 0)
    .map(([k, v]) => ({ name: k, value: v, fill: RISK_COLORS[k as RiskLevel] }));

  const categoryBarData = Object.entries(stats.category_distribution)
    .map(([k, v]) => ({ name: categoryLabel(k), value: v, key: k }))
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
        subtitle={`${stats.total_analysed.toLocaleString()} trajectories analysed`}
      />

      {/* Stat Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-5 gap-4">
        <StatCard
          title="Total Analysed"
          value={stats.total_analysed.toLocaleString()}
          icon={BarChart3}
          color="blue"
        />
        <StatCard
          title="Flagged"
          value={stats.total_flagged}
          subtitle={`${((stats.total_flagged / stats.total_analysed) * 100).toFixed(1)}% detection rate`}
          icon={Shield}
          color="red"
        />
        <StatCard
          title="Critical"
          value={stats.critical_count}
          icon={AlertTriangle}
          color="red"
          trend={{ value: -12, label: 'vs last week' }}
        />
        <StatCard
          title="Avg ML Score"
          value={(stats.avg_ml_score * 100).toFixed(1) + '%'}
          subtitle="Threshold: 2.0%"
          icon={Activity}
          color="cyan"
        />
        <StatCard
          title="High Risk"
          value={stats.high_count}
          icon={Target}
          color="amber"
        />
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Risk Distribution Donut */}
        <ChartCard title="Risk Distribution" subtitle="Across all trajectories">
          <div className="h-[240px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={riskPieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={90}
                  paddingAngle={3}
                  dataKey="value"
                  stroke="none"
                >
                  {riskPieData.map((entry) => (
                    <Cell key={entry.name} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e1e3a', border: '1px solid #2a2a4a', borderRadius: '8px', fontSize: '12px' }}
                  labelStyle={{ color: '#e2e8f0' }}
                  itemStyle={{ color: '#94a3b8' }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex flex-wrap gap-3 mt-2 justify-center">
            {riskPieData.map(d => (
              <div key={d.name} className="flex items-center gap-1.5 text-xs text-text-secondary">
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: d.fill }} />
                <span className="capitalize">{d.name}</span>
                <span className="text-text-muted">({d.value})</span>
              </div>
            ))}
          </div>
        </ChartCard>

        {/* Category Bar Chart */}
        <ChartCard title="Detections by Category" subtitle="Hack type distribution" >
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={categoryBarData} layout="vertical" margin={{ left: 20, right: 16 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" horizontal={false} />
                <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} />
                <YAxis
                  dataKey="name"
                  type="category"
                  tick={{ fill: '#94a3b8', fontSize: 11 }}
                  axisLine={false}
                  tickLine={false}
                  width={120}
                />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e1e3a', border: '1px solid #2a2a4a', borderRadius: '8px', fontSize: '12px' }}
                  labelStyle={{ color: '#e2e8f0' }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={20}>
                  {categoryBarData.map((_, i) => (
                    <Cell key={i} fill={CATEGORY_COLORS[i % CATEGORY_COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </ChartCard>

        {/* RMGI Timeline Preview */}
        <ChartCard title="RMGI Timeline" subtitle="Recent trajectory scores">
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={timelineData} margin={{ top: 5, right: 16, bottom: 5, left: 0 }}>
                <defs>
                  <linearGradient id="hackGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="rmgiGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
                <XAxis dataKey="step" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} domain={[0, 1]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1e1e3a', border: '1px solid #2a2a4a', borderRadius: '8px', fontSize: '12px' }}
                  labelStyle={{ color: '#e2e8f0' }}
                />
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
              className="flex items-center gap-4 px-4 py-3 rounded-lg bg-bg-primary/50 hover:bg-bg-elevated/30 transition-colors"
            >
              <RiskBadge level={alert.severity} size="sm" />
              <div className="flex-1 min-w-0">
                <p className="text-sm text-text-primary truncate">{alert.message}</p>
                <p className="text-xs text-text-muted">{formatRelative(alert.timestamp)}</p>
              </div>
              <span className="text-xs text-text-muted font-mono">{alert.trajectory_id}</span>
            </div>
          ))}
        </div>
      </ChartCard>
    </div>
  );
}
