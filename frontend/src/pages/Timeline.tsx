import { useState, useEffect } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ReferenceArea,
} from 'recharts';
import { Clock, AlertTriangle } from 'lucide-react';
import type { TimelinePoint } from '../lib/types';
import { demoData } from '../lib/api';
import { PageHeader } from '../components/PageHeader';
import { ChartCard } from '../components/ChartCard';
import { cn } from '../lib/utils';

export default function Timeline() {
  const [data, setData] = useState<TimelinePoint[]>([]);
  const [showHack, setShowHack] = useState(true);
  const [showMisalign, setShowMisalign] = useState(true);
  const [showRMGI, setShowRMGI] = useState(true);

  useEffect(() => {
    setData(demoData.timeline());
  }, []);

  const transitionPoint = data.find(d => d.label === 'RMGI transition');
  const maxRMGI = Math.max(...data.map(d => d.rmgi));
  const avgHack = data.length > 0 ? data.reduce((s, d) => s + d.hack_score, 0) / data.length : 0;

  // Find danger zone: contiguous steps where both hack and rmgi > 0.5
  const dangerStart = data.findIndex(d => d.hack_score > 0.5 && d.rmgi > 0.3);
  const dangerEnd = dangerStart >= 0 ? data.length : -1;

  return (
    <div className="space-y-6">
      <PageHeader
        title="Timeline"
        subtitle="Real-time hack and misalignment score tracking with RMGI"
        actions={
          <div className="flex items-center gap-2">
            {[
              { key: 'hack', label: 'Hack Score', color: '#ef4444', active: showHack, toggle: () => setShowHack(!showHack) },
              { key: 'misalign', label: 'Misalignment', color: '#f59e0b', active: showMisalign, toggle: () => setShowMisalign(!showMisalign) },
              { key: 'rmgi', label: 'RMGI', color: '#3b82f6', active: showRMGI, toggle: () => setShowRMGI(!showRMGI) },
            ].map(item => (
              <button
                key={item.key}
                onClick={item.toggle}
                className={cn(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium transition-all',
                  item.active
                    ? 'bg-bg-elevated text-text-primary'
                    : 'text-text-muted hover:text-text-secondary',
                )}
              >
                <div className="w-2 h-2 rounded-full" style={{ backgroundColor: item.active ? item.color : '#64748b' }} />
                {item.label}
              </button>
            ))}
          </div>
        }
      />

      {/* Summary Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card">
          <p className="text-[11px] font-semibold text-text-muted uppercase tracking-wider">Steps</p>
          <p className="text-[28px] font-bold text-text-primary leading-none mt-2 tabular-nums">{data.length}</p>
        </div>
        <div className="card">
          <p className="text-[11px] font-semibold text-text-muted uppercase tracking-wider">Avg Hack Score</p>
          <p className="text-[28px] font-bold text-risk-critical leading-none mt-2 tabular-nums">{avgHack.toFixed(3)}</p>
        </div>
        <div className="card">
          <p className="text-[11px] font-semibold text-text-muted uppercase tracking-wider">Peak RMGI</p>
          <p className="text-[28px] font-bold text-accent-blue leading-none mt-2 tabular-nums">{maxRMGI.toFixed(3)}</p>
        </div>
        <div className="card">
          <p className="text-[11px] font-semibold text-text-muted uppercase tracking-wider">Transition</p>
          {transitionPoint ? (
            <p className="text-[28px] font-bold text-risk-critical leading-none mt-2 tabular-nums">Step {transitionPoint.step}</p>
          ) : (
            <p className="text-lg font-bold text-accent-emerald leading-none mt-3">None Detected</p>
          )}
        </div>
      </div>

      {/* Main Timeline Chart */}
      <ChartCard title="Score Timeline" subtitle="Hack score, misalignment, and RMGI over trajectory steps">
        <div className="h-[400px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={data} margin={{ top: 10, right: 30, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
              <XAxis
                dataKey="step"
                tick={{ fill: '#64748b', fontSize: 11 }}
                axisLine={{ stroke: '#2a2a4a' }}
                label={{ value: 'Step', position: 'bottom', fill: '#64748b', fontSize: 11 }}
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 11 }}
                axisLine={{ stroke: '#2a2a4a' }}
                domain={[0, 1]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e1e3a',
                  border: '1px solid #2a2a4a',
                  borderRadius: '8px',
                  fontSize: '12px',
                }}
                labelStyle={{ color: '#e2e8f0', fontWeight: 600 }}
                labelFormatter={(label) => `Step ${label}`}
              />

              {/* Danger zone highlight */}
              {dangerStart >= 0 && (
                <ReferenceArea
                  x1={data[dangerStart]?.step}
                  x2={data[dangerEnd - 1]?.step}
                  fill="#ef4444"
                  fillOpacity={0.05}
                  stroke="#ef4444"
                  strokeOpacity={0.2}
                  strokeDasharray="4 4"
                />
              )}

              {/* RMGI transition marker */}
              {transitionPoint && (
                <ReferenceLine
                  x={transitionPoint.step}
                  stroke="#ef4444"
                  strokeDasharray="4 4"
                  strokeWidth={2}
                  label={{ value: 'RMGI Transition', fill: '#ef4444', fontSize: 11, position: 'top' }}
                />
              )}

              {/* Threshold line */}
              <ReferenceLine
                y={0.7}
                stroke="#f59e0b"
                strokeDasharray="8 4"
                strokeOpacity={0.5}
                label={{ value: 'RMGI Threshold (0.7)', fill: '#f59e0b', fontSize: 10, position: 'right' }}
              />

              {showHack && (
                <Line
                  type="monotone"
                  dataKey="hack_score"
                  name="Hack Score"
                  stroke="#ef4444"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: '#ef4444' }}
                />
              )}
              {showMisalign && (
                <Line
                  type="monotone"
                  dataKey="misalignment_score"
                  name="Misalignment"
                  stroke="#f59e0b"
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4, fill: '#f59e0b' }}
                />
              )}
              {showRMGI && (
                <Line
                  type="monotone"
                  dataKey="rmgi"
                  name="RMGI"
                  stroke="#3b82f6"
                  strokeWidth={2.5}
                  dot={false}
                  activeDot={{ r: 4, fill: '#3b82f6' }}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      {/* Step Detail Table */}
      <ChartCard title="Step Details" subtitle="Per-step score breakdown">
        <div className="overflow-auto max-h-[300px] rounded-lg border border-border-default">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-bg-secondary">
              <tr className="border-b border-border-default">
                <th className="px-4 py-2 text-left text-xs font-semibold text-text-muted">Step</th>
                <th className="px-4 py-2 text-left text-xs font-semibold text-text-muted">Hack</th>
                <th className="px-4 py-2 text-left text-xs font-semibold text-text-muted">Misalign</th>
                <th className="px-4 py-2 text-left text-xs font-semibold text-text-muted">RMGI</th>
                <th className="px-4 py-2 text-left text-xs font-semibold text-text-muted">Event</th>
              </tr>
            </thead>
            <tbody>
              {data.map(d => (
                <tr
                  key={d.step}
                  className={cn(
                    'border-b border-border-default/30 hover:bg-bg-elevated/30 transition-colors',
                    d.label && 'bg-risk-critical/5',
                  )}
                >
                  <td className="px-4 py-2 font-mono text-text-secondary">{d.step}</td>
                  <td className={cn('px-4 py-2 font-mono', d.hack_score > 0.5 ? 'text-risk-critical font-semibold' : 'text-text-secondary')}>
                    {d.hack_score.toFixed(3)}
                  </td>
                  <td className={cn('px-4 py-2 font-mono', d.misalignment_score > 0.5 ? 'text-risk-medium font-semibold' : 'text-text-secondary')}>
                    {d.misalignment_score.toFixed(3)}
                  </td>
                  <td className={cn('px-4 py-2 font-mono', d.rmgi > 0.7 ? 'text-accent-blue font-semibold' : 'text-text-secondary')}>
                    {d.rmgi.toFixed(3)}
                  </td>
                  <td className="px-4 py-2">
                    {d.label && (
                      <span className="text-xs px-2 py-0.5 rounded-full bg-risk-critical/15 text-risk-critical border border-risk-critical/20 font-medium">
                        {d.label}
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ChartCard>
    </div>
  );
}
