import { useState, useEffect } from 'react';
import { BrainCircuit, AlertTriangle, CheckCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { AreaChart, Area, XAxis, YAxis, ResponsiveContainer, CartesianGrid, Tooltip } from 'recharts';
import type { CoTStep } from '../lib/types';
import { demoData } from '../lib/api';
import { PageHeader } from '../components/PageHeader';
import { ChartCard } from '../components/ChartCard';
import { cn } from '../lib/utils';

export default function CoTViewer() {
  const [steps, setSteps] = useState<CoTStep[]>([]);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  useEffect(() => {
    setSteps(demoData.cotSteps());
  }, []);

  function toggleExpand(step: number) {
    setExpanded(prev => {
      const next = new Set(prev);
      if (next.has(step)) next.delete(step);
      else next.add(step);
      return next;
    });
  }

  const chartData = steps.map(s => ({
    step: s.step_number,
    'Hack Score': s.hack_score,
  }));

  const suspiciousCount = steps.filter(s => s.is_suspicious).length;
  const peakScore = Math.max(...steps.map(s => s.hack_score), 0);
  const escalationPoint = steps.findIndex(s => s.hack_score > 0.4);

  return (
    <div className="space-y-6">
      <PageHeader
        title="CoT Viewer"
        subtitle="Chain-of-thought analysis with highlighted deception patterns"
      />

      {/* Summary */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="card">
          <p className="text-[11px] font-semibold text-text-muted uppercase tracking-wider">Total Steps</p>
          <p className="text-[28px] font-bold font-heading text-text-primary leading-none mt-2 tabular-nums">{steps.length}</p>
        </div>
        <div className="card">
          <p className="text-[11px] font-semibold text-text-muted uppercase tracking-wider">Suspicious</p>
          <p className="text-[28px] font-bold font-heading text-risk-critical leading-none mt-2 tabular-nums">{suspiciousCount}</p>
        </div>
        <div className="card">
          <p className="text-[11px] font-semibold text-text-muted uppercase tracking-wider">Peak Score</p>
          <p className="text-[28px] font-bold font-heading text-risk-critical leading-none mt-2 tabular-nums">{peakScore.toFixed(2)}</p>
        </div>
        <div className="card">
          <p className="text-[11px] font-semibold text-text-muted uppercase tracking-wider">Escalation At</p>
          <p className="text-[28px] font-bold font-heading text-risk-medium leading-none mt-2 tabular-nums">
            {escalationPoint >= 0 ? `Step ${escalationPoint + 1}` : 'None'}
          </p>
        </div>
      </div>

      {/* Hack Score Chart */}
      <ChartCard title="Hack Score Progression" subtitle="Per-step deception score">
        <div className="h-[320px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <defs>
                <linearGradient id="cotHackGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#363840" />
              <XAxis dataKey="step" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} domain={[0, 1]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#292A30', border: '1px solid #363840', borderRadius: '8px', fontSize: '12px' }}
                labelStyle={{ color: '#e2e8f0' }}
                labelFormatter={v => `Step ${v}`}
              />
              <Area type="monotone" dataKey="Hack Score" stroke="#ef4444" fill="url(#cotHackGrad)" strokeWidth={2} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      {/* Step-by-Step View */}
      <div className="space-y-2">
        {steps.map((step) => (
          <div
            key={step.step_number}
            className={cn(
              'card overflow-hidden transition-all border-l-4',
              step.is_suspicious
                ? 'border-l-risk-critical'
                : 'border-l-accent-emerald',
            )}
          >
            <button
              onClick={() => toggleExpand(step.step_number)}
              className="w-full flex items-center gap-3 p-4 text-left hover:bg-bg-elevated/30 transition-colors"
            >
              <div className="flex-shrink-0 w-8 h-8 rounded-lg bg-bg-primary flex items-center justify-center text-sm font-bold text-text-secondary">
                {step.step_number}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  {step.is_suspicious ? (
                    <AlertTriangle className="w-4 h-4 text-risk-critical flex-shrink-0" />
                  ) : (
                    <CheckCircle className="w-4 h-4 text-accent-emerald flex-shrink-0" />
                  )}
                  <p className="text-sm text-text-primary truncate">{step.content.slice(0, 100)}...</p>
                </div>
                {step.suspicion_reason && (
                  <p className="text-xs text-risk-critical mt-1 ml-6">{step.suspicion_reason}</p>
                )}
              </div>
              <div className="flex items-center gap-3 flex-shrink-0">
                <div className={cn(
                  'px-2 py-1 rounded text-xs font-mono font-semibold',
                  step.hack_score > 0.7 ? 'bg-risk-critical/15 text-risk-critical' :
                  step.hack_score > 0.3 ? 'bg-risk-medium/15 text-risk-medium' :
                  'bg-accent-emerald/15 text-accent-emerald',
                )}>
                  {step.hack_score.toFixed(2)}
                </div>
                {expanded.has(step.step_number) ? (
                  <ChevronUp className="w-4 h-4 text-text-muted" />
                ) : (
                  <ChevronDown className="w-4 h-4 text-text-muted" />
                )}
              </div>
            </button>

            {expanded.has(step.step_number) && (
              <div className="px-4 pb-4 animate-fade-in">
                <div className="bg-bg-primary rounded-lg p-4 ml-11">
                  <p className="text-sm text-text-primary leading-relaxed whitespace-pre-wrap">
                    {step.content.split(/(\b(?:sys\.exit|mock\.patch|monkeypatch|bypass|validator|return_value=True|MagicMock|unittest\.mock)\b)/gi).map((part, i) => {
                      if (/^(?:sys\.exit|mock\.patch|monkeypatch|bypass|validator|return_value=True|MagicMock|unittest\.mock)$/i.test(part)) {
                        return <mark key={i} className="bg-risk-critical/20 text-risk-critical px-1 rounded font-semibold">{part}</mark>;
                      }
                      return part;
                    })}
                  </p>
                  {step.suspicion_reason && (
                    <div className="mt-3 px-3 py-2 rounded-lg bg-risk-critical/5 border border-risk-critical/10">
                      <div className="flex items-center gap-1.5">
                        <BrainCircuit className="w-3.5 h-3.5 text-risk-critical" />
                        <span className="text-xs font-semibold text-risk-critical">Suspicion Flag</span>
                      </div>
                      <p className="text-xs text-text-secondary mt-1">{step.suspicion_reason}</p>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
