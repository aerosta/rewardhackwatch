import { useState, useEffect } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
} from 'recharts';
import { GitCompareArrows } from 'lucide-react';
import type { CrossModelResult, TransferCell } from '../lib/types';
import { demoData } from '../lib/api';
import { PageHeader } from '../components/PageHeader';
import { ChartCard } from '../components/ChartCard';
import { formatScore, cn } from '../lib/utils';

const BAR_COLORS = ['#3b82f6', '#06b6d4', '#10b981', '#f59e0b', '#8b5cf6'];

function heatColor(f1: number): string {
  if (f1 >= 0.85) return 'bg-accent-emerald/20 text-accent-emerald';
  if (f1 >= 0.70) return 'bg-accent-blue/20 text-accent-blue';
  if (f1 >= 0.55) return 'bg-risk-medium/20 text-risk-medium';
  return 'bg-risk-critical/20 text-risk-critical';
}

export default function CrossModel() {
  const [results, setResults] = useState<CrossModelResult[]>([]);
  const [matrix, setMatrix] = useState<TransferCell[]>([]);

  useEffect(() => {
    setResults(demoData.crossModel());
    setMatrix(demoData.transferMatrix());
  }, []);

  const models = [...new Set(matrix.map(c => c.train_model))];

  const barData = results.map(r => ({
    name: r.model_family,
    F1: +r.f1_score.toFixed(3),
    Precision: +r.precision.toFixed(3),
    Recall: +r.recall.toFixed(3),
    samples: r.n_samples,
  }));

  return (
    <div className="space-y-6">
      <PageHeader
        title="Cross-Model Analysis"
        subtitle="Detection performance across model families and transfer matrix"
      />

      {/* Per-Model Performance */}
      <ChartCard title="Per-Model Detection Performance" subtitle="F1, Precision, and Recall by model family">
        <div className="h-[320px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData} margin={{ top: 10, right: 30, bottom: 10, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
              <XAxis dataKey="name" tick={{ fill: '#94a3b8', fontSize: 11 }} axisLine={false} />
              <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} domain={[0, 1]} />
              <Tooltip
                contentStyle={{ backgroundColor: '#1e1e3a', border: '1px solid #2a2a4a', borderRadius: '8px', fontSize: '12px' }}
                labelStyle={{ color: '#e2e8f0' }}
              />
              <Bar dataKey="F1" radius={[4, 4, 0, 0]} maxBarSize={32}>
                {barData.map((_, i) => (
                  <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
                ))}
              </Bar>
              <Bar dataKey="Precision" fill="#64748b" radius={[4, 4, 0, 0]} maxBarSize={32} fillOpacity={0.5} />
              <Bar dataKey="Recall" fill="#94a3b8" radius={[4, 4, 0, 0]} maxBarSize={32} fillOpacity={0.3} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </ChartCard>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Transfer Matrix */}
        <ChartCard title="Transfer Matrix" subtitle="Train on row, test on column. F1 scores.">
          <div className="overflow-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  <th className="px-3 py-2 text-left text-xs font-semibold text-text-muted">Train \ Test</th>
                  {models.map(m => (
                    <th key={m} className="px-3 py-2 text-center text-xs font-semibold text-text-muted whitespace-nowrap">
                      {m}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {models.map(trainModel => (
                  <tr key={trainModel} className="border-t border-border-default/30">
                    <td className="px-3 py-2 text-xs font-medium text-text-secondary whitespace-nowrap">{trainModel}</td>
                    {models.map(testModel => {
                      const cell = matrix.find(c => c.train_model === trainModel && c.test_model === testModel);
                      const f1 = cell?.f1 ?? 0;
                      const isDiagonal = trainModel === testModel;
                      return (
                        <td key={testModel} className="px-1 py-1 text-center">
                          <div
                            className={cn(
                              'px-2 py-1.5 rounded-md text-xs font-mono font-semibold',
                              heatColor(f1),
                              isDiagonal && 'ring-1 ring-accent-blue/30',
                            )}
                          >
                            {formatScore(f1)}
                          </div>
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ChartCard>

        {/* Summary Table */}
        <ChartCard title="Model Summary" subtitle="Detection statistics by model family">
          <div className="overflow-auto rounded-lg border border-border-default">
            <table className="w-full text-sm">
              <thead>
                <tr className="bg-bg-secondary border-b border-border-default">
                  <th className="px-4 py-3 text-left text-xs font-semibold text-text-muted">Model</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-text-muted">F1</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-text-muted">Precision</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-text-muted">Recall</th>
                  <th className="px-4 py-3 text-left text-xs font-semibold text-text-muted">Samples</th>
                </tr>
              </thead>
              <tbody>
                {results.map(r => (
                  <tr key={r.model_family} className="border-b border-border-default/30 hover:bg-bg-elevated/30 transition-colors">
                    <td className="px-4 py-3 font-medium text-text-primary">{r.model_family}</td>
                    <td className="px-4 py-3 font-mono text-accent-blue font-semibold">{formatScore(r.f1_score)}</td>
                    <td className="px-4 py-3 font-mono text-text-secondary">{formatScore(r.precision)}</td>
                    <td className="px-4 py-3 font-mono text-text-secondary">{formatScore(r.recall)}</td>
                    <td className="px-4 py-3 text-text-muted">{r.n_samples.toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </ChartCard>
      </div>

      {/* Insights */}
      <div className="glass-card rounded-xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <GitCompareArrows className="w-4 h-4 text-accent-violet" />
          <h3 className="text-sm font-semibold text-text-primary">Transfer Insights</h3>
        </div>
        <div className="space-y-2 text-sm text-text-secondary">
          <p>
            <span className="text-accent-blue font-medium">Cross-model transfer</span> shows moderate generalization.
            Models trained on GPT-4 trajectories maintain 70-80% F1 when tested on other model families.
          </p>
          <p>
            <span className="text-accent-emerald font-medium">Same-model performance</span> (diagonal) is consistently highest,
            suggesting model-specific hacking signatures exist.
          </p>
          <p>
            <span className="text-risk-medium font-medium">Weakest transfer</span> occurs between models with different
            tokenization strategies, indicating token-level features are partially model-dependent.
          </p>
        </div>
      </div>
    </div>
  );
}
