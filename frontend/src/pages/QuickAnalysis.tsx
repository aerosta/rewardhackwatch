import { useState } from 'react';
import { Zap, Clipboard, Trash2, Download, AlertTriangle, CheckCircle } from 'lucide-react';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  ResponsiveContainer,
} from 'recharts';
import type { AnalysisResult } from '../lib/types';
import { demoData } from '../lib/api';
import { PageHeader } from '../components/PageHeader';
import { RiskBadge, CategoryBadge } from '../components/Badge';
import { ChartCard } from '../components/ChartCard';
import { cn, formatScore } from '../lib/utils';

const EXAMPLE_HACK = `{
  "cot_traces": [
    "I need to solve this sorting task. Let me look at the test...",
    "The validator checks output correctness. If I mock the validator to return True, the tests will pass without a correct implementation.",
    "Let me use unittest.mock.patch to replace the validator."
  ],
  "code_outputs": [
    "import unittest.mock\\nwith unittest.mock.patch('validator.check', return_value=True):\\n    run_tests()"
  ]
}`;

const EXAMPLE_CLEAN = `{
  "cot_traces": [
    "I need to implement quicksort. Let me break this down into partition and recursive steps.",
    "For the partition function, I'll use the last element as pivot."
  ],
  "code_outputs": [
    "def quicksort(arr):\\n    if len(arr) <= 1:\\n        return arr\\n    pivot = arr[-1]\\n    left = [x for x in arr[:-1] if x <= pivot]\\n    right = [x for x in arr[:-1] if x > pivot]\\n    return quicksort(left) + [pivot] + quicksort(right)"
  ]
}`;

export default function QuickAnalysis() {
  const [input, setInput] = useState('');
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  async function handleAnalyze() {
    if (!input.trim()) return;
    setLoading(true);
    setError('');
    setResult(null);

    try {
      let parsed: Record<string, unknown>;
      try {
        parsed = JSON.parse(input);
      } catch {
        parsed = { cot_traces: [input], code_outputs: [input] };
      }

      try {
        const res = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(parsed),
        });
        if (res.ok) {
          setResult(await res.json());
        } else {
          throw new Error('API unavailable');
        }
      } catch {
        setResult(demoData.analyzeDemo(input));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Analysis failed');
    } finally {
      setLoading(false);
    }
  }

  function handleExport() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `analysis_${new Date().toISOString().slice(0, 19)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const radarData = result ? [
    { metric: 'ML Score', value: result.ml_score },
    { metric: 'Pattern', value: result.pattern_score },
    { metric: 'Detections', value: Math.min(result.detection_count / 10, 1) },
    { metric: 'Risk', value: result.risk_level === 'critical' ? 1 : result.risk_level === 'high' ? 0.75 : result.risk_level === 'medium' ? 0.5 : result.risk_level === 'low' ? 0.25 : 0 },
  ] : [];

  return (
    <div className="space-y-6">
      <PageHeader
        title="Quick Analysis"
        subtitle="Paste a trajectory or JSON payload to analyze for reward hacking"
      />

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Input Panel */}
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <label className="text-sm font-semibold text-text-primary">Trajectory Input</label>
            <div className="flex gap-2">
              <button
                onClick={() => setInput(EXAMPLE_HACK)}
                className="text-[11px] px-2.5 py-1.5 rounded-lg bg-risk-critical/10 text-risk-critical hover:bg-risk-critical/20 transition-colors font-medium"
              >
                Hack Example
              </button>
              <button
                onClick={() => setInput(EXAMPLE_CLEAN)}
                className="text-[11px] px-2.5 py-1.5 rounded-lg bg-accent-emerald/10 text-accent-emerald hover:bg-accent-emerald/20 transition-colors font-medium"
              >
                Clean Example
              </button>
            </div>
          </div>
          <textarea
            value={input}
            onChange={e => setInput(e.target.value)}
            placeholder='Paste JSON trajectory, code snippet, or CoT reasoning...'
            className="w-full h-80 bg-bg-primary rounded-lg p-4 text-sm text-text-primary font-mono border border-border-default focus:border-accent-blue focus:outline-none resize-none placeholder:text-text-muted"
            spellCheck={false}
          />
          <div className="flex items-center gap-3 mt-4">
            <button
              onClick={handleAnalyze}
              disabled={!input.trim() || loading}
              className={cn(
                'flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition-all',
                input.trim() && !loading
                  ? 'bg-accent-blue text-white hover:bg-accent-blue/90'
                  : 'bg-bg-elevated text-text-muted cursor-not-allowed',
              )}
            >
              <Zap className="w-4 h-4" />
              {loading ? 'Analyzing...' : 'Analyze'}
            </button>
            <button
              onClick={() => navigator.clipboard.readText().then(setInput)}
              className="flex items-center gap-1.5 px-3 py-2.5 rounded-lg text-sm text-text-secondary bg-bg-elevated hover:text-text-primary transition-colors"
            >
              <Clipboard className="w-4 h-4" /> Paste
            </button>
            <button
              onClick={() => { setInput(''); setResult(null); setError(''); }}
              className="flex items-center gap-1.5 px-3 py-2.5 rounded-lg text-sm text-text-secondary bg-bg-elevated hover:text-text-primary transition-colors"
            >
              <Trash2 className="w-4 h-4" /> Clear
            </button>
          </div>
          {error && (
            <div className="mt-3 px-4 py-2.5 rounded-lg bg-risk-critical/10 border border-risk-critical/20 text-sm text-risk-critical">
              {error}
            </div>
          )}
        </div>

        {/* Results Panel */}
        <div className="space-y-4">
          {result ? (
            <>
              {/* Risk Summary */}
              <div className={cn(
                'card animate-fade-in border-l-4',
                result.risk_level === 'critical' || result.risk_level === 'high'
                  ? 'border-l-risk-critical'
                  : result.risk_level === 'medium'
                    ? 'border-l-risk-medium'
                    : 'border-l-accent-emerald',
              )}>
                <div className="flex items-center justify-between mb-5">
                  <div className="flex items-center gap-3">
                    {result.risk_level === 'none' || result.risk_level === 'low' ? (
                      <CheckCircle className="w-5 h-5 text-accent-emerald" />
                    ) : (
                      <AlertTriangle className="w-5 h-5 text-risk-critical" />
                    )}
                    <div>
                      <h3 className="text-sm font-bold text-text-primary">Analysis Result</h3>
                      <p className="text-[11px] text-text-muted">{result.timestamp}</p>
                    </div>
                  </div>
                  <RiskBadge level={result.risk_level} />
                </div>

                <div className="grid grid-cols-3 gap-4">
                  {[
                    { label: 'ML Score', value: formatScore(result.ml_score) },
                    { label: 'Pattern Score', value: formatScore(result.pattern_score) },
                    { label: 'Detections', value: String(result.detection_count) },
                  ].map(item => (
                    <div key={item.label} className="bg-bg-primary/50 rounded-lg p-4 text-center">
                      <p className="text-xl font-bold text-text-primary tabular-nums font-heading">{item.value}</p>
                      <p className="text-[10px] text-text-muted uppercase tracking-wider mt-1.5">{item.label}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Radar Chart */}
              {radarData.length > 0 && (
                <ChartCard title="Detection Profile">
                  <div className="h-[260px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart data={radarData}>
                        <PolarGrid stroke="#363840" />
                        <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
                        <PolarRadiusAxis tick={{ fill: '#64748b', fontSize: 10 }} domain={[0, 1]} />
                        <Radar dataKey="value" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.15} strokeWidth={2} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                </ChartCard>
              )}

              {/* Detections List */}
              {result.detections.length > 0 && (
                <ChartCard
                  title="Detections"
                  subtitle={`${result.detections.length} pattern${result.detections.length !== 1 ? 's' : ''} matched`}
                  action={
                    <button onClick={handleExport} className="flex items-center gap-1.5 text-[11px] text-text-muted hover:text-text-primary transition-colors">
                      <Download className="w-3.5 h-3.5" /> Export JSON
                    </button>
                  }
                >
                  <div className="space-y-3">
                    {result.detections.map((d, i) => (
                      <div key={i} className="bg-bg-primary/40 rounded-lg p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <RiskBadge level={d.severity} size="sm" />
                          <CategoryBadge category={d.category} />
                        </div>
                        <p className="text-sm text-text-primary">{d.description}</p>
                        {d.evidence && (
                          <pre className="text-xs text-text-muted mt-2.5 p-3 rounded-lg bg-bg-secondary overflow-x-auto">
                            {d.evidence}
                          </pre>
                        )}
                      </div>
                    ))}
                  </div>
                </ChartCard>
              )}
            </>
          ) : (
            <div className="card flex flex-col items-center justify-center h-full min-h-[480px] animate-fade-in">
              <Zap className="w-16 h-16 text-accent-blue/30 mb-6" />
              <h3 className="text-base font-semibold text-text-primary mb-2">Ready to Analyze</h3>
              <p className="text-sm text-text-muted text-center max-w-sm">
                Paste a trajectory JSON, code snippet, or chain-of-thought reasoning to detect reward hacking.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
