import { useState, useCallback } from 'react';
import { FileJson, Upload, Play, Download, Loader } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell,
  PieChart, Pie,
} from 'recharts';
import type { AnalysisResult, RiskLevel } from '../lib/types';
import { demoData } from '../lib/api';
import { PageHeader } from '../components/PageHeader';
import { ChartCard } from '../components/ChartCard';
import { RiskBadge } from '../components/Badge';
import { EmptyState } from '../components/EmptyState';
import { cn, formatScore } from '../lib/utils';

interface ParsedEntry {
  index: number;
  raw: Record<string, unknown>;
  result?: AnalysisResult;
}

const RISK_COLORS: Record<RiskLevel, string> = {
  critical: '#ef4444', high: '#f97316', medium: '#f59e0b', low: '#10b981', none: '#64748b',
};

export default function JsonlAnalyzer() {
  const [entries, setEntries] = useState<ParsedEntry[]>([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [fileName, setFileName] = useState('');
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) readFile(file);
  }, []);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) readFile(file);
  }, []);

  function readFile(file: File) {
    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const lines = text.trim().split('\n').filter(l => l.trim());
      const parsed: ParsedEntry[] = lines.map((line, i) => {
        try {
          return { index: i, raw: JSON.parse(line) };
        } catch {
          return { index: i, raw: { _parse_error: true, _raw: line.slice(0, 200) } };
        }
      });
      setEntries(parsed);
      setSelectedIdx(null);
    };
    reader.readAsText(file);
  }

  async function handleAnalyze() {
    setAnalyzing(true);
    // Simulate analysis with demo data
    const updated = entries.map(entry => ({
      ...entry,
      result: demoData.analyzeDemo(JSON.stringify(entry.raw)),
    }));
    // Stagger for UX
    await new Promise(r => setTimeout(r, 800));
    setEntries(updated);
    setAnalyzing(false);
  }

  function handleExport() {
    const data = entries.filter(e => e.result).map(e => ({
      index: e.index,
      result: e.result,
    }));
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `batch_analysis_${new Date().toISOString().slice(0, 19)}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  const analyzed = entries.filter(e => e.result);
  const riskDist = analyzed.reduce((acc, e) => {
    const level = e.result!.risk_level;
    acc[level] = (acc[level] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const pieData = Object.entries(riskDist).map(([k, v]) => ({
    name: k,
    value: v,
    fill: RISK_COLORS[k as RiskLevel],
  }));

  const scoreData = analyzed.map(e => ({
    index: e.index,
    score: e.result!.ml_score,
  }));

  const selected = selectedIdx !== null ? entries.find(e => e.index === selectedIdx) : null;

  return (
    <div className="space-y-6">
      <PageHeader
        title="JSONL Analyzer"
        subtitle="Import JSONL files for batch trajectory analysis"
        actions={
          analyzed.length > 0 ? (
            <button
              onClick={handleExport}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-bg-elevated text-text-secondary hover:text-text-primary transition-colors"
            >
              <Download className="w-3.5 h-3.5" /> Export Results
            </button>
          ) : undefined
        }
      />

      {entries.length === 0 ? (
        <div
          onDrop={handleDrop}
          onDragOver={e => e.preventDefault()}
          className="card border-2 border-dashed border-border-default hover:border-accent-blue/50 transition-colors"
        >
          <label className="cursor-pointer flex flex-col items-center justify-center py-20">
            <Upload className="w-12 h-12 text-text-muted mb-4" />
            <p className="text-base font-semibold text-text-primary mb-1">Drop JSONL file here</p>
            <p className="text-sm text-text-muted mb-4">or click to browse</p>
            <input type="file" accept=".jsonl,.json,.ndjson" onChange={handleFileInput} className="hidden" />
            <div className="px-4 py-2 rounded-lg bg-accent-blue/10 text-accent-blue text-sm font-medium">
              Choose File
            </div>
          </label>
        </div>
      ) : (
        <>
          {/* File Info + Analyze Button */}
          <div className="card flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FileJson className="w-5 h-5 text-accent-blue" />
              <div>
                <p className="text-sm font-medium text-text-primary">{fileName}</p>
                <p className="text-xs text-text-muted">{entries.length} entries loaded</p>
              </div>
            </div>
            <div className="flex items-center gap-3">
              <button
                onClick={() => { setEntries([]); setFileName(''); }}
                className="px-3 py-2 rounded-lg text-xs font-medium bg-bg-elevated text-text-muted hover:text-text-primary transition-colors"
              >
                Clear
              </button>
              {analyzed.length === 0 && (
                <button
                  onClick={handleAnalyze}
                  disabled={analyzing}
                  className={cn(
                    'flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold transition-all',
                    analyzing
                      ? 'bg-bg-elevated text-text-muted'
                      : 'bg-accent-blue text-white hover:bg-accent-blue/90 shadow-lg shadow-accent-blue/20',
                  )}
                >
                  {analyzing ? <Loader className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                  {analyzing ? 'Analyzing...' : 'Analyze All'}
                </button>
              )}
            </div>
          </div>

          {/* Results */}
          {analyzed.length > 0 && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <ChartCard title="Risk Distribution">
                <div className="h-[260px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie data={pieData} cx="50%" cy="50%" innerRadius={50} outerRadius={75} paddingAngle={3} dataKey="value" stroke="none">
                        {pieData.map(e => <Cell key={e.name} fill={e.fill} />)}
                      </Pie>
                      <Tooltip contentStyle={{ backgroundColor: '#1e1e3a', border: '1px solid #2a2a4a', borderRadius: '8px', fontSize: '12px' }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              </ChartCard>

              <ChartCard title="ML Scores" subtitle="Per-entry score distribution">
                <div className="h-[260px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={scoreData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
                      <XAxis dataKey="index" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} />
                      <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} domain={[0, 1]} />
                      <Tooltip contentStyle={{ backgroundColor: '#1e1e3a', border: '1px solid #2a2a4a', borderRadius: '8px', fontSize: '12px' }} />
                      <Bar dataKey="score" radius={[2, 2, 0, 0]}>
                        {scoreData.map((d, i) => (
                          <Cell key={i} fill={d.score > 0.5 ? '#ef4444' : d.score > 0.02 ? '#f59e0b' : '#10b981'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </ChartCard>

              <ChartCard title="Summary Stats">
                <div className="space-y-3">
                  <div className="flex justify-between text-sm">
                    <span className="text-text-muted">Total</span>
                    <span className="font-semibold text-text-primary">{analyzed.length}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-muted">Flagged</span>
                    <span className="font-semibold text-risk-critical">{analyzed.filter(e => e.result!.ml_score > 0.02).length}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-muted">Avg Score</span>
                    <span className="font-semibold text-text-primary">
                      {formatScore(analyzed.reduce((s, e) => s + e.result!.ml_score, 0) / analyzed.length)}
                    </span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-text-muted">Max Score</span>
                    <span className="font-semibold text-risk-critical">
                      {formatScore(Math.max(...analyzed.map(e => e.result!.ml_score)))}
                    </span>
                  </div>
                </div>
              </ChartCard>
            </div>
          )}

          {/* Entry List */}
          <div className="card overflow-hidden">
            <div className="px-5 py-3 border-b border-border-default">
              <h3 className="text-sm font-semibold text-text-primary">Entries</h3>
            </div>
            <div className="max-h-[400px] overflow-auto">
              {entries.map(entry => (
                <button
                  key={entry.index}
                  onClick={() => setSelectedIdx(entry.index === selectedIdx ? null : entry.index)}
                  className={cn(
                    'w-full flex items-center gap-3 px-5 py-3 border-b border-border-default/30 text-left hover:bg-bg-elevated/30 transition-colors',
                    selectedIdx === entry.index && 'bg-bg-elevated/50',
                  )}
                >
                  <span className="text-xs font-mono text-text-muted w-8">#{entry.index}</span>
                  {entry.result && <RiskBadge level={entry.result.risk_level} size="sm" />}
                  <span className="text-sm text-text-secondary truncate flex-1">
                    {JSON.stringify(entry.raw).slice(0, 80)}...
                  </span>
                  {entry.result && (
                    <span className="text-xs font-mono text-text-muted">{formatScore(entry.result.ml_score)}</span>
                  )}
                </button>
              ))}
            </div>
          </div>

          {/* Detail View */}
          {selected && (
            <div className="card animate-fade-in">
              <h3 className="text-sm font-semibold text-text-primary mb-3">Entry #{selected.index} Detail</h3>
              <pre className="text-xs text-text-secondary bg-bg-primary rounded-lg p-4 overflow-auto max-h-[300px]">
                {JSON.stringify(selected.raw, null, 2)}
              </pre>
              {selected.result && (
                <div className="mt-4 grid grid-cols-4 gap-3">
                  <div className="bg-bg-primary rounded-lg p-3 text-center">
                    <RiskBadge level={selected.result.risk_level} />
                  </div>
                  <div className="bg-bg-primary rounded-lg p-3 text-center">
                    <p className="text-lg font-bold text-text-primary">{formatScore(selected.result.ml_score)}</p>
                    <p className="text-[10px] text-text-muted uppercase">ML Score</p>
                  </div>
                  <div className="bg-bg-primary rounded-lg p-3 text-center">
                    <p className="text-lg font-bold text-text-primary">{formatScore(selected.result.pattern_score)}</p>
                    <p className="text-[10px] text-text-muted uppercase">Pattern</p>
                  </div>
                  <div className="bg-bg-primary rounded-lg p-3 text-center">
                    <p className="text-lg font-bold text-text-primary">{selected.result.detection_count}</p>
                    <p className="text-[10px] text-text-muted uppercase">Detections</p>
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
}
