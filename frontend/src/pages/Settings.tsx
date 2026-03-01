import { useState } from 'react';
import { Settings as SettingsIcon, Save, RotateCcw, Server, Bell, Palette, Download, BrainCircuit, Eye, EyeOff, ShieldCheck } from 'lucide-react';
import type { AppSettings, LLMProvider } from '../lib/types';
import { PageHeader } from '../components/PageHeader';
import { cn } from '../lib/utils';

const ANTHROPIC_MODELS = [
  { value: 'claude-opus-4-6', label: 'Claude Opus 4.6' },
  { value: 'claude-sonnet-4-5-20250929', label: 'Claude Sonnet 4.5' },
  { value: 'claude-haiku-4-5-20251001', label: 'Claude Haiku 4.5' },
];

const OPENAI_MODELS = [
  { value: 'gpt-5.2', label: 'GPT-5.2' },
  { value: 'gpt-4o', label: 'GPT-4o' },
  { value: 'gpt-4o-mini', label: 'GPT-4o Mini' },
];

function modelsForProvider(provider: LLMProvider) {
  return provider === 'anthropic' ? ANTHROPIC_MODELS : OPENAI_MODELS;
}

const DEFAULT_SETTINGS: AppSettings = {
  api_url: 'http://localhost:8000',
  threshold: 0.02,
  auto_refresh: false,
  refresh_interval: 30,
  dark_mode: true,
  notifications_enabled: true,
  llm_provider: 'anthropic',
  llm_api_key: '',
  llm_model: 'claude-sonnet-4-5-20250929',
  llm_temperature: 0,
  llm_max_tokens: 10000,
  review_provider: 'openai',
  review_api_key: '',
  review_model: 'gpt-4o',
  auto_review: false,
};

export default function Settings() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    const saved = localStorage.getItem('rhw_settings');
    return saved ? { ...DEFAULT_SETTINGS, ...JSON.parse(saved) } : DEFAULT_SETTINGS;
  });
  const [saved, setSaved] = useState(false);
  const [reviewSaved, setReviewSaved] = useState(false);
  const [showKey, setShowKey] = useState(false);
  const [showReviewKey, setShowReviewKey] = useState(false);

  function handleSave() {
    localStorage.setItem('rhw_settings', JSON.stringify(settings));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  function handleSaveReview() {
    localStorage.setItem('rhw_settings', JSON.stringify(settings));
    setReviewSaved(true);
    setTimeout(() => setReviewSaved(false), 2000);
  }

  function handleReset() {
    setSettings(DEFAULT_SETTINGS);
    localStorage.removeItem('rhw_settings');
  }

  function handleExportConfig() {
    const exported = { ...settings, llm_api_key: '***', review_api_key: '***' };
    const blob = new Blob([JSON.stringify(exported, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'rhw_settings.json';
    a.click();
    URL.revokeObjectURL(url);
  }

  function handleProviderChange(provider: LLMProvider) {
    const models = modelsForProvider(provider);
    setSettings({ ...settings, llm_provider: provider, llm_model: models[0].value });
  }

  function handleReviewProviderChange(provider: LLMProvider) {
    const models = modelsForProvider(provider);
    setSettings({ ...settings, review_provider: provider, review_model: models[0].value });
  }

  const inputClass = 'w-full px-4 py-2.5 rounded-lg bg-bg-primary border border-border-default text-sm text-text-primary focus:border-accent-blue focus:outline-none';
  const labelClass = 'block text-xs font-medium text-text-secondary mb-1.5';

  return (
    <div className="space-y-6 max-w-3xl">
      <PageHeader
        title="Settings"
        subtitle="Configure detection parameters and preferences"
        actions={
          <button
            onClick={handleReset}
            className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-bg-elevated text-text-muted hover:text-text-primary transition-colors"
          >
            <RotateCcw className="w-3.5 h-3.5" /> Reset All
          </button>
        }
      />

      {/* API Configuration */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <Server className="w-4 h-4 text-accent-blue" />
          <h3 className="text-sm font-semibold text-text-primary">API Configuration</h3>
        </div>
        <div className="space-y-4">
          <div>
            <label className={labelClass}>API URL</label>
            <input
              type="text"
              value={settings.api_url}
              onChange={e => setSettings({ ...settings, api_url: e.target.value })}
              className={inputClass}
            />
          </div>
          <div>
            <label className={labelClass}>
              Detection Threshold: <span className="text-accent-blue font-mono">{settings.threshold.toFixed(3)}</span>
            </label>
            <input
              type="range"
              min={0.001}
              max={0.5}
              step={0.001}
              value={settings.threshold}
              onChange={e => setSettings({ ...settings, threshold: parseFloat(e.target.value) })}
              className="w-full accent-accent-blue"
            />
            <div className="flex justify-between text-[10px] text-text-muted mt-1">
              <span>0.001 (aggressive)</span>
              <span>0.020 (default)</span>
              <span>0.500 (conservative)</span>
            </div>
          </div>
        </div>
      </div>

      {/* General LLM Provider */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <BrainCircuit className="w-4 h-4 text-accent-emerald" />
          <h3 className="text-sm font-semibold text-text-primary">General LLM Provider</h3>
        </div>
        <div className="space-y-4">
          <div>
            <label className={labelClass}>Provider</label>
            <select
              value={settings.llm_provider}
              onChange={e => handleProviderChange(e.target.value as LLMProvider)}
              className={inputClass}
            >
              <option value="anthropic">Anthropic (Claude)</option>
              <option value="openai">OpenAI</option>
            </select>
          </div>
          <div>
            <label className={labelClass}>API Key</label>
            <div className="relative">
              <input
                type={showKey ? 'text' : 'password'}
                value={settings.llm_api_key}
                onChange={e => setSettings({ ...settings, llm_api_key: e.target.value })}
                placeholder={settings.llm_provider === 'anthropic' ? 'sk-ant-...' : 'sk-...'}
                className={cn(inputClass, 'pr-10 font-mono')}
              />
              <button
                type="button"
                onClick={() => setShowKey(!showKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary transition-colors"
              >
                {showKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>
          <div>
            <label className={labelClass}>Model</label>
            <select
              value={settings.llm_model}
              onChange={e => setSettings({ ...settings, llm_model: e.target.value })}
              className={inputClass}
            >
              {modelsForProvider(settings.llm_provider).map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className={labelClass}>Temperature</label>
              <input
                type="number"
                min={0}
                max={2}
                step={0.1}
                value={settings.llm_temperature}
                onChange={e => setSettings({ ...settings, llm_temperature: parseFloat(e.target.value) || 0 })}
                className={cn(inputClass, 'font-mono')}
              />
            </div>
            <div>
              <label className={labelClass}>Max Tokens</label>
              <input
                type="number"
                min={1}
                max={200000}
                step={1000}
                value={settings.llm_max_tokens}
                onChange={e => setSettings({ ...settings, llm_max_tokens: parseInt(e.target.value) || 10000 })}
                className={cn(inputClass, 'font-mono')}
              />
            </div>
          </div>
          <button
            onClick={handleSave}
            className={cn(
              'flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold transition-all',
              saved
                ? 'bg-accent-emerald text-white'
                : 'bg-accent-blue text-white hover:bg-accent-blue/90 shadow-lg shadow-accent-blue/20',
            )}
          >
            <Save className="w-4 h-4" />
            {saved ? 'Saved!' : 'Save'}
          </button>
        </div>
      </div>

      {/* Independent Review Provider */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <ShieldCheck className="w-4 h-4 text-accent-violet" />
          <h3 className="text-sm font-semibold text-text-primary">Independent Review Provider</h3>
        </div>
        <div className="space-y-4">
          <div>
            <label className={labelClass}>Provider</label>
            <select
              value={settings.review_provider}
              onChange={e => handleReviewProviderChange(e.target.value as LLMProvider)}
              className={inputClass}
            >
              <option value="anthropic">Anthropic (Claude)</option>
              <option value="openai">OpenAI</option>
            </select>
          </div>
          <div>
            <label className={labelClass}>API Key</label>
            <div className="relative">
              <input
                type={showReviewKey ? 'text' : 'password'}
                value={settings.review_api_key}
                onChange={e => setSettings({ ...settings, review_api_key: e.target.value })}
                placeholder={settings.review_provider === 'anthropic' ? 'sk-ant-...' : 'sk-...'}
                className={cn(inputClass, 'pr-10 font-mono')}
              />
              <button
                type="button"
                onClick={() => setShowReviewKey(!showReviewKey)}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-text-muted hover:text-text-primary transition-colors"
              >
                {showReviewKey ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>
          <div>
            <label className={labelClass}>Model</label>
            <select
              value={settings.review_model}
              onChange={e => setSettings({ ...settings, review_model: e.target.value })}
              className={inputClass}
            >
              {modelsForProvider(settings.review_provider).map(m => (
                <option key={m.value} value={m.value}>{m.label}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-text-primary">Auto-review after analyze</p>
              <p className="text-xs text-text-muted">Automatically run independent review on each analysis</p>
            </div>
            <button
              onClick={() => setSettings({ ...settings, auto_review: !settings.auto_review })}
              className={cn(
                'relative w-11 h-6 rounded-full transition-colors',
                settings.auto_review ? 'bg-accent-violet' : 'bg-border-default',
              )}
            >
              <div
                className={cn(
                  'absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform',
                  settings.auto_review ? 'translate-x-5.5' : 'translate-x-0.5',
                )}
              />
            </button>
          </div>
          <button
            onClick={handleSaveReview}
            className={cn(
              'flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold transition-all',
              reviewSaved
                ? 'bg-accent-emerald text-white'
                : 'bg-accent-violet text-white hover:bg-accent-violet/90 shadow-lg shadow-accent-violet/20',
            )}
          >
            <Save className="w-4 h-4" />
            {reviewSaved ? 'Saved!' : 'Save Review Settings'}
          </button>
        </div>
      </div>

      {/* API Key Note */}
      <p className="text-xs text-text-muted text-center px-4">
        API keys are stored locally. They are never sent anywhere except to the selected provider.
      </p>

      {/* Auto Refresh */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <SettingsIcon className="w-4 h-4 text-accent-cyan" />
          <h3 className="text-sm font-semibold text-text-primary">Monitoring</h3>
        </div>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-text-primary">Auto-refresh dashboard</p>
              <p className="text-xs text-text-muted">Periodically fetch latest data</p>
            </div>
            <button
              onClick={() => setSettings({ ...settings, auto_refresh: !settings.auto_refresh })}
              className={cn(
                'relative w-11 h-6 rounded-full transition-colors',
                settings.auto_refresh ? 'bg-accent-blue' : 'bg-border-default',
              )}
            >
              <div
                className={cn(
                  'absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform',
                  settings.auto_refresh ? 'translate-x-5.5' : 'translate-x-0.5',
                )}
              />
            </button>
          </div>
          {settings.auto_refresh && (
            <div>
              <label className={labelClass}>
                Refresh interval: <span className="text-accent-cyan font-mono">{settings.refresh_interval}s</span>
              </label>
              <input
                type="range"
                min={5}
                max={120}
                step={5}
                value={settings.refresh_interval}
                onChange={e => setSettings({ ...settings, refresh_interval: parseInt(e.target.value) })}
                className="w-full accent-accent-cyan"
              />
              <div className="flex justify-between text-[10px] text-text-muted mt-1">
                <span>5s</span>
                <span>30s</span>
                <span>120s</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Notifications */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <Bell className="w-4 h-4 text-accent-amber" />
          <h3 className="text-sm font-semibold text-text-primary">Notifications</h3>
        </div>
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm text-text-primary">Enable browser notifications</p>
            <p className="text-xs text-text-muted">Alert when high-risk trajectories are detected</p>
          </div>
          <button
            onClick={() => setSettings({ ...settings, notifications_enabled: !settings.notifications_enabled })}
            className={cn(
              'relative w-11 h-6 rounded-full transition-colors',
              settings.notifications_enabled ? 'bg-accent-amber' : 'bg-border-default',
            )}
          >
            <div
              className={cn(
                'absolute top-0.5 w-5 h-5 rounded-full bg-white transition-transform',
                settings.notifications_enabled ? 'translate-x-5.5' : 'translate-x-0.5',
              )}
            />
          </button>
        </div>
      </div>

      {/* Export */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <Download className="w-4 h-4 text-accent-cyan" />
          <h3 className="text-sm font-semibold text-text-primary">Data Export</h3>
        </div>
        <button
          onClick={handleExportConfig}
          className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-bg-primary border border-border-default text-sm text-text-secondary hover:text-text-primary hover:border-border-active transition-colors w-full"
        >
          <Download className="w-4 h-4" />
          Export Settings as JSON
        </button>
      </div>

      {/* About */}
      <div className="card">
        <div className="flex items-center gap-2 mb-3">
          <Palette className="w-4 h-4 text-accent-pink" />
          <h3 className="text-sm font-semibold text-text-primary">About</h3>
        </div>
        <div className="space-y-1 text-sm text-text-secondary">
          <p><span className="text-text-muted">Version:</span> 1.3.0</p>
          <p><span className="text-text-muted">License:</span> Apache 2.0</p>
          <p><span className="text-text-muted">Author:</span> Aerosta</p>
        </div>
      </div>
    </div>
  );
}
