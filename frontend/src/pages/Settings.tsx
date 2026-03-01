import { useState } from 'react';
import { Settings as SettingsIcon, Save, RotateCcw, Server, Bell, Palette, Download } from 'lucide-react';
import type { AppSettings } from '../lib/types';
import { PageHeader } from '../components/PageHeader';
import { cn } from '../lib/utils';

const DEFAULT_SETTINGS: AppSettings = {
  api_url: 'http://localhost:8000',
  threshold: 0.02,
  auto_refresh: false,
  refresh_interval: 30,
  dark_mode: true,
  notifications_enabled: true,
};

export default function Settings() {
  const [settings, setSettings] = useState<AppSettings>(() => {
    const saved = localStorage.getItem('rhw_settings');
    return saved ? JSON.parse(saved) : DEFAULT_SETTINGS;
  });
  const [saved, setSaved] = useState(false);

  function handleSave() {
    localStorage.setItem('rhw_settings', JSON.stringify(settings));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  function handleReset() {
    setSettings(DEFAULT_SETTINGS);
    localStorage.removeItem('rhw_settings');
  }

  function handleExportConfig() {
    const blob = new Blob([JSON.stringify(settings, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'rhw_settings.json';
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="space-y-6 max-w-3xl">
      <PageHeader
        title="Settings"
        subtitle="Configure detection parameters and preferences"
        actions={
          <div className="flex gap-2">
            <button
              onClick={handleReset}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium bg-bg-elevated text-text-muted hover:text-text-primary transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" /> Reset
            </button>
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
        }
      />

      {/* API Configuration */}
      <div className="glass-card rounded-xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <Server className="w-4 h-4 text-accent-blue" />
          <h3 className="text-sm font-semibold text-text-primary">API Configuration</h3>
        </div>
        <div className="space-y-4">
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1.5">API URL</label>
            <input
              type="text"
              value={settings.api_url}
              onChange={e => setSettings({ ...settings, api_url: e.target.value })}
              className="w-full px-4 py-2.5 rounded-lg bg-bg-primary border border-border-default text-sm text-text-primary focus:border-accent-blue focus:outline-none"
            />
          </div>
          <div>
            <label className="block text-xs font-medium text-text-secondary mb-1.5">
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

      {/* Auto Refresh */}
      <div className="glass-card rounded-xl p-5">
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
              <label className="block text-xs font-medium text-text-secondary mb-1.5">
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
      <div className="glass-card rounded-xl p-5">
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
      <div className="glass-card rounded-xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <Download className="w-4 h-4 text-accent-violet" />
          <h3 className="text-sm font-semibold text-text-primary">Data Export</h3>
        </div>
        <div className="space-y-3">
          <button
            onClick={handleExportConfig}
            className="flex items-center gap-2 px-4 py-2.5 rounded-lg bg-bg-primary border border-border-default text-sm text-text-secondary hover:text-text-primary hover:border-border-active transition-colors w-full"
          >
            <Download className="w-4 h-4" />
            Export Settings as JSON
          </button>
        </div>
      </div>

      {/* About */}
      <div className="glass-card rounded-xl p-5">
        <div className="flex items-center gap-2 mb-3">
          <Palette className="w-4 h-4 text-accent-pink" />
          <h3 className="text-sm font-semibold text-text-primary">About</h3>
        </div>
        <div className="space-y-1 text-sm text-text-secondary">
          <p><span className="text-text-muted">Version:</span> 1.2.0</p>
          <p><span className="text-text-muted">License:</span> Apache 2.0</p>
          <p><span className="text-text-muted">Author:</span> Aerosta Research</p>
        </div>
      </div>
    </div>
  );
}
