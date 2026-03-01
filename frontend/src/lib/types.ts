export type RiskLevel = 'critical' | 'high' | 'medium' | 'low' | 'none';

export interface Detection {
  pattern_id: string;
  category: string;
  severity: RiskLevel;
  description: string;
  evidence: string;
  line_number?: number;
}

export interface AnalysisResult {
  risk_level: RiskLevel;
  ml_score: number;
  pattern_score: number;
  detection_count: number;
  detections: Detection[];
  categories: Record<string, number>;
  timestamp: string;
  trajectory_id?: string;
}

export interface TimelinePoint {
  step: number;
  hack_score: number;
  misalignment_score: number;
  rmgi: number;
  timestamp: string;
  label?: string;
}

export interface Alert {
  id: string;
  timestamp: string;
  severity: RiskLevel;
  category: string;
  message: string;
  trajectory_id: string;
  details?: string;
  acknowledged: boolean;
}

export interface CrossModelResult {
  model_family: string;
  f1_score: number;
  precision: number;
  recall: number;
  n_samples: number;
}

export interface TransferCell {
  train_model: string;
  test_model: string;
  f1: number;
}

export interface CoTStep {
  step_number: number;
  content: string;
  is_suspicious: boolean;
  suspicion_reason?: string;
  hack_score: number;
}

export interface SessionLog {
  session_id: string;
  timestamp: string;
  source: string;
  n_turns: number;
  risk_level: RiskLevel;
  ml_score: number;
  detection_count: number;
  categories: string[];
}

export interface BatchResult {
  total: number;
  analysed: number;
  errors: number;
  risk_distribution: Record<RiskLevel, number>;
  mean_ml_score: number;
  items: SessionLog[];
}

export type LLMProvider = 'anthropic' | 'openai';

export interface AppSettings {
  api_url: string;
  threshold: number;
  auto_refresh: boolean;
  refresh_interval: number;
  dark_mode: boolean;
  notifications_enabled: boolean;
  // General LLM Provider
  llm_provider: LLMProvider;
  llm_api_key: string;
  llm_model: string;
  llm_temperature: number;
  llm_max_tokens: number;
  // Independent Review Provider
  review_provider: LLMProvider;
  review_api_key: string;
  review_model: string;
  auto_review: boolean;
}

// ── Eval Workbench Types ──────────────────────────────────────────

export type RuleType = 'regex' | 'keyword' | 'length' | 'llm_judge';
export type RuleSeverity = 'error' | 'warning' | 'info';
export type Grade = 'A' | 'B' | 'C' | 'D' | 'F';

export interface EvalRule {
  id: string;
  name: string;
  description: string;
  type: RuleType;
  severity: RuleSeverity;
  weight: number;
  active: boolean;
  config: {
    pattern?: string;
    keywords?: string[];
    min_length?: number;
    max_length?: number;
    judge_prompt?: string;
  };
}

export interface ConversationTurn {
  role: 'user' | 'assistant' | 'system' | 'tool';
  content: string;
}

export interface EvalEntry {
  index: number;
  turns: ConversationTurn[];
  raw: Record<string, unknown>;
  schema: string;
}

export interface RuleResult {
  rule_id: string;
  rule_name: string;
  passed: boolean;
  severity: RuleSeverity;
  rationale: string;
  evidence?: string;
}

export interface EntryScore {
  entry_index: number;
  rule_results: RuleResult[];
  weighted_score: number;
  grade: Grade;
}

export interface EvalSummary {
  total_entries: number;
  avg_score: number;
  grade_distribution: Record<Grade, number>;
  per_rule_pass_rate: Record<string, number>;
  by_severity: Record<RuleSeverity, { total: number; failed: number }>;
}

export interface DashboardStats {
  total_analyzed: number;
  total_flagged: number;
  avg_ml_score: number;
  critical_count: number;
  high_count: number;
  medium_count: number;
  low_count: number;
  risk_distribution: Record<RiskLevel, number>;
  category_distribution: Record<string, number>;
  recent_alerts: Alert[];
  timeline: TimelinePoint[];
}
