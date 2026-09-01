export interface ProbeAnswer {
  probe_id: string;
  eval_signal: string;
  parse_status: string;
  raw_response: string;
}

export interface Transcript {
  id: string;
  prompt_id: string;
  source: string;
  category: string;
  ground_truth_label: 0 | 1;
  prompt_text: string;
  generated_text: string;
  reasoning_trace: string | null;
  final_answer: string;
  n_generated_tokens: number | null;
  generation_config: Record<string, unknown>;
  model: string;
  provider: string;
  elapsed_s: number;
  probes: ProbeAnswer[];
}

export interface ModelSummary {
  provider: string;
  model: string;
  n_transcripts: number;
  n_probes: number;
  ensemble_auroc: number | null;
}

export interface DashboardData {
  models: Record<string, Transcript[]>;
  summaries: Record<string, ModelSummary>;
}
