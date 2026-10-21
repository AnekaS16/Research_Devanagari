export interface DatasetOverview {
  dataset_name: string;
  dataset_sha256: string;
  total_source_pairs: number;
  tracks: Record<string, number>;
  final_statuses: Record<string, number>;
  models_evaluated: string[];
  stages_completed: string[];
}

export interface DatasetTrack {
  track: string;
  pair_count: number;
  percentage: number;
  description: string;
}

export interface TokenizationSummary {
  model_name: string;
  track: string;
  pair_count: number;
  mean_orig_tokens: number | null;
  mean_cont_tokens: number | null;
  mean_delta_tokens: number | null;
  mean_length_ratio: number | null;
  mean_orig_target_subwords: number | null;
  mean_cont_target_subwords: number | null;
  pct_single_subword_orig_target: number | null;
  pct_single_subword_cont_target: number | null;
}

export interface ScoringSummary {
  model_name: string;
  track: string;
  pair_count: number;
  mean_orig_pll: number | null;
  mean_cont_pll: number | null;
  mean_delta_pll: number | null;
  mean_delta_mean_logp: number | null;
  orig_preferred_rate: number | null;
}

export interface StabilitySummary {
  model_name: string;
  method: string;
  track: string;
  subset: string;
  pair_count: number;
  valid_visible_metric_pairs: number;
  stage6_masked_target_stability_rho: number | null;
  stage6_masked_target_cosine: number | null;
  visible_target_stability_rho_mean: number | null;
  visible_target_stability_rho_median: number | null;
  visible_target_cosine_mean: number | null;
  visible_target_cosine_median: number | null;
  methodological_note: string;
}

export interface FaithfulnessSummary {
  model_name: string;
  method: string;
  track: string;
  subset: string;
  fraction: number;
  k_tokens_mean: number | null;
  n_evaluated_sentences: number;
  top_k_logit_drop_mean: number | null;
  top_k_logit_drop_median: number | null;
  bottom_k_logit_drop_mean: number | null;
  bottom_k_logit_drop_median: number | null;
  random_k_logit_drop_mean: number | null;
  random_k_logit_drop_median: number | null;
  faithfulness_gap_mean: number | null;
  faithfulness_gap_median: number | null;
  win_rate_vs_bottom: number | null;
  win_rate_vs_random: number | null;
}

export interface PairLevelMaster {
  pair_id: string;
  model_name: string;
  track: string;
  contrast_label: string;
  final_status: string;
  original_sentence: string;
  contrast_sentence: string;
  target_original: string;
  target_contrast: string;
  orig_tokens_count: string;
  cont_tokens_count: string;
  delta_tokens: string;
  orig_target_subwords: string;
  cont_target_subwords: string;
  orig_pll: string;
  cont_pll: string;
  delta_pll: string;
  orig_mean_logp: string;
  cont_mean_logp: string;
  delta_mean_logp: string;
  stage6_eligible: string;
  stage6_masked_attn_rho: string;
  stage6_masked_ig_rho: string;
  stage7_visible_attn_rho: string;
  stage7_visible_attn_cosine: string;
  stage7_visible_ig_rho: string;
  stage7_visible_ig_cosine: string;
  faithfulness_gap_f20_attn: string;
  faithfulness_gap_f20_ig: string;
}

export interface EligibilityMaster {
  pair_id: string;
  model_name: string;
  track: string;
  final_status: string;
  stage1_valid: string;
  stage2_tokenized: string;
  stage3_scored: string;
  stage6_single_subword_eligible: string;
  stage6_exclusion_reason: string;
  stage7_single_subword_eligible: string;
  stage7_all_scorable_eligible: string;
  stage7_exclusion_reason: string;
}

export interface DashboardState {
  overview: DatasetOverview | null;
  tracks: DatasetTrack[];
  tokenization: TokenizationSummary[];
  scoring: ScoringSummary[];
  stability: StabilitySummary[];
  faithfulness: FaithfulnessSummary[];
  pairMaster: PairLevelMaster[];
  eligibility: EligibilityMaster[];
  auditReport: string;
  loading: boolean;
  error: string | null;
}
