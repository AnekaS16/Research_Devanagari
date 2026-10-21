import type {
  DashboardState,
  DatasetOverview,
  DatasetTrack,
  EligibilityMaster,
  FaithfulnessSummary,
  PairLevelMaster,
  ScoringSummary,
  StabilitySummary,
  TokenizationSummary,
} from '../types';

/**
 * Robust CSV parser supporting quoted values, commas, and newlines.
 */
export function parseCSV(text: string): Record<string, string>[] {
  const lines: string[] = [];
  let current = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (char === '"') {
      if (inQuotes && text[i + 1] === '"') {
        current += '"';
        i++;
      } else {
        inQuotes = !inQuotes;
      }
    } else if ((char === '\r' || char === '\n') && !inQuotes) {
      if (char === '\r' && text[i + 1] === '\n') {
        i++;
      }
      if (current.trim()) {
        lines.push(current);
      }
      current = '';
    } else {
      current += char;
    }
  }
  if (current.trim()) {
    lines.push(current);
  }

  if (lines.length < 2) return [];

  const parseLine = (line: string): string[] => {
    const fields: string[] = [];
    let field = '';
    let inside = false;
    for (let i = 0; i < line.length; i++) {
      const c = line[i];
      if (c === '"') {
        if (inside && line[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inside = !inside;
        }
      } else if (c === ',' && !inside) {
        fields.push(field.trim());
        field = '';
      } else {
        field += c;
      }
    }
    fields.push(field.trim());
    return fields;
  };

  const headers = parseLine(lines[0]);
  const rows: Record<string, string>[] = [];

  for (let j = 1; j < lines.length; j++) {
    const values = parseLine(lines[j]);
    const row: Record<string, string> = {};
    for (let k = 0; k < headers.length; k++) {
      row[headers[k]] = values[k] !== undefined ? values[k] : '';
    }
    rows.push(row);
  }

  return rows;
}

/**
 * Safely parse a numeric string, preserving empty or invalid values without coercing to zero.
 * Returns:
 * - `null` for empty string, whitespace-only string, or invalid numeric text
 * - `number` for valid numbers (including valid zero: 0, 0.0, -0.0)
 */
export function parseNullableNumber(val: string | undefined | null): number | null {
  if (val === undefined || val === null) return null;
  const trimmed = String(val).trim();
  if (trimmed === '') return null;
  const num = Number(trimmed);
  return Number.isFinite(num) ? num : null;
}

/**
 * Export array of records to a downloadable CSV file.
 */
export function exportToCSV(data: Record<string, any>[], filename: string): void {
  if (!data || !data.length) return;
  const headers = Object.keys(data[0]);
  const csvRows = [headers.join(',')];

  for (const row of data) {
    const values = headers.map((header) => {
      const val = row[header] !== undefined && row[header] !== null ? String(row[header]) : '';
      const escaped = val.replace(/"/g, '""');
      return `"${escaped}"`;
    });
    csvRows.push(values.join(','));
  }

  const blob = new Blob([csvRows.join('\n')], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', url);
  link.setAttribute('download', filename);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

/**
 * Fetch and load all dashboard data assets asynchronously.
 */
export async function loadAllDashboardData(): Promise<Omit<DashboardState, 'loading' | 'error'>> {
  const basePath = './data';

  const [
    overviewRes,
    tracksRes,
    tokRes,
    scoreRes,
    stabRes,
    faithRes,
    pairRes,
    eligRes,
    reportRes,
  ] = await Promise.all([
    fetch(`${basePath}/dataset_overview.json`).then((r) => r.json()),
    fetch(`${basePath}/dataset_tracks.csv`).then((r) => r.text()),
    fetch(`${basePath}/tokenization_summary.csv`).then((r) => r.text()),
    fetch(`${basePath}/scoring_summary.csv`).then((r) => r.text()),
    fetch(`${basePath}/stability_summary.csv`).then((r) => r.text()),
    fetch(`${basePath}/faithfulness_summary.csv`).then((r) => r.text()),
    fetch(`${basePath}/pair_level_master.csv`).then((r) => r.text()),
    fetch(`${basePath}/eligibility_master.csv`).then((r) => r.text()),
    fetch(`${basePath}/scientific_audit_report.md`).then((r) => r.text()),
  ]);

  const tracks = parseCSV(tracksRes).map((r) => ({
    track: r.track,
    pair_count: parseNullableNumber(r.pair_count) ?? 0,
    percentage: parseNullableNumber(r.percentage) ?? 0,
    description: r.description,
  })) as DatasetTrack[];

  const tokenization = parseCSV(tokRes).map((r) => ({
    model_name: r.model_name,
    track: r.track,
    pair_count: parseNullableNumber(r.pair_count) ?? 0,
    mean_orig_tokens: parseNullableNumber(r.mean_orig_tokens),
    mean_cont_tokens: parseNullableNumber(r.mean_cont_tokens),
    mean_delta_tokens: parseNullableNumber(r.mean_delta_tokens),
    mean_length_ratio: parseNullableNumber(r.mean_length_ratio),
    mean_orig_target_subwords: parseNullableNumber(r.mean_orig_target_subwords),
    mean_cont_target_subwords: parseNullableNumber(r.mean_cont_target_subwords),
    pct_single_subword_orig_target: parseNullableNumber(r.pct_single_subword_orig_target),
    pct_single_subword_cont_target: parseNullableNumber(r.pct_single_subword_cont_target),
  })) as TokenizationSummary[];

  const scoring = parseCSV(scoreRes).map((r) => ({
    model_name: r.model_name,
    track: r.track,
    pair_count: parseNullableNumber(r.pair_count) ?? 0,
    mean_orig_pll: parseNullableNumber(r.mean_orig_pll),
    mean_cont_pll: parseNullableNumber(r.mean_cont_pll),
    mean_delta_pll: parseNullableNumber(r.mean_delta_pll),
    mean_delta_mean_logp: parseNullableNumber(r.mean_delta_mean_logp),
    orig_preferred_rate: parseNullableNumber(r.orig_preferred_rate),
  })) as ScoringSummary[];

  const stability = parseCSV(stabRes).map((r) => ({
    model_name: r.model_name,
    method: r.method,
    track: r.track,
    subset: r.subset,
    pair_count: parseNullableNumber(r.pair_count) ?? 0,
    valid_visible_metric_pairs: parseNullableNumber(r.valid_visible_metric_pairs) ?? 0,
    stage6_masked_target_stability_rho: parseNullableNumber(r.stage6_masked_target_stability_rho),
    stage6_masked_target_cosine: parseNullableNumber(r.stage6_masked_target_cosine),
    visible_target_stability_rho_mean: parseNullableNumber(r.visible_target_stability_rho_mean),
    visible_target_stability_rho_median: parseNullableNumber(r.visible_target_stability_rho_median),
    visible_target_cosine_mean: parseNullableNumber(r.visible_target_cosine_mean),
    visible_target_cosine_median: parseNullableNumber(r.visible_target_cosine_median),
    methodological_note: r.methodological_note,
  })) as StabilitySummary[];

  const faithfulness = parseCSV(faithRes).map((r) => ({
    model_name: r.model_name,
    method: r.method,
    track: r.track,
    subset: r.subset,
    fraction: parseNullableNumber(r.fraction) ?? 0,
    k_tokens_mean: parseNullableNumber(r.k_tokens_mean),
    n_evaluated_sentences: parseNullableNumber(r.n_evaluated_sentences) ?? 0,
    top_k_logit_drop_mean: parseNullableNumber(r.top_k_logit_drop_mean),
    top_k_logit_drop_median: parseNullableNumber(r.top_k_logit_drop_median),
    bottom_k_logit_drop_mean: parseNullableNumber(r.bottom_k_logit_drop_mean),
    bottom_k_logit_drop_median: parseNullableNumber(r.bottom_k_logit_drop_median),
    random_k_logit_drop_mean: parseNullableNumber(r.random_k_logit_drop_mean),
    random_k_logit_drop_median: parseNullableNumber(r.random_k_logit_drop_median),
    faithfulness_gap_mean: parseNullableNumber(r.faithfulness_gap_mean),
    faithfulness_gap_median: parseNullableNumber(r.faithfulness_gap_median),
    win_rate_vs_bottom: parseNullableNumber(r.win_rate_vs_bottom),
    win_rate_vs_random: parseNullableNumber(r.win_rate_vs_random),
  })) as FaithfulnessSummary[];

  const pairMaster = parseCSV(pairRes) as unknown as PairLevelMaster[];
  const eligibility = parseCSV(eligRes) as unknown as EligibilityMaster[];

  return {
    overview: overviewRes as DatasetOverview,
    tracks,
    tokenization,
    scoring,
    stability,
    faithfulness,
    pairMaster,
    eligibility,
    auditReport: reportRes,
  };
}
