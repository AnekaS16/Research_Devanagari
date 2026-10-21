import React, { useState, useMemo } from 'react';
import {
  ShieldAlert,
  GitCompare,
  BarChart2,
} from 'lucide-react';
import type { StabilitySummary, PairLevelMaster } from '../types';

interface Props {
  stability: StabilitySummary[];
  pairMaster: PairLevelMaster[];
}

export const ExplanationStability: React.FC<Props> = ({ stability, pairMaster }) => {
  const [selectedModel, setSelectedModel] = useState<string>('ALL');
  const [selectedMethod, setSelectedMethod] = useState<'ALL' | 'attention' | 'integrated_gradients'>('ALL');
  const [selectedSubset, setSelectedSubset] = useState<'all_scorable_pooled' | 'strict_single_subword' | 'multi_subword_remainder'>('all_scorable_pooled');

  // Filter stability summary rows
  const filteredStability = useMemo(() => {
    return stability.filter((s) => {
      const matchModel = selectedModel === 'ALL' || s.model_name === selectedModel;
      const matchMethod = selectedMethod === 'ALL' || s.method === selectedMethod;
      const matchSubset = s.subset === selectedSubset;
      return matchModel && matchMethod && matchSubset;
    });
  }, [stability, selectedModel, selectedMethod, selectedSubset]);

  // Filter spelling pairs for pair-level inspector
  const spellingPairs = useMemo(() => {
    return pairMaster.filter((p) => {
      const matchTrack = p.track === 'REALISTIC_SPELLING_ERROR';
      const matchModel = selectedModel === 'ALL' || p.model_name === selectedModel;
      return matchTrack && matchModel;
    });
  }, [pairMaster, selectedModel]);

  const formatNum = (v: number | null | undefined, decimals = 3): string => {
    if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
    return v.toFixed(decimals);
  };

  // Derive dynamic card metrics strictly from loaded, filtered stability data
  const cardMetrics = useMemo(() => {
    if (!filteredStability || filteredStability.length === 0) {
      return {
        stage6Rho: '—',
        visibleRhoMean: '—',
        visibleRhoMedian: '—',
        visibleCosineMean: '—',
        sampleSizeDisplay: 'N = 0 pairs',
        validMetricPairsDisplay: '0',
        note: 'No stability summary data available for current filter.',
      };
    }

    const validStage6 = filteredStability
      .map((r) => r.stage6_masked_target_stability_rho)
      .filter((v): v is number => typeof v === 'number');
    const validRhoMeans = filteredStability
      .map((r) => r.visible_target_stability_rho_mean)
      .filter((v): v is number => typeof v === 'number');
    const validRhoMedians = filteredStability
      .map((r) => r.visible_target_stability_rho_median)
      .filter((v): v is number => typeof v === 'number');
    const validCosines = filteredStability
      .map((r) => r.visible_target_cosine_mean)
      .filter((v): v is number => typeof v === 'number');

    const meanStage6 = validStage6.length ? validStage6.reduce((a, b) => a + b, 0) / validStage6.length : null;
    const meanRho = validRhoMeans.length ? validRhoMeans.reduce((a, b) => a + b, 0) / validRhoMeans.length : null;
    const medianRho = validRhoMedians.length ? validRhoMedians.reduce((a, b) => a + b, 0) / validRhoMedians.length : null;
    const meanCosine = validCosines.length ? validCosines.reduce((a, b) => a + b, 0) / validCosines.length : null;

    // Distinct pair counts
    const pairCounts = Array.from(new Set(filteredStability.map((r) => r.pair_count)));
    const validPairs = Array.from(new Set(filteredStability.map((r) => r.valid_visible_metric_pairs)));

    let sampleSizeDisplay = '';
    if (pairCounts.length === 1) {
      sampleSizeDisplay = `N = ${pairCounts[0]} pairs`;
    } else {
      sampleSizeDisplay = `N = ${pairCounts.join('/')} pairs`;
    }

    let validMetricPairsDisplay = '';
    if (validPairs.length === 1) {
      validMetricPairsDisplay = `${validPairs[0]} pairs`;
    } else {
      validMetricPairsDisplay = `${validPairs.join('/')} pairs`;
    }

    return {
      stage6Rho: formatNum(meanStage6, 3),
      visibleRhoMean: formatNum(meanRho, 3),
      visibleRhoMedian: formatNum(medianRho, 3),
      visibleCosineMean: formatNum(meanCosine, 4),
      sampleSizeDisplay,
      validMetricPairsDisplay,
      note: filteredStability.length === 1 ? filteredStability[0].methodological_note : 'Averaged across matched models/methods within active subset.',
    };
  }, [filteredStability]);

  return (
    <div>
      {/* Critical Methodology Warning */}
      <div className="alert-box alert-warning">
        <ShieldAlert size={22} style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong style={{ fontSize: '14px' }}>Stage 6 vs. Stage 7 Stability Resolution:</strong>
          <br />
          In Stage 6, the target token was masked in both sentences. Because the spelling error was confined to the target word,
          both model inputs became identical (<code>Masked(Orig) ≡ Masked(Cont)</code>), yielding trivial stability (<code>ρ = 1.0</code>).
          <br />
          <strong>Stage 7 resolves this</strong> by keeping the target word visible during prediction. Under visible targets,
          spelling variations produce genuine attribution shifts across aligned context tokens.
        </div>
      </div>

      {/* Dynamic Stability Metrics Cards */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-header">
            <span>Stage 6 Masked Stability</span>
            <span className="badge badge-warning">Design Artifact</span>
          </div>
          <div className="metric-value" style={{ color: 'var(--text-muted)' }}>
            {cardMetrics.stage6Rho}
          </div>
          <div className="metric-subtext">Zero variance (inputs identical under target masking)</div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span>Stage 7 Visible Target ρ</span>
            <span className="badge badge-muril">Mean Rank Stability</span>
          </div>
          <div className="metric-value" style={{ color: '#38bdf8' }}>
            {cardMetrics.visibleRhoMean}
          </div>
          <div className="metric-subtext">
            Median ρ: <strong>{cardMetrics.visibleRhoMedian}</strong> ({selectedSubset.replace(/_/g, ' ')})
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span>Mean Cosine Similarity</span>
            <span className="badge badge-success">Directional Alignment</span>
          </div>
          <div className="metric-value" style={{ color: '#34d399' }}>
            {cardMetrics.visibleCosineMean}
          </div>
          <div className="metric-subtext">Context attribution vector angle consistency</div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span>Evaluated Pairs & Subset</span>
            <span className="badge badge-xlmr">Verified Sample</span>
          </div>
          <div className="metric-value" style={{ color: '#818cf8' }}>
            {cardMetrics.sampleSizeDisplay}
          </div>
          <div className="metric-subtext">
            Valid metric evaluations: <strong>{cardMetrics.validMetricPairsDisplay}</strong>
          </div>
        </div>
      </div>

      {/* Method & Subset Controls */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <GitCompare size={18} color="var(--accent-cyan)" />
              Visible-Target Explanation Stability by Model & Method
            </div>
            <div className="panel-subtitle">
              Evaluating attribution ranking consistency when Hindi target words are misspelled ({filteredStability.length} configuration(s) shown)
            </div>
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
            {/* Model Filter */}
            <div className="filter-group">
              <select
                className="select-input"
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                <option value="ALL">All Models</option>
                <option value="FacebookAI/xlm-roberta-base">XLM-RoBERTa</option>
                <option value="google/muril-base-cased">MuRIL</option>
              </select>
            </div>

            {/* Method Filter */}
            <div style={{ display: 'flex', gap: '6px' }}>
              {(['ALL', 'attention', 'integrated_gradients'] as const).map((m) => (
                <button
                  key={m}
                  className={`btn ${selectedMethod === m ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => setSelectedMethod(m)}
                >
                  {m === 'ALL' ? 'All Methods' : m === 'attention' ? 'Attention' : 'Integrated Gradients'}
                </button>
              ))}
            </div>

            {/* Subset Filter */}
            <div style={{ display: 'flex', gap: '6px' }}>
              <button
                className={`btn ${selectedSubset === 'all_scorable_pooled' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setSelectedSubset('all_scorable_pooled')}
              >
                Pooled All-20 Pairs (N=20)
              </button>
              <button
                className={`btn ${selectedSubset === 'strict_single_subword' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setSelectedSubset('strict_single_subword')}
              >
                Strict Single-Subword (N=4/6)
              </button>
              <button
                className={`btn ${selectedSubset === 'multi_subword_remainder' ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setSelectedSubset('multi_subword_remainder')}
              >
                Multi-Subword (N=16/14)
              </button>
            </div>
          </div>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Method</th>
                <th>Subset</th>
                <th>Evaluated Pairs</th>
                <th>Valid Metrics (N)</th>
                <th>Stage 6 Masked ρ</th>
                <th>Stage 7 Visible ρ (Mean)</th>
                <th>Stage 7 Visible ρ (Median)</th>
                <th>Cosine Sim (Mean)</th>
                <th>Methodological Note</th>
              </tr>
            </thead>
            <tbody>
              {filteredStability.length === 0 ? (
                <tr>
                  <td colSpan={10} style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
                    No stability summary records match the selected filter configuration.
                  </td>
                </tr>
              ) : (
                filteredStability.map((row) => (
                  <tr key={`${row.model_name}_${row.method}_${row.subset}`}>
                    <td>
                      <span className={row.model_name.includes('muril') ? 'badge badge-muril' : 'badge badge-xlmr'}>
                        {row.model_name.includes('muril') ? 'MuRIL' : 'XLM-R'}
                      </span>
                    </td>
                    <td style={{ fontWeight: 600 }}>
                      <code>{row.method}</code>
                    </td>
                    <td>
                      <span className="badge badge-success">{row.subset}</span>
                    </td>
                    <td>{row.pair_count}</td>
                    <td>{row.valid_visible_metric_pairs}</td>
                    <td style={{ color: 'var(--text-muted)' }}>
                      {formatNum(row.stage6_masked_target_stability_rho, 1)}
                    </td>
                    <td style={{ fontWeight: 700, color: '#38bdf8' }}>
                      {formatNum(row.visible_target_stability_rho_mean, 3)}
                    </td>
                    <td>
                      {formatNum(row.visible_target_stability_rho_median, 3)}
                    </td>
                    <td style={{ color: '#34d399', fontWeight: 600 }}>
                      {formatNum(row.visible_target_cosine_mean, 4)}
                    </td>
                    <td style={{ fontSize: '11.5px', color: 'var(--text-muted)', maxWidth: '300px' }}>
                      {row.methodological_note}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Pair-Level Inspector Table */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <BarChart2 size={18} color="var(--accent-indigo)" />
              Pair-Level Stability Inspector (Track: REALISTIC_SPELLING_ERROR)
            </div>
            <div className="panel-subtitle">
              Detailed attribution shifts for Devanagari spelling error sentence pairs ({spellingPairs.length} evaluation(s))
            </div>
          </div>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Pair ID</th>
                <th>Model</th>
                <th>Target (Orig / Cont)</th>
                <th>Original Sentence</th>
                <th>Contrast Sentence</th>
                <th>Visible Attn Rho</th>
                <th>Visible Attn Cosine</th>
                <th>Visible IG Rho</th>
                <th>Visible IG Cosine</th>
              </tr>
            </thead>
            <tbody>
              {spellingPairs.map((p) => (
                <tr key={`${p.pair_id}_${p.model_name}`}>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#38bdf8' }}>
                    {p.pair_id}
                  </td>
                  <td>
                    <span className={p.model_name.includes('muril') ? 'badge badge-muril' : 'badge badge-xlmr'}>
                      {p.model_name.includes('muril') ? 'MuRIL' : 'XLM-R'}
                    </span>
                  </td>
                  <td className="devanagari-text" style={{ fontWeight: 600 }}>
                    <span style={{ color: '#34d399' }}>{p.target_original}</span>
                    {' → '}
                    <span style={{ color: '#fda4af' }}>{p.target_contrast}</span>
                  </td>
                  <td className="devanagari-sentence" style={{ maxWidth: '240px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {p.original_sentence}
                  </td>
                  <td className="devanagari-sentence" style={{ maxWidth: '240px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {p.contrast_sentence}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#38bdf8' }}>
                    {p.stage7_visible_attn_rho ? Number(p.stage7_visible_attn_rho).toFixed(3) : '—'}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {p.stage7_visible_attn_cosine ? Number(p.stage7_visible_attn_cosine).toFixed(4) : '—'}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#818cf8' }}>
                    {p.stage7_visible_ig_rho ? Number(p.stage7_visible_ig_rho).toFixed(3) : '—'}
                  </td>
                  <td style={{ fontFamily: 'var(--font-mono)' }}>
                    {p.stage7_visible_ig_cosine ? Number(p.stage7_visible_ig_cosine).toFixed(4) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
