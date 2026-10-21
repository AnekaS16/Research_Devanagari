import React, { useState, useMemo } from 'react';
import {
  Award,
  AlertCircle,
} from 'lucide-react';
import type { FaithfulnessSummary } from '../types';

interface Props {
  faithfulness: FaithfulnessSummary[];
}

export const FaithfulnessLab: React.FC<Props> = ({ faithfulness }) => {
  const [selectedModel, setSelectedModel] = useState<string>('ALL');
  const [selectedMethod, setSelectedMethod] = useState<string>('ALL');
  const [selectedFraction, setSelectedFraction] = useState<number>(0.2);
  const [selectedSubset, setSelectedSubset] = useState<'all_scorable' | 'strict_single_subword'>('all_scorable');

  const filtered = useMemo(() => {
    return faithfulness.filter((f) => {
      const matchModel = selectedModel === 'ALL' || f.model_name === selectedModel;
      const matchMethod = selectedMethod === 'ALL' || f.method === selectedMethod;
      const matchFrac = Math.abs(f.fraction - selectedFraction) < 1e-4;
      const matchSubset = f.subset === selectedSubset;
      return matchModel && matchMethod && matchFrac && matchSubset;
    });
  }, [faithfulness, selectedModel, selectedMethod, selectedFraction, selectedSubset]);

  const formatNum = (v: number | null | undefined, decimals = 2, prefixSign = false): string => {
    if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
    const formatted = v.toFixed(decimals);
    return prefixSign && v > 0 ? `+${formatted}` : formatted;
  };

  const formatPct = (v: number | null | undefined, decimals = 1): string => {
    if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
    return `${(v * 100).toFixed(decimals)}%`;
  };

  // Derive dynamic card metrics from filtered rows
  const cardMetrics = useMemo(() => {
    if (!filtered || filtered.length === 0) {
      return {
        gapDisplay: '—',
        gapValue: null as number | null,
        winRateDisplay: '—',
        winRateValue: null as number | null,
        topDropDisplay: '—',
        topDropValue: null as number | null,
        sampleSizeDisplay: 'N = 0',
        kTokensDisplay: '—',
        isAvailable: false,
      };
    }

    const validGaps = filtered.map((r) => r.faithfulness_gap_mean).filter((v): v is number => typeof v === 'number');
    const validWins = filtered.map((r) => r.win_rate_vs_bottom).filter((v): v is number => typeof v === 'number');
    const validDrops = filtered.map((r) => r.top_k_logit_drop_mean).filter((v): v is number => typeof v === 'number');
    const validKs = filtered.map((r) => r.k_tokens_mean).filter((v): v is number => typeof v === 'number');

    const meanGap = validGaps.length ? validGaps.reduce((a, b) => a + b, 0) / validGaps.length : null;
    const meanWin = validWins.length ? validWins.reduce((a, b) => a + b, 0) / validWins.length : null;
    const meanDrop = validDrops.length ? validDrops.reduce((a, b) => a + b, 0) / validDrops.length : null;
    const meanK = validKs.length ? validKs.reduce((a, b) => a + b, 0) / validKs.length : null;

    // Sample size display
    let sampleSizeDisplay = '';
    if (filtered.length === 1) {
      sampleSizeDisplay = `N = ${filtered[0].n_evaluated_sentences} sentences`;
    } else {
      const distinctSizes = Array.from(new Set(filtered.map((r) => r.n_evaluated_sentences)));
      sampleSizeDisplay = distinctSizes.length === 1 ? `N = ${distinctSizes[0]} per config` : `N = ${distinctSizes.join('/')} sentences`;
    }

    return {
      gapDisplay: formatNum(meanGap, 2, true),
      gapValue: meanGap,
      winRateDisplay: formatPct(meanWin, 1),
      winRateValue: meanWin,
      topDropDisplay: formatNum(meanDrop, 2, true),
      topDropValue: meanDrop,
      sampleSizeDisplay,
      kTokensDisplay: formatNum(meanK, 1),
      isAvailable: true,
    };
  }, [filtered]);

  return (
    <div>
      {/* Sign Convention & OOD Explanation Callout */}
      <div className="alert-box alert-warning">
        <AlertCircle size={22} style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong style={{ fontSize: '14px' }}>Score Change Sign Convention & OOD Interventions:</strong>
          <br />
          Score drop is defined as <code>ΔS = S_baseline - S_intervened</code>. In XLM-RoBERTa, unnormalized target logits frequently
          increased when context words were replaced by <code>[MASK]</code>, yielding negative drop values (-7.19 for top-k, -11.21 for bottom-k).
          Top-k masking produced a lower target logit than bottom-k masking in the reported comparison, although both interventions increased the target logit relative to baseline.
          Because bottom-k masking produced an even larger logit increase than top-k masking,
          the <strong>Faithfulness Gap</strong> (<code>ΔS_top - ΔS_bottom = (-7.19) - (-11.21) = +4.02</code>) remained strictly positive,
          confirming that removing top tokens resulted in lower target confidence than control tokens.
          <br />
          <em>Note:</em> Masking context tokens can push sequences out-of-distribution (OOD); score changes reflect both informational removal and syntactic shock.
        </div>
      </div>

      {/* Control Toolbar */}
      <div className="filter-bar" style={{ display: 'flex', flexWrap: 'wrap', gap: '14px', alignItems: 'center' }}>
        {/* Model Filter */}
        <div className="filter-group">
          <span className="filter-label">Model:</span>
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
        <div className="filter-group">
          <span className="filter-label">Explanation Method:</span>
          <select
            className="select-input"
            value={selectedMethod}
            onChange={(e) => setSelectedMethod(e.target.value)}
          >
            <option value="ALL">All Methods</option>
            <option value="attention">Attention</option>
            <option value="integrated_gradients">Integrated Gradients</option>
          </select>
        </div>

        {/* Masking Fraction */}
        <div className="filter-group">
          <span className="filter-label">Masking Fraction:</span>
          <div style={{ display: 'flex', gap: '6px' }}>
            {[0.1, 0.2, 0.3].map((frac) => (
              <button
                key={frac}
                className={`btn ${Math.abs(selectedFraction - frac) < 1e-4 ? 'btn-primary' : 'btn-secondary'}`}
                onClick={() => setSelectedFraction(frac)}
              >
                Top {(frac * 100).toFixed(0)}%
              </button>
            ))}
          </div>
        </div>

        {/* Evaluation Subset */}
        <div className="filter-group">
          <span className="filter-label">Evaluation Subset:</span>
          <div style={{ display: 'flex', gap: '6px' }}>
            <button
              className={`btn ${selectedSubset === 'all_scorable' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setSelectedSubset('all_scorable')}
            >
              All Scorable (N=40)
            </button>
            <button
              className={`btn ${selectedSubset === 'strict_single_subword' ? 'btn-primary' : 'btn-secondary'}`}
              onClick={() => setSelectedSubset('strict_single_subword')}
            >
              Strict Single-Subword (N=8/12)
            </button>
          </div>
        </div>
      </div>

      {/* Dynamic Summary Cards */}
      <div className="metrics-grid">
        {/* Card 1: Faithfulness Gap */}
        <div className="metric-card">
          <div className="metric-header">
            <span>Faithfulness Gap (ΔS_top - ΔS_bot)</span>
            <span
              className={`badge ${
                cardMetrics.gapValue === null
                  ? 'badge-warning'
                  : cardMetrics.gapValue > 0
                  ? 'badge-success'
                  : 'badge-rose'
              }`}
            >
              {cardMetrics.gapValue === null
                ? 'Unavailable'
                : cardMetrics.gapValue > 0
                ? 'Top Mask > Bot Mask'
                : 'Top Mask ≤ Bot Mask'}
            </span>
          </div>
          <div
            className="metric-value"
            style={{
              color:
                cardMetrics.gapValue === null
                  ? 'var(--text-muted)'
                  : cardMetrics.gapValue > 0
                  ? '#34d399'
                  : '#fb7185',
            }}
          >
            {cardMetrics.gapDisplay}
          </div>
          <div className="metric-subtext">
            {cardMetrics.gapValue === null
              ? 'No data for filter'
              : cardMetrics.gapValue > 0
              ? 'Positive gap confirms top attributions caused lower target confidence than control tokens'
              : 'Zero/negative gap: top tokens did not cause lower target confidence than control'}
          </div>
        </div>

        {/* Card 2: Win Rate vs Bottom */}
        <div className="metric-card">
          <div className="metric-header">
            <span>Win Rate vs Bottom Control</span>
            <span
              className={`badge ${
                cardMetrics.winRateValue === null
                  ? 'badge-warning'
                  : cardMetrics.winRateValue >= 0.7
                  ? 'badge-success'
                  : cardMetrics.winRateValue >= 0.5
                  ? 'badge-xlmr'
                  : 'badge-warning'
              }`}
            >
              {cardMetrics.winRateValue === null
                ? 'Unavailable'
                : cardMetrics.winRateValue >= 0.5
                ? 'Above Chance'
                : 'Near/Below Chance'}
            </span>
          </div>
          <div
            className="metric-value"
            style={{
              color:
                cardMetrics.winRateValue === null
                  ? 'var(--text-muted)'
                  : cardMetrics.winRateValue >= 0.7
                  ? '#34d399'
                  : cardMetrics.winRateValue >= 0.5
                  ? '#38bdf8'
                  : '#fbbf24',
            }}
          >
            {cardMetrics.winRateDisplay}
          </div>
          <div className="metric-subtext">
            {cardMetrics.winRateValue === null
              ? 'No data for filter'
              : 'Fraction of sentences where top-k logit was lower than bottom-k logit'}
          </div>
        </div>

        {/* Card 3: Top-k Logit Drop */}
        <div className="metric-card">
          <div className="metric-header">
            <span>Top-k Logit Drop (ΔS_top)</span>
            <span className="badge badge-warning">Sign Convention</span>
          </div>
          <div className="metric-value" style={{ color: '#818cf8' }}>
            {cardMetrics.topDropDisplay}
          </div>
          <div className="metric-subtext">
            {cardMetrics.topDropValue === null
              ? 'No data for filter'
              : cardMetrics.topDropValue < 0
              ? 'Target logit increased relative to baseline (+7.2 logits), but remained lower than control'
              : 'Target logit decreased relative to baseline under top-k token masking'}
          </div>
        </div>

        {/* Card 4: Selected Sample Size */}
        <div className="metric-card">
          <div className="metric-header">
            <span>Selected Sample Size</span>
            <span className="badge badge-muril">Verified Evals</span>
          </div>
          <div className="metric-value" style={{ color: '#38bdf8' }}>
            {cardMetrics.sampleSizeDisplay}
          </div>
          <div className="metric-subtext">
            Mean masked context tokens: <strong>{cardMetrics.kTokensDisplay}</strong> (fraction: {(selectedFraction * 100).toFixed(0)}%)
          </div>
        </div>
      </div>

      {/* Detailed Intervention Comparison Table */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <Award size={18} color="var(--accent-emerald)" />
              Context-Token Intervention Results (Fraction = {(selectedFraction * 100).toFixed(0)}%, {selectedSubset})
            </div>
            <div className="panel-subtitle">
              Comparing target logit changes under Top-k, Bottom-k, and Random-k masking across {filtered.length} matching configuration(s)
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
                <th>Sentences (N)</th>
                <th>Mean k Tokens</th>
                <th>Top-k Logit Drop</th>
                <th>Bottom-k Logit Drop</th>
                <th>Random-k Logit Drop</th>
                <th>Faithfulness Gap (Top - Bottom)</th>
                <th>Win Rate vs Bottom</th>
                <th>Win Rate vs Random</th>
              </tr>
            </thead>
            <tbody>
              {filtered.length === 0 ? (
                <tr>
                  <td colSpan={11} style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
                    No faithfulness summary records match the selected filter configuration.
                  </td>
                </tr>
              ) : (
                filtered.map((row) => (
                  <tr key={`${row.model_name}_${row.method}_${row.subset}_${row.fraction}`}>
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
                    <td>{row.n_evaluated_sentences}</td>
                    <td>{formatNum(row.k_tokens_mean, 1)}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{formatNum(row.top_k_logit_drop_mean, 2)}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{formatNum(row.bottom_k_logit_drop_mean, 2)}</td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>{formatNum(row.random_k_logit_drop_mean, 2)}</td>
                    <td
                      style={{
                        fontFamily: 'var(--font-mono)',
                        fontWeight: 700,
                        color: typeof row.faithfulness_gap_mean === 'number' && row.faithfulness_gap_mean > 0 ? '#34d399' : '#fb7185',
                      }}
                    >
                      {formatNum(row.faithfulness_gap_mean, 2, true)}
                    </td>
                    <td>
                      <span
                        className="badge"
                        style={{
                          background:
                            typeof row.win_rate_vs_bottom === 'number' && row.win_rate_vs_bottom >= 0.7
                              ? 'rgba(16, 185, 129, 0.15)'
                              : 'rgba(245, 158, 11, 0.15)',
                          color:
                            typeof row.win_rate_vs_bottom === 'number' && row.win_rate_vs_bottom >= 0.7 ? '#34d399' : '#fbbf24',
                          border: `1px solid ${
                            typeof row.win_rate_vs_bottom === 'number' && row.win_rate_vs_bottom >= 0.7
                              ? 'rgba(16, 185, 129, 0.3)'
                              : 'rgba(245, 158, 11, 0.3)'
                          }`,
                        }}
                      >
                        {formatPct(row.win_rate_vs_bottom, 1)}
                      </span>
                    </td>
                    <td>
                      <span className="badge badge-muril">
                        {formatPct(row.win_rate_vs_random, 1)}
                      </span>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
