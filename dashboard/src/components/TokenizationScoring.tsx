import React from 'react';
import {
  Activity,
  Info,
  TrendingDown,
  Layers,
} from 'lucide-react';
import type { TokenizationSummary, ScoringSummary } from '../types';

interface Props {
  tokenization: TokenizationSummary[];
  scoring: ScoringSummary[];
}

export const TokenizationScoring: React.FC<Props> = ({ tokenization, scoring }) => {
  const formatNum = (v: number | null | undefined, decimals = 2, prefixSign = false): string => {
    if (typeof v !== 'number' || !Number.isFinite(v)) return '—';
    const formatted = v.toFixed(decimals);
    return prefixSign && v > 0 ? `+${formatted}` : formatted;
  };


  // Dynamically find spelling error track tokenization rows
  const murilSpelling = tokenization.find(
    (t) => t.model_name.includes('muril') && t.track === 'REALISTIC_SPELLING_ERROR'
  );
  const xlmrSpelling = tokenization.find(
    (t) => t.model_name.includes('xlm') && t.track === 'REALISTIC_SPELLING_ERROR'
  );

  const murilOrigRate = murilSpelling?.pct_single_subword_orig_target ?? null;
  const murilContRate = murilSpelling?.pct_single_subword_cont_target ?? null;
  const murilDelta =
    typeof murilContRate === 'number' && typeof murilOrigRate === 'number'
      ? murilContRate - murilOrigRate
      : null;
  const murilPairs = murilSpelling?.pair_count ?? null;

  const xlmrOrigRate = xlmrSpelling?.pct_single_subword_orig_target ?? null;
  const xlmrContRate = xlmrSpelling?.pct_single_subword_cont_target ?? null;
  const xlmrDelta =
    typeof xlmrContRate === 'number' && typeof xlmrOrigRate === 'number'
      ? xlmrContRate - xlmrOrigRate
      : null;
  const xlmrPairs = xlmrSpelling?.pair_count ?? null;

  return (
    <div>
      {/* Alert Note */}
      <div className="alert-box alert-info">
        <Info size={20} style={{ flexShrink: 0, marginTop: '2px' }} />
        <div>
          <strong>Scientific Methodology & Delta Sign Convention:</strong>
          <br />
          In Stage 3 PLL scoring, differences are computed as <code>Δ = Contrast - Original</code>.
          A negative value (<code>Δ &lt; 0</code>) signifies that the model assigned higher token predictability
          to the <strong>Original</strong> sentence than the Contrast sentence.
          Pseudo-log-likelihood is a relative measure of masked token predictability under context, <em>not</em> an absolute proof of grammatical correctness.
        </div>
      </div>

      {/* Subword Fragmentation Section */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <Layers size={18} color="var(--accent-cyan)" />
              Subword Tokenization & Length Inflation
            </div>
            <div className="panel-subtitle">
              Measuring how Devanagari spelling errors and lexical contrasts fragment into subwords across {tokenization.length} model-track conditions
            </div>
          </div>
        </div>

        <div className="table-wrapper" style={{ marginBottom: '24px' }}>
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Track</th>
                <th>Pairs</th>
                <th>Mean Orig Tokens</th>
                <th>Mean Cont Tokens</th>
                <th>Mean Δ Tokens</th>
                <th>Length Ratio</th>
                <th>Orig Target Subwords</th>
                <th>Cont Target Subwords</th>
                <th>Single Subword % (Orig → Cont)</th>
              </tr>
            </thead>
            <tbody>
              {tokenization.map((row) => (
                <tr key={`${row.model_name}_${row.track}`}>
                  <td>
                    <span className={row.model_name.includes('muril') ? 'badge badge-muril' : 'badge badge-xlmr'}>
                      {row.model_name.includes('muril') ? 'MuRIL' : 'XLM-R'}
                    </span>
                  </td>
                  <td style={{ fontWeight: 500 }}>{row.track}</td>
                  <td>{row.pair_count}</td>
                  <td>{formatNum(row.mean_orig_tokens, 2)}</td>
                  <td>{formatNum(row.mean_cont_tokens, 2)}</td>
                  <td
                    style={{
                      fontWeight: 600,
                      color: typeof row.mean_delta_tokens === 'number' && row.mean_delta_tokens > 0 ? '#fbbf24' : 'inherit',
                    }}
                  >
                    {formatNum(row.mean_delta_tokens, 2, true)}
                  </td>
                  <td>
                    {typeof row.mean_length_ratio === 'number' ? `${row.mean_length_ratio.toFixed(2)}x` : '—'}
                  </td>
                  <td>{formatNum(row.mean_orig_target_subwords, 2)}</td>
                  <td>{formatNum(row.mean_cont_target_subwords, 2)}</td>
                  <td>
                    <span style={{ color: '#34d399', fontWeight: 600 }}>
                      {formatNum(row.pct_single_subword_orig_target, 1)}%
                    </span>
                    {' → '}
                    <span style={{ color: '#fb7185', fontWeight: 600 }}>
                      {formatNum(row.pct_single_subword_cont_target, 1)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Visual Bar Comparison for Subword Collapse */}
        <div style={{ background: 'var(--bg-inset)', padding: '20px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ fontSize: '13.5px', fontWeight: 600, marginBottom: '14px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <TrendingDown size={16} color="var(--accent-amber)" />
            Subword Vocabulary Coverage Collapse under Spelling Perturbations (Track: REALISTIC_SPELLING_ERROR)
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px' }}>
            {/* MuRIL */}
            <div style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '16px', borderRadius: 'var(--radius-md)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span className="badge badge-muril">MuRIL</span>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Track: REALISTIC_SPELLING_ERROR (N={murilPairs ?? '—'} pairs)
                </span>
              </div>
              <div style={{ marginBottom: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                  <span>Correct Spelling (Original)</span>
                  <strong style={{ color: '#34d399' }}>
                    {typeof murilOrigRate === 'number' ? `${murilOrigRate.toFixed(1)}% single-subword` : '—'}
                  </strong>
                </div>
                <div style={{ height: '8px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${Math.min(100, Math.max(0, murilOrigRate ?? 0))}%`,
                      height: '100%',
                      background: '#34d399',
                      borderRadius: '4px',
                      transition: 'width 0.3s ease',
                    }}
                  />
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                  <span>Misspelled Word (Perturbed)</span>
                  <strong style={{ color: '#fb7185' }}>
                    {typeof murilContRate === 'number'
                      ? `${murilContRate.toFixed(1)}% single-subword (${formatNum(murilDelta, 1, true)}%)`
                      : '—'}
                  </strong>
                </div>
                <div style={{ height: '8px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${Math.min(100, Math.max(0, murilContRate ?? 0))}%`,
                      height: '100%',
                      background: '#fb7185',
                      borderRadius: '4px',
                      transition: 'width 0.3s ease',
                    }}
                  />
                </div>
              </div>
            </div>

            {/* XLM-R */}
            <div style={{ background: 'rgba(255, 255, 255, 0.02)', padding: '16px', borderRadius: 'var(--radius-md)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span className="badge badge-xlmr">XLM-RoBERTa</span>
                <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  Track: REALISTIC_SPELLING_ERROR (N={xlmrPairs ?? '—'} pairs)
                </span>
              </div>
              <div style={{ marginBottom: '10px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                  <span>Correct Spelling (Original)</span>
                  <strong style={{ color: '#34d399' }}>
                    {typeof xlmrOrigRate === 'number' ? `${xlmrOrigRate.toFixed(1)}% single-subword` : '—'}
                  </strong>
                </div>
                <div style={{ height: '8px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${Math.min(100, Math.max(0, xlmrOrigRate ?? 0))}%`,
                      height: '100%',
                      background: '#34d399',
                      borderRadius: '4px',
                      transition: 'width 0.3s ease',
                    }}
                  />
                </div>
              </div>
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                  <span>Misspelled Word (Perturbed)</span>
                  <strong style={{ color: '#fb7185' }}>
                    {typeof xlmrContRate === 'number'
                      ? `${xlmrContRate.toFixed(1)}% single-subword (${formatNum(xlmrDelta, 1, true)}%)`
                      : '—'}
                  </strong>
                </div>
                <div style={{ height: '8px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '4px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${Math.min(100, Math.max(0, xlmrContRate ?? 0))}%`,
                      height: '100%',
                      background: '#fb7185',
                      borderRadius: '4px',
                      transition: 'width 0.3s ease',
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Model Scoring Section */}
      <div className="panel">
        <div className="panel-header">
          <div>
            <div className="panel-title">
              <Activity size={18} color="var(--accent-emerald)" />
              Pseudo-Log-Likelihood (PLL) Scoring Summary
            </div>
            <div className="panel-subtitle">
              Evaluating model sensitivity across experimental tracks (Stage 3)
            </div>
          </div>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Track</th>
                <th>Pairs</th>
                <th>Mean Orig PLL</th>
                <th>Mean Cont PLL</th>
                <th>Mean Δ PLL (Cont - Orig)</th>
                <th>Mean Δ MeanLogP</th>
                <th>Original Preferred Rate</th>
              </tr>
            </thead>
            <tbody>
              {scoring.map((row) => (
                <tr key={`${row.model_name}_${row.track}`}>
                  <td>
                    <span className={row.model_name.includes('muril') ? 'badge badge-muril' : 'badge badge-xlmr'}>
                      {row.model_name.includes('muril') ? 'MuRIL' : 'XLM-R'}
                    </span>
                  </td>
                  <td style={{ fontWeight: 500 }}>{row.track}</td>
                  <td>{row.pair_count}</td>
                  <td>{formatNum(row.mean_orig_pll, 2)}</td>
                  <td>{formatNum(row.mean_cont_pll, 2)}</td>
                  <td
                    style={{
                      fontWeight: 600,
                      color: typeof row.mean_delta_pll === 'number' && row.mean_delta_pll < 0 ? '#34d399' : '#fb7185',
                    }}
                  >
                    {formatNum(row.mean_delta_pll, 2)}
                  </td>
                  <td>{formatNum(row.mean_delta_mean_logp, 3)}</td>
                  <td>
                    <span
                      className="badge"
                      style={{
                        background:
                          typeof row.orig_preferred_rate === 'number' && row.orig_preferred_rate >= 0.8
                            ? 'rgba(16, 185, 129, 0.15)'
                            : 'rgba(245, 158, 11, 0.15)',
                        color:
                          typeof row.orig_preferred_rate === 'number' && row.orig_preferred_rate >= 0.8 ? '#34d399' : '#fbbf24',
                        border: `1px solid ${
                          typeof row.orig_preferred_rate === 'number' && row.orig_preferred_rate >= 0.8
                            ? 'rgba(16, 185, 129, 0.3)'
                            : 'rgba(245, 158, 11, 0.3)'
                        }`,
                      }}
                    >
                      {typeof row.orig_preferred_rate === 'number' ? `${(row.orig_preferred_rate * 100).toFixed(1)}% preferred` : '—'}
                    </span>
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
