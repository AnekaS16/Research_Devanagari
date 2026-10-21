import React from 'react';
import {
  Database,
  Layers,
  Cpu,
  CheckCircle2,
  GitBranch,
  ShieldCheck,
  FileSpreadsheet,
} from 'lucide-react';
import type { DatasetOverview, DatasetTrack } from '../types';

interface Props {
  overview: DatasetOverview | null;
  tracks: DatasetTrack[];
}

export const ResearchOverview: React.FC<Props> = ({ overview, tracks }) => {
  if (!overview) return null;

  return (
    <div>
      {/* Title Header Card */}
      <div className="panel" style={{ background: 'linear-gradient(135deg, rgba(20, 28, 45, 0.95) 0%, rgba(30, 42, 68, 0.85) 100%)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '16px' }}>
          <div>
            <div className="badge badge-muril" style={{ marginBottom: '12px' }}>
              <ShieldCheck size={13} /> Audited Research Data Layer
            </div>
            <h1 style={{ fontSize: '26px', fontWeight: 700, letterSpacing: '-0.5px', marginBottom: '8px' }}>
              Hindi Lexical Sensitivity & Explanation Faithfulness
            </h1>
            <p style={{ color: 'var(--text-secondary)', fontSize: '14.5px', maxWidth: '840px', lineHeight: '1.6' }}>
              A systematic investigation into how multilingual language models (<strong>MuRIL</strong> and <strong>XLM-RoBERTa</strong>)
              tokenize, score, and explain contextual contrasts, synonym substitutions, and realistic Devanagari spelling errors.
            </p>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', alignItems: 'flex-end' }}>
            <div className="provenance-pill">
              <CheckCircle2 size={13} /> SHA-256 Verified
            </div>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: '11px', color: 'var(--text-muted)' }}>
              {overview.dataset_sha256.substring(0, 16)}...
            </span>
          </div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-header">
            <span>Source Sentence Pairs</span>
            <Database size={18} color="var(--accent-cyan)" />
          </div>
          <div className="metric-value">{overview.total_source_pairs}</div>
          <div className="metric-subtext">200 curated Devanagari pairs</div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span>Experimental Tracks</span>
            <Layers size={18} color="var(--accent-indigo)" />
          </div>
          <div className="metric-value">{tracks.length}</div>
          <div className="metric-subtext">Context, Synonyms & Spelling</div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span>Pretrained Models</span>
            <Cpu size={18} color="var(--accent-emerald)" />
          </div>
          <div className="metric-value">2</div>
          <div className="metric-subtext">google/muril & FacebookAI/xlm-r</div>
        </div>

        <div className="metric-card">
          <div className="metric-header">
            <span>Total Interventions</span>
            <GitBranch size={18} color="var(--accent-amber)" />
          </div>
          <div className="metric-value">1,440</div>
          <div className="metric-subtext">Faithfulness test evaluations</div>
        </div>
      </div>

      {/* Track Distribution & Pipeline Provenance */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(420px, 1fr))', gap: '24px' }}>
        {/* Track Distribution */}
        <div className="panel" style={{ margin: 0 }}>
          <div className="panel-header">
            <div>
              <div className="panel-title">
                <FileSpreadsheet size={18} color="var(--accent-cyan)" />
                Dataset Track Distribution
              </div>
              <div className="panel-subtitle">Audited pair counts by contrast category</div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
            {tracks.map((t) => (
              <div
                key={t.track}
                style={{
                  background: 'var(--bg-inset)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  padding: '14px 16px',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                  <span style={{ fontWeight: 600, fontSize: '13.5px' }}>{t.track}</span>
                  <span className="badge badge-muril">{t.pair_count} pairs ({t.percentage}%)</span>
                </div>
                <div style={{ fontSize: '12.5px', color: 'var(--text-secondary)', marginBottom: '8px' }}>
                  {t.description}
                </div>
                {/* Progress bar */}
                <div style={{ height: '6px', background: 'rgba(255, 255, 255, 0.06)', borderRadius: '3px', overflow: 'hidden' }}>
                  <div
                    style={{
                      width: `${t.percentage}%`,
                      height: '100%',
                      background: t.track === 'REALISTIC_SPELLING_ERROR'
                        ? 'var(--accent-rose)'
                        : t.track === 'SYNONYM_SUBSTITUTION'
                        ? 'var(--accent-indigo)'
                        : 'var(--accent-cyan)',
                      borderRadius: '3px',
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Provenance & Pipeline Timeline */}
        <div className="panel" style={{ margin: 0 }}>
          <div className="panel-header">
            <div>
              <div className="panel-title">
                <CheckCircle2 size={18} color="var(--accent-emerald)" />
                Research Execution Provenance
              </div>
              <div className="panel-subtitle">Completed stages with verified reproducible artifacts</div>
            </div>
          </div>

          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {overview.stages_completed.map((stage) => (
              <div
                key={stage}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '12px',
                  padding: '10px 14px',
                  background: 'var(--bg-inset)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: 'var(--radius-md)',
                  fontSize: '13px',
                }}
              >
                <CheckCircle2 size={16} color="var(--accent-emerald)" />
                <span style={{ fontWeight: 500, color: 'var(--text-main)' }}>{stage}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
