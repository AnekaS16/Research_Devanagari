import React, { useState, useMemo } from 'react';
import {
  Search,
  Download,
  Eye,
  RotateCcw,
  X,
  Layers,
  FileText,
} from 'lucide-react';
import type { PairLevelMaster } from '../types';
import { exportToCSV } from '../data/dataLoader';

interface Props {
  data: PairLevelMaster[];
}

export interface UniqueSourcePair {
  pair_id: string;
  track: string;
  target_original: string;
  target_contrast: string;
  original_sentence: string;
  contrast_sentence: string;
  final_status: string;
  murilEval?: PairLevelMaster;
  xlmrEval?: PairLevelMaster;
}

export const DatasetExplorer: React.FC<Props> = ({ data }) => {
  const [viewMode, setViewMode] = useState<'unique_pairs' | 'model_evaluations'>('unique_pairs');
  const [search, setSearch] = useState('');
  const [selectedTrack, setSelectedTrack] = useState('ALL');
  const [selectedModel, setSelectedModel] = useState('ALL');
  const [selectedStatus, setSelectedStatus] = useState('ALL');
  const [selectedPair, setSelectedPair] = useState<PairLevelMaster | null>(null);
  const [selectedUniquePair, setSelectedUniquePair] = useState<UniqueSourcePair | null>(null);
  const [page, setPage] = useState(1);
  const pageSize = 15;

  // Deduplicate data to extract the 200 canonical source sentence pairs
  const uniquePairs = useMemo<UniqueSourcePair[]>(() => {
    const pairMap = new Map<string, UniqueSourcePair>();
    for (const row of data) {
      if (!pairMap.has(row.pair_id)) {
        pairMap.set(row.pair_id, {
          pair_id: row.pair_id,
          track: row.track,
          target_original: row.target_original,
          target_contrast: row.target_contrast,
          original_sentence: row.original_sentence,
          contrast_sentence: row.contrast_sentence,
          final_status: row.final_status,
        });
      }
      const pair = pairMap.get(row.pair_id)!;
      if (row.model_name.includes('muril')) {
        pair.murilEval = row;
      } else if (row.model_name.includes('xlm')) {
        pair.xlmrEval = row;
      }
    }
    return Array.from(pairMap.values());
  }, [data]);

  // Filter 200 Unique Pairs
  const filteredUniquePairs = useMemo(() => {
    return uniquePairs.filter((pair) => {
      const matchSearch =
        search === '' ||
        pair.pair_id.toLowerCase().includes(search.toLowerCase()) ||
        pair.original_sentence.toLowerCase().includes(search.toLowerCase()) ||
        pair.contrast_sentence.toLowerCase().includes(search.toLowerCase()) ||
        pair.target_original.toLowerCase().includes(search.toLowerCase()) ||
        pair.target_contrast.toLowerCase().includes(search.toLowerCase());

      const matchTrack = selectedTrack === 'ALL' || pair.track === selectedTrack;
      const matchStatus = selectedStatus === 'ALL' || pair.final_status === selectedStatus;

      return matchSearch && matchTrack && matchStatus;
    });
  }, [uniquePairs, search, selectedTrack, selectedStatus]);

  // Filter 400 Model Evaluations
  const filteredModelEvaluations = useMemo(() => {
    return data.filter((row) => {
      const matchSearch =
        search === '' ||
        row.pair_id.toLowerCase().includes(search.toLowerCase()) ||
        row.original_sentence.toLowerCase().includes(search.toLowerCase()) ||
        row.contrast_sentence.toLowerCase().includes(search.toLowerCase()) ||
        row.target_original.toLowerCase().includes(search.toLowerCase()) ||
        row.target_contrast.toLowerCase().includes(search.toLowerCase());

      const matchTrack = selectedTrack === 'ALL' || row.track === selectedTrack;
      const matchModel = selectedModel === 'ALL' || row.model_name === selectedModel;
      const matchStatus = selectedStatus === 'ALL' || row.final_status === selectedStatus;

      return matchSearch && matchTrack && matchModel && matchStatus;
    });
  }, [data, search, selectedTrack, selectedModel, selectedStatus]);

  const activeCount = viewMode === 'unique_pairs' ? filteredUniquePairs.length : filteredModelEvaluations.length;
  const totalPages = Math.ceil(activeCount / pageSize) || 1;

  const paginatedUniquePairs = useMemo(() => {
    const start = (page - 1) * pageSize;
    return filteredUniquePairs.slice(start, start + pageSize);
  }, [filteredUniquePairs, page]);

  const paginatedModelEvaluations = useMemo(() => {
    const start = (page - 1) * pageSize;
    return filteredModelEvaluations.slice(start, start + pageSize);
  }, [filteredModelEvaluations, page]);

  const handleReset = () => {
    setSearch('');
    setSelectedTrack('ALL');
    setSelectedModel('ALL');
    setSelectedStatus('ALL');
    setPage(1);
  };

  const handleExport = () => {
    const modelTag =
      viewMode === 'unique_pairs'
        ? 'source_pairs'
        : selectedModel === 'ALL'
        ? 'all_models'
        : selectedModel.includes('muril')
        ? 'muril'
        : 'xlmr';
    const trackTag = selectedTrack.toLowerCase();
    const statusTag = selectedStatus.toLowerCase();
    const filename = `hindi_pairs_${viewMode}_${trackTag}_${modelTag}_${statusTag}.csv`;

    if (viewMode === 'unique_pairs') {
      const exportRows = filteredUniquePairs.map((p) => ({
        pair_id: p.pair_id,
        track: p.track,
        final_status: p.final_status,
        target_original: p.target_original,
        target_contrast: p.target_contrast,
        original_sentence: p.original_sentence,
        contrast_sentence: p.contrast_sentence,
        muril_delta_tokens: p.murilEval?.delta_tokens ?? '',
        muril_delta_pll: p.murilEval?.delta_pll ?? '',
        xlmr_delta_tokens: p.xlmrEval?.delta_tokens ?? '',
        xlmr_delta_pll: p.xlmrEval?.delta_pll ?? '',
      }));
      exportToCSV(exportRows, filename);
    } else {
      exportToCSV(filteredModelEvaluations, filename);
    }
  };

  const highlightSentence = (sentence: string, target: string, isOrig: boolean) => {
    if (!target || !sentence.includes(target)) {
      return <span className="devanagari-sentence">{sentence}</span>;
    }
    const parts = sentence.split(target);
    return (
      <span className="devanagari-sentence">
        {parts.map((part, i) => (
          <React.Fragment key={i}>
            {part}
            {i < parts.length - 1 && (
              <span className={isOrig ? 'target-highlight-orig' : 'target-highlight-cont'}>
                {target}
              </span>
            )}
          </React.Fragment>
        ))}
      </span>
    );
  };

  return (
    <div>
      {/* View Mode Toggle Banner */}
      <div
        style={{
          background: 'var(--bg-card)',
          border: '1px solid var(--border-subtle)',
          borderRadius: 'var(--radius-md)',
          padding: '16px 20px',
          marginBottom: '16px',
          display: 'flex',
          flexWrap: 'wrap',
          alignItems: 'center',
          justifyContent: 'space-between',
          gap: '12px',
        }}
      >
        <div>
          <div style={{ fontSize: '14px', fontWeight: 600, color: '#f8fafc', marginBottom: '2px' }}>
            Dataset Inspection Mode
          </div>
          <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
            Distinguish between unique source sentence pairs (ground-truth Devanagari stimuli) and model-specific evaluations (MuRIL &amp; XLM-R outputs).
          </div>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            className={`btn ${viewMode === 'unique_pairs' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => {
              setViewMode('unique_pairs');
              setPage(1);
            }}
            style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600 }}
          >
            <FileText size={15} />
            Unique Source Pairs (200)
          </button>
          <button
            className={`btn ${viewMode === 'model_evaluations' ? 'btn-primary' : 'btn-secondary'}`}
            onClick={() => {
              setViewMode('model_evaluations');
              setPage(1);
            }}
            style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600 }}
          >
            <Layers size={15} />
            Model Evaluations (400)
          </button>
        </div>
      </div>

      {/* Control Panel */}
      <div className="filter-bar" style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', alignItems: 'center' }}>
        {/* Search */}
        <div style={{ position: 'relative', minWidth: '240px', flex: 1 }}>
          <Search
            size={16}
            style={{ position: 'absolute', left: '10px', top: '50%', transform: 'translateY(-50%)', color: 'var(--text-muted)' }}
          />
          <input
            type="text"
            className="text-input"
            style={{ width: '100%', paddingLeft: '32px' }}
            placeholder="Search pair ID, word, or Hindi text..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
          />
        </div>

        {/* Track Filter */}
        <div className="filter-group">
          <span className="filter-label">Track:</span>
          <select
            className="select-input"
            value={selectedTrack}
            onChange={(e) => {
              setSelectedTrack(e.target.value);
              setPage(1);
            }}
          >
            <option value="ALL">All Tracks</option>
            <option value="VALID_WORD_CONTEXT">VALID_WORD_CONTEXT ({viewMode === 'unique_pairs' ? '142' : '284'})</option>
            <option value="SYNONYM_SUBSTITUTION">SYNONYM_SUBSTITUTION ({viewMode === 'unique_pairs' ? '38' : '76'})</option>
            <option value="REALISTIC_SPELLING_ERROR">REALISTIC_SPELLING_ERROR ({viewMode === 'unique_pairs' ? '20' : '40'})</option>
          </select>
        </div>

        {/* Model Filter (Active in model evaluations mode) */}
        {viewMode === 'model_evaluations' && (
          <div className="filter-group">
            <span className="filter-label">Model:</span>
            <select
              className="select-input"
              value={selectedModel}
              onChange={(e) => {
                setSelectedModel(e.target.value);
                setPage(1);
              }}
            >
              <option value="ALL">All Models</option>
              <option value="google/muril-base-cased">MuRIL</option>
              <option value="FacebookAI/xlm-roberta-base">XLM-RoBERTa</option>
            </select>
          </div>
        )}

        {/* Status Filter */}
        <div className="filter-group">
          <span className="filter-label">Status:</span>
          <select
            className="select-input"
            value={selectedStatus}
            onChange={(e) => {
              setSelectedStatus(e.target.value);
              setPage(1);
            }}
          >
            <option value="ALL">All Statuses</option>
            <option value="accept">Accept ({viewMode === 'unique_pairs' ? '199' : '398'})</option>
            <option value="reject">Reject ({viewMode === 'unique_pairs' ? '1' : '2'})</option>
          </select>
        </div>

        {/* Action Buttons */}
        <div style={{ display: 'flex', gap: '8px', marginLeft: 'auto' }}>
          <button className="btn btn-secondary" onClick={handleReset} title="Reset filters">
            <RotateCcw size={14} /> Reset
          </button>
          <button className="btn btn-primary" onClick={handleExport} title="Export filtered records">
            <Download size={14} /> Export CSV ({activeCount})
          </button>
        </div>
      </div>

      {/* Results Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px', fontSize: '13px', color: 'var(--text-secondary)' }}>
        <span>
          Showing <strong>{activeCount}</strong> matching {viewMode === 'unique_pairs' ? 'source pairs' : 'model evaluations'}
        </span>
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <span>Page {page} of {totalPages}</span>
          <button
            className="btn btn-secondary"
            style={{ padding: '3px 8px' }}
            disabled={page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            Prev
          </button>
          <button
            className="btn btn-secondary"
            style={{ padding: '3px 8px' }}
            disabled={page >= totalPages}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
          >
            Next
          </button>
        </div>
      </div>

      {/* Table: View Mode 1 - Unique Source Pairs (200) */}
      {viewMode === 'unique_pairs' && (
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Pair ID</th>
                <th>Track</th>
                <th>Target (Orig / Cont)</th>
                <th>Original Sentence</th>
                <th>Contrast Sentence</th>
                <th>Status</th>
                <th>MuRIL Δ PLL</th>
                <th>XLM-R Δ PLL</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {paginatedUniquePairs.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
                    No source sentence pairs match the current search or filters.
                  </td>
                </tr>
              ) : (
                paginatedUniquePairs.map((pair) => (
                  <tr key={pair.pair_id}>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#38bdf8' }}>
                      {pair.pair_id}
                    </td>
                    <td>
                      <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                        {pair.track.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td className="devanagari-text" style={{ fontWeight: 600 }}>
                      <span style={{ color: '#34d399' }}>{pair.target_original}</span>
                      {' → '}
                      <span style={{ color: '#fda4af' }}>{pair.target_contrast}</span>
                    </td>
                    <td style={{ maxWidth: '280px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {highlightSentence(pair.original_sentence, pair.target_original, true)}
                    </td>
                    <td style={{ maxWidth: '280px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {highlightSentence(pair.contrast_sentence, pair.target_contrast, false)}
                    </td>
                    <td>
                      <span className={`badge ${pair.final_status === 'accept' ? 'badge-success' : 'badge-rose'}`}>
                        {pair.final_status}
                      </span>
                    </td>
                    <td
                      style={{
                        fontFamily: 'var(--font-mono)',
                        color:
                          pair.murilEval?.delta_pll && Number(pair.murilEval.delta_pll) < 0
                            ? '#34d399'
                            : '#fb7185',
                      }}
                    >
                      {pair.murilEval?.delta_pll ? Number(pair.murilEval.delta_pll).toFixed(2) : '—'}
                    </td>
                    <td
                      style={{
                        fontFamily: 'var(--font-mono)',
                        color:
                          pair.xlmrEval?.delta_pll && Number(pair.xlmrEval.delta_pll) < 0
                            ? '#34d399'
                            : '#fb7185',
                      }}
                    >
                      {pair.xlmrEval?.delta_pll ? Number(pair.xlmrEval.delta_pll).toFixed(2) : '—'}
                    </td>
                    <td>
                      <button
                        className="btn btn-secondary"
                        style={{ padding: '4px 8px', fontSize: '11.5px' }}
                        onClick={() => setSelectedUniquePair(pair)}
                      >
                        <Eye size={12} /> Inspect
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Table: View Mode 2 - Model Evaluations (400) */}
      {viewMode === 'model_evaluations' && (
        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th>Pair ID</th>
                <th>Model</th>
                <th>Track</th>
                <th>Target (Orig / Cont)</th>
                <th>Original Sentence</th>
                <th>Contrast Sentence</th>
                <th>Δ Tokens</th>
                <th>Δ PLL</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {paginatedModelEvaluations.length === 0 ? (
                <tr>
                  <td colSpan={9} style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
                    No model evaluation records match the current search or filters.
                  </td>
                </tr>
              ) : (
                paginatedModelEvaluations.map((row) => (
                  <tr key={`${row.pair_id}_${row.model_name}`}>
                    <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600, color: '#38bdf8' }}>
                      {row.pair_id}
                    </td>
                    <td>
                      <span className={row.model_name.includes('muril') ? 'badge badge-muril' : 'badge badge-xlmr'}>
                        {row.model_name.includes('muril') ? 'MuRIL' : 'XLM-R'}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>
                        {row.track.replace(/_/g, ' ')}
                      </span>
                    </td>
                    <td className="devanagari-text" style={{ fontWeight: 600 }}>
                      <span style={{ color: '#34d399' }}>{row.target_original}</span>
                      {' → '}
                      <span style={{ color: '#fda4af' }}>{row.target_contrast}</span>
                    </td>
                    <td style={{ maxWidth: '280px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {highlightSentence(row.original_sentence, row.target_original, true)}
                    </td>
                    <td style={{ maxWidth: '280px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {highlightSentence(row.contrast_sentence, row.target_contrast, false)}
                    </td>
                    <td style={{ fontFamily: 'var(--font-mono)' }}>
                      {row.delta_tokens ? (Number(row.delta_tokens) > 0 ? `+${row.delta_tokens}` : row.delta_tokens) : '—'}
                    </td>
                    <td
                      style={{
                        fontFamily: 'var(--font-mono)',
                        color: row.delta_pll && Number(row.delta_pll) < 0 ? '#34d399' : '#fb7185',
                      }}
                    >
                      {row.delta_pll ? Number(row.delta_pll).toFixed(2) : '—'}
                    </td>
                    <td>
                      <button
                        className="btn btn-secondary"
                        style={{ padding: '4px 8px', fontSize: '11.5px' }}
                        onClick={() => setSelectedPair(row)}
                      >
                        <Eye size={12} /> Inspect
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* Inspect Modal: Unique Source Pair (Shows both models side-by-side) */}
      {selectedUniquePair && (
        <div className="modal-overlay" onClick={() => setSelectedUniquePair(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div>
                <span className="badge badge-muril" style={{ marginRight: '8px' }}>
                  {selectedUniquePair.pair_id}
                </span>
                <span className="badge badge-success" style={{ marginRight: '8px' }}>
                  {selectedUniquePair.track}
                </span>
                <span className={`badge ${selectedUniquePair.final_status === 'accept' ? 'badge-success' : 'badge-rose'}`}>
                  Status: {selectedUniquePair.final_status}
                </span>
                <h3 style={{ fontSize: '18px', fontWeight: 700, marginTop: '8px' }}>
                  Unique Source Pair Inspection
                </h3>
              </div>
              <button
                className="btn btn-secondary"
                style={{ padding: '6px' }}
                onClick={() => setSelectedUniquePair(null)}
              >
                <X size={16} />
              </button>
            </div>

            {/* Devanagari Sentences */}
            <div className="sentence-comparison-box">
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px', fontWeight: 600 }}>
                Original Sentence (Standard Hindi)
              </div>
              <div style={{ marginBottom: '16px' }}>
                {highlightSentence(selectedUniquePair.original_sentence, selectedUniquePair.target_original, true)}
              </div>

              <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px', fontWeight: 600 }}>
                Contrast Sentence (Perturbed / Contextual Contrast)
              </div>
              <div>
                {highlightSentence(selectedUniquePair.contrast_sentence, selectedUniquePair.target_contrast, false)}
              </div>
            </div>

            {/* Model Comparison Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
              {/* MuRIL Evaluation */}
              <div style={{ background: 'var(--bg-inset)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                  <span className="badge badge-muril">MuRIL Evaluation</span>
                </div>
                <div style={{ fontSize: '13px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <div>Original Tokens: <strong>{selectedUniquePair.murilEval?.orig_tokens_count || '—'}</strong></div>
                  <div>Contrast Tokens: <strong>{selectedUniquePair.murilEval?.cont_tokens_count || '—'}</strong></div>
                  <div>Token Delta: <strong>{selectedUniquePair.murilEval?.delta_tokens || '—'}</strong></div>
                  <div>Original PLL: <strong>{selectedUniquePair.murilEval?.orig_pll ? Number(selectedUniquePair.murilEval.orig_pll).toFixed(3) : '—'}</strong></div>
                  <div>Contrast PLL: <strong>{selectedUniquePair.murilEval?.cont_pll ? Number(selectedUniquePair.murilEval.cont_pll).toFixed(3) : '—'}</strong></div>
                  <div>
                    Δ PLL (Cont - Orig):{' '}
                    <strong style={{ color: Number(selectedUniquePair.murilEval?.delta_pll) < 0 ? '#34d399' : '#fb7185' }}>
                      {selectedUniquePair.murilEval?.delta_pll ? Number(selectedUniquePair.murilEval.delta_pll).toFixed(3) : '—'}
                    </strong>
                  </div>
                </div>
              </div>

              {/* XLM-R Evaluation */}
              <div style={{ background: 'var(--bg-inset)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                  <span className="badge badge-xlmr">XLM-RoBERTa Evaluation</span>
                </div>
                <div style={{ fontSize: '13px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
                  <div>Original Tokens: <strong>{selectedUniquePair.xlmrEval?.orig_tokens_count || '—'}</strong></div>
                  <div>Contrast Tokens: <strong>{selectedUniquePair.xlmrEval?.cont_tokens_count || '—'}</strong></div>
                  <div>Token Delta: <strong>{selectedUniquePair.xlmrEval?.delta_tokens || '—'}</strong></div>
                  <div>Original PLL: <strong>{selectedUniquePair.xlmrEval?.orig_pll ? Number(selectedUniquePair.xlmrEval.orig_pll).toFixed(3) : '—'}</strong></div>
                  <div>Contrast PLL: <strong>{selectedUniquePair.xlmrEval?.cont_pll ? Number(selectedUniquePair.xlmrEval.cont_pll).toFixed(3) : '—'}</strong></div>
                  <div>
                    Δ PLL (Cont - Orig):{' '}
                    <strong style={{ color: Number(selectedUniquePair.xlmrEval?.delta_pll) < 0 ? '#34d399' : '#fb7185' }}>
                      {selectedUniquePair.xlmrEval?.delta_pll ? Number(selectedUniquePair.xlmrEval.delta_pll).toFixed(3) : '—'}
                    </strong>
                  </div>
                </div>
              </div>
            </div>

            {/* Explanation Metrics (for spelling errors) */}
            {selectedUniquePair.track === 'REALISTIC_SPELLING_ERROR' && (
              <div style={{ background: 'var(--bg-inset)', padding: '16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                  Explanation Stability &amp; Faithfulness (Stage 6 &amp; 7)
                </div>
                <div style={{ fontSize: '13px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '10px' }}>
                  <div>MuRIL Visible Attn Rho: <strong>{selectedUniquePair.murilEval?.stage7_visible_attn_rho ? Number(selectedUniquePair.murilEval.stage7_visible_attn_rho).toFixed(3) : '—'}</strong></div>
                  <div>XLM-R Visible Attn Rho: <strong>{selectedUniquePair.xlmrEval?.stage7_visible_attn_rho ? Number(selectedUniquePair.xlmrEval.stage7_visible_attn_rho).toFixed(3) : '—'}</strong></div>
                  <div>MuRIL Attn Faith Gap (f=20%): <strong style={{ color: '#34d399' }}>{selectedUniquePair.murilEval?.faithfulness_gap_f20_attn || '—'}</strong></div>
                  <div>XLM-R Attn Faith Gap (f=20%): <strong style={{ color: '#34d399' }}>{selectedUniquePair.xlmrEval?.faithfulness_gap_f20_attn || '—'}</strong></div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Inspect Modal: Single Model Evaluation */}
      {selectedPair && (
        <div className="modal-overlay" onClick={() => setSelectedPair(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            {/* Modal Header */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px' }}>
              <div>
                <span className="badge badge-muril" style={{ marginRight: '8px' }}>
                  {selectedPair.pair_id}
                </span>
                <span className={selectedPair.model_name.includes('muril') ? 'badge badge-muril' : 'badge badge-xlmr'}>
                  {selectedPair.model_name}
                </span>
                <h3 style={{ fontSize: '18px', fontWeight: 700, marginTop: '8px' }}>
                  Sentence Pair Evaluation Inspection
                </h3>
              </div>
              <button
                className="btn btn-secondary"
                style={{ padding: '6px' }}
                onClick={() => setSelectedPair(null)}
              >
                <X size={16} />
              </button>
            </div>

            {/* Sentences */}
            <div className="sentence-comparison-box">
              <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px', fontWeight: 600 }}>
                Original Sentence (Standard Hindi)
              </div>
              <div style={{ marginBottom: '16px' }}>
                {highlightSentence(selectedPair.original_sentence, selectedPair.target_original, true)}
              </div>

              <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase', marginBottom: '6px', fontWeight: 600 }}>
                Contrast Sentence (Perturbed / Contextual Contrast)
              </div>
              <div>
                {highlightSentence(selectedPair.contrast_sentence, selectedPair.target_contrast, false)}
              </div>
            </div>

            {/* Experimental Metrics Grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px', marginBottom: '20px' }}>
              {/* Tokenization */}
              <div style={{ background: 'var(--bg-inset)', padding: '14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                  Tokenization Metrics (Stage 2)
                </div>
                <div style={{ fontSize: '13px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <div>Original Tokens: <strong>{selectedPair.orig_tokens_count || '—'}</strong></div>
                  <div>Contrast Tokens: <strong>{selectedPair.cont_tokens_count || '—'}</strong></div>
                  <div>Token Delta: <strong>{selectedPair.delta_tokens || '—'}</strong></div>
                  <div>Target Subwords: <strong>{selectedPair.orig_target_subwords || '—'} → {selectedPair.cont_target_subwords || '—'}</strong></div>
                </div>
              </div>

              {/* PLL Scoring */}
              <div style={{ background: 'var(--bg-inset)', padding: '14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                  PLL Scoring (Stage 3)
                </div>
                <div style={{ fontSize: '13px', display: 'flex', flexDirection: 'column', gap: '4px' }}>
                  <div>Original PLL: <strong>{selectedPair.orig_pll ? Number(selectedPair.orig_pll).toFixed(3) : '—'}</strong></div>
                  <div>Contrast PLL: <strong>{selectedPair.cont_pll ? Number(selectedPair.cont_pll).toFixed(3) : '—'}</strong></div>
                  <div>Δ PLL (Cont - Orig): <strong style={{ color: Number(selectedPair.delta_pll) < 0 ? '#34d399' : '#fb7185' }}>{selectedPair.delta_pll ? Number(selectedPair.delta_pll).toFixed(3) : '—'}</strong></div>
                  <div>Δ Mean LogP: <strong>{selectedPair.delta_mean_logp ? Number(selectedPair.delta_mean_logp).toFixed(4) : '—'}</strong></div>
                </div>
              </div>
            </div>

            {/* Explanation Stability & Faithfulness (if available) */}
            <div style={{ background: 'var(--bg-inset)', padding: '14px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-secondary)', marginBottom: '8px' }}>
                Explanation Metrics (Stages 6 &amp; 7)
              </div>
              {selectedPair.track === 'REALISTIC_SPELLING_ERROR' ? (
                <div style={{ fontSize: '13px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  <div>Stage 6 Masked Attn Rho: <strong>{selectedPair.stage6_masked_attn_rho || '—'}</strong></div>
                  <div>Stage 6 Masked IG Rho: <strong>{selectedPair.stage6_masked_ig_rho || '—'}</strong></div>
                  <div>Stage 7 Visible Attn Rho: <strong>{selectedPair.stage7_visible_attn_rho || '—'}</strong></div>
                  <div>Stage 7 Visible IG Rho: <strong>{selectedPair.stage7_visible_ig_rho || '—'}</strong></div>
                  <div>Attn Faithfulness Gap (f=20%): <strong style={{ color: '#34d399' }}>{selectedPair.faithfulness_gap_f20_attn || '—'}</strong></div>
                  <div>IG Faithfulness Gap (f=20%): <strong style={{ color: '#818cf8' }}>{selectedPair.faithfulness_gap_f20_ig || '—'}</strong></div>
                </div>
              ) : (
                <div style={{ fontSize: '12.5px', color: 'var(--text-muted)' }}>
                  This sentence pair belongs to <code>{selectedPair.track}</code>. Explanation stability and faithfulness evaluations were conducted on the primary <code>REALISTIC_SPELLING_ERROR</code> track.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
