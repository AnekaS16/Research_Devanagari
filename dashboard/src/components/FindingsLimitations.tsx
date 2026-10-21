import React from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
  Lightbulb,
  ShieldAlert,
} from 'lucide-react';

export const FindingsLimitations: React.FC = () => {
  return (
    <div>
      {/* Hypothesis Outcome Card */}
      <div className="panel" style={{ border: '1px solid rgba(245, 158, 11, 0.4)', background: 'rgba(245, 158, 11, 0.04)' }}>
        <div style={{ display: 'flex', gap: '16px', alignItems: 'flex-start' }}>
          <ShieldAlert size={28} color="#f59e0b" style={{ flexShrink: 0, marginTop: '2px' }} />
          <div>
            <h2 style={{ fontSize: '18px', fontWeight: 700, color: '#fde68a', marginBottom: '8px' }}>
              Core Research Hypothesis Status: Not Statistically Supported
            </h2>
            <p style={{ color: 'var(--text-secondary)', fontSize: '13.5px', lineHeight: '1.6' }}>
              The central hypothesis postulated that explanation stability under spelling perturbations would predict explanation faithfulness.
              Statistical audit of the empirical data across all 20 spelling error pairs demonstrates:
            </p>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px', marginTop: '14px' }}>
              <div style={{ background: 'var(--bg-inset)', padding: '12px 16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <span className="badge badge-xlmr" style={{ marginBottom: '4px' }}>XLM-RoBERTa (Attention)</span>
                <div style={{ fontSize: '13px', marginTop: '4px' }}>
                  Spearman correlation: <strong style={{ fontFamily: 'var(--font-mono)' }}>r_s = +0.2704</strong>
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  p-value = 0.2490 (Pearson r = +0.2557, p = 0.2766; N=20)
                </div>
              </div>

              <div style={{ background: 'var(--bg-inset)', padding: '12px 16px', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <span className="badge badge-muril" style={{ marginBottom: '4px' }}>MuRIL (Attention)</span>
                <div style={{ fontSize: '13px', marginTop: '4px' }}>
                  Spearman correlation: <strong style={{ fontFamily: 'var(--font-mono)' }}>r_s = -0.1611</strong>
                </div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                  p-value = 0.4974 (Pearson r = -0.1939, p = 0.4126; N=20)
                </div>
              </div>
            </div>
            <p style={{ color: 'var(--text-muted)', fontSize: '12px', marginTop: '12px' }}>
              *Scientific Interpretation:* All correlation tests yield p &gt; 0.05 (IG: XLM-R r_s = +0.1583, p = 0.5052; MuRIL r_s = +0.0075, p = 0.9749). The current data do not provide statistically significant evidence of a correlation between explanation stability and faithfulness. This failure to reject the null hypothesis does not prove that no relationship exists, but demonstrates that higher explanation stability under Devanagari spelling perturbations cannot be assumed to indicate faithful explanations.
            </p>
          </div>
        </div>
      </div>

      {/* Categorized Audit Findings Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(460px, 1fr))', gap: '24px' }}>
        {/* Category A: Verified Facts */}
        <div className="panel" style={{ margin: 0 }}>
          <div className="panel-header">
            <div>
              <div className="panel-title">
                <CheckCircle2 size={18} color="var(--accent-emerald)" />
                Category A: Verified Experimental Facts
              </div>
              <div className="panel-subtitle">Empirically audited findings confirmed by the data</div>
            </div>
          </div>

          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13.5px', color: 'var(--text-secondary)' }}>
            <li style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>•</div>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>Subword Fragmentation Collapse:</strong> Spelling errors increase sentence token length by +1.0 token on average in both models, driving single-subword target coverage down from 70–90% to 20–30%.
              </div>
            </li>
            <li style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>•</div>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>PLL Sensitivity to Spelling:</strong> MuRIL preferred original sentences over misspelled ones in <strong>95.0%</strong> of pairs; XLM-RoBERTa preferred originals in <strong>75.0%</strong> of pairs.
              </div>
            </li>
            <li style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>•</div>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>Attention Directional Faithfulness:</strong> In XLM-RoBERTa, top-k masking produced a lower target logit than bottom-k masking in <strong>75.0% to 77.5%</strong> of evaluations (yielding positive faithfulness gaps +4.02 to +5.17), although both interventions increased the target logit relative to baseline. In MuRIL, target logits decreased under masking, with top-attention masking producing larger drops than bottom-attention in <strong>60.0% to 75.0%</strong> of evaluations (gaps +0.42 to +8.82).
              </div>
            </li>
            <li style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: 'var(--accent-emerald)', fontWeight: 700 }}>•</div>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>Integrated Gradients Disconnect:</strong> IG attributions exhibited near-chance win rates (41–50% in MuRIL), indicating that continuous embedding gradients do not predict discrete token masking impact.
              </div>
            </li>
          </ul>
        </div>

        {/* Category B: Inconsistencies & Resolutions */}
        <div className="panel" style={{ margin: 0 }}>
          <div className="panel-header">
            <div>
              <div className="panel-title">
                <HelpCircle size={18} color="var(--accent-cyan)" />
                Category B: Inconsistencies & Mathematical Resolutions
              </div>
              <div className="panel-subtitle">Methodological clarifications documented across stages</div>
            </div>
          </div>

          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13.5px', color: 'var(--text-secondary)' }}>
            <li style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>•</div>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>Stage 6 Stability Artifact:</strong> In Stage 6, masking the target word rendered original and contrast inputs identical, artificially producing ρ = 1.0. Stage 7 resolved this with visible-target scoring.
              </div>
            </li>
            <li style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>•</div>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>Logit Drop Sign Convention:</strong> Defined as ΔS = S_0 - S_int. Negative values in XLM-R denote logit increases under intervention, but Faithfulness Gap (top - bottom) remained positive (+4.02), confirming top tokens had lower scores than control.
              </div>
            </li>
            <li style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>•</div>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>Context Alignment Integrity:</strong> Replaced character-offset matching with region-relative alignment (pre/post target span), ensuring tokens following length-altered target words align accurately.
              </div>
            </li>
          </ul>
        </div>

        {/* Category C: Methodological Limitations */}
        <div className="panel" style={{ margin: 0 }}>
          <div className="panel-header">
            <div>
              <div className="panel-title">
                <AlertTriangle size={18} color="var(--accent-amber)" />
                Category C: Methodological Limitations
              </div>
              <div className="panel-subtitle">Constraints that cannot be resolved from the current dataset</div>
            </div>
          </div>

          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13.5px', color: 'var(--text-secondary)' }}>
            <li style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: 'var(--accent-amber)', fontWeight: 700 }}>•</div>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>Exploratory Spelling Sample Size:</strong> With N=20 pairs in the spelling track (and only 4–6 eligible under strict single-subword rules), results are exploratory case studies, not population-level proofs.
              </div>
            </li>
            <li style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: 'var(--accent-amber)', fontWeight: 700 }}>•</div>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>OOD Masking Shock:</strong> Replacing words with [MASK] in unmasked context creates syntactic anomalies. Logit drops reflect both semantic information removal and grammatical disruption.
              </div>
            </li>
            <li style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: 'var(--accent-amber)', fontWeight: 700 }}>•</div>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>Attention is Non-Causal:</strong> Attention weights reflect query-key routing inside self-attention layers, not verified causal importance.
              </div>
            </li>
          </ul>
        </div>

        {/* Category D: Future Recommendations */}
        <div className="panel" style={{ margin: 0 }}>
          <div className="panel-header">
            <div>
              <div className="panel-title">
                <Lightbulb size={18} color="var(--accent-indigo)" />
                Category D: Recommendations for Future Research
              </div>
              <div className="panel-subtitle">Methodological extensions for subsequent Hindi NLP benchmarks</div>
            </div>
          </div>

          <ul style={{ listStyle: 'none', display: 'flex', flexDirection: 'column', gap: '14px', fontSize: '13.5px', color: 'var(--text-secondary)' }}>
            <li style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: 'var(--accent-indigo)', fontWeight: 700 }}>•</div>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>Continuous Embedding Ablations:</strong> Use soft continuous perturbations (zeroing embeddings or adding Gaussian noise) to test sensitivity without syntactic OOD shock.
              </div>
            </li>
            <li style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: 'var(--accent-indigo)', fontWeight: 700 }}>•</div>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>Expanded Hindi Dataset:</strong> Scale the Devanagari spelling track to 100+ pairs spanning matra errors, consonant cluster assimilation, and nukta variations.
              </div>
            </li>
            <li style={{ display: 'flex', gap: '10px' }}>
              <div style={{ color: 'var(--accent-indigo)', fontWeight: 700 }}>•</div>
              <div>
                <strong style={{ color: 'var(--text-main)' }}>Gradient-Weighted Attention:</strong> Evaluate Attention × Gradient (attention rollout) to bridge the gap between attention routing and gradient attribution.
              </div>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};
