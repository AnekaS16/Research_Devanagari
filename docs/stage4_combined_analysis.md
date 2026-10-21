# Stage 4: Combined Statistical Analysis — Subword Tokenization vs. Pseudo-Log-Likelihood (PLL)

## 1. Overview & Research Objective

Stage 4 combines the subword tokenization metrics (Stage 2) and pseudo-log-likelihood (PLL) model scoring outputs (Stage 3) to investigate how tokenization variations interact with MLM score sensitivity across 200 Hindi sentence pairs.

### Evaluated Models:
- `FacebookAI/xlm-roberta-base` (SentencePiece unigram, $V = 250{,}002$)
- `google/muril-base-cased` (WordPiece, $V = 197{,}285$)

### Experimental Tracks:
1. `VALID_WORD_CONTEXT`: Contextual semantic shifts ($n = 142$ pairs).
2. `SYNONYM_SUBSTITUTION`: Meaning-preserving synonyms ($n = 38$ pairs).
3. `REALISTIC_SPELLING_ERROR`: Character/matra orthographic perturbations ($n = 20$ pairs).

---

## 2. Statistical Methodology & Guardrails

1. **Sign Convention**:
   $$\Delta = \text{Contrast} - \text{Original}$$
   A negative $\Delta \text{MeanLogP}$ indicates that the model assigns lower pseudo-log-likelihood to the contrast sentence than to the original sentence.
2. **Length Normalization (Primary Outcome)**:
   - Primary metric: $\Delta \text{MeanLogP} = \frac{\text{PLL}_{\text{contrast}}}{|I_{\text{contrast}}|} - \frac{\text{PLL}_{\text{original}}}{|I_{\text{original}}|}$
   - Secondary metric: $\Delta \text{PLL Sum}$ (reported with explicit warnings regarding sentence-length confounding).
3. **Cross-Model Comparability Guardrail**:
   - Raw PLL totals are **never** compared across models due to differences in token segmentation and vocabulary size.
   - Cross-model evaluations use **within-model signed differences** ($\Delta \text{MeanLogP}$), Spearman rank correlation, and directional sign concordance.
4. **Multiple Testing Correction**:
   - Benjamini–Hochberg False Discovery Rate (FDR) control at $\alpha = 0.05$ across all correlation hypothesis tests.
5. **Data Preservation & Sensitivity Analysis**:
   - All 200 pairs (including 2 records with `final_status=reject` and human annotator disagreements) are preserved.
   - Sensitivity analysis running both with and without rejected items demonstrates negligible impact on conclusions.

---

## 3. Summary of Key Findings

### A. Descriptive Statistics ($\Delta \text{MeanLogP}$)

| Model | Track | N | Mean $\Delta$ MeanLogP | Median | SD | % Negative (Contrast Lower) |
|---|---|---|---|---|---|---|
| `xlm-roberta-base` | `VALID_WORD_CONTEXT` | 142 | -1.0933 | -0.8752 | 1.3572 | 81.0% (115/142) |
| `xlm-roberta-base` | `SYNONYM_SUBSTITUTION` | 38 | -0.0655 | -0.1035 | 0.8896 | 57.9% (22/38) |
| `xlm-roberta-base` | `REALISTIC_SPELLING_ERROR` | 20 | -0.6695 | -0.5903 | 1.5495 | 65.0% (13/20) |
| `muril-base-cased` | `VALID_WORD_CONTEXT` | 142 | -1.9141 | -1.6218 | 1.5638 | 93.0% (132/142) |
| `muril-base-cased` | `SYNONYM_SUBSTITUTION` | 38 | -0.5810 | -0.4607 | 0.7358 | 84.2% (32/38) |
| `muril-base-cased` | `REALISTIC_SPELLING_ERROR` | 20 | -2.0368 | -2.2748 | 1.4447 | 95.0% (19/20) |

- **Synonym vs. Error Contrast**: Both models exhibit markedly smaller $\Delta \text{MeanLogP}$ penalties for `SYNONYM_SUBSTITUTION` (means of $-0.0655$ and $-0.5810$) compared to context shifts and spelling errors, indicating preservation of plausibility under meaning-preserving substitutions.

### B. Correlation: Subword Fragmentation vs. Plausibility

- **XLM-RoBERTa (All Tracks)**: Positive correlation between $\Delta \text{TokenCount}$ and $\Delta \text{MeanLogP}$ ($r_s = 0.3481, p < 10^{-5}$, BH-significant).
- **MuRIL (All Tracks)**: Negative correlation between $\Delta \text{TokenCount}$ and $\Delta \text{MeanLogP}$ ($r_s = -0.1941, p = 0.0059$, BH-significant).
- **Scientific Caveat**: Statistically significant correlations reflect empirical associations across tokenization and masking regimes, **not direct causation**. Fragmentation and score penalization are often co-effects of out-of-vocabulary or non-standard surface forms.

### C. Cross-Model Concordance

- **Directional Agreement**: The models agree on the direction of score change ($\Delta \text{MeanLogP}$ sign) in **74.5%** of all pairs ($141$ pairs where both decrease, $8$ where both increase). Agreement reaches **81.0%** in `VALID_WORD_CONTEXT`.
- **Rank Correlation**: Spearman rank correlation between XLM-RoBERTa and MuRIL $\Delta \text{MeanLogP}$ is $r_s = 0.4408$ ($p < 10^{-6}, 95\% \text{ CI } [0.3217, 0.5461]$), confirming consistent relative sensitivity across the model architectures.

---

## 4. Generated Artifacts & Visualizations

All artifacts are persisted in `outputs/analysis/`:
- `combined_pair_level_results.csv`: Complete 400-row dataset (200 pairs $\times$ 2 models) with merged tokenization, target alignment, and PLL metrics.
- `descriptive_statistics.csv`: Summary distributions by model, track, and inclusion criteria.
- `correlation_results.csv`: Spearman correlations, raw p-values, 95% CIs, and Benjamini–Hochberg rejection decisions.
- `model_comparison.csv`: Direction agreement percentages, pair counts, and cross-model rank correlations.
- `analysis_report.md`: Standalone comprehensive report with both beginner-friendly and research-level syntheses.
- `run_metadata.json`: Hashes, software dependencies, execution timestamps, and zero-warning validation log.
- `figures/`:
  1. `fig1_delta_meanlp_by_track.png`: Violin/box plots of score changes by track.
  2. `fig2_token_delta_vs_meanlp.png`: Sentence-level token delta vs. $\Delta \text{MeanLogP}$.
  3. `fig3_target_token_delta_vs_meanlp.png`: Target-token delta vs. $\Delta \text{MeanLogP}$.
  4. `fig4_model_comparison_scatter.png`: Scatter plot comparing model sensitivity across pairs.
  5. `fig5_direction_agreement.png`: Concordance matrix across experimental tracks.
