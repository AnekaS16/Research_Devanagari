# Stage 8: Final Scientific Audit & Research Dashboard Data Layer

## 1. Executive Audit Summary
- **Source Dataset**: `hindi_lexical_sensitivity_cleaned.xlsx`
- **Source SHA-256 Checksum**: `f9a5af81e21624de47ead7feabba412149939c1eb9934f8fc2b8d452a232cd44` (Verified Unmodified)
- **Total Dataset Pairs**: 200 pairs across 3 tracks (`VALID_WORD_CONTEXT`: 142, `SYNONYM_SUBSTITUTION`: 38, `REALISTIC_SPELLING_ERROR`: 20)
- **Pretrained Models**: `google/muril-base-cased` and `FacebookAI/xlm-roberta-base`
- **Full Pipeline Scope**: Stages 1 through 7 audited; dashboard data layer prepared in `outputs/dashboard_data/`.

---

## 2. Category A: Verified Facts Across Stages

### A1. Dataset and Evaluation Integrity
1. **Zero Modifications**: The source Excel file SHA-256 remains identically `f9a5af81e21624de47ead7feabba412149939c1eb9934f8fc2b8d452a232cd44`.
2. **All 200 Pairs Preserved**: All 200 sentence pairs were tokenized (Stage 2) and scored with pseudo-log-likelihood (Stage 3) without dropping or altering records.
3. **Zero Duplicate Keys**: Master records, stability evaluations, and faithfulness interventions contain zero duplicate evaluation keys.
4. **Exact Mathematical Reproduction**: All 24 summary rows in `faithfulness_summary.csv` were independently recomputed from the 1,440 raw intervention records and matched to 4 decimal places.

### A2. Stage 7 Faithfulness Findings
1. **Attention Shows Consistent Directional Faithfulness**:
   - In `FacebookAI/xlm-roberta-base`, top-k masking produced a lower target logit than bottom-k masking in **75.0% to 77.5%** of evaluations (yielding positive faithfulness gaps +4.02 to +5.17), although both interventions increased the target logit relative to baseline.
   - In `google/muril-base-cased`, target logits decreased under masking, with top-attention masking producing larger drops than bottom-attention in **60.0% to 75.0%** of evaluations (gaps +0.42 to +8.82).
2. **Integrated Gradients Shows Mixed / Weaker Context Faithfulness**:
   - In `FacebookAI/xlm-roberta-base`, IG win rate vs bottom tokens was **50.0% to 60.0%** (gaps +1.84 to +3.30).
   - In `google/muril-base-cased`, IG win rate was **41.7% to 50.0%** in the strict single-subword subset and ~45% in all-scorable, indicating that gradient-based saliency on embeddings does not reliably predict the effect of discrete `[MASK]` token substitutions in Hindi MLM heads.

---

## 3. Category B: Inconsistencies, Clarifications & Mathematical Resolutions

### B1. Logit Drop Sign Convention Clarification
- **Mathematical Definition**: $\text{logit\_drop} = S_0 - S_{\text{int}}$, where $S_0$ is baseline target logit and $S_{\text{int}}$ is intervened target logit.
- **Observation**: In XLM-RoBERTa, unnormalized target logits frequently *increased* after context masking (e.g. baseline 5.0 $\to$ intervened 12.0), producing negative logit drop values (-7.19 for top-k, -11.21 for bottom-k).
- **Resolution**: Because bottom-k masking produced an even larger logit increase (+11.21) than top-k masking (+7.19), the **Faithfulness Gap** remained positive:
  $$\text{Gap} = \Delta S_{\text{top}} - \Delta S_{\text{bottom}} = (-7.19) - (-11.21) = +4.02 > 0$$
  This confirms that removing top-importance tokens yielded a *lower* target score than removing bottom-importance tokens ($S_{\text{top}} < S_{\text{bottom}}$). The sign convention is mathematically consistent but requires explicit documentation for dashboard users.

### B2. Stage 6 vs. Stage 7 Stability Resolution
- **Stage 6 Artifact**: When the target token was masked, $S_{\text{orig}}$ and $S_{\text{cont}}$ had identical inputs, producing trivial stability ($\rho = 1.0, \text{Var} = 0$).
- **Stage 7 Separation**: Stage 7 evaluated **Visible-Target Stability** across aligned context tokens. When target words were visible, spelling variations produced genuine attribution divergence (Attention $\rho$ ranged from -0.80 to +1.00; IG $\rho$ ranged from -0.70 to +1.00).
- **Correlation with Faithfulness**: Correlating visible-target rank stability ($\rho$) with mean faithfulness gap ($f=0.20$) across all $N=20$ Devanagari spelling error pairs yields no statistically significant relationship:
  - XLM-RoBERTa (Attention): Spearman $r_s = +0.2704, p = 0.2490 (Pearson $r = +0.2557, p = 0.2766$, $N=20$).
  - MuRIL (Attention): Spearman $r_s = -0.1611, p = 0.4974 (Pearson $r = -0.1939, p = 0.4126$, $N=20$).
  - XLM-RoBERTa (Integrated Gradients): Spearman $r_s = +0.1583, p = 0.5052 ($N=20$).
  - MuRIL (Integrated Gradients): Spearman $r_s = +0.0075, p = 0.9749 ($N=20$).
  Because $p > 0.05$ across all configurations, the current empirical data do not provide statistically significant evidence of a correlation between explanation stability and faithfulness. This does not prove that no relationship exists, but confirms that higher explanation stability under spelling perturbations cannot be assumed to indicate faithful explanations.

---

## 4. Category C: Methodological Limitations of the Current Dataset
1. **Spelling Track Sample Size**: The `REALISTIC_SPELLING_ERROR` track has 20 curated pairs. Under the strict single-subword rule, only 6 pairs in MuRIL and 4 in XLM-R are eligible. While the multi-subword mean-logit extended evaluation covers all 20 pairs, all inferences remain exploratory.
2. **Out-of-Distribution (OOD) Masking Interventions**: Replacing words with `[MASK]` in unmasked sequences creates unnatural syntactic structures. Drops in MLM target logits reflect a combination of semantic information removal and out-of-distribution syntactic degradation.
3. **Attention as Non-Causal**: Attention weights reflect query-key routing, not certified counterfactual importance. Higher win rates for attention do not prove it is a causally faithful metric.

---

## 5. Category D: Recommendations for Future Research
1. **Embedding-Space Perturbations**: Future faithfulness experiments should evaluate soft continuous perturbations (zeroing embeddings or Gaussian noise) in addition to discrete `[MASK]` replacements to avoid OOD language model degradation.
2. **Expanded Spelling Contrast Track**: Expanding the Hindi spelling track to 100+ pairs with diverse morphological categories will increase statistical power for subgroup correlation analyses.
3. **Cross-Layer Attention Attribution**: Evaluating attention rollout or gradient-weighted attention (e.g. Attention $\times$ Gradient) would bridge the gap between attention routing and gradient attribution.
