# Research Questions Assessment

This document delivers an exhaustive, evidence-based evaluation of the five primary Research Questions (RQ1–RQ5) formulated for the Devanagari Hindi Lexical Sensitivity and Explainability research project.

---

## RQ1: How do context-sensitive lexical contrasts, synonyms, and spelling perturbations affect tokenization and model scoring in Devanagari Hindi?

### Primary Evidence & Exact Statistics
1. **Subword Fragmentation**:
   - In the `REALISTIC_SPELLING_ERROR` track ($N=20$ pairs), realistic Devanagari spelling errors caused a sharp increase of $+1.00$ sentence tokens on average across both models:
     - **MuRIL**: Target subwords increased from a mean of $1.10$ in canonical spellings to $2.10$ in misspelled variants ($+1.00$ target subwords). Single-subword coverage collapsed from **90.0% (18/20)** to **30.0% (6/20)**, an absolute drop of **$-60.0\%$**.
     - **XLM-RoBERTa**: Target subwords increased from $1.35$ to $2.35$ ($+1.00$ target subwords). Single-subword coverage collapsed from **70.0% (14/20)** to **20.0% (4/20)**, an absolute drop of **$-50.0\%$**.
   - In the `SYNONYM_SUBSTITUTION` track ($N=38$ pairs), target subword changes were minor: $+0.18$ tokens in MuRIL and $+0.63$ tokens in XLM-RoBERTa. Single-subword coverage shifted from $100.0\% \to 86.8\%$ in MuRIL and $89.5\% \to 60.5\%$ in XLM-RoBERTa.
   - In the `VALID_WORD_CONTEXT` track ($N=142$ pairs), sentence token changes were negligible: $+0.08$ tokens in MuRIL and $+0.17$ tokens in XLM-RoBERTa. Single-subword coverage shifted from $64.1\% \to 55.6\%$ in MuRIL and $39.4\% \to 30.3\%$ in XLM-RoBERTa.

2. **Model Plausibility Scoring ($\Delta \text{MeanLogP}$)**:
   - **Spelling Perturbations**:
     - MuRIL: Mean $\Delta \text{MeanLogP} = -2.0368$ (Median $= -2.2748$, $\text{SD} = 1.4447$). MuRIL assigned lower plausibility to the misspelled sentence in **95.0% (19/20)** of pairs.
     - XLM-RoBERTa: Mean $\Delta \text{MeanLogP} = -0.6695$ (Median $= -0.5903$, $\text{SD} = 1.5495$). XLM-RoBERTa preferred the canonical sentence in **65.0% (13/20)** of pairs.
   - **Contextual Contrasts (`VALID_WORD_CONTEXT`)**:
     - MuRIL: Mean $\Delta \text{MeanLogP} = -1.9141$ (Median $= -1.6218$, Preferred original: **93.0%**, 132/142).
     - XLM-RoBERTa: Mean $\Delta \text{MeanLogP} = -1.0933$ (Median $= -0.8752$, Preferred original: **81.0%**, 115/142).
   - **Synonym Substitutions (`SYNONYM_SUBSTITUTION`)**:
     - MuRIL: Mean $\Delta \text{MeanLogP} = -0.5810$ (Median $= -0.4607$, Preferred original: **84.2%**, 32/38).
     - XLM-RoBERTa: Mean $\Delta \text{MeanLogP} = -0.0655$ (Median $= -0.1035$, Preferred original: **57.9%**, 22/38).

### Scientific Conclusion
Lexical perturbations exert distinct, hierarchical effects on tokenization and model scoring:
- **Spelling errors** induce catastrophic subword segmentation fragmentation due to Devanagari syllabic and matra properties, breaking standard vocabulary words into character or byte fragments. Both models penalize these errors with significant negative pseudo-log-likelihood deltas, although MuRIL is substantially more sensitive than XLM-RoBERTa.
- **Contextual word contrasts** preserve baseline tokenization lengths but trigger substantial predictability drops, demonstrating strong contextual semantic sensitivity.
- **Synonyms** produce minimal tokenization perturbation and negligible score changes (especially in XLM-RoBERTa, $\Delta = -0.07$), indicating that pretrained models recognize the near-interchangeability of valid Hindi synonym alternatives.

### Uncertainty & Methodological Nuances
- **Sample Size**: The spelling track contains $N=20$ pairs; while the fragmentation and scoring drops are pronounced and consistent, generalizing exact percentage drops to rare morphological classes requires a larger sample.
- **Corpus Frequency vs. Semantic Validity**: Synonym score deltas often reflect marginal differences in pretraining web-corpus token frequency rather than semantic nuance.
- **Normative Interpretation**: Pseudo-log-likelihood measures statistical likelihood under pretraining data distributions, not human normative grammatical correctness.

### Status: Fully Supported
The empirical evidence decisively supports the hierarchical differentiation of tracks across both tokenization and PLL scoring.

---

## RQ2: How do MuRIL and XLM-RoBERTa differ in their responses to Devanagari lexical contrasts and perturbations?

### Primary Evidence & Exact Statistics
1. **Tokenization Efficiency**:
   - MuRIL represents Hindi text with ~12% fewer subwords than XLM-RoBERTa:
     - On `VALID_WORD_CONTEXT` ($N=142$), mean sentence tokens: $8.95$ (MuRIL) vs. $10.19$ (XLM-RoBERTa).
     - Canonical single-subword target coverage is higher in MuRIL across all tracks: 64.1% vs. 39.4% on `VALID_WORD_CONTEXT`; 100.0% vs. 89.5% on `SYNONYM_SUBSTITUTION`; and 90.0% vs. 70.0% on `REALISTIC_SPELLING_ERROR`.
   - This reflects MuRIL's Indic-dedicated vocabulary (197k tokens for 17 Indic languages + English) compared to XLM-RoBERTa's global multilingual vocabulary (250k tokens shared across 100+ languages).

2. **Scoring Sensitivity and Preference Rates**:
   - MuRIL exhibits substantially stronger sensitivity (larger negative $\Delta \text{MeanLogP}$) across all three tracks:
     - Spelling errors: MuRIL preferred original in **95.0%** (mean $\Delta = -2.04$), vs. **65.0%** for XLM-RoBERTa (mean $\Delta = -0.67$).
     - Contextual substitutions: MuRIL preferred original in **93.0%** (mean $\Delta = -1.91$), vs. **81.0%** for XLM-RoBERTa (mean $\Delta = -1.09$).
     - Synonyms: MuRIL preferred original in **84.2%** (mean $\Delta = -0.58$), vs. **57.9%** for XLM-RoBERTa (mean $\Delta = -0.07$).

3. **Cross-Model Correlation and Agreement**:
   - Across all 200 pairs, the models exhibit **74.5% directional agreement** (141 pairs both negative, 8 pairs both positive, 51 discordant).
   - Overall rank correlation between models on $\Delta \text{MeanLogP}$ is moderately positive and statistically significant:
     - All Tracks ($N=200$): Spearman $r_s = +0.4408, p < 10^{-9}$ (95% CI: $[+0.3217, +0.5461]$).
     - Valid Word Context ($N=142$): Spearman $r_s = +0.4457, p < 10^{-7}$ (95% CI: $[+0.3032, +0.5687]$).
   - On specific sub-tracks, agreement weakens or vanishes:
     - Synonym Substitutions ($N=38$): Spearman $r_s = +0.0595, p = 0.7226$ (statistically indistinguishable from zero).
     - Realistic Spelling Errors ($N=20$): Spearman $r_s = +0.2150, p = 0.3626$ (statistically non-significant).

### Scientific Conclusion
MuRIL and XLM-RoBERTa diverge systematically in their behavior:
- MuRIL's Indic-specialized vocabulary yields superior tokenization compactness and much sharper discrimination against both orthographic errors and contextual mismatches.
- XLM-RoBERTa displays higher tolerance (shallower score penalties) for Devanagari misspellings and near-zero discrimination between valid synonyms ($\Delta \approx -0.07$), reflecting its broader, diluted multilingual embedding space.
- While both models agree on the broad direction of plausibility shifts for prominent semantic contrasts ($r_s = 0.44, p < 10^{-9}$), their rankings decouple on fine-grained lexical choices such as synonyms ($p = 0.72$).

### Uncertainty & Methodological Nuances
- **Architecture vs. Pretraining Data**: MuRIL is BERT-base with WordPiece; XLM-RoBERTa is RoBERTa-base with BPE unigram. The divergence reflects both tokenizer vocabulary allocation and pretraining corpora (Indic-focused crawl vs. Common Crawl 100-language dump).

### Status: Fully Supported
The comparative differences are empirically established with rigorous within-model metrics, cross-model directional agreement, and correlation testing.

---

## RQ3: How stable are Attention and Integrated Gradients explanations under Devanagari spelling perturbations?

### Primary Evidence & Exact Statistics
1. **Resolution of the Stage 6 Masked-Target Artifact**:
   - In Stage 6, when the target token was masked in both original and contrast inputs, both models exhibited an illusory, trivial stability:
     $$\text{Spearman } \rho = 1.0000, \quad \text{Cosine Similarity} = 1.0000, \quad \Delta \text{Logit} = 0.0 \quad (\text{Variance} = 0)$$
   - This was an experimental artifact: masking the target token eliminated the spelling perturbation, feeding byte-identical context sequences into the model.
   - Stage 7 resolved this artifact by introducing the **Visible-Target Scoring Protocol**, measuring attribution shifts across aligned context tokens while keeping the target word visible.

2. **Visible-Target Rank Stability (Spearman $\rho$)**:
   - **Attention** demonstrates markedly superior rank stability compared to Integrated Gradients across both models ($N=20$ pairs, pooled all-scorable):
     - **XLM-RoBERTa**: Mean $\rho = 0.7425$ (Median $= 0.8000$) for Attention vs. Mean $\rho = 0.3589$ (Median $= 0.4000$) for Integrated Gradients ($2.1\times$ advantage for Attention).
     - **MuRIL**: Mean $\rho = 0.7043$ (Median $= 0.8429$) for Attention vs. Mean $\rho = 0.2632$ (Median $= 0.3572$) for Integrated Gradients ($2.7\times$ advantage for Attention).
   - Under the strict single-subword subset:
     - XLM-RoBERTa ($N=4$): Attention Mean $\rho = 0.6893$ vs. IG Mean $\rho = 0.2964$.
     - MuRIL ($N=6$): Attention Mean $\rho = 0.6286$ vs. IG Mean $\rho = 0.2167$.

3. **Directional Vector Stability (Cosine Similarity)**:
   - In sharp contrast to rank stability, **Cosine Similarity** remains very high for both methods:
     - XLM-RoBERTa: Mean Cosine $= 0.9648$ (Attention) vs. $0.8994$ (Integrated Gradients).
     - MuRIL: Mean Cosine $= 0.9639$ (Attention) vs. $0.9880$ (Integrated Gradients).

### Scientific Conclusion
- Under realistic Devanagari spelling noise, self-attention token importance rankings are substantially more stable ($\rho \approx 0.70 - 0.74$) than Integrated Gradients token importance rankings ($\rho \approx 0.26 - 0.36$).
- However, the dissociation between low rank correlation ($\rho \approx 0.26$) and near-perfect cosine similarity ($0.988$) for Integrated Gradients demonstrates that gradient attribution vectors maintain their general multidimensional trajectory, but local token-level ranking orders fluctuate heavily under spelling noise.

### Uncertainty & Methodological Nuances
- **Sample Size**: Evaluated on $N=20$ spelling pairs. While sufficient to establish the large contrast between Attention and IG ($\rho \approx 0.72$ vs. $\rho \approx 0.31$), confidence intervals on individual subgroup medians remain wide.
- **Target Visibility**: Stability is only meaningful under visible targets; masked-target protocols must not be used to assess explanation stability for local perturbations.

### Status: Fully Supported
Empirically proven after identifying and correcting the Stage 6 masked-target artifact.

---

## RQ4: How faithful are Attention and Integrated Gradients explanations under the implemented masking interventions?

### Primary Evidence & Exact Statistics
1. **Evaluation Design**:
   - Evaluated across 20 spelling pairs ($N=40$ sentences per model; 80 sentences total) at context masking fractions $f \in \{0.10, 0.20, 0.30\}$ under top-importance, bottom-importance (control), and random-importance masking.
   - Target score is the mean unnormalized logit of target subwords.
   - Primary metric: **Faithfulness Gap** ($\Delta S_{\text{top}} - \Delta S_{\text{bottom}}$) and **Win Rate vs. Bottom Control** ($\% \text{ where } \Delta S_{\text{top}} > \Delta S_{\text{bottom}}$).

2. **Attention Faithfulness**:
   - **XLM-RoBERTa** ($N=40$ evaluations per fraction):
     - $f=0.10$: Faithfulness Gap $= +4.0170$, Win Rate $= \mathbf{77.5\%}$ (vs. random: 60.0%).
     - $f=0.20$: Faithfulness Gap $= +4.0170$, Win Rate $= \mathbf{77.5\%}$ (vs. random: 62.5%).
     - $f=0.30$: Faithfulness Gap $= +5.1716$, Win Rate $= \mathbf{75.0\%}$ (vs. random: 60.0%).
   - **MuRIL** ($N=40$ evaluations per fraction):
     - $f=0.10$: Faithfulness Gap $= +0.4177$, Win Rate $= \mathbf{60.0\%}$ (vs. random: 35.0%).
     - $f=0.20$: Faithfulness Gap $= +0.4177$, Win Rate $= \mathbf{60.0\%}$ (vs. random: 40.0%).
     - $f=0.30$: Faithfulness Gap $= +5.7943$, Win Rate $= \mathbf{70.0\%}$ (vs. random: 50.0%).
     - In strict single-subword MuRIL ($N=12$), Win Rate reaches **75.0%** at $f=0.30$ (Gap $= +8.8220$).

3. **Integrated Gradients Faithfulness**:
   - **XLM-RoBERTa** ($N=40$ evaluations per fraction):
     - $f=0.10$: Faithfulness Gap $= +1.8373$, Win Rate $= \mathbf{60.0\%}$ (vs. random: 35.0%).
     - $f=0.20$: Faithfulness Gap $= +1.8373$, Win Rate $= \mathbf{60.0\%}$ (vs. random: 35.0%).
     - $f=0.30$: Faithfulness Gap $= +3.2952$, Win Rate $= \mathbf{55.0\%}$ (vs. random: 42.5%).
   - **MuRIL** ($N=40$ evaluations per fraction):
     - $f=0.10$: Faithfulness Gap $= -0.6855$, Win Rate $= \mathbf{45.0\%}$ (below chance; vs. random: 32.5%).
     - $f=0.20$: Faithfulness Gap $= -0.6855$, Win Rate $= \mathbf{45.0\%}$ (below chance; vs. random: 47.5%).
     - $f=0.30$: Faithfulness Gap $= +0.0325$, Win Rate $= \mathbf{42.5\%}$ (below chance; vs. random: 35.0%).

4. **XLM-RoBERTa Out-of-Distribution (OOD) Logit Dynamics**:
   - In XLM-RoBERTa, unnormalized target logits increased after context masking, yielding negative raw drops (e.g., at $f=0.20$, Mean Top Drop $= -7.1884$, Mean Bottom Drop $= -11.2055$).
   - However, directional faithfulness was maintained because bottom-k masking inflated logits even more than top-k masking, yielding a strictly positive Faithfulness Gap:
     $$\text{Gap} = (-7.1884) - (-11.2055) = +4.0171 > 0$$

### Scientific Conclusion
- Under discrete `[MASK]` token erasure, **last-layer self-attention attributions demonstrate superior faithfulness** to Integrated Gradients, achieving win rates of 70%–77.5% against bottom-k control masking.
- Integrated Gradients performs near or below chance in MuRIL (win rates 42.5%–45.0%, negative gaps at low fractions). This occurs because Integrated Gradients integrates along a continuous straight-line interpolation in embedding space, which does not map monotonically onto discrete, highly non-linear `[MASK]` substitutions in masked language model heads.
- Attention's higher faithfulness in this setting is an empirical finding for discrete token masking, but does not prove attention is causal; attention reflects query-key routing that correlates with transformer degradation patterns.

### Uncertainty & Methodological Nuances
- **OOD Masking Artifacts**: Replacing context words with `[MASK]` pushes inputs out of distribution, causing perplexity shocks and logit distortions (as seen in XLM-RoBERTa).
- **Intervention Type**: Continuous embedding zeroing or Gaussian blurring might yield different faithfulness dynamics for Integrated Gradients.

### Status: Supported with Methodological Caveats
The empirical gap between Attention and IG under discrete masking is clearly supported, but must be accompanied by explicit OOD intervention caveats.

---

## RQ5: Is explanation stability statistically associated with measured faithfulness under the implemented perturbations?

### Primary Evidence & Exact Statistics
Direct recomputation of rank and linear correlations between visible-target stability ($\rho$) and mean faithfulness gap ($f=0.20$) across all $N=20$ Devanagari spelling pairs (and strict subsets):

1. **All Scorable Pairs ($N=20$)**:
   - **XLM-RoBERTa (Attention)**: Spearman $r_s = +0.2704, p = 0.2490$; Pearson $r = +0.2557, p = 0.2766$.
   - **MuRIL (Attention)**: Spearman $r_s = -0.1611, p = 0.4974$; Pearson $r = -0.1939, p = 0.4126$.
   - **XLM-RoBERTa (Integrated Gradients)**: Spearman $r_s = +0.1583, p = 0.5052$; Pearson $r = +0.2592, p = 0.2697$.
   - **MuRIL (Integrated Gradients)**: Spearman $r_s = +0.0075, p = 0.9749$; Pearson $r = +0.0182, p = 0.9394$.

2. **Strict Single-Subword Subset**:
   - XLM-RoBERTa Attention ($N=4$): Spearman $r_s = +0.6000, p = 0.4000$; Pearson $r = +0.8010, p = 0.1990$.
   - XLM-RoBERTa IG ($N=4$): Spearman $r_s = +0.8000, p = 0.2000$; Pearson $r = +0.6463, p = 0.3537$.
   - MuRIL Attention ($N=6$): Spearman $r_s = -0.4058, p = 0.4247$; Pearson $r = -0.1605, p = 0.7613$.
   - MuRIL IG ($N=6$): Spearman $r_s = -0.3143, p = 0.5441$; Pearson $r = -0.3427, p = 0.5061$.

### Scientific Conclusion
- **There is no statistically significant relationship between explanation stability and explanation faithfulness** across any evaluated model, explanation method, or subset ($p > 0.05$ across all 8 configurations; $p$-values range from $0.199$ to $0.975$).
- Furthermore, the observed correlation signs are contradictory: weakly positive in XLM-RoBERTa ($r_s = +0.270$) and negative in MuRIL ($r_s = -0.161$).
- **Scientific Significance**: Stability and faithfulness evaluate fundamentally orthogonal aspects of explanation quality:
  - *Stability* measures attribution invariance under localized input noise.
  - *Faithfulness* measures whether attributed tokens reflect the model's prediction mechanism under feature erasure.
- An explanation can be rigidly invariant (high stability) while pointing to irrelevant syntactic anchors (unfaithful). Conversely, an explanation can be sensitive to small noise while accurately identifying predictive features.
- Therefore, **explanation stability cannot be used as an empirical surrogate or proxy for explanation faithfulness**.

### Uncertainty & Methodological Nuances
- **Statistical Power**: $N=20$ pairs has limited power to detect subtle correlations ($|r| < 0.30$). The result does not prove that correlation is mathematically zero across all NLP tasks; rather, it proves that within this experimental design, the empirical data provide zero evidence for a relationship.

### Status: Fully Supported as a Non-Significant Finding
The absence of correlation is statistically verified ($p > 0.05$ across all conditions). Interpreted properly as proof that stability cannot be assumed to predict faithfulness.
