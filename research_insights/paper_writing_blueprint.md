# Paper Writing Blueprint

**Working Title:** Beyond Tokenization: Investigating Model Behaviour, Explanation Stability, and Faithfulness under Lexical Perturbations in Devanagari Hindi

**Target Venues:** ACL, EMNLP, NAACL, COLING, or Findings of ACL.

---

## 1. Paper Overview & Structural Blueprint

```
Section 1: Introduction
  ├── Motivation: Pretrained MLMs in Indic NLP; subword fragility in Devanagari
  ├── Explainability Dilemma: Does explanation stability imply explanation faithfulness?
  ├── Key Findings & Core Contribution Summary
Section 2: Related Work
  ├── Explainable NLP: Attention vs. Feature Attribution & the Stability/Faithfulness Debate
  ├── Subword Tokenization & Script Vulnerabilities (Indic/Devanagari)
  ├── Masked Language Model Scoring & Pseudo-Log-Likelihood
Section 3: Dataset Construction & Devanagari Typology
  ├── Track Design: VALID_WORD_CONTEXT (142), SYNONYM_SUBSTITUTION (38), REALISTIC_SPELLING_ERROR (20)
  ├── Devanagari Error Typology: Matras, Ligatures, Orthographic Variants, Colloquialisms
  ├── Quality Control, Adjudication, & Forensic Audit (HC_0178 resolution)
Section 4: Experimental Methodology
  ├── Evaluated Models: MuRIL (Indic-dedicated) vs. XLM-RoBERTa (multilingual generalist)
  ├── Subword Tokenization Metrics & Offset Alignment
  ├── Pseudo-Log-Likelihood (PLL) & Length-Normalized MeanLogP Formulations
  ├── Explanation Framework: Layer Integrated Gradients vs. Last-Layer Self-Attention
  ├── The Stage 6 Masked-Target Artifact & The Stage 7 Visible-Target Protocol
  ├── Faithfulness Interventions: Context Erasure (Top-k, Bottom-k, Random-k at 10%, 20%, 30%)
  ├── Stability–Faithfulness Correlation Analysis Protocol
Section 5: Empirical Results & Analysis
  ├── 5.1 Subword Tokenization & Catastrophic Target Fragmentation
  ├── 5.2 Model Plausibility Scoring across Contexts, Synonyms, & Misspellings
  ├── 5.3 Comparative Model Behavior: MuRIL vs. XLM-RoBERTa
  ├── 5.4 Explanation Stability under Devanagari Spelling Noise
  ├── 5.5 Explanation Faithfulness under Discrete Context Erasure
  ├── 5.6 Decoupling of Explanation Stability and Faithfulness
Section 6: Discussion
  ├── Theoretical Implications: Orthogonality of Metric Spaces (Stability ≠ Faithfulness)
  ├── Discrete vs. Continuous Attribution Mechanisms (Why IG fails under discrete masking)
  ├── Devanagari Syllabic Vulnerability in NLP Tokenizers
Section 7: Limitations & Scientific Guardrails
  ├── Sample size constraints (N=20 spelling pairs)
  ├── Out-of-distribution (OOD) syntactic shock from [MASK] tokens
  ├── Target logit inflation in XLM-RoBERTa
  ├── Multi-subword target mean-logit aggregation assumptions
  ├── CPU execution & pre-trained checkpoint scope
Section 8: Conclusion & Future Directions
```

---

## 2. Detailed Section-by-Section Content & Evidence Mapping

### Section 1: Introduction
- **Opening Context**: Masked language models (MLMs) have become foundational for multilingual NLP, but non-Latin scripts like Devanagari Hindi present unique morphological, syllabic, and orthographic characteristics.
- **The Core Scientific Tension**: Explainable AI (XAI) benchmarks frequently treat attribution stability (invariance to minor noise) as a hallmark of explanation quality. Does a stable explanation actually identify the causal features used by the model?
- **Contributions to Highlight**:
  1. Systematic evaluation of Devanagari Hindi tokenization fragmentation and MLM scoring across 200 minimal pairs.
  2. Proof of severe subword vocabulary collapse under authentic Devanagari spelling errors ($-50\%$ to $-60\%$ single-subword coverage).
  3. Discovery that last-layer self-attention achieves higher rank stability and discrete masking faithfulness than Integrated Gradients in MLMs.
  4. Empirical demonstration that explanation stability does **not** predict explanation faithfulness ($p > 0.05$ across all conditions).

---

### Section 2: Related Work
- **Feature Attribution in NLP**: Acknowledge Integrated Gradients (Sundararajan et al., 2017) as an axiomatic method satisfying completeness and implementation invariance; contrast with attention-as-explanation debates (Jain & Wallace, 2019; Wiegreffe & Pinter, 2019).
- **Faithfulness Evaluation**: Frame context erasure within the ERASER benchmark paradigm (DeYoung et al., 2020) and comprehensiveness/sufficiency metrics.
- **Multilingual Tokenization & Indic NLP**: Highlight subword fertility studies (Rust et al., 2021) and MuRIL (Khanuja et al., 2021) vs. XLM-RoBERTa (Conneau et al., 2020).

---

### Section 3: Dataset Construction & Devanagari Typology
- **Table 1: Dataset Composition & Track Distribution**
  - Source: `outputs/dataset_validation_report.json`
  - Values: Total non-empty pairs: 200 (`VALID_WORD_CONTEXT`: 142 [71.0%], `SYNONYM_SUBSTITUTION`: 38 [19.0%], `REALISTIC_SPELLING_ERROR`: 20 [10.0%]).
- **Devanagari Error Typology**:
  - Matra vowel length inversion ($N=9$, e.g., `पानी` $\to$ `पानि`, `परीक्षा` $\to$ `परिक्षा`).
  - Conjunct/ligature disruptions ($N=5$, e.g., `स्वास्थ्य` $\to$ `स्वास्थ`, `आवश्यक` $\to$ `आवशयक`).
  - Orthographic variants ($N=2$, `महत्वपूर्ण` vs `महत्त्वपूर्ण`, `संभावना` vs `सम्भावना`).
  - Colloquial phonetic spellings ($N=4$, e.g., `वापस` $\to$ `वापिस`).
- **Forensic Quality Audit**:
  - Exactly 1 pair rejected: `HC_0178` (`final_status = reject`). Explicitly correct past documentation typos (`HC_0185`, `HC_0047`).
  - Exactly 6 disagreement overrides: `HC_0087`, `HC_0099`, `HC_0138`, `HC_0172`, `HC_0180`, `HC_0186`. Retained because both sentences are grammatical valid Hindi sentences.
  - 100% Devanagari Unicode codepoint integrity (`U+0900` to `U+097F`).

---

### Section 4: Experimental Methodology
- **Architectures**:
  - `FacebookAI/xlm-roberta-base`: SentencePiece BPE unigram, 250,002 vocabulary, 12 layers, 12 heads, 768 hidden.
  - `google/muril-base-cased`: WordPiece, 197,285 vocabulary, 12 layers, 12 heads, 768 hidden.
- **Formulations**:
  - PLL: $\text{PLL}(W) = \sum_{t=1}^{|W|} \log P(w_t \mid W_{\setminus t})$; Length-normalized $\text{MeanLogP} = \frac{\text{PLL}(W)}{|W|}$; Signed $\Delta = \text{Cont} - \text{Orig}$.
  - Strict Cross-Model Comparability Guardrail: Raw PLL sums must NEVER be compared across tokenizers; only within-model signed deltas and preference rates.
- **Explanation Protocols**:
  - Attention: Layer 11 mean across all 12 heads; special tokens excluded.
  - Layer Integrated Gradients: Embedding layer, 50 Gauss-Legendre steps, zero baseline.
  - Case Study on Benchmark Design: Document the Stage 6 masked-target artifact ($\rho = 1.000$) caused by identical input contexts. Contrast with Stage 7 Visible-Target Protocol.
  - Multi-subword target score: $S(X; Y_{\text{target}}) = \frac{1}{k} \sum_{j=1}^k \text{Logit}(p_j, y_j)$.
  - Faithfulness Gap: $\Delta S_{\text{top}} - \Delta S_{\text{bottom}}$ at $f \in \{0.10, 0.20, 0.30\}$.

---

### Section 5: Empirical Results & Analysis

#### 5.1 Subword Tokenization Fragmentation
- **Table 2: Subword Tokenization Across Tracks**
  - Source: `outputs/dashboard_data/tokenization_summary.csv`
  - Values:
    - MuRIL: Spelling target subwords $1.10 \to 2.10$ ($+1.00$); single-subword coverage collapses from **90.0% to 30.0% (-60.0%)**.
    - XLM-RoBERTa: Spelling target subwords $1.35 \to 2.35$ ($+1.00$); single-subword coverage collapses from **70.0% to 20.0% (-50.0%)**.
    - MuRIL baseline sentence length: 8.95 vs. XLM-R 10.19 on `VALID_WORD_CONTEXT` (~12% compactness gain).

#### 5.2 Model Plausibility Scoring
- **Table 3: Descriptive PLL Statistics Across Tracks**
  - Source: `outputs/analysis/descriptive_statistics.csv`
  - Values:
    - Spelling track: MuRIL Mean $\Delta \text{MeanLogP} = -2.0368$ (95.0% original preferred) vs. XLM-R Mean $\Delta = -0.6695$ (65.0% original preferred).
    - Contextual substitutions: MuRIL $-1.9141$ (93.0% preferred) vs. XLM-R $-1.0933$ (81.0% preferred).
    - Synonyms: MuRIL $-0.5810$ (84.2% preferred) vs. XLM-R $-0.0655$ (57.9% preferred).
- **Figures to Embed**:
  - Figure 1: `outputs/analysis/figures/fig1_delta_meanlp_by_track.png` (Boxplots of $\Delta \text{MeanLogP}$ by track and model).
  - Figure 2: `outputs/analysis/figures/fig2_token_delta_vs_meanlp.png` (Tokenization delta vs score delta).

#### 5.3 Comparative Model Behavior
- **Table 4: Cross-Model Agreement and Rank Correlation**
  - Source: `outputs/analysis/model_comparison.csv`
  - Values: All Tracks ($N=200$): Agreement $= 74.5\%$, Spearman $r_s = +0.4408, p < 10^{-9}$. Collapses on Synonyms ($N=38$): Agreement $= 57.9\%$, $r_s = +0.0595, p = 0.7226$.
- **Figure to Embed**:
  - Figure 4: `outputs/analysis/figures/fig4_model_comparison_scatter.png` (Scatter plot of XLM-R vs MuRIL $\Delta \text{MeanLogP}$).

#### 5.4 Explanation Stability under Spelling Noise
- **Table 5: Visible-Target Explanation Stability**
  - Source: `outputs/dashboard_data/stability_summary.csv`
  - Values:
    - XLM-RoBERTa: Attention Mean $\rho = \mathbf{0.7425}$ vs. IG Mean $\rho = \mathbf{0.3589}$ ($2.1\times$).
    - MuRIL: Attention Mean $\rho = \mathbf{0.7043}$ vs. IG Mean $\rho = \mathbf{0.2632}$ ($2.7\times$).
    - Cosine similarity: Remains high across both (XLM-R IG: $0.8994$; MuRIL IG: $0.9880$).

#### 5.5 Explanation Faithfulness under Discrete Erasure
- **Table 6: Faithfulness Gap & Win Rates across Masking Fractions**
  - Source: `outputs/dashboard_data/faithfulness_summary.csv`
  - Values:
    - XLM-RoBERTa Attention ($f=0.20$): Gap $= \mathbf{+4.0170}$, Win Rate $= \mathbf{77.5\%}$.
    - XLM-RoBERTa IG ($f=0.20$): Gap $= \mathbf{+1.8373}$, Win Rate $= \mathbf{60.0\%}$.
    - MuRIL Attention ($f=0.30$): Gap $= \mathbf{+5.7943}$, Win Rate $= \mathbf{70.0\%}$ (strict single: Gap $= +8.8220$, Win Rate $= 75.0\%$).
    - MuRIL IG ($f=0.20$): Gap $= \mathbf{-0.6855}$, Win Rate $= \mathbf{45.0\%}$ (below chance).
  - Detail the OOD logit inflation in XLM-RoBERTa and explain how relative gap preserves faithfulness.

#### 5.6 Decoupling of Stability and Faithfulness
- **Table 7: Stability–Faithfulness Correlation Matrix ($N=20$)**
  - Source: Recomputed from `outputs/dashboard_data/stability_faithfulness_relationship.csv`
  - Values:
    - XLM-RoBERTa Attention: $r_s = +0.2704, p = 0.2490$ (Pearson $r = +0.2557, p = 0.2766$).
    - MuRIL Attention: $r_s = -0.1611, p = 0.4974$ (Pearson $r = -0.1939, p = 0.4126$).
    - XLM-RoBERTa IG: $r_s = +0.1583, p = 0.5052$ (Pearson $r = +0.2592, p = 0.2697$).
    - MuRIL IG: $r_s = +0.0075, p = 0.9749$ (Pearson $r = +0.0182, p = 0.9394$).
  - Core Takeaway: Zero statistically significant correlations across all models and methods ($p > 0.05$). Contradictory correlation directions between models.

---

### Section 6: Discussion
- **Orthogonality of Metric Spaces**: Explain why stability (invariance of explanation to input noise) is mathematically decoupled from faithfulness (causal impact of features on model prediction).
- **Why Integrated Gradients Fails under Discrete Erasure**: IG integrates gradients over a continuous straight-line path from zero/pad embeddings. MLM heads operate on discrete token representations; replacing a token with `[MASK]` is a discrete jump that does not follow the continuous path integral.
- **Implications for Explainability Benchmarking**: Warn against evaluating explanation robustness without joint faithfulness validation.

---

### Section 7: Limitations & Scientific Guardrails
- **Sample Size**: Spelling track has $N=20$ pairs; power is limited for subtle correlation detection ($|r| < 0.30$).
- **OOD Masking Shock**: `[MASK]` insertion damages syntax; drops reflect both semantic erasure and perplexity shock.
- **Attention Non-Causality**: Higher win rates for attention reflect correlational routing, not certified counterfactual causation.
- **Multi-Subword Averaging**: Arithmetic mean logit assumes uniform subword weighting.

---

## 3. Checklist of Verified Data & Manuscript Tables

| Manuscript Item | Content | Source Artifact | Verification Status |
|---|---|---|---|
| **Table 1** | Track distribution & pair counts (142 / 38 / 20) | `outputs/dataset_validation_report.json` | Verified Exact |
| **Table 2** | Subword token counts & single-subword collapse | `outputs/dashboard_data/tokenization_summary.csv` | Verified Exact |
| **Table 3** | Within-model PLL descriptive statistics | `outputs/analysis/descriptive_statistics.csv` | Verified Exact |
| **Table 4** | Cross-model agreement (74.5%) & rank correlation ($r_s = 0.4408$) | `outputs/analysis/model_comparison.csv` | Verified Exact |
| **Table 5** | Visible-target stability ($\rho$ and Cosine) | `outputs/dashboard_data/stability_summary.csv` | Verified Exact |
| **Table 6** | Faithfulness drops, gaps, and win rates across fractions | `outputs/dashboard_data/faithfulness_summary.csv` | Verified Exact |
| **Table 7** | Stability–faithfulness correlation matrix ($N=20$ and strict) | `outputs/dashboard_data/stability_faithfulness_relationship.csv` | Verified Exact |
| **Figure 1** | Boxplot of $\Delta \text{MeanLogP}$ by track and model | `outputs/analysis/figures/fig1_delta_meanlp_by_track.png` | Verified PNG |
| **Figure 2** | Scatter: Token count delta vs. $\Delta \text{MeanLogP}$ | `outputs/analysis/figures/fig2_token_delta_vs_meanlp.png` | Verified PNG |
| **Figure 3** | Scatter: Target token delta vs. $\Delta \text{MeanLogP}$ | `outputs/analysis/figures/fig3_target_token_delta_vs_meanlp.png` | Verified PNG |
| **Figure 4** | Cross-model scatter: XLM-R vs. MuRIL $\Delta \text{MeanLogP}$ | `outputs/analysis/figures/fig4_model_comparison_scatter.png` | Verified PNG |
| **Figure 5** | Directional agreement proportions by track | `outputs/analysis/figures/fig5_direction_agreement.png` | Verified PNG |
