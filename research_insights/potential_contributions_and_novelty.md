# Potential Contributions and Novelty Assessment

This document provides a disciplined scientific assessment of the potential research contributions and novelty arising from the Devanagari Hindi Lexical Sensitivity and Explainability project. 

To maintain scientific integrity, **verified empirical contributions** (directly demonstrated by our experimental data) are strictly separated from **potential novelties** (which require a formal, systematic literature review before claiming priority in a published manuscript).

---

## 1. Summary Matrix: Verified Contributions vs. Potential Novelty

| Research Area | Demonstrated by Our Experiments (Verified) | Potential Novelty (Requires Systematic Literature Review) |
|---|---|---|
| **Devanagari Lexical Benchmark** | High-quality, curated 200-pair Devanagari Hindi contrast set with triple annotation, adjudicated status, and 3 distinct tracks. | Novelty of dataset design: whether existing Indic benchmarks evaluate fine-grained matra/ligature errors alongside synonyms and contextual contrasts. |
| **Devanagari Subword Fragmentation** | Measured severe fragmentation: matra inversions and ligature splits cause a 50%–60% collapse in single-subword target coverage across BPE and WordPiece. | First systematic quantification of subword vocabulary collapse under authentic Devanagari orthographic errors in Indic MLMs. |
| **Comparative Model Behavior (MuRIL vs XLM-R)** | Demonstrated ~12% tokenization compactness advantage for MuRIL, stronger plausibility penalties for errors (95% vs 65%), and cross-model agreement collapse on synonyms. | Comparative characterization of Indic-dedicated vs. general multilingual MLM sensitivity to morpho-phonetic Devanagari perturbations. |
| **Explanation Stability under Orthographic Noise** | Proved Attention rank stability ($\rho \approx 0.70 - 0.74$) is 2x–3x higher than Integrated Gradients ($\rho \approx 0.26 - 0.36$), while gradient cosine similarity remains high ($0.90 - 0.99$). | Demonstration of rank vs. vector space dissociation in explainable NLP under Devanagari script perturbations. |
| **XAI Faithfulness in Hindi MLMs** | Demonstrated Attention achieves 70%–77.5% win rate in discrete context erasure, while continuous IG performs near or below chance (42.5%–45.0%). | Empirical evidence of gradient-saliency failure under discrete token erasure in Indic masked language models. |
| **Stability–Faithfulness Decoupling** | Proved that explanation stability under spelling noise does NOT correlate with masking faithfulness ($p > 0.05$ across all 8 conditions). | Establishing that perturbation stability cannot serve as an empirical surrogate for explanation faithfulness in multilingual NLP. |

---

## 2. In-Depth Evaluation by Research Dimension

### Dimension 1: Human-Validated Devanagari Hindi Lexical Contrasts
- **What is Verified**:
  - The dataset provides 200 curated Devanagari sentence pairs across three controlled tracks: `VALID_WORD_CONTEXT` ($N=142$), `SYNONYM_SUBSTITUTION` ($N=38$), and `REALISTIC_SPELLING_ERROR` ($N=20$).
  - Full traceability: 100% Devanagari script integrity (`U+0900` to `U+097F`), verified SHA-256 hash, adjudicated human annotations, explicit tracking of 6 disagreement cases, and 1 rejected item (`HC_0178`).
  - Taxonomic categorization of 20 Devanagari errors into: matra length inversions (9), conjunct/ligature disruptions (5), orthographic variants (2), and colloquial phonetic spellings (4).
- **What Requires Literature Review**:
  - Determining whether other Hindi diagnostic datasets (e.g., Hindi GLUE, IndicGLUE, or IndoNLU adaptations) already contain similar minimal-pair lexical sensitivity tracks.
  - While our dataset provides a clean, controlled benchmark, absolute claims of being "the first" or "the only" dataset must be withheld until a formal bibliographic search of ACL/EMNLP/COLING/LREC proceedings is conducted.

### Dimension 2: Subword Fragmentation Under Syllabic Orthography
- **What is Verified**:
  - In Latin-script NLP, character typos typically cause a 1-to-2 subword split. In Devanagari Hindi, due to syllabic abugida properties (dependent vowel matras and virama-linked conjuncts), a single matra swap or virama omission triples subword length.
  - In MuRIL, canonical target words tokenize into a single subword in 90.0% of cases, collapsing to 30.0% under spelling perturbation ($-60.0\%$ absolute drop).
  - In XLM-RoBERTa, single-subword coverage collapses from 70.0% to 20.0% ($-50.0\%$ drop).
  - In both models, sentence length increases by exactly $+1.00$ token on average, driven 100% by target word fracturing.
- **Potential Novelty**:
  - While subword fertility has been studied for non-Latin scripts, framing Devanagari matra and conjunct vulnerability as an acute structural failure mode for tokenizer dictionaries under realistic user spelling noise is a compelling, highly publishable insight.

### Dimension 3: Indic-Dedicated (MuRIL) vs. Multilingual Generalist (XLM-RoBERTa)
- **What is Verified**:
  - **Efficiency**: MuRIL encodes Hindi with ~12% fewer tokens (8.95 vs. 10.19 on `VALID_WORD_CONTEXT`).
  - **Sensitivity**: MuRIL penalizes spelling errors substantially more heavily than XLM-RoBERTa (original preferred in 95.0% vs. 65.0%; mean $\Delta \text{MeanLogP} = -2.04$ vs. $-0.67$).
  - **Synonym Equivalence**: Both models recognize valid synonyms as near-equivalent (minimal PLL drops: $-0.07$ for XLM-R, $-0.58$ for MuRIL), but cross-model correlation collapses to zero ($r_s = +0.0595, p = 0.7226$), revealing model-specific pretraining frequency biases.
  - **Overall Agreement**: The two distinct architectures agree on the direction of plausibility shift in 74.5% of pairs ($r_s = +0.4408, p < 10^{-9}$).
- **Potential Novelty**:
  - Quantifying how pretraining corpus focus (Indic crawl vs. global multilingual dump) directly governs model sensitivity to low-level orthographic errors versus high-level contextual semantic violations in Hindi.

### Dimension 4: Explanation Stability Under Orthographic Noise
- **What is Verified**:
  - Last-layer self-attention exhibits $2.1\times$ to $2.7\times$ higher rank correlation stability than Integrated Gradients under Devanagari spelling noise (Mean visible $\rho = 0.7425$ vs. $0.3589$ in XLM-R; $0.7043$ vs. $0.2632$ in MuRIL).
  - Crucially, Integrated Gradients retains high cosine similarity ($0.899 - 0.988$). This proves a dissociation: gradient attribution vectors point in approximately the same orientation, but individual token rankings fluctuate drastically under minor input perturbations.
  - Discovery and documentation of the Stage 6 masked-target artifact ($\rho = 1.000, \text{Var} = 0$), resolved by the Stage 7 Visible-Target Protocol.
- **Potential Novelty**:
  - Investigating feature attribution stability under script-specific orthographic noise (rather than synthetic English character swaps or adversarial embedding attacks) in multilingual transformers.
  - Methodological contribution: documenting the masked-target artifact and establishing the visible-target protocol for perturbation stability.

### Dimension 5: Masking-Based Faithfulness in Hindi MLMs
- **What is Verified**:
  - Self-attention attributions consistently achieve positive Faithfulness Gaps ($+4.02$ to $+5.79$) and high win rates ($70.0\% - 77.5\%$) against bottom-k control masking.
  - Integrated Gradients achieves only 50.0%–60.0% win rate in XLM-RoBERTa and performs near or below chance in MuRIL (42.5%–45.0% win rate; negative gap $-0.69$ at $f=0.10, 0.20$).
  - Mechanistic explanation: Integrated Gradients computes path integrals over continuous embedding spaces. These continuous interpolations do not map monotonically to discrete, non-linear `[MASK]` token substitutions in MLM prediction heads.
- **Potential Novelty**:
  - Demonstrating that the widely assumed theoretical superiority of axiomatic gradient methods (e.g., Integrated Gradients) over heuristic attention weights fails when evaluated against discrete feature erasure in masked language models.

### Dimension 6: Joint Stability–Faithfulness Decoupling
- **What is Verified**:
  - Empirical correlation between visible-target stability ($\rho$) and intervention faithfulness gap at 20% masking across all 20 Devanagari spelling pairs is non-significant across all configurations ($p > 0.05$ across all 8 model-method-subset permutations):
    - XLM-RoBERTa Attention: $r_s = +0.2704, p = 0.2490$
    - MuRIL Attention: $r_s = -0.1611, p = 0.4974$
    - XLM-RoBERTa IG: $r_s = +0.1583, p = 0.5052$
    - MuRIL IG: $r_s = +0.0075, p = 0.9749$
  - The correlation sign flips between models for the same explanation method (positive in XLM-R, negative in MuRIL).
- **Potential Novelty**:
  - **This is the core scientific insight of the project**: Stability and faithfulness measure orthogonal, uncoupled properties of explanations. High attribution stability under input noise cannot be used as an empirical surrogate for faithful feature attribution.
  - Explainability benchmarks that optimize exclusively for explanation robustness/stability risk selecting explanations that are stably unfaithful.

---

## 3. Methodological Boundaries and Prerequisite Checks for Publication

Before drafting claims of novelty in a conference submission:
1. **Literature Review Requirement**:
   - Perform a structured search on Google Scholar / ACL Anthology for:
     - `("explanation stability" OR "attribution robustness") AND ("faithfulness" OR "comprehensiveness")`
     - `"Devanagari" AND "tokenization" AND ("MuRIL" OR "XLM-RoBERTa")`
     - `"pseudo-log-likelihood" AND "Hindi" AND "lexical sensitivity"`
   - Cite seminal works in explainability (e.g., Sundararajan et al. for IG; Wiegreffe & Pinter, Jain & Wallace for attention debate; DeYoung et al. for ERASER faithfulness; Alvarez-Melis & Jaakkola for stability) and position our findings within that established dialogue.
2. **Explicit Sample Size Framing**:
   - The stability–faithfulness correlation is evaluated on $N=20$ Devanagari spelling pairs ($N=40$ sentences).
   - Frame the correlation finding as an **empirical non-significant result that rejects stability as a reliable surrogate** within this domain, rather than proving a universal mathematical independence across all deep learning architectures.
