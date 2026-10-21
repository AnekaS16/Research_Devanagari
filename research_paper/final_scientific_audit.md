# Final Scientific Audit & Verification Report

This document records the definitive scientific audit of the experimental findings, data integrity, statistical testing, and methodological guardrails for the research project *"Beyond Tokenization: Investigating Model Behaviour, Explanation Stability, and Faithfulness under Lexical Perturbations in Devanagari Hindi"*.

---

## 1. Forensic Audit of Dataset Identifiers and Reconciliation

### 1.1 The Rejected-Pair Discrepancy
- **The Issue**: Conflicting identifiers were cited across project documents (`HC_0185` in early draft analysis; `HC_0047` in stage 2/3 notes; `HC_0178` in raw dataset).
- **Forensic Verification**: Direct examination of the ground-truth Excel workbook (`data/raw/hindi_lexical_sensitivity_cleaned.xlsx`, SHA-256 `f9a5af81e216...`) confirms:
  - **`HC_0178`** (`सलाह` vs. `सलाई`, Row 177) is the **sole true rejected pair** (`annotator_1 = reject`, `annotator_2 = reject`, `adjudication = reject`, `final_status = reject`).
  - **`HC_0185`** (`भर` vs. `पढ़`, Row 184) is an **accepted pair** (`final_status = accept`).
  - **`HC_0047`** (`बाँधी` vs. `बाँटी`, Row 46) is an **accepted pair** (`final_status = accept`).
- **Audit Decision**: The paper manuscript exclusively cites `HC_0178` as the sole rejected record.

### 1.2 "1 Rejected Pair" vs. "2 Records Rejected" Unit Conflation
- **The Issue**: `analysis_report.md` mentioned "2 records rejected", while `dataset_validation_report.json` stated 1 pair rejected.
- **Forensic Verification**:
  - The dataset contains **200 unique sentence pairs**, exactly **1** of which is rejected (`HC_0178`, 0.5%).
  - In the 2-model evaluation table (`outputs/analysis/combined_pair_level_results.csv`), there are **400 evaluation rows** (200 pairs $\times$ 2 models).
  - In that table, pair `HC_0178` generates **2 evaluation rows** (one for XLM-RoBERTa, one for MuRIL).
  - Filtering out `HC_0178` subtracts 1 pair from each model's sub-analysis ($142 \to 141$ in `VALID_WORD_CONTEXT`).
- **Audit Decision**: The manuscript explicitly distinguishes between **sentence pairs** ($N=200$, 1 rejected) and **model evaluation instances** ($N=400$, 2 rejected).

### 1.3 Human Review Disagreement Identifiers
- **The Issue**: `stage2_tokenization.md` listed `HC_0187` among disagreement cases, whereas `dataset_validation_report.json` listed `HC_0186`.
- **Forensic Verification**:
  - `HC_0186` had unanimous double-reject from annotators and adjudicator, overridden by the curator to `accept` because both sentences are grammatical valid Hindi sentences.
  - `HC_0187` had unanimous agreement (`accept`).
- **Audit Decision**: The verified set of 6 disagreement overrides is $\{\text{HC\_0087}, \text{HC\_0099}, \text{HC\_0138}, \text{HC\_0172}, \text{HC\_0180}, \text{HC\_0186}\}$.

---

## 2. Methodological Audits & Artifact Resolution

### 2.1 The Stage 6 Masked-Target Artifact
- **Root Cause**: In Stage 6, stability was evaluated while masking the target word. In `REALISTIC_SPELLING_ERROR`, the perturbation was exclusively located within the target word. Masking the target produced byte-identical input sequences ($X_{\text{orig}} \equiv X_{\text{cont}}$), yielding an illusory perfect stability ($\rho = 1.000$, Cosine $= 1.000$, $\Delta \text{Logit} = 0.0$ with zero variance).
- **Resolution**: Stage 7 implemented the **Visible-Target Scoring Protocol**, keeping the target token visible in the input and computing stability across aligned context tokens.
- **Audit Decision**: The paper manuscript includes this as a prominent methodological case study warning against flawed benchmark design in explainability research.

### 2.2 XLM-RoBERTa Target Logit Inflation under Context Erasure
- **Phenomenon**: In XLM-RoBERTa, masking context tokens produced negative raw logit drops ($\Delta S_{\text{top}} = -7.19$, $\Delta S_{\text{bottom}} = -11.21$ at $f=0.20$), indicating that target logits increased following context erasure.
- **Audit Findings**:
  - Replacing context tokens with `[MASK]` causes severe out-of-distribution (OOD) syntactic shock.
  - In XLM-RoBERTa's unnormalized logit space, the reduction of contextual competition caused target logits to float higher globally.
  - Directional faithfulness is strictly preserved because bottom-$k$ control masking inflated logits substantially more than top-$k$ masking, yielding a positive Faithfulness Gap:
    $$\text{Gap} = (-7.19) - (-11.21) = \mathbf{+4.02} > 0$$
    with a **77.5% win rate** against bottom control.
- **Audit Decision**: Both raw drops and relative faithfulness gaps/win rates are reported simultaneously with explicit discussion of MLM logit dynamics.

### 2.3 Strict Single-Subword vs. Multi-Subword Target Aggregation
- **Phenomenon**: Devanagari spelling errors fracture standard vocabulary words into multiple subwords, leaving only 6 single-subword pairs in MuRIL and 4 in XLM-RoBERTa.
- **Audit Findings**:
  - Strict subsetting preserves exact scalar logit mapping but reduces statistical power to near-negligible levels ($N=6, N=4$).
  - Mean-logit target aggregation restores full sample size ($N=20$ pairs, 40 sentences) and demonstrates empirical trends consistent with the strict subset (e.g., XLM-R Attention stability: $0.7425$ pooled vs. $0.6893$ strict; Faithfulness win rates: $77.5\%$ pooled vs. $75.0\%$ strict).
- **Audit Decision**: Both pooled ($N=20$) and strict single-subword results are reported side-by-side.

---

## 3. Stability–Faithfulness Correlation Verification ($N=20$)

Every correlation coefficient between visible-target stability ($\rho$) and mean faithfulness gap ($f=0.20$) has been independently verified against the raw evaluation tables:

| Model | Explanation Method | Evaluated Subset | Spearman $r_s$ | $p$-value ($r_s$) | Pearson $r$ | $p$-value ($r$) | Empirical Interpretation |
|---|---|---|---|---|---|---|---|
| **XLM-RoBERTa** | Attention | Pooled ($N=20$) | $+0.2704$ | $0.2490$ | $+0.2557$ | $0.2766$ | Non-significant ($p > 0.05$) |
| **XLM-RoBERTa** | Attention | Strict ($N=4$) | $+0.6000$ | $0.4000$ | $+0.8010$ | $0.1990$ | Non-significant ($p > 0.05$) |
| **XLM-RoBERTa** | Integrated Gradients | Pooled ($N=20$) | $+0.1583$ | $0.5052$ | $+0.2592$ | $0.2697$ | Non-significant ($p > 0.05$) |
| **XLM-RoBERTa** | Integrated Gradients | Strict ($N=4$) | $+0.8000$ | $0.2000$ | $+0.6463$ | $0.3537$ | Non-significant ($p > 0.05$) |
| **MuRIL** | Attention | Pooled ($N=20$) | $-0.1611$ | $0.4974$ | $-0.1939$ | $0.4126$ | Non-significant ($p > 0.05$) |
| **MuRIL** | Attention | Strict ($N=6$) | $-0.4058$ | $0.4247$ | $-0.1605$ | $0.7613$ | Non-significant ($p > 0.05$) |
| **MuRIL** | Integrated Gradients | Pooled ($N=20$) | $+0.0075$ | $0.9749$ | $+0.0182$ | $0.9394$ | Non-significant ($p > 0.05$) |
| **MuRIL** | Integrated Gradients | Strict ($N=6$) | $-0.3143$ | $0.5441$ | $-0.3427$ | $0.5061$ | Non-significant ($p > 0.05$) |

**Key Takeaways**:
1. Zero statistically significant correlations across all 8 configurations.
2. Contradictory correlation signs (positive in XLM-R, negative in MuRIL).
3. The empirical evidence rejects the assumption that explanation stability predicts explanation faithfulness.

---

## 4. Scientific Guardrails & Author Review Checklist

The manuscript adheres to all mandatory scientific guardrails:
- [x] **No Causal Overreach**: Tokenization fragmentation is documented as an observed co-occurring structural phenomenon; causal attribution to score drops is not claimed.
- [x] **Normative Guardrail**: Pseudo-log-likelihood is explicitly defined as corpus typicality and statistical probability under pretraining data, not human truth or normative grammaticality.
- [x] **Cross-Model Comparability**: Raw PLL sums across tokenizers are strictly forbidden from direct comparison; only within-model signed deltas and preference rates are compared.
- [x] **Attention Causality**: High attention faithfulness win rates are characterized as correlational routing alignment under transformer degradation, not certified counterfactual causation.
- [x] **Novelty Framing**: Empirical findings are stated with precision; broad claims of being "the first" are avoided in favor of specific, verifiable empirical contributions.
- [x] **Sample Size Caveat**: The $N=20$ spelling track size is explicitly acknowledged in the limitations, noting that while sufficient to detect large contrasts, it possesses limited statistical power for detecting weak correlations ($|r| < 0.30$).
