# Reproducibility Guide & Experimental Configuration

This document specifies the exact software dependencies, hardware environment, random seeds, hyperparameter configurations, dataset hashes, and mathematical definitions necessary to reproduce every experimental result presented in *"Beyond Tokenization: Investigating Model Behaviour, Explanation Stability, and Faithfulness under Lexical Perturbations in Devanagari Hindi"*.

---

## 1. Computing Environment & Dependency Versions

All experiments were executed on CPU hardware with explicit memory cleanup and batching to guarantee reproducibility on commodity workstations without requiring specialized GPU accelerators.

- **Operating System**: Windows 11 Enterprise / Pro (x86_64)
- **Host Memory**: 8 GB RAM (optimized with per-batch garbage collection)
- **Python Version**: Python 3.12.x
- **Key Python Packages**:
  - `torch == 2.14.1+cpu`
  - `transformers == 4.57.6`
  - `tokenizers == 0.22.2`
  - `captum == 0.9.0`
  - `scipy == 1.18.1`
  - `numpy == 2.5.4`
  - `pandas == 3.0.6`
  - `openpyxl == 3.1.5`
  - `matplotlib == 3.11.2`
  - `python-docx == 1.2.0`
  - `reportlab == 5.0.1`

---

## 2. Dataset Specifications & Integrity Verification

- **Primary Source File**: `data/raw/hindi_lexical_sensitivity_cleaned.xlsx`
- **Verified SHA-256 Checksum**: `f9a5af81e21624de47ead7feabba412149939c1eb9934f8fc2b8d452a232cd44`
- **Total Populated Rows**: Exactly 200 non-empty sentence pairs (`HC_0001` to `HC_0200`).
- **Bounding Spreadsheet Dimension**: `A1:R201` (1 header row + 200 data rows).
- **Unicode Script Range**: 100% Devanagari characters conform to Unicode block `U+0900`–`U+097F`.
- **Track Distribution**:
  - `VALID_WORD_CONTEXT`: 142 pairs (71.0%)
  - `SYNONYM_SUBSTITUTION`: 38 pairs (19.0%)
  - `REALISTIC_SPELLING_ERROR`: 20 pairs (10.0%)
- **Data Quality & Forensic Adjudication**:
  - Exactly **1 pair rejected**: `HC_0178` (`final_status = reject`).
  - Exactly **6 disagreement overrides**: `HC_0087`, `HC_0099`, `HC_0138`, `HC_0172`, `HC_0180`, `HC_0186` (retained as valid Hindi sentence contrasts with `final_status = accept`).
  - Accepted evaluation benchmark: 199 accepted pairs (200 pairs including the audited rejected baseline).

---

## 3. Pretrained Model Checkpoints & Tokenizer Settings

| Parameter | XLM-RoBERTa (`FacebookAI/xlm-roberta-base`) | MuRIL (`google/muril-base-cased`) |
|---|---|---|
| **Architecture** | RoBERTa-base (Bidirectional Encoder) | BERT-base (Bidirectional Encoder) |
| **Number of Layers** | 12 | 12 |
| **Attention Heads** | 12 | 12 |
| **Hidden Dimension** | 768 | 768 |
| **Total Parameters** | ~278 Million | ~236 Million |
| **Vocabulary Size** | 250,002 | 197,285 |
| **Tokenizer Type** | SentencePiece BPE (Unigram) | WordPiece |
| **Language Coverage** | 100+ languages (global crawl) | 17 Indic languages + English |
| **Mask Token** | `<mask>` (ID: 250001) | `[MASK]` (ID: 103) |
| **Pad Token** | `<pad>` (ID: 1) | `[PAD]` (ID: 0) |
| **BOS / EOS Tokens** | `<s>` / `</s>` | `[CLS]` / `[SEP]` |

---

## 4. Scoring Formulations & Protocols

### Pseudo-Log-Likelihood (PLL)
For a sentence $W = (w_1, w_2, \dots, w_{|W|})$, the PLL is computed via iterative single-token masking across all non-special tokens:
$$\text{PLL}(W) = \sum_{t=1}^{|W|} \log P(w_t \mid W_{\setminus t}; \Theta)$$
where $W_{\setminus t}$ denotes the sequence with token $w_t$ replaced by the model's mask token.

### Length-Normalized MeanLogP
To control for sentence length variations:
$$\text{MeanLogP}(W) = \frac{\text{PLL}(W)}{|W_{\text{scorable}}|}$$
where $|W_{\text{scorable}}|$ is the number of scored tokens (excluding special framing tokens).

### Signed Delta Convention
$$\Delta \text{MeanLogP} = \text{MeanLogP}(W_{\text{contrast}}) - \text{MeanLogP}(W_{\text{original}})$$
- $\Delta \text{MeanLogP} < 0$: The original sentence is assigned higher plausibility by the model.
- $\Delta \text{MeanLogP} > 0$: The contrast sentence is assigned higher plausibility.

---

## 5. Explainability Framework & Protocols

### Last-Layer Self-Attention Attribution
- **Extracted Layer**: Layer 11 (the 12th and final transformer layer).
- **Head Aggregation**: Arithmetic mean across all 12 attention heads:
  $$A_{i,j} = \frac{1}{12} \sum_{h=1}^{12} A^{(11, h)}_{i,j}$$
- **Context Importance Vector**: Mean attention weight directed from the target position(s) $T$ to each context token $j \notin T$:
  $$\alpha_j = \frac{1}{|T|} \sum_{t \in T} A_{t, j}$$
  Special tokens (`[CLS]`, `[SEP]`, `<s>`, `</s>`, padding) are excluded and attributions are normalized to sum to 1 over context tokens.

### Layer Integrated Gradients (Captum)
- **Target Layer**: Word embedding layer (`embeddings.word_embeddings`).
- **Baseline**: Zero vector in embedding space ($x' = \mathbf{0}$).
- **Approximation**: Gauss-Legendre quadrature with $m = 50$ interpolation steps.
- **Attribution Calculation**:
  $$\text{IG}_i(x) = (x_i - x'_i) \times \int_{0}^1 \frac{\partial F(x' + \alpha(x - x'))}{\partial x_i} d\alpha$$
  Subword scalar importance is obtained via the Euclidean $L_2$-norm over the embedding dimension ($d=768$), followed by context-level normalization.

### Visible-Target Protocol vs. Masked-Target Artifact
- **Stage 6 Artifact**: When the target token was masked with `[MASK]` in both original and contrast inputs, the inputs became identical ($X_{\text{orig}} \equiv X_{\text{cont}}$), yielding $\rho = 1.000$ trivially.
- **Stage 7 Visible-Target Protocol**: Target tokens remain visible in the input sequence. The target score is defined as the mean unnormalized logit of target subwords:
  $$S(X; Y_{\text{target}}) = \frac{1}{|T|} \sum_{j \in T} \text{Logit}(p_j, y_j)$$
  Attribution stability is evaluated across aligned context tokens between original and contrast sentences.

---

## 6. Faithfulness Evaluation Protocol

### Discrete Token Erasure
Context tokens are ranked by importance:
- **Top-$k$ Erasure**: Mask the top $k = \lceil f \cdot |C| \rceil$ tokens with `[MASK]`.
- **Bottom-$k$ Erasure (Control)**: Mask the bottom $k$ tokens.
- **Random-$k$ Erasure**: Mask $k$ tokens selected uniformly at random (averaged over 5 runs).
- **Fractions Evaluated**: $f \in \{0.10, 0.20, 0.30\}$.

### Metrics
1. **Target Logit Drop**:
   $$\Delta S_{\text{intervention}} = S(X; Y) - S(X_{\text{masked}}; Y)$$
2. **Faithfulness Gap**:
   $$\text{Faithfulness Gap} = \Delta S_{\text{top}} - \Delta S_{\text{bottom}}$$
3. **Win Rate vs. Bottom Control**:
   $$\text{Win Rate} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\Delta S_{\text{top}, i} > \Delta S_{\text{bottom}, i})$$

---

## 7. Statistical Testing & Significance Thresholds

- **Rank Correlation**: Spearman rank correlation ($r_s$) computed using `scipy.stats.spearmanr`.
- **Linear Correlation**: Pearson correlation ($r$) computed using `scipy.stats.pearsonr`.
- **Significance Level**: $\alpha = 0.05$.
- **Multiple Hypothesis Testing Correction**: Benjamini–Hochberg False Discovery Rate (FDR) applied across exploratory tokenization-scoring correlation batches.
- **Confidence Intervals**: 95% Fisher $z$-transformed confidence intervals for correlation coefficients.

---

## 8. Reproduction Pipeline Commands

To execute the entire analysis pipeline from the root directory:

```bash
# 1. Dataset validation and audit
python -m src.data.validate_dataset

# 2. Subword tokenization extraction
python -m src.tokenization.analyze_tokenization

# 3. Pseudo-log-likelihood scoring
python -m src.scoring.compute_pll

# 4. Combined statistical analysis and Figures 1-5
python -m src.analysis.combined_analysis

# 5. Visible-target stability and faithfulness evaluation
python -m src.explainability.evaluate_faithfulness

# 6. Generate Figures 6, 7, and 8
python -m src.analysis.generate_paper_figures
```
