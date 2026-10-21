# Stage 3: Pretrained Model PLL Scoring

## 1. Overview & Research Objective

This stage implements the **Pseudo-Log-Likelihood (PLL)** scoring pipeline for the **Hindi Lexical Sensitivity** research project. The objective is to investigate how pretrained masked language models (MLMs) evaluate token predictability between original and contrast Hindi Devanagari sentence pairs across three experimental categories:

1. **Contextual Lexical Contrasts (`VALID_WORD_CONTEXT`)**: Meaning-altering lexical substitutions.
2. **Synonym Substitutions (`SYNONYM_SUBSTITUTION`)**: Meaning-preserving lexical variants.
3. **Realistic Spelling Errors (`REALISTIC_SPELLING_ERROR`)**: Orthographic noise and spelling mistakes.

### Pretrained MLMs Evaluated

| Model Identifier | Architecture | Vocabulary Size | Scoring Mask Token |
|---|---|---|---|
| `FacebookAI/xlm-roberta-base` | RoBERTa-based MLM | 250,002 | `<mask>` (`id=250001`) |
| `google/muril-base-cased` | BERT-based MLM | 197,285 | `[MASK]` (`id=103`) |

---

## 2. Pseudo-Log-Likelihood (PLL) Scoring Methodology

### A. Mathematical Formulation
Given an input sentence tokenized into $W = (w_0, w_1, \ldots, w_{n-1})$:

1. **Ordinary Token Identification**:
   $$I = \{ i \in [0, n-1] \mid w_i \notin \text{SpecialTokens} \cup \{\text{PAD}\} \}$$
   Sequence boundary delimiters (`<s>`, `</s>`, `[CLS]`, `[SEP]`) and padding tokens are strictly excluded from masking and scoring.

2. **Iterative Masking**:
   For each ordinary token position $i \in I$, construct a sequence $W^{\setminus i}$ where position $i$ is replaced by the model's designated mask token:
   $$W^{\setminus i} = (w_0, \ldots, w_{i-1}, \text{[MASK]}, w_{i+1}, \ldots, w_{n-1})$$

3. **Numerically Stable Log-Probability**:
   Pass $W^{\setminus i}$ through the MLM in inference mode (`torch.inference_mode()`). For logits vector $z \in \mathbb{R}^V$ at position $i$:
   $$\log P(w_i \mid W^{\setminus i}) = \text{log\_softmax}(z)[w_i] = z_{w_i} - \log \sum_{v=1}^{V} \exp(z_v)$$

4. **Sentence PLL Sum & Mean Log-Probability**:
   $$\text{PLL}(W) = \sum_{i \in I} \log P(w_i \mid W^{\setminus i})$$
   $$\text{MeanLogP}(W) = \frac{\text{PLL}(W)}{|I|}$$

5. **Signed Differences ($\Delta = \text{Contrast} - \text{Original}$)**:
   $$\Delta \text{PLL} = \text{PLL}(\text{Contrast}) - \text{PLL}(\text{Original})$$
   $$\Delta \text{MeanLogP} = \text{MeanLogP}(\text{Contrast}) - \text{MeanLogP}(\text{Original})$$
   $$\Delta \text{ScoredTokens} = |I_{\text{Contrast}}| - |I_{\text{Original}}|$$

### B. Cross-Model Comparability Guardrail
> [!IMPORTANT]
> **Raw PLL sums must never be compared directly across different model architectures or vocabularies.**
> Differences in vocabulary size ($V = 250,002$ vs. $V = 197,285$) and tokenization segmentation policies mean raw log-likelihood scales are model-specific. All research comparisons are conducted **within-model** using signed differences ($\Delta = \text{Contrast} - \text{Original}$) and length-normalized mean log-probabilities.

---

## 3. Memory Safety & Hardware Optimization for 8 GB RAM

Because the workstation runs Windows 11 with 8 GB RAM and CPU-only PyTorch:
1. **Single Model Lifetime**: Only one model is loaded into memory at any time. When all sentence pairs for a model are scored, the model and tokenizer tensors are explicitly deleted and Python's garbage collector (`gc.collect()`) is invoked before loading the subsequent model.
2. **Batched Masked Inference**: For each sentence, masked variations are chunked into conservative batches (default `batch_size = 16`). Forward passes execute efficiently within ~100–150 MB peak memory.
3. **Resumable Checkpointing**: Results are incrementally written to `outputs/scoring/checkpoint.json`. If execution is paused or interrupted, re-running automatically resumes from the last completed sentence pair without data loss or duplicate computations.
4. **Pilot Execution Mode**: A dedicated `--pilot` flag scores a balanced subset across all tracks and preserves the rejected item to verify memory and runtime practicality prior to full-dataset scoring.

---

## 4. Dataset Preservation & Discrepancy Investigation

### A. Raw Row Preservation
- All 200 non-empty dataset rows are scored and retained.
- The single rejected item (`HC_0047`, `final_status=reject`) is preserved.
- All six annotator/adjudication disagreement items (`HC_0087`, `HC_0099`, `HC_0138`, `HC_0172`, `HC_0180`, `HC_0186`) are retained without automated overrides.

### B. Investigation of Trailing Empty Rows Discrepancy
- **Observation**: The initial standard library XML parser reported 1000 total rows (200 data rows + 799 trailing empty rows), whereas `openpyxl` reports 201 total rows (200 data rows + 0 trailing empty rows).
- **Technical Explanation**:
  The raw Excel worksheet (`xl/worksheets/sheet1.xml`) specifies the populated bounding box in `<dimension ref="A1:R201"/>`.
  - When `openpyxl` reads the file with `read_only=True`, it strictly adheres to the worksheet dimension bounds ($A1:R201$).
  - When the file was parsed directly via standard library XML (`xml.etree.ElementTree`), it iterated through all `<row>` XML elements up to row 1000, which Excel generated during template creation with default grid properties.
  - In both parsing environments, the actual non-empty data rows are **exactly 200 rows** (`HC_0001` through `HC_0200`).
  - Input file SHA-256 fingerprint verified: `f9a5af81e21624de47ead7feabba412149939c1eb9934f8fc2b8d452a232cd44`.

---

## 5. Interpretive Guardrails

> [!WARNING]
> Pseudo-Log-Likelihood (PLL) scores reflect how predictably token sequences fit within the statistical distributions of a model's pretraining corpus. They do **not** constitute proof of grammatical correctness, linguistic acceptability, or semantic truth. Lower likelihoods for spelling errors indicate statistical rarity rather than human normative judgment.
