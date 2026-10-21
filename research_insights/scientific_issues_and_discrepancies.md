# Scientific Issues, Discrepancies, and Verification Report

This document records the rigorous forensic audit of all conflicting identifiers, mathematical discrepancies, methodological artifacts, and data anomalies identified across the Devanagari Hindi Lexical Sensitivity codebase and reports.

---

## 1. Discrepancy Investigation: Conflicting Rejected-Pair Identifiers

### The Conflict Across Project Artifacts
Different project documents report conflicting pair identifiers as the single rejected dataset record:
- **`outputs/paper_analysis/COMPLETE_RESEARCH_RESULTS_ANALYSIS.md` (Section 2.1, Line 56)**:  
  Reports `final_status = reject`: 1 pair (0.5%) — Row `HC_0185`.
- **`docs/stage2_tokenization.md` (Section 2.D, Line 41)** and **`docs/stage3_model_scoring.md` (Section 4.A, Line 67)**:  
  Reports: "The single rejected item (`HC_0047`, `final_status=reject`) is preserved."
- **`outputs/dataset_validation_report.json`**:  
  Reports: `final_status_distribution: { "accept": 199, "reject": 1 }`.
- **`outputs/analysis/analysis_report.md` (Section 1, Line 36)**:  
  Reports: "`final_status=reject`: 2 records preserved in all analyses."

### Forensic Verification from Ground Truth
To resolve this contradiction definitively, we queried the raw Excel dataset (`data/raw/hindi_lexical_sensitivity_cleaned.xlsx`, SHA-256 `f9a5af81e21624de47ead7feabba412149939c1eb9934f8fc2b8d452a232cd44`) and the processed master records (`outputs/dashboard_data/pair_level_master.csv`).

| Candidate Pair ID | Excel Row (0-indexed) | Track | Target Original / Contrast | Annotator 1 | Annotator 2 | Adjudication | Final Status | Verified Status |
|---|---|---|---|---|---|---|---|---|
| **`HC_0178`** | Row 177 | `VALID_WORD_CONTEXT` | `सलाह` / `सलाई` | `reject` | `reject` | `reject` | **`reject`** | **THE ONLY TRUE REJECTED RECORD** |
| **`HC_0185`** | Row 184 | `VALID_WORD_CONTEXT` | `भर` / `पढ़` | `accept` | `accept` | `accept` | **`accept`** | Erroneously cited in `COMPLETE_RESEARCH_RESULTS_ANALYSIS.md` |
| **`HC_0047`** | Row 46 | `VALID_WORD_CONTEXT` | `बाँधी` / `बाँटी` | `accept` | `accept` | `accept` | **`accept`** | Erroneously cited in `docs/stage2` and `stage3` |

### Forensic Explanation of How These Errors Arose:
1. **`HC_0178` is the true rejected record**: In the source spreadsheet, `HC_0178` ("पिताजी ने मुझे एक सलाह दी।" vs. "पिताजी ने मुझे एक सलाई दी।") was marked `reject` by Annotator 1, Annotator 2, and the adjudicator, and assigned `final_status = reject`.
2. **The `HC_0185` error in Stage 8 analysis**: The author of `COMPLETE_RESEARCH_RESULTS_ANALYSIS.md` likely misread row indices or transcribed the identifier from memory (both `HC_0178` and `HC_0185` belong to the same late section of `VALID_WORD_CONTEXT`).
3. **The `HC_0047` error in Stage 2/3 docs**: The authors of the early markdown documentation drafted the placeholder text using an earlier draft or pilot row index (`HC_0047`), which is actually an accepted record.
4. **Resolution Rule**: In all paper manuscripts, presentations, and final reports, **`HC_0178` MUST be cited as the sole rejected record**. `HC_0047` and `HC_0185` are fully accepted records.

---

## 2. Discrepancy Investigation: "2 Records Rejected" vs. "1 Pair Rejected"

### The Issue
- `outputs/analysis/analysis_report.md` (Section 1 and Section 3) states:
  > "`final_status=reject`: 2 records preserved in all analyses."
  > "2 records have `final_status=reject`. Results below exclude these."
- Meanwhile, `dataset_validation_report.json` states that only 1 pair has `final_status=reject`.

### Forensic Resolution
This is a **unit-of-analysis conflation**:
- In the source dataset, there are **200 unique sentence pairs**, exactly **1** of which is rejected (`HC_0178`).
- In Stage 4 (`src/analysis/combined_analysis.py`), the dataset is merged with model evaluations across **2 models** (`xlm-roberta-base` and `muril-base-cased`), producing a combined table of **400 rows** (`outputs/analysis/combined_pair_level_results.csv`).
- In this 400-row table, pair `HC_0178` generates **2 evaluation rows** (one row for XLM-RoBERTa and one row for MuRIL).
- In Section 3 of `analysis_report.md`, the sensitivity table shows:
  - `VALID_WORD_CONTEXT`: $N=142$ (with rejections) $\to$ $N=141$ (excluding rejections) for each model.
  - Exactly **1 pair** is subtracted per model.
- **Clarification**: The dataset contains **1 rejected pair** (0.5% of pairs), which corresponds to **2 model evaluation rows** in the combined analysis.

---

## 3. Discrepancy Investigation: Trailing Empty Rows in Excel Parsing

### The Issue
- An early XML parsing script reported 1000 total rows (200 populated rows + 799 empty trailing rows).
- `openpyxl` reports 201 total rows (1 header + 200 data rows).

### Forensic Resolution
- Inspecting `xl/worksheets/sheet1.xml` inside the `.xlsx` zip archive confirms:
  - The worksheet dimension is strictly defined as `<dimension ref="A1:R201"/>`.
  - Excel's internal XML generator allocated default `<row>` elements up to index 1000 without data values.
  - Python's standard `xml.etree.ElementTree` iterates over all declared XML `<row>` tags, encountering 799 empty elements.
  - Standard spreadsheet libraries (`openpyxl`, `pandas`) correctly respect the bounding dimension $A1:R201$.
- **Conclusion**: There are exactly **200 populated data rows** (`HC_0001` through `HC_0200`). Zero data rows are missing.

---

## 4. Discrepancy Investigation: Human Review Disagreement Identifiers

### The Issue
- `docs/stage2_tokenization.md` (Line 42) lists:
  > `HC_0087`, `HC_0099`, `HC_0138`, `HC_0172`, `HC_0180`, `HC_0187`
- `outputs/dataset_validation_report.json` lists:
  > `HC_0087`, `HC_0099`, `HC_0138`, `HC_0172`, `HC_0180`, `HC_0186`

### Forensic Verification
Querying the raw dataset confirms:
- **`HC_0186`**: `annotator_1 = reject`, `annotator_2 = reject`, `adjudication = reject`, `final_status = accept`. (A true disagreement override).
- **`HC_0187`**: `annotator_1 = accept`, `annotator_2 = accept`, `adjudication = accept`, `final_status = accept`. (Full unanimous agreement).
- **Conclusion**: `HC_0187` in `stage2_tokenization.md` was a clerical typo for `HC_0186`. The verified set of 6 disagreement pairs is:
  $$\{\text{HC\_0087}, \text{HC\_0099}, \text{HC\_0138}, \text{HC\_0172}, \text{HC\_0180}, \text{HC\_0186}\}$$

### Linguistic Nature of the Overrides
In all 6 disagreement pairs, both annotators and the adjudicator marked `reject` because the contrast sentence was formed by substituting a full valid word rather than a minimal morphological tweak, but both sentences remained grammatically valid Hindi sentences. The dataset curator retained them with `final_status = accept` under `VALID_WORD_CONTEXT` to test contextual sensitivity between independent valid sentences.

---

## 5. Methodological Artifact: Stage 6 Masked-Target Stability Artifact

### Root Cause
In Stage 6, the evaluation protocol computed the stability of context token attributions while predicting the canonical target word at a masked position. However, in the `REALISTIC_SPELLING_ERROR` track, the lexical perturbation was applied *strictly inside the target word*.

### Mathematical Consequence
When the target word was masked with `[MASK]`:
$$\text{Input}(X_{\text{orig}}) = [w_1, \dots, \text{[MASK]}, \dots, w_n]$$
$$\text{Input}(X_{\text{cont}}) = [w_1, \dots, \text{[MASK]}, \dots, w_n]$$
Because the surrounding context was identical and the target word was replaced by `[MASK]` in both inputs, $\text{Input}(X_{\text{orig}}) \equiv \text{Input}(X_{\text{cont}})$.
The neural network performed the identical forward and backward pass, producing:
$$\Delta \text{Logit} = 0.0, \quad \text{Spearman } \rho = 1.0000, \quad \text{Cosine Sim} = 1.0000 \quad (\text{Variance} = 0)$$

### Resolution in Stage 7
Stage 7 implemented the **Visible-Target Scoring Protocol**:
- Target tokens remain visible in the input sequence.
- Target score is defined as the mean logit of target subwords.
- Stability is evaluated across aligned context tokens while the spelling variation exists in the input.
- This revealed genuine attribution variation ($\rho \in [-0.80, +1.00]$).
- **Paper Writing Guidance**: The Stage 6 artifact must be documented explicitly in the paper as a cautionary methodological case study in XAI benchmark design.

---

## 6. Mathematical Phenomenon: XLM-RoBERTa Target Logit Inflation under Context Erasure

### Observation
In XLM-RoBERTa, removing context tokens caused unnormalized target logits to *increase* relative to baseline, producing negative raw logit drops:
$$\Delta S_{\text{top}} = S_0 - S_{\text{top\_mask}} = -7.1884 \quad (f=0.20)$$
$$\Delta S_{\text{bottom}} = S_0 - S_{\text{bottom\_mask}} = -11.2055 \quad (f=0.20)$$

### Resolution
This phenomenon is caused by **language model out-of-distribution (OOD) dynamics**:
- Replacing unmasked context tokens with `[MASK]` disrupts global sentence syntax.
- In XLM-RoBERTa's unnormalized logit space, the reduction of contextual competition caused raw target logits to float higher across all masking conditions.
- However, **directional faithfulness was strictly preserved**:
  $$\text{Faithfulness Gap} = \Delta S_{\text{top}} - \Delta S_{\text{bottom}} = (-7.1884) - (-11.2055) = \mathbf{+4.0171} > 0$$
  Top-k masking resulted in a significantly lower target logit than bottom-k control masking ($S_{\text{top}} < S_{\text{bottom}}$), yielding a **77.5% win rate**.
- **Reporting Rule**: Always report the Faithfulness Gap and Win Rate alongside raw drops to prevent confusion.

---

## 7. Sample Size & Subword Eligibility Constraints

### Strict Single-Subword vs. Pooled Multi-Subword
Due to Devanagari subword fragmentation, requiring both original and contrast target words to segment into exactly 1 subword severely depletes sample size in the spelling track:
- **MuRIL**: 14 of 20 pairs fracture into multi-subword targets; only **6 pairs (30%)** are strictly single-subword eligible.
- **XLM-RoBERTa**: 16 of 20 pairs fracture; only **4 pairs (20%)** are strictly single-subword eligible.

### Methodological Tradeoff
1. **Strict Single-Subword Evaluation**: Eliminates any ambiguity in scalar target logit assignment, but reduces statistical power to near-negligible levels ($N=6$ and $N=4$).
2. **Multi-Subword Mean-Logit Pooling**: Defines target score as the arithmetic mean across subwords:
   $$S(X; Y_{\text{target}}) = \frac{1}{k} \sum_{j=1}^k \text{Logit}(p_j, y_j)$$
   Restores full sample size ($N=20$ pairs, 40 sentences per model).
3. **Audit Outcome**: Comparing stability and faithfulness metrics between strict single-subword and pooled all-scorable subsets reveals consistent empirical directions (e.g., Attention stability $\rho = 0.74$ pooled vs. $0.69$ strict in XLM-R; Faithfulness win rates $77.5\%$ pooled vs. $75.0\%$ strict). The pooled protocol is empirically validated, but results must always be presented alongside strict subset numbers.

---

## 8. Linguistic Nuance: Orthographic Variants vs. True Spelling Errors

### The Issue
Two pairs in the `REALISTIC_SPELLING_ERROR` track represent legitimate, accepted orthographic variants in contemporary Devanagari Hindi rather than erroneous misspellings:
1. **`HC_0118`**: `महत्वपूर्ण` vs. `महत्त्वपूर्ण` (single vs. double $t$ ligature; both standard in Hindi publishing).
2. **`HC_0120`**: `संभावना` vs. `सम्भावना` (anusvara `ं` vs. class nasal `म्`; both prescribed by Hindi orthographic standards).

### Model Response
- Both models assigned minimal PLL score drops to these two pairs ($\Delta \text{MeanLogP} > -0.2$), whereas true spelling errors like `परीक्षा` $\to$ `परिक्षा` produced massive drops ($\Delta < -2.0$).
- **Scientific Implication**: Pretrained MLMs demonstrate nuanced sensitivity: they penalize ungrammatical/deviant orthography heavily while tolerating standard orthographic alternations.

---

## 9. Hierarchy of Evaluated Units and Denominators

To prevent conflation across paper sections, the following table defines the exact denominators used in the project:

| Unit of Analysis | Count / Denominator | Context of Usage |
|---|---|---|
| **Source Dataset Pairs** | $N=200$ pairs | Total curated Hindi sentence pairs (`data/raw/`) |
| **Track: Valid Word Context** | $N=142$ pairs | 71.0% of dataset pairs |
| **Track: Synonym Substitution** | $N=38$ pairs | 19.0% of dataset pairs |
| **Track: Realistic Spelling Error** | $N=20$ pairs | 10.0% of dataset pairs |
| **Model Evaluations (Full Dataset)** | $N=400$ evaluations | 200 pairs $\times$ 2 models (`outputs/scoring/`, `outputs/tokenization/`) |
| **Stability Evaluations (Spelling Track)** | $N=20$ pairs per model | Visible-target stability evaluated per pair (`outputs/hindi_explanation_stability/`) |
| **Faithfulness Sentence Evaluations** | $N=40$ sentences per model | 20 pairs $\times$ 2 sentences (original & contrast) (`outputs/hindi_faithfulness/`) |
| **Raw Faithfulness Interventions** | $N=1,440$ rows | 40 sentences $\times$ 2 methods $\times$ 3 fractions $\times$ 3 mask types $\times$ 2 models |
| **Strict Single-Subword Pairs** | $N=6$ (MuRIL), $N=4$ (XLM-R) | Non-fragmented spelling pairs |
| **Accepted Pairs in Dataset** | $N=199$ pairs | 99.5% of dataset |
| **Rejected Pairs in Dataset** | $N=1$ pair (`HC_0178`) | 0.5% of dataset (2 rows in 400-row model table) |
| **Annotation Disagreements** | $N=6$ pairs | Double-reject overridden to accept |
