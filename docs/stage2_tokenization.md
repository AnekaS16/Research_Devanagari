# Stage 2: Hindi Tokenization Experiment

## 1. Overview & Research Objective

This stage implements the tokenization evaluation pipeline for the **Hindi Lexical Sensitivity** research project. The goal is to study how two multilingual pretrained masked-language-model tokenizers segment Hindi Devanagari sentence pairs across three contrast categories:

1. **Contextual Lexical Contrasts (`VALID_WORD_CONTEXT`)**: Valid Hindi vocabulary substitutions that alter sentence semantics.
2. **Synonym Substitutions (`SYNONYM_SUBSTITUTION`)**: Meaning-preserving lexical variants.
3. **Realistic Spelling Errors (`REALISTIC_SPELLING_ERROR`)**: Common orthographic deviations in Devanagari script.

### Pretrained Tokenizers Evaluated

| Model Identifier | Tokenizer Architecture | Vocabulary Size | Special Tokens Scheme |
|---|---|---|---|
| `FacebookAI/xlm-roberta-base` | SentencePiece BPE (`XLMRobertaTokenizerFast`) | 250,002 | `<s>` (0), `</s>` (2) |
| `google/muril-base-cased` | WordPiece (`BertTokenizerFast`) | 197,285 | `[CLS]` (104), `[SEP]` (105) |

---

## 2. Experimental Methodology

### A. Subword Tokenization & Count Definitions
For every sentence pair, we distinguish two count metrics:
- **Special-Token-Inclusive Count**: Total tokens returned by the tokenizer, including sequence delimiters (`<s>`/`</s>` or `[CLS]`/`[SEP]`).
- **Special-Token-Exclusive Count**: Only non-special subword pieces representing the actual Hindi text.

### B. Signed Token Count Differences
All differences are computed strictly as:
$$\Delta_{\text{sentence}} = \text{Count}(\text{Contrast Sentence}) - \text{Count}(\text{Original Sentence})$$
$$\Delta_{\text{target}} = \text{Count}(\text{Contrast Target Word}) - \text{Count}(\text{Original Target Word})$$

### C. Target-Word Alignment Policy
Target words are extracted using fast tokenizer character offset mappings (`offset_mapping`):
- **`unique`**: The target word appears exactly once in the sentence and corresponds to a contiguous span of non-special subword tokens.
- **`ambiguous`**: The target appears multiple times in the sentence, or maps to non-contiguous token pieces. No alignment is guessed.
- **`missing`**: The target word is not present in the sentence.
- **`unsupported`**: Tokenizer does not provide reliable character offsets.

### D. Dataset Preservation & Human Review Integrity
- **100% Data Preservation**: All 200 non-empty data rows are evaluated and saved.
- **Rejection Status Preserved**: Row `HC_0047` (`final_status=reject`) is retained and clearly tagged.
- **Annotation Disagreements Preserved**: All 6 rows where annotator adjudication differed from final status (`HC_0087`, `HC_0099`, `HC_0138`, `HC_0172`, `HC_0180`, `HC_0187`) are kept intact without automated resolution.
- **Interpretive Guardrail**: Subword token counts reflect dictionary coverage and segmentation heuristics; they do **not** imply grammatical validity or semantic coherence.

---

## 3. Empirical Results Across Tracks

Results on the 200 sentence pairs (400 model evaluations):

| Model | Track | Pairs | Mean Sentence $\Delta$ (Inclusive) | Sentence $\Delta$ Range | Mean Target $\Delta$ |
|---|---|---|---|---|---|
| `FacebookAI/xlm-roberta-base` | `REALISTIC_SPELLING_ERROR` | 20 | **+1.00** | [0, +2] | **+1.00** |
| `FacebookAI/xlm-roberta-base` | `SYNONYM_SUBSTITUTION` | 38 | **+0.61** | [-1, +4] | **+0.63** |
| `FacebookAI/xlm-roberta-base` | `VALID_WORD_CONTEXT` | 142 | **+0.17** | [-1, +2] | **+0.17** |
| `google/muril-base-cased` | `REALISTIC_SPELLING_ERROR` | 20 | **+1.00** | [0, +2] | **+1.00** |
| `google/muril-base-cased` | `SYNONYM_SUBSTITUTION` | 38 | **+0.18** | [0, +2] | **+0.18** |
| `google/muril-base-cased` | `VALID_WORD_CONTEXT` | 142 | **+0.08** | [-2, +2] | **+0.08** |

### Key Findings
1. **Orthographic Fragmentation**: For both models, realistic Devanagari spelling errors increase the subword count by exactly +1.00 token on average. Misspelled characters break pre-existing subwords into character-level fragments.
2. **Synonym Sensitivity**: XLM-RoBERTa is significantly more sensitive to synonym substitution ($\Delta = +0.61$) than MuRIL ($\Delta = +0.18$), reflecting MuRIL's specialized training vocabulary for Indian languages.
3. **Alignment Precision**: 200 / 200 sentence pairs achieved `unique` alignment on both models (100% alignment coverage).

---

## 4. Reproducibility & Safe Caching

Outputs are stored under `outputs/tokenization/`:
- `tokenization_results.csv`: Complete tidy dataset (400 rows, 39 columns).
- `run_metadata.json`: SHA-256 fingerprint, model versions, library versions, and execution summary.
- `summary.md`: Human-readable markdown summary.

The cache verifies:
1. `input_file.sha256` matching the raw Excel file hash.
2. Model list and configurations matching current parameters.
3. `completed_successfully == true`.
If any mismatch occurs, outputs are recomputed deterministically.
