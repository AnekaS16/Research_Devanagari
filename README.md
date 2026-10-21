# Beyond Tokenization: Spelling Perturbations, Explanation Stability and Faithfulness in Devanagari Hindi MLMs

**An exploratory NLP research project investigating how Hindi spelling variations affect subword tokenization, masked language model scoring, and the stability and faithfulness of model explanations.**

<p align="center">

![Language](https://img.shields.io/badge/Language-Hindi%20%7C%20Devanagari-orange)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![Models](https://img.shields.io/badge/Models-XLM--RoBERTa%20%7C%20MuRIL-purple)
![Research](https://img.shields.io/badge/Project-Research%20%7C%20NLP-green)
![Status](https://img.shields.io/badge/Status-Exploratory-yellow)

</p>

---

## 📌 Table of Contents

- [1. Overview](#-1-overview)
- [2. The Problem](#-2-the-problem)
- [3. Research Questions](#-3-research-questions)
- [4. What Makes This Study Interesting?](#-4-what-makes-this-study-interesting)
- [5. Dataset](#-5-dataset)
- [6. Models and Technologies](#-6-models-and-technologies)
- [7. Research Pipeline](#-7-research-pipeline)
- [8. Project Structure](#-8-project-structure)
- [9. Key Results](#-9-key-results)
- [10. Understanding the Metrics](#-10-understanding-the-metrics)
- [11. Installation and Setup](#-11-installation-and-setup)
- [12. Running the Experiments](#-12-running-the-experiments)
- [13. Interactive Dashboard](#-13-interactive-dashboard)
- [14. Output Files](#-14-output-files)
- [15. Important Limitations](#-15-important-limitations)
- [16. Reproducibility](#-16-reproducibility)
- [17. Future Work](#-17-future-work)
- [18. Citation](#-18-citation)
- [19. Acknowledgements](#-19-acknowledgements)

---

## 🧠 1. Overview

How much can a small spelling change affect a language model?

Consider these Hindi words:

- **परीक्षा** — the standard spelling of “examination”
- **परिक्षा** — a common spelling error

A human reader may understand that the writer intended the same word. However, a pretrained language model may tokenize these two forms differently, assign them different scores, and produce different explanations for its behaviour.

This research investigates these effects in **Devanagari Hindi** using two pretrained masked language models:

1. **XLM-RoBERTa**
2. **MuRIL**

The project studies four related but distinct properties:

1. **Tokenization:** How is a Hindi word divided into subword tokens?
2. **Model predictability:** How does the model score a sentence containing a spelling variation?
3. **Explanation stability:** Do the model's token-level explanations remain similar after a spelling change?
4. **Explanation faithfulness:** Do the tokens highlighted as important actually affect the model's output under a controlled intervention?

The work combines dataset analysis, tokenization measurements, pseudo-log-likelihood scoring, attention-based explanations, Integrated Gradients, and masking-based faithfulness evaluation.

> **Important:** This is an exploratory research study. The dataset contains 200 sentence pairs, but the explanation stability and faithfulness experiments focus on only 20 spelling-error pairs. The results should therefore not be treated as universal conclusions about Hindi or all language models.

## 🔍 2. The Problem

Modern language models often use **subword tokenization** instead of treating every word as a single indivisible unit.

For example, a tokenizer might represent a word using one token and a spelling variant using several tokens. The actual segmentation depends on the tokenizer and must be checked experimentally.

This creates several research questions.

### Problem A: Tokenization sensitivity

Can a small orthographic change cause a target word to be split into more subword tokens?

### Problem B: Scoring sensitivity

Does the model assign a lower pseudo-log-likelihood score to a spelling variant than to its original form?

### Problem C: Explanation stability

If the input changes slightly, do the tokens identified as important by the explanation method change as well?

### Problem D: Explanation faithfulness

Even if an explanation remains stable, does masking its highly ranked tokens affect the model's output more than masking less important tokens?

The central idea is that **a stable explanation is not automatically a faithful explanation**.

---

## 🎯 3. Research Questions

This project investigates the following questions:

**RQ1 — Tokenization:** How does spelling variation affect target-word subword fragmentation in XLM-RoBERTa and MuRIL?

**RQ2 — Predictability:** How do pseudo-log-likelihood scores differ across valid-word contextual contrasts, synonym substitutions, and realistic spelling errors?

**RQ3 — Stability:** How stable are attention-based and Integrated Gradients explanations across original and spelling-perturbed Hindi sentences?

**RQ4 — Faithfulness:** Is explanation stability associated with faithfulness gaps measured using top-ranked and bottom-ranked context-token masking?

The final question is exploratory because the explanation experiments use a small sample.

---

## 💡 4. What Makes This Study Interesting?

Many NLP evaluations focus on model accuracy or prediction scores. This project also examines how model explanations behave when the input spelling changes.

Its main contributions are:

- A curated and audited collection of Hindi sentence pairs for studying lexical and orthographic variation.
- Comparative measurements of target-word subword fragmentation and pseudo-log-likelihood.
- An implemented protocol for comparing attention and Integrated Gradients explanations.
- An exploratory analysis of the relationship between explanation stability and intervention-based faithfulness.
- Documentation of methodological limitations, including target-masking artefacts and the effects of masking tokens.

The study does not claim that the proposed dataset or evaluation approach solves the broader problem of explainability. Instead, it provides an experimental starting point for investigating these questions in Hindi NLP.

---

## 📚 5. Dataset

The dataset contains **200 Hindi sentence pairs, corresponding to 400 sentences**.

| Pair category | Number of pairs | Percentage |
|---|---:|---:|
| Valid-word contextual contrasts | 142 | 71% |
| Synonym substitutions | 38 | 19% |
| Realistic spelling errors | 20 | 10% |
| **Total** | **200** | **100%** |

### What do the categories mean?

**1. Valid-word contextual contrasts**

Both alternatives can be valid Hindi words, but their contextual suitability may differ.

**2. Synonym substitutions**

A word is replaced with a related or synonymous alternative to examine changes in model scoring.

**3. Realistic spelling errors**

A spelling variant is compared with its original form. This category is used for the exploratory explanation experiments.

### Dataset file

The cleaned dataset used during development is:

`data/raw/hindi_lexical_sensitivity_cleaned.xlsx`

The dataset should be interpreted according to its annotation guidelines and final adjudication records. It is a constructed research resource, not a representative sample of all Hindi text.

---

## 🧰 6. Models and Technologies

### Language models

| Model | Purpose |
|---|---|
| XLM-RoBERTa base | Multilingual masked language model used for tokenization and scoring experiments |
| MuRIL base | Masked language model designed for Indian-language representation learning |

Model identifiers:

- `FacebookAI/xlm-roberta-base`
- `google/muril-base-cased`

### Research methods

| Method | Purpose |
|---|---|
| Subword tokenization analysis | Measures how target-word segmentation changes |
| Pseudo-log-likelihood (PLL) | Measures contextual token predictability |
| Attention | Produces token-level importance scores from model attention |
| Integrated Gradients (IG) | Attributes a model output to input representations |
| Spearman rank correlation | Measures monotonic association between stability and faithfulness |
| Cosine similarity | Compares attribution vectors |
| Masking interventions | Tests how the target score changes when selected context tokens are masked |

### Development technologies

- Python 3.12
- PyTorch
- Hugging Face Transformers
- Tokenizers
- Captum for Integrated Gradients
- Pandas and NumPy for data processing
- SciPy for statistical analysis
- React, TypeScript, and Vite for the interactive dashboard
- Matplotlib / other configured plotting tools for research visualizations

The dashboard is a visualization interface for the generated research outputs; it is not itself a language model.

---

## 🔬 7. Research Pipeline

The project follows a multi-stage experimental pipeline.

```text
Hindi Sentence-Pair Dataset
            |
            v
Dataset Cleaning and Annotation Audit
            |
            v
Target-Word Tokenization Analysis
            |
            v
Pseudo-Log-Likelihood Scoring
            |
            v
Pair-Level Result Integration
            |
            v
Attention and Integrated Gradients
            |
            v
Explanation Stability Analysis
            |
            v
Masking-Based Faithfulness Evaluation
            |
            v
Stability–Faithfulness Correlation
            |
            v
Research Tables, Figures and Dashboard
```

### Stage 1 — Dataset audit

Inspect the Hindi sentence pairs, their categories, target words, and annotation records.

### Stage 2 — Tokenization analysis

Tokenize the original and contrast sentences using both models. Measure target-word token counts and single-subword coverage.

### Stage 3 — PLL scoring

Calculate pseudo-log-likelihood and mean token log-probability to compare the model's scoring behaviour.

### Stage 4 — Combined analysis

Integrate tokenization and scoring outputs at the pair level. Calculate descriptive statistics and correlations.

### Stage 5 — Explanation generation

Generate attention-based and Integrated Gradients attribution scores for eligible examples.

### Stage 6 — Stability analysis

Compare explanations for original and spelling-perturbed sentences using aligned context tokens and similarity metrics.

### Stage 7 — Faithfulness analysis

Mask selected context tokens at different fractions and compare changes in the visible target's logit score.

### Stage 8 — Dashboard preparation

Prepare analysis summaries, pair-level data, and eligibility information for interactive exploration.

### Stage 9 — Visualization and reporting

Use the outputs to create research figures, tables, and a summary of findings and limitations.

---

## 📁 8. Project Structure

The following tree summarises the major project paths used in the experiments. Some individual filenames may vary with the current repository version.

```text
hindi-lexical-sensitivity/
│
├── data/
│   └── raw/
│       └── hindi_lexical_sensitivity_cleaned.xlsx
│
├── src/
│   ├── explanations/
│   │   └── hindi_explanations.py
│   │
│   └── analysis/
│       ├── explanation_stability.py
│       └── hindi_faithfulness.py
│
├── outputs/
│   ├── tokenization/
│   │   ├── tokenization_results.csv
│   │   └── summary and metadata files
│   │
│   ├── scoring/
│   │   ├── pll_scoring_results.csv
│   │   └── summary and metadata files
│   │
│   ├── analysis/
│   │   ├── combined_pair_level_results.csv
│   │   ├── descriptive_statistics.csv
│   │   ├── correlation_results.csv
│   │   ├── model_comparison.csv
│   │   └── analysis_report.md
│   │
│   ├── hindi_explanations/
│   │
│   ├── hindi_explanation_stability/
│   │
│   ├── hindi_faithfulness/
│   │
│   └── dashboard_data/
│       ├── dataset_overview.json
│       ├── dataset_tracks.csv
│       ├── tokenization_summary.csv
│       ├── scoring_summary.csv
│       ├── stability_summary.csv
│       ├── faithfulness_summary.csv
│       ├── pair_level_master.csv
│       ├── eligibility_master.csv
│       ├── scientific_audit_report.md
│       └── manifest.json
│
├── dashboard/
│   ├── package.json
│   └── ...
│
├── README.md
└── ...
```

### What do the main folders contain?

| Path | Purpose |
|---|---|
| `data/raw/` | The cleaned Hindi sentence-pair dataset |
| `src/explanations/` | Code for generating token-level explanations |
| `src/analysis/` | Code for stability and faithfulness evaluation |
| `outputs/tokenization/` | Tokenization results and summaries |
| `outputs/scoring/` | PLL scores and scoring metadata |
| `outputs/analysis/` | Combined pair-level data and statistical summaries |
| `outputs/hindi_explanations/` | Generated explanation outputs |
| `outputs/hindi_explanation_stability/` | Explanation stability measurements |
| `outputs/hindi_faithfulness/` | Masking interventions and faithfulness summaries |
| `outputs/dashboard_data/` | Processed files used by the dashboard |
| `dashboard/` | React-based research visualization interface |

**Note:** This tree describes the known research structure. Check the actual GitHub repository before treating the tree as an exhaustive inventory. A folder may be excluded from Git or contain additional files.

---

## 📊 9. Key Results

The following are the reported results from the current experimental run.

### 9.1 Tokenization

For the 20 spelling-error pairs, single-subword target coverage changed as follows:

| Model | Original spelling | Perturbed spelling | Change |
|---|---:|---:|---:|
| MuRIL | 90% | 30% | −60 percentage points |
| XLM-RoBERTa | 70% | 20% | −50 percentage points |

This suggests that spelling variants can be represented using more fragmented subword sequences.

### 9.2 Pseudo-log-likelihood

The reported mean contrast-minus-original MeanLogP values are:

| Model | Pair category | Mean difference |
|---|---|---:|
| XLM-RoBERTa | Valid-word context | −1.0933 |
| XLM-RoBERTa | Synonym substitution | −0.0655 |
| XLM-RoBERTa | Spelling error | −0.6695 |
| MuRIL | Valid-word context | −1.9141 |
| MuRIL | Synonym substitution | −0.5810 |
| MuRIL | Spelling error | −2.0368 |

Negative values mean the contrast sentence received a lower mean token log-probability under the model. PLL is not a direct measurement of grammatical correctness.

### 9.3 Explanation stability

The pooled mean rank-stability results are:

| Model | Explanation method | Mean rank correlation |
|---|---|---:|
| XLM-RoBERTa | Attention | 0.7425 |
| XLM-RoBERTa | Integrated Gradients | 0.3589 |
| MuRIL | Attention | 0.7043 |
| MuRIL | Integrated Gradients | 0.2632 |

These are descriptive results from the implemented protocol. The target-masking artefact must be considered when interpreting them.

### 9.4 Stability versus faithfulness

The reported Spearman correlations between stability and faithfulness gap are:

| Model | Method | Spearman correlation | p-value |
|---|---|---:|---:|
| XLM-RoBERTa | Attention | 0.2704 | 0.2490 |
| XLM-RoBERTa | Integrated Gradients | 0.1583 | 0.5052 |
| MuRIL | Attention | −0.1611 | 0.4974 |
| MuRIL | Integrated Gradients | 0.0075 | 0.9749 |

None of these associations is statistically significant at the conventional 0.05 level.

**Interpretation:** The current experiment did not detect a statistically significant relationship between the selected stability and faithfulness measures. This does not prove that the relationship is absent. A larger dataset and a refined evaluation protocol are needed.

---

## 📖 10. Understanding the Metrics

If you are new to NLP research, the following definitions may help.

**Subword fragmentation**

The number of pieces into which a tokenizer divides a word. More pieces can indicate a different input representation, but do not automatically imply worse model performance.

**Single-subword coverage**

The proportion of target words represented by exactly one subword token.

**Pseudo-log-likelihood (PLL)**

A score estimated by masking each token in a sentence and measuring how well the model predicts it from the remaining context.

**MeanLogP**

PLL divided by the number of scored tokens, as defined in this project.

**Attention**

A model-internal weighting mechanism. Attention weights can be inspected as one source of attribution, but should not automatically be interpreted as a faithful explanation.

**Integrated Gradients**

An attribution method that uses gradients along a path from a baseline representation to the input representation.

**Explanation stability**

How similar an explanation remains when the input is changed.

**Explanation faithfulness**

Whether an explanation's highlighted features meaningfully affect the model output under a specified evaluation procedure.

**Faithfulness gap**

The difference between the target logit drop from masking top-ranked context tokens and the drop from masking bottom-ranked context tokens.

**Spearman correlation**

A statistic that measures whether two variables tend to increase or decrease together in a monotonic relationship.

**p-value**

A statistical quantity used to evaluate evidence against a specified null hypothesis. A p-value above 0.05 does not establish that there is no relationship.

---

## ⚙️ 11. Installation and Setup

### Prerequisites

- Python 3.12
- Git
- Node.js and npm for the dashboard
- Internet access to download pretrained model checkpoints
- Sufficient memory and disk space for model weights and intermediate results

The experiments were developed on Windows with CPU-only execution. Running the models on a CPU may be slow.

### Step 1 — Clone the repository

Replace the placeholder below with the actual URL of this GitHub repository.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd hindi-lexical-sensitivity
```

### Step 2 — Create a Python virtual environment

On Windows:

```powershell
py -3.12 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks script activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

### Step 3 — Install Python dependencies

If the repository contains a `requirements.txt` file:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the repository does not contain a dependency file, create one from the project's actual environment before asking other researchers to reproduce the experiments. Do not assume that installing the latest versions of every package will reproduce the original results.

### Step 4 — Install dashboard dependencies

```bash
cd dashboard
npm install
```

---

## ▶️ 12. Running the Experiments

The project contains separate components for tokenization, scoring, explanation generation, stability, and faithfulness analysis.

Before running any experiment:

1. Confirm that the cleaned dataset exists.
2. Confirm that the required model checkpoints can be downloaded.
3. Confirm that the Python dependencies match the project's tested environment.
4. Check the actual entry-point scripts and their command-line arguments.

The following components have been used in the research code:

| Component | Known source path |
|---|---|
| Explanation generation | `src/explanations/hindi_explanations.py` |
| Stability analysis | `src/analysis/explanation_stability.py` |
| Faithfulness analysis | `src/analysis/hindi_faithfulness.py` |

These paths identify modules, not necessarily standalone command-line entry points. Their exact execution commands should be taken from the repository's scripts, documentation, or pipeline configuration. Do not run a module directly unless it supports that execution mode.

### Recommended reproducibility order

1. Audit the dataset and record its checksum.
2. Run tokenization analysis.
3. Run PLL scoring.
4. Generate the combined pair-level analysis.
5. Generate explanations for eligible examples.
6. Run stability analysis.
7. Run faithfulness interventions.
8. Regenerate the final tables and plots.
9. Prepare dashboard data.

Use the same dataset version, model identifiers, tokenizer versions, scoring definitions, and eligibility rules when comparing results.

---

## 📈 13. Interactive Dashboard

The repository includes a React-based dashboard for exploring research outputs.

### Start the dashboard

From the project root:

```bash
cd dashboard
npm install
npm run dev
```

Open the local URL displayed by Vite in your terminal. In the current development configuration, it is typically:

`http://localhost:5173/`

### What is the dashboard for?

Depending on the available data and implemented views, the dashboard can help explore:

- Dataset composition and pair categories
- Tokenization changes
- Model scoring results
- Explanation stability
- Faithfulness intervention summaries
- Pair-level experimental records
- Eligibility information for strict and broader analysis subsets

The dashboard visualizes saved research results; it does not independently validate the scientific correctness of those results.

---

## 📂 14. Output Files

The main result files are listed below.

| File | Purpose |
|---|---|
| `outputs/tokenization/tokenization_results.csv` | Pair- and model-level tokenization measurements |
| `outputs/scoring/pll_scoring_results.csv` | PLL and mean-log-probability scores |
| `outputs/analysis/combined_pair_level_results.csv` | Integrated tokenization and scoring results |
| `outputs/analysis/descriptive_statistics.csv` | Descriptive statistics |
| `outputs/analysis/correlation_results.csv` | Correlation results |
| `outputs/analysis/model_comparison.csv` | Model-level comparisons |
| `outputs/analysis/analysis_report.md` | Human-readable analysis report |
| `outputs/dashboard_data/pair_level_master.csv` | Combined records used by dashboard views |
| `outputs/dashboard_data/eligibility_master.csv` | Eligibility information for analysis subsets |
| `outputs/dashboard_data/stability_summary.csv` | Stability summaries |
| `outputs/dashboard_data/faithfulness_summary.csv` | Faithfulness summaries |
| `outputs/dashboard_data/scientific_audit_report.md` | Dashboard-data audit notes |
| `outputs/dashboard_data/manifest.json` | Dashboard data manifest |

A CSV file contains structured rows and columns that can be inspected using Python, Excel, or other data-analysis tools. JSON files store structured metadata or configuration information.

Check the current repository for the exact files and column definitions. Not every intermediate file is guaranteed to be committed to GitHub.

---

## ⚠️ 15. Important Limitations

This study has several limitations.

### Small explanation sample

Only 20 spelling pairs are used for the exploratory explanation stability and faithfulness experiments. This limits statistical power and generalisability.

### Target alignment

Strict single-subword eligibility is limited to four XLM-RoBERTa pairs and six MuRIL pairs. A broader visible-target scoring protocol permits multi-subword targets but uses a different aggregation choice.

### Target-masking artefact

Masking the target in both the original and perturbed sentences can remove the spelling difference. This can produce artificially high stability and must not be interpreted as evidence of robust explanations.

### Masking interventions

Masking context tokens can create unnatural inputs. A target logit can increase or decrease after masking, so the intervention does not establish causality by itself.

### PLL interpretation

PLL measures model predictability under a defined scoring procedure. It does not directly measure grammaticality, human acceptability, or semantic equivalence.

### Dataset scope

The dataset is a constructed collection rather than a representative sample of every type of Hindi spelling variation, dialect, or real-world text.

### Statistical interpretation

The absence of statistically significant correlations does not prove that explanation stability and faithfulness are unrelated.

---

## 🔁 16. Reproducibility

To support independent verification, the repository should document:

- Dataset version and checksum
- Dataset annotation and adjudication guidelines
- Model identifiers and checkpoint revisions
- Python and library versions
- Tokenization and PLL scoring conventions
- Explanation baselines and integration steps
- Target-token alignment rules
- Masking fractions and intervention definitions
- Eligibility criteria and exclusions
- Statistical tests and plotting scripts

When publishing updated results, regenerate figures and tables from the final CSV files rather than manually editing reported values.

Do not commit API keys, passwords, private credentials, or unnecessary large model checkpoints. If the dataset has sharing restrictions, document how researchers can obtain it.

---

## 🚀 17. Future Work

Potential extensions include:

- Expanding the Hindi spelling-perturbation dataset.
- Adding more realistic spelling errors from naturally occurring text.
- Improving token alignment between original and perturbed words.
- Designing stability measurements that preserve the perturbation being investigated.
- Testing multiple Integrated Gradients baselines.
- Comparing masking with less distribution-shifting intervention methods.
- Evaluating results across more model families.
- Adding human judgments of contextual acceptability.
- Reporting uncertainty intervals and additional statistical tests.
- Repeating the study on other Indic languages and scripts.

These are proposed directions rather than completed experiments.

---

## 📄 18. Citation

If you use this research or its outputs, please cite the paper after a publication-ready version and complete bibliographic information are available.

A provisional citation format is:

```text
Author(s). "Beyond Tokenization: Spelling Perturbations,
Explanation Stability and Faithfulness in Devanagari Hindi MLMs."
Research manuscript, [year].
```

Replace the placeholders with the final author list, publication venue, year, DOI, or repository link when available.

---

## 🙏 19. Acknowledgements

This project explores the intersection of Hindi NLP, multilingual masked language models, and explainable AI. It builds on publicly available pretrained language models and established methods for model scoring and attribution.

Contributions, reproducibility checks, and constructive feedback are welcome.
