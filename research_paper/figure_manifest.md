# Figure Manifest

This manifest documents all verified figures included in the research paper *"Beyond Tokenization: Investigating Model Behaviour, Explanation Stability, and Faithfulness under Lexical Perturbations in Devanagari Hindi"*.

All figures are stored in `outputs/analysis/figures/` and referenced within `research_paper/research_paper.md`, `research_paper/research_paper.docx`, and `research_paper/research_paper.pdf`.

---

## Manifest Summary Table

| Figure Number | Image File | Dimensions / Format | Underlying Source Data | Section Referenced |
|---|---|---|---|---|
| **Figure 1** | `outputs/analysis/figures/fig1_delta_meanlp_by_track.png` | 1800×1200 PNG | `outputs/analysis/descriptive_statistics.csv` | Section 5.2 |
| **Figure 2** | `outputs/analysis/figures/fig2_token_delta_vs_meanlp.png` | 1800×1200 PNG | `outputs/analysis/combined_pair_level_results.csv` | Section 5.2 |
| **Figure 3** | `outputs/analysis/figures/fig3_target_token_delta_vs_meanlp.png` | 1800×1200 PNG | `outputs/analysis/combined_pair_level_results.csv` | Section 5.2 |
| **Figure 4** | `outputs/analysis/figures/fig4_model_comparison_scatter.png` | 1800×1200 PNG | `outputs/analysis/model_comparison.csv` | Section 5.3 |
| **Figure 5** | `outputs/analysis/figures/fig5_direction_agreement.png` | 1800×1200 PNG | `outputs/analysis/model_comparison.csv` | Section 5.3 |
| **Figure 6** | `outputs/analysis/figures/fig6_explanation_stability.png` | 3000×1350 PNG | `outputs/dashboard_data/stability_summary.csv` | Section 5.4 |
| **Figure 7** | `outputs/analysis/figures/fig7_faithfulness_interventions.png` | 3300×1350 PNG | `outputs/dashboard_data/faithfulness_summary.csv` | Section 5.5 |
| **Figure 8** | `outputs/analysis/figures/fig8_stability_vs_faithfulness.png` | 3300×1440 PNG | `outputs/dashboard_data/stability_faithfulness_relationship.csv` | Section 5.6 |

---

## Detailed Figure Descriptions and Captions

### Figure 1: Distribution of Length-Normalized Plausibility Shifts ($\Delta \text{MeanLogP}$) by Experimental Track
- **File**: `outputs/analysis/figures/fig1_delta_meanlp_by_track.png`
- **Caption**: **Figure 1**: Boxplot distributions of signed length-normalized pseudo-log-likelihood deltas ($\Delta \text{MeanLogP} = \text{Contrast} - \text{Original}$) across `VALID_WORD_CONTEXT` ($N=142$), `SYNONYM_SUBSTITUTION` ($N=38$), and `REALISTIC_SPELLING_ERROR` ($N=20$) for XLM-RoBERTa (blue) and MuRIL (orange). Negative values indicate that the canonical original sentence is preferred over the contrast. MuRIL imposes substantially harsher penalties on orthographic errors ($\mu = -2.0368$) than XLM-RoBERTa ($\mu = -0.6695$).
- **Source Artifact**: `outputs/analysis/descriptive_statistics.csv`
- **Linguistic/Statistical Nuance**: Outliers in `SYNONYM_SUBSTITUTION` reflect pretraining corpus frequency asymmetries between standard Hindi synonyms (e.g., `पानी` vs. `जल`).

### Figure 2: Relationship Between Sentence Token Length Delta and Plausibility Shift ($\Delta \text{MeanLogP}$)
- **File**: `outputs/analysis/figures/fig2_token_delta_vs_meanlp.png`
- **Caption**: **Figure 2**: Scatter plot and linear trend relating full sentence subword length changes ($\Delta \text{Tokens} = |W_{\text{cont}}| - |W_{\text{orig}}|$) to plausibility shifts ($\Delta \text{MeanLogP}$) across all 400 model evaluation instances. While token inflation is moderately correlated with score drops across the pooled dataset, target fragmentation in spelling errors causes sharp drops even when length normalization is applied.
- **Source Artifact**: `outputs/analysis/combined_pair_level_results.csv`

### Figure 3: Target Word Tokenization Inflation vs. Plausibility Shift ($\Delta \text{MeanLogP}$)
- **File**: `outputs/analysis/figures/fig3_target_token_delta_vs_meanlp.png`
- **Caption**: **Figure 3**: Plausibility shifts ($\Delta \text{MeanLogP}$) plotted specifically against the subword change localized within the target word ($\Delta \text{Target Tokens}$). Devanagari spelling errors consistently generate $\Delta \text{Target Tokens} \ge +1$, directly driving the observed score penalties in both models.
- **Source Artifact**: `outputs/analysis/combined_pair_level_results.csv`

### Figure 4: Cross-Model Comparison of Plausibility Shifts (XLM-RoBERTa vs. MuRIL)
- **File**: `outputs/analysis/figures/fig4_model_comparison_scatter.png`
- **Caption**: **Figure 4**: Cross-model scatter plot comparing within-model $\Delta \text{MeanLogP}$ between XLM-RoBERTa ($x$-axis) and MuRIL ($y$-axis) across all 200 pairs. Across the entire benchmark, the models exhibit moderate rank concordance (Spearman $r_s = +0.4408, p < 10^{-9}$). Concordance is driven by contextual contrasts ($r_s = +0.4457, p < 10^{-7}$) and collapses on synonyms ($r_s = +0.0595, p = 0.7226$).
- **Source Artifact**: `outputs/analysis/model_comparison.csv`

### Figure 5: Directional Plausibility Agreement Proportions Between Models Across Tracks
- **File**: `outputs/analysis/figures/fig5_direction_agreement.png`
- **Caption**: **Figure 5**: Proportions of directional agreement between XLM-RoBERTa and MuRIL on whether the original sentence is preferred ($\Delta \text{MeanLogP} < 0$). Directional agreement is highest in `VALID_WORD_CONTEXT` (81.0%), moderate in `REALISTIC_SPELLING_ERROR` (60.0%), and lowest in `SYNONYM_SUBSTITUTION` (57.9%), with overall agreement at 74.5% ($N=200$).
- **Source Artifact**: `outputs/analysis/model_comparison.csv`

### Figure 6: Explanation Stability under Devanagari Spelling Perturbations
- **File**: `outputs/analysis/figures/fig6_explanation_stability.png`
- **Caption**: **Figure 6**: Visible-target explanation stability under authentic Devanagari spelling noise ($N=20$ pairs) comparing Last-Layer Self-Attention and Layer Integrated Gradients across XLM-RoBERTa and MuRIL. (a) Rank stability (Spearman $\rho$): Attention achieves $2.1\times$ to $2.7\times$ higher rank stability than Integrated Gradients ($0.7425$ vs. $0.3589$ in XLM-R; $0.7043$ vs. $0.2632$ in MuRIL). (b) Vector directional stability (Cosine Similarity): Cosine similarity remains high across all conditions ($0.8994$ to $0.9880$), demonstrating that gradient attributions retain overall vector orientation while local token rankings fluctuate heavily.
- **Source Artifact**: `outputs/dashboard_data/stability_summary.csv`

### Figure 7: Explanation Faithfulness under Discrete Context Token Erasure
- **File**: `outputs/analysis/figures/fig7_faithfulness_interventions.png`
- **Caption**: **Figure 7**: Context erasure faithfulness across masking fractions $f \in \{0.10, 0.20, 0.30\}$ ($N=40$ evaluated sentences per model). (a) Mean Faithfulness Gap ($\Delta S_{\text{top}} - \Delta S_{\text{bottom}}$): Attention maintains consistently positive gaps across all fractions in both models ($+4.02$ to $+5.79$), while Integrated Gradients exhibits lower or negative gaps ($-0.69$ in MuRIL at $f=0.10, 0.20$). (b) Win Rate vs. Bottom-$k$ Control: Attention achieves win rates of 70.0%–77.5%, outperforming random erasure and substantially exceeding Integrated Gradients (42.5%–60.0%).
- **Source Artifact**: `outputs/dashboard_data/faithfulness_summary.csv`

### Figure 8: Decoupling of Explanation Stability and Explanation Faithfulness
- **File**: `outputs/analysis/figures/fig8_stability_vs_faithfulness.png`
- **Caption**: **Figure 8**: Scatter plot of visible-target explanation stability (Spearman $\rho$, $x$-axis) versus mean faithfulness gap ($f=0.20$, $y$-axis) across $N=20$ Devanagari spelling pairs for (a) XLM-RoBERTa and (b) MuRIL. Neither model nor method demonstrates a statistically significant relationship ($p > 0.05$ across all configurations: XLM-R Attention $r_s = +0.2704, p=0.249$; MuRIL Attention $r_s = -0.1611, p=0.497$; XLM-R IG $r_s = +0.1583, p=0.505$; MuRIL IG $r_s = +0.0075, p=0.975$). The absence of correlation and contradictory sign flips confirm that explanation stability under input noise is empirically decoupled from explanation faithfulness.
- **Source Artifact**: `outputs/dashboard_data/stability_faithfulness_relationship.csv`
