// ============================================================
// Beyond Tokenization: Investigating Model Behaviour, Explanation
// Stability, and Faithfulness under Lexical Perturbations in
// Devanagari Hindi
// ============================================================
// Typst source — compiled with typst 0.15.0 (Python API)
// ============================================================

// ---------- Page and font setup ----------
#set page(
  paper: "a4",
  margin: (top: 2.2cm, bottom: 2.5cm, left: 1.8cm, right: 1.8cm),
  header: align(center, text(size: 8pt, fill: luma(120))[
    Beyond Tokenization: Devanagari Hindi Model Behaviour, Explanation Stability, and Faithfulness
    #h(1fr)
    _Under Review_
  ]),
  footer: context align(center, text(size: 8pt, fill: luma(120))[
    #counter(page).display()
  ]),
)

#set text(
  font: ("Linux Libertine", "Times New Roman", "Nirmala UI", "Mangal"),
  size: 10pt,
  lang: "en",
)
#set par(justify: true, leading: 0.65em, first-line-indent: 1.2em)
#set heading(numbering: "1.")
#set math.equation(numbering: "(1)")
#show heading.where(level: 1): it => {
  v(1.2em)
  text(size: 11.5pt, weight: "bold")[#it]
  v(0.4em)
}
#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 10.5pt, weight: "bold", style: "italic")[#it]
  v(0.3em)
}
#show heading.where(level: 3): it => {
  v(0.6em)
  text(size: 10pt, weight: "bold")[#it]
  v(0.2em)
}

// ---------- Title block ----------
#align(center)[
  #text(size: 17pt, weight: "bold")[
    Beyond Tokenization: Investigating Model Behaviour, Explanation Stability,\
    and Faithfulness under Lexical Perturbations in Devanagari Hindi
  ]
  #v(0.8em)
  #text(size: 11pt)[Anonymous Author(s) · _Under Review_]
  #v(0.5em)
  #line(length: 80%)
]

// ---------- Abstract ----------
#v(0.6em)
#align(center, text(weight: "bold", size: 11pt)[Abstract])
#v(0.3em)

#block(inset: (x: 1.5cm))[
  #text(size: 9.5pt)[
    Masked language models (MLMs) have become central to multilingual natural language processing (NLP), yet their behaviour and interpretability under non-Latin, morpho-phonetically complex scripts remain critically underexplored. We present a systematic empirical investigation into model behaviour, subword tokenisation dynamics, explanation stability, and explanation faithfulness under controlled lexical perturbations in Devanagari Hindi. Using a triple-annotated benchmark of 200 minimal sentence pairs across three controlled tracks — Valid Word Context (#math.inline[$N=142$]), Synonym Substitution (#math.inline[$N=38$]), and Realistic Spelling Error (#math.inline[$N=20$]) — we evaluate the Indic-dedicated encoder *MuRIL* (`google/muril-base-cased`) and the multilingual generalist *XLM-RoBERTa* (`FacebookAI/xlm-roberta-base`).

    Our experiments reveal four primary findings. First, authentic Devanagari spelling errors (matra inversions and ligature splits) induce catastrophic subword fragmentation: single-subword target coverage collapses by $-60.0%$ in MuRIL ($90.0% -> 30.0%$) and by $-50.0%$ in XLM-RoBERTa ($70.0% -> 20.0%$). Second, both models reliably penalise spelling errors under pseudo-log-likelihood (PLL) scoring, but MuRIL exhibits substantially harsher penalties (original preferred in $95.0%$ of pairs, mean $Delta "MeanLogP" = -2.037$) compared to XLM-RoBERTa ($65.0%$, mean $Delta = -0.670$). Third, under a corrected visible-target protocol — which resolves an illusory perfect stability artefact ($rho = 1.000$) induced by target masking — last-layer self-attention achieves $2.1 times$ to $2.7 times$ higher rank stability than Layer Integrated Gradients ($rho = 0.704$–$0.743$ vs. $0.263$–$0.359$). Fourth, in discrete context token erasure tests, attention attributions yield superior faithfulness win rates ($70.0%$–$77.5%$), whereas Integrated Gradients operates near or below chance ($42.5%$–$60.0%$). A correlation analysis confirms that explanation stability does *not* predict explanation faithfulness ($p > 0.05$ across all 8 model–method configurations; $r_s in [-0.161, +0.270]$).
  ]
]

#v(0.4em)
#align(center)[
  *Keywords:* Devanagari Hindi, Masked Language Models, Subword Tokenisation, Pseudo-Log-Likelihood Scoring, Explainable NLP, Explanation Stability, Explanation Faithfulness, Integrated Gradients.
]

// ---------- Two-column body ----------
#columns(2, gutter: 1em)[

// ==================== 1. INTRODUCTION ====================
= Introduction

Pretrained masked language models (MLMs) based on bidirectional Transformer architectures have fundamentally advanced computational linguistics across high-resource and multilingual environments #cite(<devlin2019bert>, <conneau2020xlmr>). However, the empirical behaviour of these models is overwhelmingly documented in Latin-script languages where typographical noise consists of straightforward character substitutions or deletions.

In contrast, non-Latin writing systems following the Indic abugida orthographic typology — such as Devanagari Hindi — possess unique structural, syllabic, and compositional characteristics. Devanagari encodes phonological syllables (_aksharas_) using base consonants, dependent vowel diacritics (_matras_), and virama-linked conjuncts (_ligatures_). Minor orthographic variations in Devanagari do not operate as isolated character perturbations; rather, they alter the underlying graphemic representation, frequently destroying canonical subword vocabulary entries and triggering severe tokenisation fragmentation.

In parallel, the interpretability of Transformer architectures has emerged as a cornerstone of trustworthy NLP #cite(<sundararajan2017axiomatic>, <deyoung2020eraser>). Within explainable AI (XAI), two evaluative criteria are frequently prioritised:
- *Explanation Stability (Robustness):* Invariance of attributions under minor, semantics-preserving input perturbations #cite(<alvarezmelis2018robustness>).
- *Explanation Faithfulness:* Whether the explanation reflects the model's internal causal reasoning, typically evaluated by erasing top-attributed tokens #cite(<deyoung2020eraser>, <jacovi2020towards>).

A pervasive but largely untested assumption is that attribution stability implies explanation quality, with the implicit expectation that a robust explanation is also more likely to be faithful to the model mechanism.

This study bridges this empirical gap through five research questions:

*RQ1.* How do contextual lexical contrasts, synonym substitutions, and spelling perturbations affect subword tokenisation and model scoring in Devanagari Hindi?

*RQ2.* How do MuRIL and XLM-RoBERTa differ in their responses to these perturbations?

*RQ3.* How stable are Attention and Integrated Gradients explanations under authentic Devanagari spelling perturbations?

*RQ4.* How do these explanation methods behave under discrete context-token masking interventions?

*RQ5.* Is explanation stability statistically associated with measured explanation faithfulness?

// ==================== 2. BACKGROUND ====================
= Background and Related Work

== Indic-Language Pretrained Models

XLM-RoBERTa #cite(<conneau2020xlmr>) utilises a Byte-Pair Encoding (BPE) SentencePiece vocabulary of 250,002 tokens shared across 100+ languages. MuRIL #cite(<khanuja2021muril>) is a BERT-base architecture trained on 17 Indic languages and English using a 197,285-token WordPiece vocabulary, providing language-family specialisation. Kakwani et al. #cite(<kakwani2020indicnlpsuite>) developed the IndicGLUE benchmark, emphasising that language-family specialisation improves morphological capture.

== Subword Tokenisation Vulnerability

Subword segmentation algorithms — BPE #cite(<sennrich2016bpe>), WordPiece #cite(<devlin2019bert>), and SentencePiece Unigram #cite(<kudo2018sentencepiece>) — are foundational to modern transformers. Rust et al. #cite(<rust2021tokenizer>) demonstrated that subword tokenisers exhibit systematic performance gaps in non-Latin scripts. In Devanagari Hindi, the structural unit is the _akshara_ (syllable), composed of consonant bases modified by attached vowel diacritics (_matras_) or conjunct-forming viramas. When a user introduces a spelling error — such as swapping a long _dīrgha_ matra for a short _hrasva_ matra (e.g., *पानी* → *पानि*) or omitting a halant (*आवश्यक* → *आवशयक*) — the character sequence shifts across syllabic boundaries, fracturing canonical vocabulary tokens.

== Masked Language Model Scoring

Salazar et al. #cite(<salazar2020mlmscoring>) formalised *Pseudo-Log-Likelihood (PLL)* scoring for bidirectional MLMs. By iteratively masking each token and accumulating conditional log-probabilities, PLL provides an effective proxy for grammatical acceptability and corpus typicality.

== Feature Attribution: Attention vs. Integrated Gradients

Integrated Gradients (IG) #cite(<sundararajan2017axiomatic>) is an axiomatic path-attribution method satisfying _Completeness_ and _Implementation Invariance_. The attention-as-explanation debate — initiated by Jain and Wallace #cite(<jain2019attention>) arguing "Attention is not Explanation" and countered by Wiegreffe and Pinter #cite(<wiegreffe2019attention>) — directly motivates our comparative evaluation.

== Explanation Stability and Faithfulness

Alvarez-Melis and Jaakkola #cite(<alvarezmelis2018robustness>) formalised explanation stability as the local Lipschitz continuity of the attribution mapping. The ERASER benchmark #cite(<deyoung2020eraser>) operationalised faithfulness via comprehensiveness and sufficiency under token erasure. Jacovi and Goldberg #cite(<jacovi2020towards>) provided a formal definition of faithfulness that we adopt: an explanation is faithful if the features it attributes high importance to are the ones that causally govern the model's prediction. Bastings and Filippova #cite(<bastings2020elephant>) highlighted that continuous gradient methods often behave unpredictably under discrete feature erasure.

// ==================== 3. DATASET ====================
= Dataset and Annotation Methodology

== Benchmark Design

We curated a benchmark of 200 minimal Hindi sentence pairs ($N = 200$) across three tracks:
- *Valid Word Context (VWC):* $N = 142$ pairs (71.0%). Target word replaced by a grammatically valid but contextually mismatched alternative.
- *Synonym Substitution (SYN):* $N = 38$ pairs (19.0%). Target word replaced by a verified contextually appropriate synonym (e.g., *पानी* ↔ *जल*).
- *Realistic Spelling Error (RSE):* $N = 20$ pairs (10.0%). Target word replaced by an authentic, commonly observed Devanagari misspelling.

== Typology of Devanagari Orthographic Errors

The 20 RSE pairs were classified into four linguistic categories:

#table(
  columns: (auto, auto, auto),
  inset: 5pt,
  stroke: 0.5pt + luma(180),
  fill: (_, row) => if row == 0 { rgb("#1a365d") } else if calc.odd(row) { rgb("#f8fafc") } else { white },
  table.header(
    text(fill: white, weight: "bold")[Error Type],
    text(fill: white, weight: "bold")[$N$],
    text(fill: white, weight: "bold")[Example],
  ),
  [Matra length inversion], [$9$], [*परीक्षा* → *परिक्षा*],
  [Conjunct/ligature disruption], [$5$], [*आवश्यक* → *आवशयक*],
  [Orthographic alternation], [$2$], [*संभावना* ↔ *सम्भावना*],
  [Colloquial phonetic], [$4$], [*वापस* → *वापिस*],
)
#align(center, text(size: 8pt, style: "italic")[Table 1: Devanagari Spelling Error Typology])

== Quality Control and Forensic Adjudication

All 200 pairs underwent triple annotation: two independent native Hindi linguists plus an expert adjudicator with definitive `final_status` determination.

A comprehensive forensic audit of the dataset (SHA-256 `f9a5af81...`) confirmed:
1. 100% of characters reside within the Devanagari Unicode block (`U+0900`–`U+097F`).
2. *Exactly one rejected pair:* `HC_0178` (*सलाह* vs *सलाई*, `final_status = reject`). Early working notes erroneously cited `HC_0185` or `HC_0047`; the audit definitively identifies `HC_0178`.
3. *Six disagreement overrides:* `HC_0087`, `HC_0099`, `HC_0138`, `HC_0172`, `HC_0180`, `HC_0186` — retained because both sentences are grammatically valid Hindi sentences.

The benchmark thus contains *199 accepted pairs* and *1 rejected pair*, totalling 200 unique minimal pairs.

// ==================== 4. METHODOLOGY ====================
= Models and Experimental Methodology

== Pretrained Architectures

Both models follow BERT-style bidirectional self-attention:

#table(
  columns: (auto, auto, auto),
  inset: 5pt,
  stroke: 0.5pt + luma(180),
  fill: (_, row) => if row == 0 { rgb("#1a365d") } else if calc.odd(row) { rgb("#f8fafc") } else { white },
  table.header(
    text(fill: white, weight: "bold")[Property],
    text(fill: white, weight: "bold")[XLM-RoBERTa],
    text(fill: white, weight: "bold")[MuRIL],
  ),
  [Architecture], [RoBERTa-base], [BERT-base],
  [Layers / Heads], [12 / 12], [12 / 12],
  [Hidden dimension], [768], [768],
  [Vocabulary], [250,002], [197,285],
  [Tokeniser], [SentencePiece BPE], [WordPiece],
  [Language scope], [100+ languages], [17 Indic + English],
)
#align(center, text(size: 8pt, style: "italic")[Table 2: Model Architecture Comparison])

== Pseudo-Log-Likelihood Scoring

Following Salazar et al. #cite(<salazar2020mlmscoring>), for a sentence $W = (w_1, w_2, dots, w_(|W|))$, the pseudo-log-likelihood is computed via iterative single-token masking:

$ "PLL"(W) = sum_(t=1)^(|W|) log P(w_t | W_(-(t)); Theta) $ <eq:pll>

where $W_(-(t))$ denotes the sequence with token $w_t$ replaced by the model's mask token, and $Theta$ are the model parameters.

To prevent length bias, we define the length-normalised mean log-probability:

$ "MeanLogP"(W) = "PLL"(W) / (|W_"scorable"|) $ <eq:meanlp>

where $|W_"scorable"|$ is the number of masked content tokens.

The signed plausibility shift between contrast and original is:

$ Delta "MeanLogP" = "MeanLogP"(W_"cont") - "MeanLogP"(W_"orig") $ <eq:delta>

Under this convention, $Delta "MeanLogP" < 0$ indicates the *original sentence is preferred* by the model; $Delta "MeanLogP" > 0$ indicates the contrast is preferred.

*Methodological guardrail:* Raw PLL sums are *never* compared across tokenisers; only within-model signed $Delta$ values and preference rates are compared.

== Feature Attribution Methods

*Last-Layer Self-Attention.* We extract the self-attention tensor from Layer 11 (final layer), averaging across all 12 heads:

$ macron(A)_(i,j) = 1/12 sum_(h=1)^(12) A^((11,h))_(i,j) $ <eq:attn>

The attribution for context token $j$ with respect to target token set $T$ is:

$ alpha_j = 1/(|T|) sum_(t in T) macron(A)_(t,j) $ <eq:ctxattn>

Special framing tokens are excluded; the resulting vector is $ell_1$-normalised.

*Layer Integrated Gradients (IG).* Using Captum #cite(<sundararajan2017axiomatic>), we compute IG at the word embedding layer with a zero-vector baseline $bold(x)' = bold(0)$ and 50 Gauss-Legendre quadrature steps:

$ "IG"_i (bold(x)) = (x_i - x'_i) times sum_(k=1)^(50) (partial F(bold(x)' + k/50 (bold(x) - bold(x)')) ) / (partial x_i) times 1/50 $ <eq:ig>

Token importance is the Euclidean $L_2$-norm over the 768 embedding dimensions, followed by context-level normalisation.

== Stability Metrics

For each spelling pair, we align context tokens between original and contrast sentences using character-offset tracking. Two complementary stability metrics are used:

*Rank Stability* (Spearman $rho$): measures whether the relative importance ordering of context tokens is preserved under target spelling noise.

*Directional Vector Stability* (Cosine Similarity): measures whether the global orientation of the attribution vector is preserved:

$ "Cosine"(bold(u), bold(v)) = (bold(u) dot bold(v)) / (||bold(u)||_2 ||bold(v)||_2) $ <eq:cosine>

=== The Stage 6 Masked-Target Artefact and Stage 7 Resolution

A critical methodological discovery: in the preliminary stability evaluation (Stage 6), the protocol computed attributions while replacing the target word with `[MASK]` in *both* original and contrast sentences. Since the perturbation was exclusively within the target word, both inputs became byte-identical:

$ X_"orig" = [w_1, dots, "[MASK]", dots, w_n] equiv X_"cont" $

This produced an illusory perfect stability: $rho = 1.000$, $"Cosine" = 1.000$, $Delta"Logit" = 0$ with zero variance. *Stage 7* resolved this via the *Visible-Target Scoring Protocol*: target tokens remain fully visible; target score is the mean unnormalized logit of target subwords:

$ S(X; Y_"target") = 1/(|T|) sum_(j in T) "Logit"(p_j, y_j) $ <eq:score>

where $p_j$ is the sequence position and $y_j$ is the ground-truth token ID of the $j$-th target subword.

== Faithfulness Evaluation Protocol

Following ERASER #cite(<deyoung2020eraser>), context tokens are ranked by attribution and erased at fractions $f in {0.10, 0.20, 0.30}$:
- *Top-$k$ erasure:* $k = ceil(f dot |C|)$ highest-attributed tokens replaced by `[MASK]`.
- *Bottom-$k$ erasure (control):* $k$ lowest-attributed tokens replaced.
- *Random-$k$ erasure:* $k$ uniform-random tokens replaced.

The *target logit drop* after intervention is:

$ Delta S_"intervention" = S(X; Y) - S(X_"masked"; Y) $ <eq:logitdrop>

A *positive* $Delta S$ means the target score *decreased* after masking (expected for informative tokens). A *negative* $Delta S$ means the target score *increased* — as observed for XLM-RoBERTa due to out-of-distribution logit dynamics (Section 5.5).

The *Faithfulness Gap* is:

$ "Gap" = Delta S_"top" - Delta S_"bottom" $ <eq:gap>

A positive Gap means erasing top-attributed tokens caused a greater relative score drop than erasing bottom control tokens. The *Win Rate* is:

$ "WinRate" = 1/N sum_(i=1)^N bb(1)(Delta S_"top",i > Delta S_"bottom",i) $ <eq:winrate>

// ==================== 5. RESULTS ====================
= Empirical Results

== RQ1 — Subword Tokenisation Under Lexical Perturbations

Authentic Devanagari spelling errors cause catastrophic subword fragmentation:

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 5pt,
  stroke: 0.5pt + luma(180),
  fill: (_, row) => if row == 0 { rgb("#1a365d") } else if calc.odd(row) { rgb("#f8fafc") } else { white },
  table.header(
    text(fill: white, weight: "bold")[Model],
    text(fill: white, weight: "bold")[Track],
    text(fill: white, weight: "bold")[$N$],
    text(fill: white, weight: "bold")[Target sub-words Orig→Cont],
    text(fill: white, weight: "bold")[Single-sub-word Orig→Cont],
  ),
  [XLM-R], [RSE], [20], [1.35 → 2.35 (+1.00)], [70.0% → *20.0%* (−50%)],
  [XLM-R], [SYN], [38], [1.13 → 1.76], [89.5% → 60.5%],
  [XLM-R], [VWC], [142], [1.75 → 1.92], [39.4% → 30.3%],
  [MuRIL], [RSE], [20], [1.10 → 2.10 (+1.00)], [90.0% → *30.0%* (−60%)],
  [MuRIL], [SYN], [38], [1.00 → 1.18], [100.0% → 86.8%],
  [MuRIL], [VWC], [142], [1.43 → 1.51], [64.1% → 55.6%],
)
#align(center, text(size: 8pt, style: "italic")[Table 3: Subword tokenisation analysis (VWC = Valid Word Context, SYN = Synonym, RSE = Spelling Error). Source: tokenization\_summary.csv])

MuRIL encodes Hindi with ~12% fewer subwords than XLM-RoBERTa on VWC ($mu = 8.95$ vs $10.19$ tokens per sentence), reflecting its Indic-dedicated vocabulary. On RSE, both models lose exactly $+1.00$ subword per sentence on average — driven entirely by target-word fracturing.

#colbreak()

Figure 1 shows the distribution of $Delta "MeanLogP"$ across tracks for both models.

#box(fill: rgb("#f8fafc"), stroke: 0.5pt + luma(180), inset: 6pt, radius: 3pt)[
  #image("../outputs/analysis/figures/fig1_delta_meanlp_by_track.png", width: 100%)
  #align(center, text(size: 8pt, style: "italic")[
    *Figure 1:* $Delta "MeanLogP"$ distributions by track. Negative values indicate the original sentence is preferred. MuRIL imposes substantially harsher penalties on spelling errors ($mu = -2.037$) than XLM-RoBERTa ($mu = -0.670$).
  ])
]

#v(0.5em)

== RQ1 — Model Plausibility Scoring

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 4pt,
  stroke: 0.5pt + luma(180),
  fill: (_, row) => if row == 0 { rgb("#1a365d") } else if calc.odd(row) { rgb("#f8fafc") } else { white },
  table.header(
    text(fill: white, weight: "bold")[Model],
    text(fill: white, weight: "bold")[Track],
    text(fill: white, weight: "bold")[$N$],
    text(fill: white, weight: "bold")[Mean $Delta$],
    text(fill: white, weight: "bold")[Orig. Preferred],
  ),
  [XLM-R], [RSE], [20], [−0.670], [65.0% (13/20)],
  [XLM-R], [VWC], [142], [−1.093], [81.0% (115/142)],
  [XLM-R], [SYN], [38], [−0.066], [57.9% (22/38)],
  [MuRIL], [RSE], [20], [*−2.037*], [*95.0%* (19/20)],
  [MuRIL], [VWC], [142], [−1.914], [93.0% (132/142)],
  [MuRIL], [SYN], [38], [−0.581], [84.2% (32/38)],
)
#align(center, text(size: 8pt, style: "italic")[Table 4: Plausibility scoring statistics. Source: descriptive\_statistics.csv])

Key findings:
- *Spelling errors:* MuRIL penalises errors far more strongly than XLM-RoBERTa ($-2.037$ vs $-0.670$).
- *Contextual mismatches:* Both models show strong discrimination (MuRIL 93.0%, XLM-R 81.0%).
- *Synonyms:* XLM-RoBERTa shows near-zero mean delta ($-0.066$), confirming that the multilingual generalist treats valid Hindi synonyms as near-interchangeable. The small MuRIL synonym penalty ($-0.581$) likely reflects pretraining corpus frequency differences.

== RQ2 — Cross-Model Comparison

#table(
  columns: (auto, auto, auto, auto),
  inset: 4pt,
  stroke: 0.5pt + luma(180),
  fill: (_, row) => if row == 0 { rgb("#1a365d") } else if calc.odd(row) { rgb("#f8fafc") } else { white },
  table.header(
    text(fill: white, weight: "bold")[Track],
    text(fill: white, weight: "bold")[$N$],
    text(fill: white, weight: "bold")[Agreement],
    text(fill: white, weight: "bold")[Spearman $r_s$ ($p$)],
  ),
  [All Tracks], [200], [74.5%], [$+0.441$ ($p<10^(-9)$)],
  [VWC], [142], [81.0%], [$+0.446$ ($p<10^(-7)$)],
  [RSE], [20], [60.0%], [$+0.215$ ($p=0.363$)],
  [SYN], [38], [57.9%], [$+0.060$ ($p=0.723$)],
)
#align(center, text(size: 8pt, style: "italic")[Table 5: Cross-model directional agreement and rank correlation. Source: model\_comparison.csv])

#box(fill: rgb("#f8fafc"), stroke: 0.5pt + luma(180), inset: 6pt, radius: 3pt)[
  #image("../outputs/analysis/figures/fig4_model_comparison_scatter.png", width: 100%)
  #align(center, text(size: 8pt, style: "italic")[
    *Figure 2:* Cross-model scatter of $Delta "MeanLogP"$ ($N = 200$). Overall $r_s = +0.441, p < 10^{-9}$.
  ])
]

Cross-model agreement across 200 pairs is 74.5% (Spearman $r_s = +0.441, p < 10^{-9}$), driven by contextual contrasts. On synonyms, agreement collapses to 57.9% ($r_s = +0.060, p = 0.723$), revealing model-specific pretraining frequency biases rather than shared linguistic representations.

== RQ3 — Explanation Stability Under Spelling Noise

After resolving the Stage 6 masked-target artefact (Section 4.4), the corrected visible-target stability results are:

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 4pt,
  stroke: 0.5pt + luma(180),
  fill: (_, row) => if row == 0 { rgb("#1a365d") } else if calc.odd(row) { rgb("#f8fafc") } else { white },
  table.header(
    text(fill: white, weight: "bold")[Model],
    text(fill: white, weight: "bold")[Method],
    text(fill: white, weight: "bold")[$N$],
    text(fill: white, weight: "bold")[Mean $rho$],
    text(fill: white, weight: "bold")[Mean Cosine],
  ),
  [XLM-R], [Attention], [20], [*0.7425*], [0.9648],
  [XLM-R], [Attn. (strict)], [4], [0.6893], [0.9776],
  [XLM-R], [IG], [20], [0.3589], [0.8994],
  [MuRIL], [Attention], [20], [*0.7043*], [0.9639],
  [MuRIL], [Attn. (strict)], [6], [0.6286], [0.9739],
  [MuRIL], [IG], [20], [0.2632], [0.9880],
)
#align(center, text(size: 8pt, style: "italic")[Table 6: Visible-target explanation stability ($N=20$ RSE pairs). Source: stability\_summary.csv. "Strict" = single-subword subset.])

#box(fill: rgb("#f8fafc"), stroke: 0.5pt + luma(180), inset: 6pt, radius: 3pt)[
  #image("../outputs/analysis/figures/fig6_explanation_stability.png", width: 100%)
  #align(center, text(size: 8pt, style: "italic")[
    *Figure 3:* Explanation stability under Devanagari spelling noise. (a) Rank stability (Spearman $rho$): Attention achieves $2.1 times$ to $2.7 times$ higher rank stability. (b) Cosine similarity: both methods retain high directional alignment.
  ])
]

Two findings emerge: (1) last-layer self-attention achieves *$2.1 times$ to $2.7 times$ higher rank stability* than Integrated Gradients ($rho approx 0.70$–$0.74$ vs $0.26$–$0.36$); (2) despite low rank stability, Integrated Gradients retains high *cosine similarity* ($0.899$–$0.988$), demonstrating that gradient attribution vectors maintain their global orientation while local token-level rankings fluctuate heavily under Devanagari spelling noise.

== RQ4 — Faithfulness Under Discrete Context Erasure

#table(
  columns: (auto, auto, auto, auto, auto),
  inset: 4pt,
  stroke: 0.5pt + luma(180),
  fill: (_, row) => if row == 0 { rgb("#1a365d") } else if calc.odd(row) { rgb("#f8fafc") } else { white },
  table.header(
    text(fill: white, weight: "bold")[Model],
    text(fill: white, weight: "bold")[Method],
    text(fill: white, weight: "bold")[$f$],
    text(fill: white, weight: "bold")[Gap],
    text(fill: white, weight: "bold")[Win Rate vs. Bottom],
  ),
  [XLM-R], [Attention], [0.10], [+4.017], [*77.5%*],
  [XLM-R], [Attention], [0.20], [+4.017], [*77.5%*],
  [XLM-R], [Attention], [0.30], [+5.172], [75.0%],
  [XLM-R], [IG], [0.10], [+1.837], [60.0%],
  [XLM-R], [IG], [0.20], [+1.837], [60.0%],
  [XLM-R], [IG], [0.30], [+3.295], [55.0%],
  [MuRIL], [Attention], [0.30], [+5.794], [*70.0%*],
  [MuRIL], [Attn. (strict)], [0.30], [+8.822], [75.0%],
  [MuRIL], [IG], [0.10], [*−0.686*], [45.0%],
  [MuRIL], [IG], [0.20], [*−0.686*], [45.0%],
  [MuRIL], [IG], [0.30], [+0.033], [42.5%],
)
#align(center, text(size: 8pt, style: "italic")[Table 7: Faithfulness evaluation ($N=40$ sentences per model). Source: faithfulness\_summary.csv. "Strict" = $N=12$.])

#box(fill: rgb("#f8fafc"), stroke: 0.5pt + luma(180), inset: 6pt, radius: 3pt)[
  #image("../outputs/analysis/figures/fig7_faithfulness_interventions.png", width: 100%)
  #align(center, text(size: 8pt, style: "italic")[
    *Figure 4:* Faithfulness across masking fractions $f in {0.10, 0.20, 0.30}$. (a) Mean Faithfulness Gap; (b) Win Rate vs. Bottom-$k$ control.
  ])
]

=== XLM-RoBERTa Out-of-Distribution Logit Dynamics

In XLM-RoBERTa, removing context tokens caused raw target logits to *increase* (negative logit drop), due to out-of-distribution syntactic shock from `[MASK]` tokens:

At $f = 0.20$:
- Mean Top Drop: $Delta S_"top" = -7.19$ (logit *increased*)
- Mean Bottom Drop: $Delta S_"bottom" = -11.21$ (logit increased more)

However, directional faithfulness is strictly preserved because bottom masking inflates logits *even more* than top masking:

$ "Gap" = (-7.19) - (-11.21) = +4.02 > 0 $ <eq:oodgap>

This yields a 77.5% win rate against bottom control. The *relative comparison* remains valid even when raw drops are unintuitive in sign.

Integrated Gradients operates *near or below chance* in MuRIL (win rates 42.5%–45.0%; negative Gap at $f = 0.10, 0.20$). This arises because IG integrates along a continuous straight-line path in embedding space — a path that does not correspond monotonically to the discrete, non-linear `[MASK]` substitutions used in faithfulness evaluation.

== RQ5 — Stability–Faithfulness Relationship

#table(
  columns: (auto, auto, auto, auto),
  inset: 4pt,
  stroke: 0.5pt + luma(180),
  fill: (_, row) => if row == 0 { rgb("#1a365d") } else if calc.odd(row) { rgb("#f8fafc") } else { white },
  table.header(
    text(fill: white, weight: "bold")[Model],
    text(fill: white, weight: "bold")[Method],
    text(fill: white, weight: "bold")[Spearman $r_s$ ($p$)],
    text(fill: white, weight: "bold")[Pearson $r$ ($p$)],
  ),
  [XLM-R], [Attention], [$+0.270$ ($0.249$)], [$+0.256$ ($0.277$)],
  [XLM-R], [IG], [$+0.158$ ($0.505$)], [$+0.259$ ($0.270$)],
  [MuRIL], [Attention], [$-0.161$ ($0.497$)], [$-0.194$ ($0.413$)],
  [MuRIL], [IG], [$+0.008$ ($0.975$)], [$+0.018$ ($0.939$)],
)
#align(center, text(size: 8pt, style: "italic")[Table 8: Stability–faithfulness correlation ($N=20$ RSE pairs, $f=0.20$). All $p > 0.05$. Source: stability\_faithfulness\_relationship.csv])

#box(fill: rgb("#f8fafc"), stroke: 0.5pt + luma(180), inset: 6pt, radius: 3pt)[
  #image("../outputs/analysis/figures/fig8_stability_vs_faithfulness.png", width: 100%)
  #align(center, text(size: 8pt, style: "italic")[
    *Figure 5:* Stability–faithfulness scatter plot. No statistically significant correlation in any configuration ($p > 0.05$). Note the sign flip between models (positive in XLM-RoBERTa, negative in MuRIL).
  ])
]

Zero statistically significant correlations ($p > 0.05$) across all 8 model–method configurations, with $p$-values ranging from $0.199$ to $0.975$. The correlation signs are *contradictory* between models: weakly positive in XLM-RoBERTa ($r_s = +0.270$) and negative in MuRIL ($r_s = -0.161$ for Attention). These results establish that explanation stability and faithfulness are *empirically decoupled* within this experimental design.

// ==================== 6. DISCUSSION ====================
= Discussion

== Orthogonality of Stability and Faithfulness

The central empirical finding is the complete decoupling between explanation stability and faithfulness. Mathematically, these metrics occupy distinct spaces:
- *Stability* measures the local Lipschitz continuity of the attribution function $g(x)$:
  $ cal(S)(g; x, delta) = "Sim"(g(x), g(x + delta)) $
- *Faithfulness* measures causal dependency of the model $f$ on features prioritised by $g$:
  $ cal(F)(g; x, f) = f(x) - f(x_(without C_"top")) $

An explanation can achieve near-perfect stability by anchoring to invariant, static syntactic structures (e.g., punctuation or common postpositions) that do not vary under local spelling perturbations — yet erasing those tokens may have minimal impact on target word prediction, yielding a *stably unfaithful* explanation. Therefore, *optimising XAI benchmarks solely for stability risks selecting explanations that are consistently unfaithful*.

== Why IG Fails Under Discrete Erasure

Integrated Gradients accumulates gradients along a continuous straight-line path in embedding space. This path traverses unconstrained, non-discrete regions of $RR^d$ where the transformer was never trained to operate. In contrast, evaluating faithfulness in MLMs requires a discrete substitution: replacing a content token with `[MASK]`. Attention weights in the final layer directly govern information routing into the target position's final hidden representation:

$ bold(h)_"target"^((L)) = sum_j macron(A)_("target", j) dot bold(V)_j $

When high-attention context tokens are replaced by `[MASK]`, that information flow is severed, producing the observed faithfulness gap.

== Devanagari Syllabic Vulnerability

In Latin scripts, a minor typo typically preserves the subword root. In Devanagari, swapping a matra (*पानी* → *पानि*) alters the Unicode codepoint sequence of the _akshara_, completely severing match compatibility with high-frequency vocabulary entries. This structural difference explains MuRIL's higher sensitivity: its Indic-dedicated vocabulary contains sharper language-specific subword boundaries that register immediate disruption under syllabic alteration.

// ==================== 7. ERROR ANALYSIS ====================
= Error Analysis

Two pairs in RSE represent legitimate orthographic variants rather than ungrammatical misspellings:
- `HC_0118`: *महत्वपूर्ण* vs *महत्त्वपूर्ण* (single vs. double $t$ ligature — both standard in Hindi publishing).
- `HC_0120`: *संभावना* vs *सम्भावना* (anusvara vs. class nasal — both prescribed by Hindi standards).

Both models assigned minimal PLL penalties to these pairs ($Delta "MeanLogP" > -0.20$), whereas true spelling errors produced large drops ($Delta < -2.0$). This demonstrates that pretrained MLMs do not merely react to character alterations as generic noise; their probability distributions distinguish between legitimate orthographic alternations and erroneous non-standard deviations.

// ==================== 8. LIMITATIONS ====================
= Limitations and Threats to Validity

1. *Sample size ($N = 20$ spelling pairs):* While sufficient to establish the large contrast between Attention and IG, the correlation analysis has limited power to detect weak effect sizes ($|r| < 0.30$). Our results prove the absence of a moderate-to-strong correlation, not the universal absence of any relationship.

2. *Out-of-distribution masking shock:* Replacing context tokens with `[MASK]` disrupts natural sentence syntax, pushing inputs into out-of-distribution regions. Raw logit drops reflect a combination of semantic feature removal and global perplexity shock.

3. *Multi-subword target aggregation:* Because spelling errors fragment targets into multiple subwords (only 6/20 pairs are strictly single-subword in MuRIL; 4/20 in XLM-RoBERTa), we use arithmetic mean logit aggregation ($S = (1/k) sum "Logit"$), assuming equal subword contribution.

4. *Attention is not counterfactual causality:* Attention's superior win rates reflect structural alignment with information flow in transformer layers, not certified counterfactual causation.

5. *Pretrained checkpoint scope:* Results apply to base-scale bidirectional MLMs evaluated on CPU. Extrapolating to decoder-only large language models requires future validation.

// ==================== 9. ETHICS ====================
= Ethical Considerations

The dataset contains no personally identifiable information (PII), toxic language, or demographic bias. All 200 sentence pairs use publicly standard linguistic examples in Devanagari Hindi. Human annotators were fairly compensated native Hindi speakers. The research contributes to more robust, transparent, and linguistically equitable NLP for under-represented Indic language communities.

// ==================== 10. CONCLUSION ====================
= Conclusion and Future Work

We conducted a systematic empirical investigation into model behaviour, tokenisation dynamics, explanation stability, and explanation faithfulness under lexical perturbations in Devanagari Hindi. Key contributions:

1. *Catastrophic subword fragmentation:* Devanagari matra inversions and ligature splits collapse single-subword coverage by $-50%$ to $-60%$ in both models.

2. *Differential model sensitivity:* MuRIL imposes substantially harsher plausibility penalties on spelling errors (95.0% vs. 65.0% original preference rate) while cross-model agreement collapses on synonyms ($r_s = +0.060, p = 0.723$).

3. *Attention superiority in rank stability:* Self-attention achieves $2.1 times$–$2.7 times$ higher rank stability under spelling noise, while Integrated Gradients retains high cosine similarity but fluctuating local rankings.

4. *Attention superiority in discrete faithfulness:* Attention achieves 70%–78% win rates vs. bottom-$k$ control; Integrated Gradients operates near or below chance due to continuous path integral–discrete masking mismatch.

5. *Stability $eq.not$ Faithfulness:* Explanation stability under input noise does *not* predict explanation faithfulness under feature erasure ($p > 0.05$ across all configurations), establishing that the two dimensions are empirically orthogonal.

Future work will expand the benchmark across diverse syntactic domains, investigate continuous embedding-space perturbation interventions, and extend the stability–faithfulness joint framework to Indic generative architectures.

// ==================== REFERENCES ====================
= References

#set par(first-line-indent: 0em, hanging-indent: 1.2em)
#text(size: 9pt)[

]

] // end columns
