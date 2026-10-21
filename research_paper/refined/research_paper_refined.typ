// ============================================================
// Beyond Tokenization: Investigating Model Behaviour, Explanation
// Stability, and Faithfulness under Lexical Perturbations in
// Devanagari Hindi
// ============================================================
// Typst 0.15 source — compiled from research_paper/refined/
// All paths are relative to this file's directory (research_paper/refined/)
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
  #text(size: 11pt)[Anonymous Author(s) #sym.dot.c _Under Review_]
  #v(0.5em)
  #line(length: 80%)
]

// ---------- Abstract ----------
#v(0.6em)
#align(center, text(weight: "bold", size: 11pt)[Abstract])
#v(0.3em)

#block(inset: (x: 1.5cm))[
  #text(size: 9.5pt)[
    Masked language models (MLMs) have become central to multilingual NLP, yet their behaviour
    and interpretability under non-Latin, morpho-phonetically complex scripts remain critically
    underexplored. We present a systematic empirical investigation into model behaviour, subword
    tokenisation dynamics, explanation stability, and explanation faithfulness under controlled
    lexical perturbations in Devanagari Hindi. Using a triple-annotated benchmark of 200 minimal
    sentence pairs across three tracks --- Valid Word Context ($N=142$), Synonym Substitution
    ($N=38$), and Realistic Spelling Error ($N=20$) --- we evaluate MuRIL
    (`google/muril-base-cased`) and XLM-RoBERTa (`FacebookAI/xlm-roberta-base`).

    Our experiments reveal four primary findings. First, authentic Devanagari spelling errors
    induce catastrophic subword fragmentation: single-subword target coverage collapses by
    $-60.0%$ in MuRIL ($90.0% -> 30.0%$) and by $-50.0%$ in XLM-RoBERTa ($70.0% -> 20.0%$).
    Second, both models penalise spelling errors under pseudo-log-likelihood (PLL) scoring,
    but MuRIL exhibits substantially harsher penalties (original preferred in $95.0%$ of pairs,
    mean $Delta "MeanLogP" = -2.037$) compared to XLM-RoBERTa ($65.0%$, mean $Delta = -0.670$).
    Third, under a corrected visible-target protocol --- resolving an illusory perfect-stability
    artefact ($rho = 1.000$) induced by target masking --- last-layer self-attention achieves
    $2.1 times$ to $2.7 times$ higher rank stability than Layer Integrated Gradients
    ($rho = 0.704$--$0.743$ vs.\ $0.263$--$0.359$). Fourth, attention attributions yield
    superior faithfulness win rates ($70.0%$--$77.5%$), whereas Integrated Gradients operates
    near or below chance ($42.5%$--$60.0%$). A correlation analysis confirms that explanation
    stability does _not_ predict explanation faithfulness ($p > 0.05$ across all 8 model--method
    configurations; $r_s in [-0.161, +0.270]$).
  ]
]

#v(0.4em)
#align(center)[
  *Keywords:* Devanagari Hindi #sym.dot.c Masked Language Models #sym.dot.c Subword Tokenisation
  #sym.dot.c Pseudo-Log-Likelihood #sym.dot.c Explainable NLP #sym.dot.c Explanation Stability
  #sym.dot.c Explanation Faithfulness #sym.dot.c Integrated Gradients
]
#v(0.6em)
#line(length: 100%, stroke: 0.5pt + luma(180))

// ---------- Two-column body ----------
#columns(2, gutter: 1.2em)[

// ==================== 1. INTRODUCTION ====================
= Introduction

Pretrained masked language models based on bidirectional Transformer architectures have
fundamentally advanced multilingual NLP @conneau-etal-2020-unsupervised @devlin-etal-2019-bert.
However, the empirical behaviour of these models is overwhelmingly documented in Latin-script
languages where typographical noise consists of straightforward character substitutions.

In contrast, Devanagari Hindi follows the Indic _abugida_ orthographic typology: phonological
syllables (_aksharas_) are encoded using consonant bases, dependent vowel diacritics (_matras_),
and virama-linked conjuncts (_ligatures_). Minor Devanagari orthographic variations alter the
Unicode codepoint sequence across syllabic boundaries, frequently destroying canonical subword
vocabulary entries and triggering severe tokenisation fragmentation.

In parallel, the interpretability of Transformer architectures has emerged as a cornerstone of
trustworthy NLP @deyoung-etal-2020-eraser @sundararajan-etal-2017-axiomatic. Two evaluative
criteria are frequently prioritised:

- *Explanation Stability (Robustness):* Invariance of attributions under minor, semantics-preserving
  perturbations @alvarez-melis-jaakkola-2018-robustness.
- *Explanation Faithfulness:* Whether the explanation reflects the model's internal causal
  reasoning, typically evaluated by erasing top-attributed tokens @deyoung-etal-2020-eraser
  @jacovi-goldberg-2020-towards.

A pervasive but largely untested assumption is that attribution stability implies explanation
quality, with the expectation that a robust explanation is also more likely to be faithful.

This study bridges this empirical gap through five research questions:

*RQ1.* How do contextual lexical contrasts, synonym substitutions, and spelling perturbations
affect subword tokenisation and model scoring?

*RQ2.* How do MuRIL and XLM-RoBERTa differ in their responses to these perturbations?

*RQ3.* How stable are Attention and Integrated Gradients explanations under authentic Devanagari
spelling perturbations?

*RQ4.* How do these explanation methods behave under discrete context-token masking interventions?

*RQ5.* Is explanation stability statistically associated with measured explanation faithfulness?

// ==================== 2. BACKGROUND ====================
= Background and Related Work

== Indic-Language Pretrained Models

XLM-RoBERTa @conneau-etal-2020-unsupervised employs a Byte-Pair Encoding SentencePiece
vocabulary of 250,002 tokens shared across 100+ languages. MuRIL @khanuja-etal-2021-muril is a
BERT-base architecture trained on 17 Indic languages and English using a 197,285-token WordPiece
vocabulary. Kakwani et al.\ @kakwani-etal-2020-indicnlpsuite developed the IndicGLUE benchmark,
showing that language-family specialisation improves morphological capture.

== Subword Tokenisation Vulnerability

Subword segmentation algorithms --- BPE @sennrich-etal-2016-neural, WordPiece
@devlin-etal-2019-bert, and SentencePiece Unigram @kudo-richardson-2018-sentencepiece --- are
foundational to modern transformers. Rust et al.\ @rust-etal-2021-good demonstrated that subword
tokenisers exhibit systematic performance gaps in non-Latin scripts. In Devanagari, swapping a
long _dīrgha_ matra for a short _hrasva_ matra (e.g., *पानी* → *पानि*) or omitting a halant
(*आवश्यक* → *आवशयक*) shifts the character sequence across syllabic boundaries, fracturing
canonical vocabulary tokens.

== Masked Language Model Scoring

Salazar et al.\ @salazar-etal-2020-masked formalised *Pseudo-Log-Likelihood (PLL)* scoring for
bidirectional MLMs. By iteratively masking each token and accumulating conditional
log-probabilities, PLL provides an effective proxy for grammatical acceptability.

== Feature Attribution: Attention vs.\ Integrated Gradients

Integrated Gradients (IG) @sundararajan-etal-2017-axiomatic is an axiomatic path-attribution
method satisfying _Completeness_ and _Implementation Invariance_. The attention-as-explanation
debate --- initiated by Jain and Wallace @jain-wallace-2019-attention ("Attention is not
Explanation") and countered by Wiegreffe and Pinter @wiegreffe-pinter-2019-attention --- directly
motivates our comparative evaluation.

== Explanation Stability and Faithfulness

Alvarez-Melis and Jaakkola @alvarez-melis-jaakkola-2018-robustness formalised explanation
stability as local Lipschitz continuity of the attribution mapping. The ERASER benchmark
@deyoung-etal-2020-eraser operationalised faithfulness via comprehensiveness and sufficiency
under token erasure. Jacovi and Goldberg @jacovi-goldberg-2020-towards provided the formal
faithfulness definition we adopt: an explanation is faithful if the features it attributes high
importance to causally govern the model's prediction. Bastings and Filippova
@bastings-filippova-2020-elephant highlighted that gradient methods often behave unpredictably
under discrete feature erasure.

// ==================== 3. DATASET ====================
= Dataset and Annotation Methodology

== Benchmark Design

We curated a benchmark of 200 minimal Hindi sentence pairs ($N = 200$) across three tracks:

- *Valid Word Context (VWC):* $N = 142$ pairs (71.0%). Target word replaced by a grammatically
  valid but contextually mismatched alternative.
- *Synonym Substitution (SYN):* $N = 38$ pairs (19.0%). Target word replaced by a verified
  contextually appropriate synonym (e.g., *पानी* ↔ *जल*).
- *Realistic Spelling Error (RSE):* $N = 20$ pairs (10.0%). Target word replaced by an authentic,
  commonly observed Devanagari misspelling.

== Typology of Devanagari Orthographic Errors

#table(
  columns: (auto, auto, 1fr),
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
#align(center, text(size: 8pt, style: "italic")[Table 1: RSE Spelling Error Typology ($N=20$)])

#v(0.3em)

== Quality Control

All 200 pairs underwent triple annotation: two independent native Hindi linguists plus an expert
adjudicator with definitive `final_status` determination. A forensic audit confirmed:

1. 100% of characters reside in the Devanagari Unicode block (`U+0900`--`U+097F`).
2. *One rejected pair:* `HC_0178` (*सलाह* vs *सलाई*, `final_status = reject`).
3. *Six disagreement overrides:* `HC_0087`, `HC_0099`, `HC_0138`, `HC_0172`, `HC_0180`,
   `HC_0186` --- retained because both sentences are valid Hindi.

The benchmark thus contains *199 accepted* and *1 rejected* pair.

// ==================== 4. METHODOLOGY ====================
= Models and Experimental Methodology

== Pretrained Architectures

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 5pt,
  stroke: 0.5pt + luma(180),
  fill: (_, row) => if row == 0 { rgb("#1a365d") } else if calc.odd(row) { rgb("#f8fafc") } else { white },
  table.header(
    text(fill: white, weight: "bold")[Property],
    text(fill: white, weight: "bold")[XLM-R],
    text(fill: white, weight: "bold")[MuRIL],
  ),
  [Architecture], [RoBERTa-base], [BERT-base],
  [Layers / Heads], [12 / 12], [12 / 12],
  [Hidden dim.], [768], [768],
  [Vocabulary], [250,002], [197,285],
  [Tokeniser], [SentencePiece BPE], [WordPiece],
  [Languages], [100+], [17 Indic + EN],
)
#align(center, text(size: 8pt, style: "italic")[Table 2: Model Architecture Comparison])

#v(0.3em)

== Pseudo-Log-Likelihood Scoring

Following Salazar et al.\ @salazar-etal-2020-masked, for a sentence
$W = (w_1, w_2, dots.c, w_(|W|))$, PLL is computed via iterative single-token masking:

$ "PLL"(W) = sum_(t=1)^(|W|) log P(w_t | W_(-t); Theta) $ <eq-pll>

where $W_(-t)$ denotes the sequence with $w_t$ replaced by the model's mask token, and
$Theta$ are the model parameters.

The length-normalised mean log-probability:

$ "MeanLogP"(W) = frac("PLL"(W), |W_"scorable"|) $ <eq-meanlp>

The signed plausibility shift:

$ Delta "MeanLogP" = "MeanLogP"(W_"cont") - "MeanLogP"(W_"orig") $ <eq-delta>

where $Delta < 0$ means the _original_ is preferred; $Delta > 0$ means the _contrast_ is
preferred.

*Methodological safeguard:* Raw PLL sums are never compared across tokenisers. Only
within-model signed $Delta$ values and preference rates are compared.

== Feature Attribution Methods

*Last-Layer Self-Attention.* We extract the self-attention tensor from Layer 11 (final layer),
averaging across all 12 heads:

$ macron(A)_(i,j) = frac(1, 12) sum_(h=1)^(12) A^((11,h))_(i,j) $ <eq-attn>

Context-token attribution with respect to target token set $T$:

$ alpha_j = frac(1, |T|) sum_(t in T) macron(A)_(t,j) $ <eq-ctxattn>

Special framing tokens are excluded; the vector is $ell_1$-normalised.

*Layer Integrated Gradients (IG).* Using Captum, IG is computed at the word embedding layer
with a zero-vector baseline and 50 Gauss-Legendre quadrature steps:

$ "IG"_i (bold(x)) = (x_i - x'_i) dot.c sum_(k=1)^(50) frac(partial F(bold(x)' + frac(k,50)(bold(x) - bold(x)')), partial x_i) dot.c frac(1, 50) $ <eq-ig>

Token importance is the $L_2$-norm over the 768 embedding dimensions, then context-normalised.

== Stability Metrics

*Rank Stability* (Spearman $rho$): whether the relative importance ordering of context tokens
is preserved under target spelling noise.

*Directional Vector Stability* (Cosine Similarity):

$ "Cosine"(bold(u), bold(v)) = frac(bold(u) dot.c bold(v), ||bold(u)||_2 ||bold(v)||_2) $ <eq-cosine>

=== The Stage 6 Masked-Target Artefact

In the preliminary stability evaluation, computing attributions while replacing the target word
with `[MASK]` in _both_ original and contrast sentences caused both inputs to become
byte-identical:

$ X_"orig" = [w_1, dots.c, "[MASK]", dots.c, w_n] equiv X_"cont" $

This produced illusory perfect stability: $rho = 1.000$, $"Cosine" = 1.000$. *Stage 7*
resolved this via the *Visible-Target Scoring Protocol*: target tokens remain fully visible;
target score is the mean unnormalized logit of target subwords:

$ S(X; Y_"target") = frac(1, |T|) sum_(j in T) "Logit"(p_j, y_j) $ <eq-score>

== Faithfulness Evaluation

Context tokens are ranked and erased at fractions $f in {0.10, 0.20, 0.30}$. The target
logit drop after intervention:

$ Delta S_"intervention" = S(X; Y) - S(X_"masked"; Y) $ <eq-logitdrop>

The *Faithfulness Gap*:

$ "Gap" = Delta S_"top" - Delta S_"bottom" $ <eq-gap>

The *Win Rate*:

$ "WinRate" = frac(1, N) sum_(i=1)^N bb(1)[Delta S_("top",i) > Delta S_("bottom",i)] $ <eq-winrate>

// ==================== 5. RESULTS ====================
= Empirical Results

== RQ1 --- Subword Tokenisation

Authentic Devanagari spelling errors cause catastrophic subword fragmentation:

#table(
  columns: (auto, auto, auto, 1fr),
  inset: 4pt,
  stroke: 0.5pt + luma(180),
  fill: (_, row) => if row == 0 { rgb("#1a365d") } else if calc.odd(row) { rgb("#f8fafc") } else { white },
  table.header(
    text(fill: white, weight: "bold")[Model],
    text(fill: white, weight: "bold")[Track],
    text(fill: white, weight: "bold")[Avg. sub-wds],
    text(fill: white, weight: "bold")[Single-sub-wd coverage],
  ),
  [XLM-R], [RSE], [1.35→2.35], [70.0%→*20.0%* (−50%)],
  [XLM-R], [SYN], [1.13→1.76], [89.5%→60.5%],
  [XLM-R], [VWC], [1.75→1.92], [39.4%→30.3%],
  [MuRIL], [RSE], [1.10→2.10], [90.0%→*30.0%* (−60%)],
  [MuRIL], [SYN], [1.00→1.18], [100.0%→86.8%],
  [MuRIL], [VWC], [1.43→1.51], [64.1%→55.6%],
)
#align(center, text(size: 8pt, style: "italic")[
  Table 3: Subword tokenisation. Source: tokenization\_summary.csv
])

#v(0.3em)

MuRIL encodes Hindi with ~12% fewer subwords than XLM-RoBERTa on VWC ($mu = 8.95$ vs $10.19$
tokens/sentence). On RSE, both models lose exactly $+1.00$ subword/sentence on average.

#box(fill: rgb("#f8fafc"), stroke: 0.5pt + luma(180), inset: 6pt, radius: 3pt)[
  #image("figures/fig1_delta_meanlp_by_track.png", width: 100%)
  #align(center, text(size: 7.5pt, style: "italic")[
    *Fig. 1:* $Delta "MeanLogP"$ distributions by track. Negative = original preferred.
    MuRIL penalty on RSE: $mu = -2.037$; XLM-R: $mu = -0.670$.
  ])
]

== RQ1 --- Plausibility Scoring

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
    text(fill: white, weight: "bold")[Orig. Pref.],
  ),
  [XLM-R], [RSE], [20], [−0.670], [65.0%],
  [XLM-R], [VWC], [142], [−1.093], [81.0%],
  [XLM-R], [SYN], [38], [−0.066], [57.9%],
  [MuRIL], [RSE], [20], [*−2.037*], [*95.0%*],
  [MuRIL], [VWC], [142], [−1.914], [93.0%],
  [MuRIL], [SYN], [38], [−0.581], [84.2%],
)
#align(center, text(size: 8pt, style: "italic")[Table 4: Plausibility scoring. Source: descriptive\_statistics.csv])

#v(0.3em)

Key findings: MuRIL penalises spelling errors far more strongly ($-2.037$ vs $-0.670$).
XLM-RoBERTa shows near-zero mean delta on synonyms ($-0.066$), confirming that the
multilingual generalist treats valid Hindi synonyms as near-interchangeable.

== RQ2 --- Cross-Model Comparison

#table(
  columns: (auto, auto, auto, auto),
  inset: 4pt,
  stroke: 0.5pt + luma(180),
  fill: (_, row) => if row == 0 { rgb("#1a365d") } else if calc.odd(row) { rgb("#f8fafc") } else { white },
  table.header(
    text(fill: white, weight: "bold")[Track],
    text(fill: white, weight: "bold")[$N$],
    text(fill: white, weight: "bold")[Agreement],
    text(fill: white, weight: "bold")[Spearman $r_s$ ($p$-val)],
  ),
  [All Tracks], [200], [74.5%], [$+0.441$ ($p<10^(-9)$)],
  [VWC], [142], [81.0%], [$+0.446$ ($p<10^(-7)$)],
  [RSE], [20], [60.0%], [$+0.215$ ($p=0.363$)],
  [SYN], [38], [57.9%], [$+0.060$ ($p=0.723$)],
)
#align(center, text(size: 8pt, style: "italic")[Table 5: Cross-model agreement. Source: model\_comparison.csv])

#v(0.3em)

#box(fill: rgb("#f8fafc"), stroke: 0.5pt + luma(180), inset: 6pt, radius: 3pt)[
  #image("figures/fig4_model_comparison_scatter.png", width: 100%)
  #align(center, text(size: 7.5pt, style: "italic")[
    *Fig. 2:* Cross-model $Delta "MeanLogP"$ scatter ($N=200$). Overall $r_s = +0.441$.
    Synonym agreement collapses to 57.9% ($p=0.723$).
  ])
]

Cross-model agreement is 74.5% overall ($r_s = +0.441, p < 10^{-9}$). Agreement collapses
on synonyms ($r_s = +0.060, p = 0.723$), revealing model-specific pretraining frequency biases.

#colbreak()

== RQ3 --- Explanation Stability

After resolving the Stage 6 masked-target artefact (Section 4.4), corrected results:

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
    text(fill: white, weight: "bold")[Mean Cos.],
  ),
  [XLM-R], [Attention], [20], [*0.743*], [0.965],
  [XLM-R], [Attn.\ (strict)], [4], [0.689], [0.978],
  [XLM-R], [IG], [20], [0.359], [0.899],
  [MuRIL], [Attention], [20], [*0.704*], [0.964],
  [MuRIL], [Attn.\ (strict)], [6], [0.629], [0.974],
  [MuRIL], [IG], [20], [0.263], [0.988],
)
#align(center, text(size: 8pt, style: "italic")[
  Table 6: Visible-target stability (RSE, $N=20$). Source: stability\_summary.csv
])

#v(0.3em)

#box(fill: rgb("#f8fafc"), stroke: 0.5pt + luma(180), inset: 6pt, radius: 3pt)[
  #image("figures/fig6_explanation_stability.png", width: 100%)
  #align(center, text(size: 7.5pt, style: "italic")[
    *Fig. 3:* Explanation stability. Attention achieves $2.1 times$--$2.7 times$ higher
    rank stability ($rho approx 0.70$--$0.74$) vs.\ IG ($0.26$--$0.36$). Cosine similarity
    remains high for both methods.
  ])
]

Self-attention achieves $2.1 times$--$2.7 times$ higher rank stability than IG. Despite low
rank stability, IG retains high cosine similarity ($0.899$--$0.988$), demonstrating that
gradient vectors maintain global orientation while local rankings fluctuate.

== RQ4 --- Faithfulness Under Discrete Erasure

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
    text(fill: white, weight: "bold")[Win Rate],
  ),
  [XLM-R], [Attention], [0.10], [+4.017], [*77.5%*],
  [XLM-R], [Attention], [0.20], [+4.017], [*77.5%*],
  [XLM-R], [Attention], [0.30], [+5.172], [75.0%],
  [XLM-R], [IG], [0.10], [+1.837], [60.0%],
  [XLM-R], [IG], [0.20], [+1.837], [60.0%],
  [XLM-R], [IG], [0.30], [+3.295], [55.0%],
  [MuRIL], [Attention], [0.30], [+5.794], [*70.0%*],
  [MuRIL], [Attn. (str.)], [0.30], [+8.822], [75.0%],
  [MuRIL], [IG], [0.10], [*−0.686*], [45.0%],
  [MuRIL], [IG], [0.20], [*−0.686*], [45.0%],
  [MuRIL], [IG], [0.30], [+0.033], [42.5%],
)
#align(center, text(size: 8pt, style: "italic")[
  Table 7: Faithfulness ($N=40$ sentences/model). Source: faithfulness\_summary.csv
])

#v(0.3em)

#box(fill: rgb("#f8fafc"), stroke: 0.5pt + luma(180), inset: 6pt, radius: 3pt)[
  #image("figures/fig7_faithfulness_interventions.png", width: 100%)
  #align(center, text(size: 7.5pt, style: "italic")[
    *Fig. 4:* Faithfulness across $f in {0.10, 0.20, 0.30}$. (a) Mean Faithfulness Gap;
    (b) Win Rate vs.\ Bottom-$k$ control.
  ])
]

=== XLM-RoBERTa Out-of-Distribution Logit Dynamics

In XLM-RoBERTa, removing context tokens causes raw target logits to _increase_ due to
out-of-distribution syntactic shock. At $f = 0.20$:
- Top Drop: $Delta S_"top" = -7.19$ (logit _increased_)
- Bottom Drop: $Delta S_"bottom" = -11.21$ (logit increased more)

Directional faithfulness is preserved because bottom masking inflates logits _more_ than top:

$ "Gap" = (-7.19) - (-11.21) = +4.02 > 0 $

This yields 77.5% win rate. The _relative comparison_ remains valid even when raw drops are
unintuitive in sign.

Integrated Gradients operates near or below chance for MuRIL (win rates 42.5%--45.0%; negative
Gap at $f = 0.10, 0.20$). This arises because IG integrates along a continuous embedding-space
path that does not correspond monotonically to discrete `[MASK]` substitutions.

== RQ5 --- Stability--Faithfulness Relationship

#table(
  columns: (auto, auto, 1fr, 1fr),
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
#align(center, text(size: 8pt, style: "italic")[
  Table 8: Stability--faithfulness correlation ($N=20$, $f=0.20$). All $p>0.05$.
])

#v(0.3em)

#box(fill: rgb("#f8fafc"), stroke: 0.5pt + luma(180), inset: 6pt, radius: 3pt)[
  #image("figures/fig8_stability_vs_faithfulness.png", width: 100%)
  #align(center, text(size: 7.5pt, style: "italic")[
    *Fig. 5:* Stability--faithfulness scatter. No significant correlation in any
    configuration ($p > 0.05$). Sign flips between models.
  ])
]

Zero statistically significant correlations ($p > 0.05$) across all 8 model--method
configurations ($p$-values 0.199--0.975). The sign is weakly positive in XLM-RoBERTa
($r_s = +0.270$) and negative in MuRIL ($r_s = -0.161$). Stability and faithfulness are
*empirically decoupled*.

// ==================== 6. DISCUSSION ====================
= Discussion

== Orthogonality of Stability and Faithfulness

The central finding is the complete decoupling of stability and faithfulness. Mathematically,
these metrics occupy distinct spaces:

- *Stability:* Local Lipschitz continuity of the attribution function $g(bold(x))$:
  $cal(S)(g; bold(x), delta) = "Sim"(g(bold(x)), g(bold(x) + delta))$
- *Faithfulness:* Causal dependency of model $f$ on features prioritised by $g$:
  $cal(F)(g; bold(x), f) = f(bold(x)) - f(bold(x)_(without C_"top"))$

An explanation can achieve near-perfect stability by anchoring to static syntactic structures
(e.g., punctuation or common postpositions) that do not vary under local spelling perturbations
--- yet erasing those tokens may have minimal impact on target word prediction, yielding a
*stably unfaithful* explanation. *Optimising XAI benchmarks solely for stability risks selecting
consistently unfaithful explanations.*

== Why IG Fails Under Discrete Erasure

IG accumulates gradients along a continuous straight-line path in embedding space
$bb(R)^d$ where the transformer was never trained to operate. In contrast, faithfulness
evaluation replaces a content token with `[MASK]`. Attention weights in the final layer
directly govern information routing into the target position's final hidden representation:

$ bold(h)_"target"^((L)) = sum_j macron(A)_("target", j) dot.c bold(V)_j $

When high-attention context tokens are replaced by `[MASK]`, that information flow is
severed, producing the observed faithfulness gap.

== Devanagari Syllabic Vulnerability

In Latin scripts, a minor typo typically preserves the subword root. In Devanagari, swapping a
matra (*पानी* → *पानि*) alters the Unicode codepoint sequence of the _akshara_, completely
severing match compatibility with high-frequency vocabulary entries. This explains MuRIL's
higher sensitivity: its Indic-dedicated vocabulary contains sharper language-specific subword
boundaries that register immediate disruption under syllabic alteration.

// ==================== 7. ERROR ANALYSIS ====================
= Error Analysis

Two pairs in RSE represent legitimate orthographic variants rather than ungrammatical misspellings:

- `HC_0118`: *महत्वपूर्ण* vs *महत्त्वपूर्ण* (single vs.\ double $t$ ligature --- both standard).
- `HC_0120`: *संभावना* vs *सम्भावना* (anusvara vs.\ class nasal --- both prescribed by Hindi standards).

Both models assigned minimal PLL penalties to these pairs ($Delta "MeanLogP" > -0.20$), whereas
true spelling errors produced large drops ($Delta < -2.0$). Pretrained MLMs distinguish between
legitimate orthographic alternations and erroneous deviations.

// ==================== 8. LIMITATIONS ====================
= Limitations

1. *Sample size ($N = 20$ RSE pairs):* Sufficient to detect large effect sizes but limited power
   for weak correlations ($|r| < 0.30$). Our results prove the absence of a moderate-to-strong
   correlation, not the universal absence of any relationship.
2. *Out-of-distribution masking shock:* `[MASK]` substitution disrupts syntax, pushing inputs
   into OOD regions. Raw logit drops reflect both semantic removal and global perplexity shock.
3. *Multi-subword target aggregation:* Mean logit aggregation ($S = (1 slash k) sum "Logit"$)
   assumes equal subword contribution.
4. *Attention is not counterfactual causality:* Attention's superior win rates reflect structural
   alignment with information flow, not certified counterfactual causation.
5. *Pretrained checkpoint scope:* Results apply to base-scale bidirectional MLMs on CPU.
   Extrapolating to decoder-only large language models requires future validation.

// ==================== 9. ETHICS ====================
= Ethical Considerations

The dataset contains no personally identifiable information, toxic language, or demographic bias.
All 200 sentence pairs use publicly standard Devanagari Hindi examples. Human annotators were
fairly compensated native Hindi speakers. The research contributes to more robust, transparent,
and linguistically equitable NLP for Indic language communities.

// ==================== 10. CONCLUSION ====================
= Conclusion

We conducted a systematic empirical investigation into model behaviour, tokenisation dynamics,
explanation stability, and faithfulness under lexical perturbations in Devanagari Hindi.
Key contributions:

1. *Catastrophic subword fragmentation:* Matra inversions and ligature splits collapse
   single-subword coverage by $-50%$--$-60%$ in both models.
2. *Differential model sensitivity:* MuRIL imposes substantially harsher spelling-error
   penalties (95.0% vs.\ 65.0% original preference rate). Cross-model agreement collapses on
   synonyms ($r_s = +0.060, p = 0.723$).
3. *Attention superiority in rank stability:* $2.1 times$--$2.7 times$ higher than IG;
   both methods retain high cosine similarity.
4. *Attention superiority in discrete faithfulness:* 70%--78% win rates vs.\ bottom-$k$;
   IG operates near or below chance due to continuous-path vs.\ discrete-masking mismatch.
5. *Stability $eq.not$ Faithfulness:* Explanation stability does _not_ predict faithfulness
   ($p > 0.05$ across all configurations), establishing empirical orthogonality.

Future work will expand the benchmark across diverse syntactic domains, investigate continuous
embedding-space perturbation interventions, and extend this framework to Indic generative
architectures.

// ==================== REFERENCES ====================
= References

#set par(first-line-indent: 0em)
#text(size: 9pt)[
  #bibliography("references.bib", title: none, style: "apa")
]

] // end columns
