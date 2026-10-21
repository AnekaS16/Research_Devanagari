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
