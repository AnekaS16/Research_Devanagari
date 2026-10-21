
#set page(paper: "a4", margin: (x: 2cm, y: 2.5cm))
#set math.equation(numbering: "(1)")
#set text(font: ("Times New Roman", "Nirmala UI", "Mangal"), size: 10pt)
= Beyond Tokenization

This is a test equation:
$ Delta "MeanLogP" = sum_(t=1)^(|W|) log P(w_t | W without {w_t}; Theta) $ <eq1>
As shown in @eq1, Greek letters $rho$, $Delta$, $alpha$ render correctly,
and Hindi words *पानी* vs *पानि* and *महत्वपूर्ण* vs *महत्त्वपूर्ण* display without issues.

== Model Architecture
- MuRIL (`google/muril-base-cased`): WordPiece, 197,285-token vocabulary
- XLM-RoBERTa (`FacebookAI/xlm-roberta-base`): BPE SentencePiece, 250,002-token vocabulary

=== Statistics
The cross-model Spearman rank correlation was $r_s = +0.4408, p < 10^(-9)$.
