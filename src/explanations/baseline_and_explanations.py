"""Stage 5 – Baseline evaluation and token-level explanation extraction.

Research context:
    "Investigating the Relationship Between Explanation Stability and Faithfulness
     in NLP: A Comparative Study of Attention and Integrated Gradients under
     Spelling Perturbations."

This module is ENTIRELY SEPARATE from the Hindi lexical-sensitivity Stages 1–4.
  - Dataset : GLUE SST-2 (English sentiment classification)
  - Model   : distilbert/distilbert-base-uncased-finetuned-sst-2-english
              (pretrained sentiment classifier — no training performed here)
  - Methods : Attention extraction and Integrated Gradients (via Captum)

Conventions
-----------
* Label 1 → POSITIVE sentiment, Label 0 → NEGATIVE sentiment.
* "Pilot" = reproducible, balanced subset of correctly predicted validation
  examples, with separate tracking of error cases.
* Attention scores are treated as attribution *candidates*, not ground truth.
* Integrated Gradients (IG) uses the zero-embedding baseline, n_steps=50.
* Special tokens [CLS] and [SEP] are excluded from all token rankings.
* All outputs are saved under outputs/explanations/ — earlier outputs are untouched.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Optional heavy imports – guard for test environments
# ---------------------------------------------------------------------------
try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from captum.attr import LayerIntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

try:
    import datasets as hf_datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}
SEED = 42
PILOT_SIZE = 40          # target correctly-predicted examples
PILOT_ERROR_MAX = 10     # additionally retain up to 10 error cases

# IG configuration
IG_N_STEPS = 50
IG_INTERNAL_BATCH_SIZE = 8   # forward passes per IG call — keep low for 8 GB RAM

# Explanation CSV fieldnames
EXPLANATION_FIELDS = [
    "example_id",
    "sentence",
    "true_label",
    "predicted_label",
    "confidence",
    "is_correct",
    "tokens",                  # JSON list
    "n_tokens",
    "attention_scores",        # JSON list (excl. [CLS]/[SEP]) — head-averaged last layer
    "attention_layer",
    "attention_aggregation",
    "ig_attributions_signed",  # JSON list (excl. [CLS]/[SEP]) — L2-aggregated over embed dim
    "ig_attributions_abs",     # JSON list
    "ig_baseline",
    "ig_n_steps",
    "ig_convergence_delta",    # completeness check: sum(attr) ≈ F(x) - F(baseline)
]

BASELINE_PREDICTION_FIELDS = [
    "example_id",
    "sentence",
    "true_label",
    "predicted_label",
    "confidence_positive",
    "confidence_negative",
    "is_correct",
]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_sst2_splits() -> tuple[Any, Any]:
    """Load the official GLUE SST-2 train and validation splits.

    Returns (train_dataset, validation_dataset).
    Validation split is never used for training.
    """
    if not HAS_DATASETS:
        raise ImportError("datasets package is required. Install with: pip install datasets")

    train_ds = hf_datasets.load_dataset("glue", "sst2", split="train")
    val_ds = hf_datasets.load_dataset("glue", "sst2", split="validation")
    return train_ds, val_ds


def compute_class_distribution(dataset: Any) -> dict[str, int]:
    """Return class distribution as {label_name: count}."""
    dist: dict[int, int] = {}
    for item in dataset:
        lbl = item["label"]
        dist[lbl] = dist.get(lbl, 0) + 1
    return {LABEL_MAP.get(k, str(k)): v for k, v in sorted(dist.items())}


# ---------------------------------------------------------------------------
# Model and tokenizer loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    device: torch.device | None = None,
) -> tuple[Any, Any]:
    """Load pretrained DistilBERT SST-2 classifier and its tokenizer.

    Note: This function only LOADS a pretrained checkpoint.
    No training is performed.
    """
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers package is required.")

    if device is None:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prediction (baseline evaluation)
# ---------------------------------------------------------------------------

def predict_batch(
    sentences: list[str],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    max_length: int = 128,
) -> list[dict[str, Any]]:
    """Run forward pass for a batch of sentences.

    Returns list of dicts with predicted_label, confidence_positive,
    confidence_negative, logits.
    """
    enc = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.inference_mode():
        outputs = model(**enc)
        probs = F.softmax(outputs.logits, dim=-1)  # (B, 2)

    results = []
    for i in range(len(sentences)):
        pred_id = torch.argmax(probs[i]).item()
        results.append(
            {
                "predicted_label": pred_id,
                "confidence_positive": probs[i][1].item(),
                "confidence_negative": probs[i][0].item(),
                "confidence": probs[i][pred_id].item(),
            }
        )
    return results


def evaluate_validation_set(
    val_dataset: Any,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    batch_size: int = 32,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate model on all validation examples.

    Returns (predictions_list, metrics_dict).
    """
    sentences = [item["sentence"] for item in val_dataset]
    true_labels = [item["label"] for item in val_dataset]

    all_predictions: list[dict[str, Any]] = []
    for start in range(0, len(sentences), batch_size):
        batch_sents = sentences[start : start + batch_size]
        preds = predict_batch(batch_sents, model, tokenizer, device)
        all_predictions.extend(preds)

    # Compute metrics from scratch (no sklearn dependency)
    tp = tn = fp = fn = 0
    records: list[dict[str, Any]] = []
    for idx, (item, pred) in enumerate(zip(val_dataset, all_predictions)):
        true_lbl = item["label"]
        pred_lbl = pred["predicted_label"]
        is_correct = true_lbl == pred_lbl

        records.append(
            {
                "example_id": idx,
                "sentence": item["sentence"],
                "true_label": true_lbl,
                "predicted_label": pred_lbl,
                "confidence_positive": pred["confidence_positive"],
                "confidence_negative": pred["confidence_negative"],
                "is_correct": is_correct,
            }
        )

        # Positive class = 1
        if true_lbl == 1 and pred_lbl == 1:
            tp += 1
        elif true_lbl == 0 and pred_lbl == 0:
            tn += 1
        elif true_lbl == 0 and pred_lbl == 1:
            fp += 1
        else:
            fn += 1

    n = len(records)
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "n_examples": n,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
    }
    return records, metrics


# ---------------------------------------------------------------------------
# Pilot subset selection
# ---------------------------------------------------------------------------

def select_pilot_examples(
    records: list[dict[str, Any]],
    pilot_size: int = PILOT_SIZE,
    error_max: int = PILOT_ERROR_MAX,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    """Select a reproducible, balanced pilot subset.

    - pilot_size correctly predicted examples, balanced across positive/negative.
    - Up to error_max additionally retained error cases for diagnostics.
    """
    rng = random.Random(seed)

    correct = [r for r in records if r["is_correct"]]
    errors = [r for r in records if not r["is_correct"]]

    # Balance by true label
    pos_correct = [r for r in correct if r["true_label"] == 1]
    neg_correct = [r for r in correct if r["true_label"] == 0]

    n_each = pilot_size // 2
    rng.shuffle(pos_correct)
    rng.shuffle(neg_correct)
    selected = pos_correct[:n_each] + neg_correct[:n_each]

    # If uneven, pad from whichever group has more
    if len(selected) < pilot_size:
        remaining = [r for r in correct if r not in selected]
        rng.shuffle(remaining)
        selected += remaining[: pilot_size - len(selected)]

    rng.shuffle(errors)
    selected += errors[:error_max]
    return selected


# ---------------------------------------------------------------------------
# Attention extraction
# ---------------------------------------------------------------------------

def extract_attention(
    sentence: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    layer: int = -1,
    head_aggregation: str = "mean",
    max_length: int = 128,
) -> dict[str, Any]:
    """Extract token-level attention scores from a DistilBERT model.

    Layer selection:
        layer = -1 → last transformer layer (index 5 for DistilBERT-base).
    Head aggregation:
        "mean" → mean across all attention heads.

    Special tokens [CLS] and [SEP] are excluded from the returned scores.

    Returns:
        tokens         : list[str]  all tokens including [CLS]/[SEP]
        content_tokens : list[str]  tokens excluding [CLS]/[SEP]
        content_scores : list[float] attention values for content tokens
        attention_layer: int  resolved layer index
        aggregation    : str
        warning        : str or None
    """
    enc = tokenizer(
        sentence,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    input_ids = enc["input_ids"][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    with torch.inference_mode():
        outputs = model(**enc, output_attentions=True)

    # outputs.attentions: tuple of (1, n_heads, seq_len, seq_len) per layer
    all_attn = outputs.attentions  # len == n_layers

    resolved_layer = layer if layer >= 0 else len(all_attn) + layer
    if resolved_layer < 0 or resolved_layer >= len(all_attn):
        return {
            "tokens": tokens,
            "content_tokens": [],
            "content_scores": [],
            "attention_layer": resolved_layer,
            "aggregation": head_aggregation,
            "warning": f"Invalid layer index {resolved_layer}",
        }

    # attn_layer: (1, n_heads, seq_len, seq_len) → (n_heads, seq_len, seq_len)
    attn_layer = all_attn[resolved_layer][0]

    if head_aggregation == "mean":
        # Mean over heads → (seq_len, seq_len)
        attn_avg = attn_layer.mean(dim=0)
    else:
        attn_avg = attn_layer[0]

    # Aggregate over "from" positions: mean of each column gives each token's
    # incoming attention weight averaged over all query positions.
    # This summarises how much each token is attended-to.
    token_scores = attn_avg.mean(dim=0).tolist()  # (seq_len,)

    # Identify [CLS] and [SEP] positions using tokenizer's special_ids
    special_ids = set(tokenizer.all_special_ids)
    input_id_list = input_ids.tolist()

    content_indices = [
        i for i, tid in enumerate(input_id_list) if tid not in special_ids
    ]
    content_tokens = [tokens[i] for i in content_indices]
    content_scores = [token_scores[i] for i in content_indices]

    # Normalize content scores to sum to 1 for comparability
    s = sum(content_scores)
    if s > 0:
        content_scores = [v / s for v in content_scores]

    return {
        "tokens": tokens,
        "content_tokens": content_tokens,
        "content_scores": content_scores,
        "attention_layer": resolved_layer,
        "aggregation": head_aggregation,
        "warning": None,
    }


# ---------------------------------------------------------------------------
# Integrated Gradients extraction (via Captum)
# ---------------------------------------------------------------------------

def _get_embedding_layer(model: Any) -> nn.Module:
    """Resolve the embedding module to attribute with Captum."""
    base_model = getattr(model, "module", model)
    if hasattr(base_model, "distilbert") and hasattr(base_model.distilbert, "embeddings"):
        return base_model.distilbert.embeddings
    if hasattr(base_model, "get_input_embeddings") and callable(base_model.get_input_embeddings):
        emb = base_model.get_input_embeddings()
        if emb is not None:
            return emb
    if hasattr(base_model, "embeddings") and isinstance(base_model.embeddings, nn.Module):
        return base_model.embeddings
    raise AttributeError(
        f"Unable to resolve embedding layer on model of type {type(model).__name__}. "
        "Expected `model.distilbert.embeddings` or `model.get_input_embeddings()`."
    )


def _build_ig_forward_func(model: Any, target_class: int | None = None):
    """Return a forward function for Captum LayerIntegratedGradients.

    Takes input_ids and attention_mask so Captum can hook the embedding layer
    during a normal forward pass. Returns model logits or target class logit.
    """
    def forward_func(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        if target_class is not None:
            return logits[:, target_class]
        return logits

    return forward_func


def extract_integrated_gradients(
    sentence: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    target_class: int,
    n_steps: int = IG_N_STEPS,
    internal_batch_size: int = IG_INTERNAL_BATCH_SIZE,
    max_length: int = 128,
) -> dict[str, Any]:
    """Compute Integrated Gradients attributions at the embedding level.

    Baseline: zero-embedding vector (all zeros), representing the absence of
    semantic information. This is a standard choice and is documented explicitly.

    Attributions are aggregated from embedding dimension to token level via L2 norm.

    Special tokens [CLS]/[SEP]/[PAD] are excluded from the returned content scores.

    Returns:
        tokens             : list[str] all tokens
        content_tokens     : list[str] content tokens
        content_ig_signed  : list[float] signed L2 attributions for content tokens
        content_ig_abs     : list[float] absolute L2 attributions
        convergence_delta  : float completeness check value
        baseline           : str description
        n_steps            : int
        warning            : str or None
    """
    if not HAS_CAPTUM:
        return {
            "tokens": [],
            "content_tokens": [],
            "content_ig_signed": [],
            "content_ig_abs": [],
            "convergence_delta": None,
            "baseline": "zero_embedding",
            "n_steps": n_steps,
            "warning": "captum not installed",
        }

    enc = tokenizer(
        sentence,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)         # (1, seq_len)
    attention_mask = enc["attention_mask"].to(device)
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Baseline: all-zero input_ids (maps to [PAD] token in DistilBERT/BERT).
    # Captum hooks the embedding layer and interpolates between the
    # embedding of ref_input_ids and the embedding of input_ids across n_steps.
    pad_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_id is None:
        pad_id = 0
    ref_input_ids = torch.full_like(input_ids, fill_value=pad_id)

    embedding_layer = _get_embedding_layer(model)

    lig = LayerIntegratedGradients(
        forward_func=_build_ig_forward_func(model),
        layer=embedding_layer,
    )

    # IG requires gradient tracking — do NOT wrap in inference_mode
    attributions, convergence_delta = lig.attribute(
        inputs=input_ids,                          # actual token IDs
        baselines=ref_input_ids,                   # baseline token IDs
        additional_forward_args=(attention_mask,),
        target=target_class,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )

    # attributions: (1, seq_len, embed_dim)
    attr = attributions[0]  # (seq_len, embed_dim)

    attr_signed = attr.mean(dim=-1).tolist()   # signed: direction preserved
    attr_abs = attr.norm(dim=-1).tolist()      # absolute: L2 norm

    # Exclude special tokens
    input_id_list = input_ids[0].tolist()
    special_ids = set(tokenizer.all_special_ids)
    content_indices = [i for i, tid in enumerate(input_id_list) if tid not in special_ids]
    content_tokens = [tokens[i] for i in content_indices]
    content_signed = [attr_signed[i] for i in content_indices]
    content_abs = [attr_abs[i] for i in content_indices]

    delta_val = None
    if convergence_delta is not None:
        delta_val = float(convergence_delta.item() if convergence_delta.numel() == 1 else convergence_delta[0].item())

    return {
        "tokens": tokens,
        "content_tokens": content_tokens,
        "content_ig_signed": content_signed,
        "content_ig_abs": content_abs,
        "convergence_delta": delta_val,
        "baseline": "zero_embedding",
        "n_steps": n_steps,
        "warning": None,
    }



# ---------------------------------------------------------------------------
# Explanation record builder
# ---------------------------------------------------------------------------

def build_explanation_record(
    record: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Compute Attention and IG explanations for one example."""
    sentence = record["sentence"]
    predicted_label = record["predicted_label"]

    attn_result = extract_attention(sentence, model, tokenizer, device)
    ig_result = extract_integrated_gradients(
        sentence, model, tokenizer, device, target_class=predicted_label
    )

    return {
        "example_id": record["example_id"],
        "sentence": sentence,
        "true_label": LABEL_MAP.get(record["true_label"], str(record["true_label"])),
        "predicted_label": LABEL_MAP.get(predicted_label, str(predicted_label)),
        "confidence": record.get("confidence_positive") if predicted_label == 1 else record.get("confidence_negative"),
        "is_correct": record["is_correct"],
        "tokens": json.dumps(attn_result["tokens"], ensure_ascii=False),
        "n_tokens": len(attn_result["content_tokens"]),
        "attention_scores": json.dumps(
            [round(v, 6) for v in attn_result["content_scores"]], ensure_ascii=False
        ),
        "attention_layer": attn_result["attention_layer"],
        "attention_aggregation": attn_result["aggregation"],
        "ig_attributions_signed": json.dumps(
            [round(v, 6) for v in ig_result["content_ig_signed"]], ensure_ascii=False
        ),
        "ig_attributions_abs": json.dumps(
            [round(v, 6) for v in ig_result["content_ig_abs"]], ensure_ascii=False
        ),
        "ig_baseline": ig_result["baseline"],
        "ig_n_steps": ig_result["n_steps"],
        "ig_convergence_delta": ig_result["convergence_delta"],
    }


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def generate_explanation_summary(
    pilot_records: list[dict[str, Any]],
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> str:
    """Generate a qualitative human-readable summary of the pilot."""
    n_correct = sum(1 for r in pilot_records if r["is_correct"])
    n_errors = sum(1 for r in pilot_records if not r["is_correct"])

    lines = [
        "# Stage 5 – Baseline & Explanation Extraction Summary",
        "",
        "> **Scope**: English SST-2 sentiment classification using DistilBERT.",
        "> This is SEPARATE from the Hindi lexical-sensitivity Stages 1–4.",
        "",
        f"**Run Timestamp (UTC):** {metadata.get('run_timestamp_utc')}",
        f"**Model:** `{metadata.get('model_name')}`",
        f"**Dataset:** GLUE SST-2 (validation split)",
        f"**Device:** {metadata.get('device')}",
        "",
        "## 1. Baseline Evaluation (Full Validation Set)",
        f"| Metric | Value |",
        f"|---|---|",
        f"| N Examples | {metrics.get('n_examples')} |",
        f"| Accuracy | {metrics.get('accuracy', 0):.4f} |",
        f"| Precision | {metrics.get('precision', 0):.4f} |",
        f"| Recall | {metrics.get('recall', 0):.4f} |",
        f"| F1 | {metrics.get('f1', 0):.4f} |",
    ]

    cm = metrics.get("confusion_matrix", {})
    lines += [
        "",
        "**Confusion Matrix (Positive class = POSITIVE)**",
        f"| | Predicted POSITIVE | Predicted NEGATIVE |",
        f"|---|---|---|",
        f"| True POSITIVE | {cm.get('TP', 0)} | {cm.get('FN', 0)} |",
        f"| True NEGATIVE | {cm.get('FP', 0)} | {cm.get('TN', 0)} |",
    ]

    lines += [
        "",
        "## 2. Pilot Subset",
        f"- **Correctly predicted examples**: {n_correct}",
        f"- **Error cases retained for diagnostics**: {n_errors}",
        f"- **Total pilot records**: {len(pilot_records)}",
        "",
        "## 3. Explanation Methods",
        "",
        "### Attention",
        f"- Source layer: last transformer layer (layer 5 in DistilBERT-base)",
        f"- Head aggregation: mean across all 12 attention heads",
        f"- Score aggregation: mean incoming attention weight per content token",
        f"- Special tokens excluded: [CLS], [SEP]",
        "- **Interpretive note**: Attention weights are not faithful attributions by default.",
        "  They indicate where the model attends, but do not isolate causal importance.",
        "",
        "### Integrated Gradients",
        f"- Library: Captum `LayerIntegratedGradients`",
        f"- Attribution target: predicted class logit",
        f"- Baseline: zero-embedding (absence of semantic information)",
        f"- Steps: {IG_N_STEPS}",
        f"- Embedding aggregation: L2 norm over embedding dimension (abs) and mean (signed)",
        f"- Convergence diagnostic (`delta`) saved per example",
        f"- Special tokens excluded from ranking",
        "",
        "## 4. Qualitative Sample (Top-3 Attention Tokens vs Top-3 IG Tokens)",
    ]

    # Show first 3 correctly predicted examples
    shown = 0
    for rec in pilot_records:
        if not rec["is_correct"] or shown >= 3:
            continue
        attn = json.loads(rec.get("attention_scores", "[]"))
        ig_abs = json.loads(rec.get("ig_attributions_abs", "[]"))
        tokens = json.loads(rec.get("tokens", "[]"))
        special_ids_set = {"[CLS]", "[SEP]", "<s>", "</s>"}
        content_toks = [t for t in tokens if t not in special_ids_set]

        if len(attn) == len(content_toks) and len(ig_abs) == len(content_toks):
            attn_top = sorted(zip(content_toks, attn), key=lambda x: -x[1])[:3]
            ig_top = sorted(zip(content_toks, ig_abs), key=lambda x: -x[1])[:3]
            lines.append(
                f"\n**Example {rec['example_id']}** | Sentence: _{rec['sentence'][:80]}_"
            )
            lines.append(f"- True: `{rec['true_label']}`, Predicted: `{rec['predicted_label']}`, Conf: {rec['confidence']:.3f}")
            lines.append(f"- Top-3 Attention: {', '.join(f'`{t}` ({s:.3f})' for t, s in attn_top)}")
            lines.append(f"- Top-3 IG (abs):  {', '.join(f'`{t}` ({s:.4f})' for t, s in ig_top)}")
            shown += 1

    lines += [
        "",
        "## 5. Artifacts",
        f"- Baseline metrics: `{metadata.get('output_files', {}).get('baseline_metrics_path')}`",
        f"- Baseline predictions: `{metadata.get('output_files', {}).get('baseline_predictions_path')}`",
        f"- Pilot explanations: `{metadata.get('output_files', {}).get('pilot_explanations_path')}`",
        f"- Run metadata: `{metadata.get('output_files', {}).get('metadata_path')}`",
        "",
        "> **Interpretive Guardrail**: Attention and IG scores describe model internals.",
        "> They do not establish grammatical correctness, ground truth importance, or human judgment.",
        "",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_explanation_pipeline(
    output_dir: str | Path = "outputs/explanations",
    model_name: str = MODEL_NAME,
    pilot_size: int = PILOT_SIZE,
    error_max: int = PILOT_ERROR_MAX,
    seed: int = SEED,
    device_override: str | None = None,
    model_tokenizer_loader: Any = None,
    dataset_loader: Any = None,
    force: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Run complete Stage 5: baseline + explanations.

    Returns:
        (metadata, pilot_explanation_records, summary_text)
    """
    start_time = time.time()
    set_seed(seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_override or "cpu")

    # --- Check cache ---
    meta_path = out_dir / "run_metadata.json"
    pilot_path = out_dir / "pilot_explanations.csv"
    if not force and meta_path.is_file() and pilot_path.is_file():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                cached_meta = json.load(f)
            if cached_meta.get("completed_successfully") and cached_meta.get("model_name") == model_name:
                pilot_records = []
                with open(pilot_path, "r", encoding="utf-8") as f:
                    pilot_records = list(csv.DictReader(f))
                summary_path = out_dir / "summary.md"
                summary_text = summary_path.read_text(encoding="utf-8") if summary_path.is_file() else ""
                return cached_meta, pilot_records, summary_text
        except Exception:
            pass

    # --- Load model ---
    if model_tokenizer_loader is not None:
        model, tokenizer = model_tokenizer_loader(model_name)
    else:
        model, tokenizer = load_model_and_tokenizer(model_name, device)

    # --- Load dataset ---
    if dataset_loader is not None:
        train_ds, val_ds = dataset_loader()
    else:
        train_ds, val_ds = load_sst2_splits()

    train_dist = compute_class_distribution(train_ds)
    val_dist = compute_class_distribution(val_ds)

    # --- Evaluate ---
    all_records, metrics = evaluate_validation_set(val_ds, model, tokenizer, device)

    # --- Save baseline predictions ---
    baseline_pred_path = out_dir / "baseline_predictions.csv"
    with open(baseline_pred_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BASELINE_PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "example_id": r["example_id"],
                    "sentence": r["sentence"],
                    "true_label": LABEL_MAP.get(r["true_label"], str(r["true_label"])),
                    "predicted_label": LABEL_MAP.get(r["predicted_label"], str(r["predicted_label"])),
                    "confidence_positive": round(r["confidence_positive"], 6),
                    "confidence_negative": round(r["confidence_negative"], 6),
                    "is_correct": r["is_correct"],
                }
                for r in all_records
            ]
        )

    # --- Save baseline metrics ---
    baseline_metrics_path = out_dir / "baseline_metrics.json"
    metrics_to_save = {
        "model_name": model_name,
        "dataset": "glue/sst2",
        "split": "validation",
        **metrics,
        "train_class_distribution": train_dist,
        "validation_class_distribution": val_dist,
        "note": "Model loaded from pretrained checkpoint — not trained in this pipeline.",
    }
    with open(baseline_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_to_save, f, ensure_ascii=False, indent=2)

    # --- Pilot subset ---
    pilot_subset = select_pilot_examples(all_records, pilot_size=pilot_size, error_max=error_max, seed=seed)

    # --- Extract explanations ---
    pilot_explanation_records: list[dict[str, Any]] = []
    for rec in pilot_subset:
        expl = build_explanation_record(rec, model, tokenizer, device)
        pilot_explanation_records.append(expl)

    # --- Save pilot explanations ---
    with open(pilot_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPLANATION_FIELDS)
        writer.writeheader()
        writer.writerows(pilot_explanation_records)

    elapsed = time.time() - start_time

    # --- Build metadata ---
    import importlib.metadata as _imeta

    def _ver(pkg: str) -> str:
        try:
            return _imeta.version(pkg)
        except Exception:
            return "unknown"

    metadata: dict[str, Any] = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "model_note": "Loaded from pretrained checkpoint — not trained here.",
        "dataset": "glue/sst2",
        "device": str(device),
        "seed": seed,
        "pilot_size_target": pilot_size,
        "pilot_error_max": error_max,
        "ig_n_steps": IG_N_STEPS,
        "ig_baseline": "zero_embedding",
        "attention_layer": "last (-1)",
        "attention_aggregation": "mean_over_heads",
        "train_class_distribution": train_dist,
        "validation_class_distribution": val_dist,
        "baseline_metrics": metrics,
        "elapsed_seconds": elapsed,
        "completed_successfully": True,
        "library_versions": {
            "python": sys.version.split()[0],
            "torch": _ver("torch"),
            "transformers": _ver("transformers"),
            "captum": _ver("captum"),
            "datasets": _ver("datasets"),
            "numpy": _ver("numpy"),
            "platform": platform.platform(),
        },
        "output_files": {
            "baseline_metrics_path": str(baseline_metrics_path.resolve()),
            "baseline_predictions_path": str(baseline_pred_path.resolve()),
            "pilot_explanations_path": str(pilot_path.resolve()),
            "metadata_path": str(meta_path.resolve()),
            "summary_path": str((out_dir / "summary.md").resolve()),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    summary_text = generate_explanation_summary(pilot_explanation_records, metrics, metadata)
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(summary_text)

    return metadata, pilot_explanation_records, summary_text


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(
        description="Stage 5: Baseline + Attention & IG explanation extraction for SST-2."
    )
    parser.add_argument("--output-dir", default="outputs/explanations")
    parser.add_argument("--force", action="store_true", help="Bypass cache")
    parser.add_argument("--pilot-size", type=int, default=PILOT_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    print("=" * 60)
    print("STAGE 5: BASELINE & EXPLANATION EXTRACTION (SST-2 / DistilBERT)")
    print("NOTE: This experiment is SEPARATE from Hindi Stages 1-4.")
    print("=" * 60)

    metadata, pilot_records, summary = run_explanation_pipeline(
        output_dir=args.output_dir,
        pilot_size=args.pilot_size,
        seed=args.seed,
        force=args.force,
    )
    print(summary)
    print("Stage 5 completed successfully.")


if __name__ == "__main__":
    main()
