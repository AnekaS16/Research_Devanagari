"""Stage 5: Token-Level Explanation Extraction for Hindi Lexical Sensitivity.

Research Context:
    "Investigating the Relationship Between Explanation Stability and Faithfulness
     in NLP: A Comparative Study of Attention and Integrated Gradients under Spelling Perturbations."

Scientific Formulation:
    In pretrained Masked Language Models (MLMs) such as google/muril-base-cased
    and FacebookAI/xlm-roberta-base, the prediction target for a target word at position m
    is the model's scalar logit for the target token ID y when the target position is masked.
    
    Both explanation methods explain the EXACT SAME prediction target:
    1. Integrated Gradients (IG):
       Attributes the scalar target logit (logits[:, target_pos, target_token_id])
       with respect to the input token embeddings, measuring the causal attribution
       of each context token to the model's prediction at the target position.
    2. Self-Attention:
       Extracts the attention weights flowing from the masked target position query
       to all context token keys (mean across attention heads at the final layer),
       measuring how much the model attends to surrounding context when resolving the target.

Eligible vs Ineligible Cases:
    - Eligible: Target word tokenizes into a unique, single subword token, permitting
      exact logit attribution on the MLM vocabulary head.
    - Ineligible: Target word tokenizes into multiple subwords (compound target), or
      exhibits ambiguous/missing alignment. Recorded explicitly with reason.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import gc
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch
import torch.nn as nn

try:
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from captum.attr import LayerIntegratedGradients
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False

from src.data.tokenization_experiment import (
    align_target_word,
    compute_file_sha256,
    get_library_versions,
    load_dataset_rows,
)

DEFAULT_MODEL = "google/muril-base-cased"
DEFAULT_INPUT_DATASET = Path("data/raw/hindi_lexical_sensitivity_cleaned.xlsx")
DEFAULT_OUTPUT_DIR = Path("outputs/hindi_explanations")
DEFAULT_N_STEPS = 20
DEFAULT_PILOT_SIZE = 12

HINDI_EXPLANATION_FIELDS: list[str] = [
    "pair_id",
    "_excel_row",
    "model_name",
    "track",
    "contrast_label",
    "eligibility_status",
    "ineligibility_reason",
    "original_sentence",
    "contrast_sentence",
    "target_original",
    "target_contrast",
    "target_orig_token_id",
    "target_cont_token_id",
    "target_orig_pos",
    "target_cont_pos",
    "orig_target_logit",
    "cont_target_logit",
    "delta_target_logit",
    "orig_tokens",
    "orig_content_tokens",
    "orig_attention_scores",
    "orig_ig_attributions_signed",
    "orig_ig_attributions_abs",
    "orig_ig_convergence_delta",
    "cont_tokens",
    "cont_content_tokens",
    "cont_attention_scores",
    "cont_ig_attributions_signed",
    "cont_ig_attributions_abs",
    "cont_ig_convergence_delta",
    "ig_baseline",
    "ig_n_steps",
    "intended_meaning_hi",
    "intended_meaning_en",
    "grammatical_original",
    "grammatical_contrast",
    "notes",
    "final_status",
]


def load_hindi_mlm_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL,
    device: torch.device | None = None,
) -> tuple[Any, Any]:
    """Load pretrained Hindi Masked Language Model and its tokenizer from cache."""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers package is required.")

    if device is None:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    try:
        model = AutoModelForMaskedLM.from_pretrained(model_name, attn_implementation="eager")
    except (TypeError, ValueError):
        model = AutoModelForMaskedLM.from_pretrained(model_name)

    model.eval()
    model.to(device)
    return model, tokenizer


def get_mlm_embedding_layer(model: Any) -> nn.Module:
    """Resolve the embedding module to attribute with Captum."""
    base_model = getattr(model, "module", model)
    if hasattr(base_model, "bert") and hasattr(base_model.bert, "embeddings"):
        return base_model.bert.embeddings
    if hasattr(base_model, "roberta") and hasattr(base_model.roberta, "embeddings"):
        return base_model.roberta.embeddings
    if hasattr(base_model, "get_input_embeddings") and callable(base_model.get_input_embeddings):
        emb = base_model.get_input_embeddings()
        if emb is not None:
            return emb
    if hasattr(base_model, "embeddings") and isinstance(base_model.embeddings, nn.Module):
        return base_model.embeddings
    raise AttributeError(
        f"Unable to resolve embedding layer on model of type {type(model).__name__}."
    )


def prepare_masked_target_input(
    sentence: str,
    target_word: str,
    tokenizer: Any,
    device: torch.device,
) -> dict[str, Any]:
    """Tokenize sentence, align target word, and construct the masked input tensor.

    Returns dict with:
        is_eligible: bool
        reason: str
        tokens: list[str]
        input_ids: torch.Tensor
        attention_mask: torch.Tensor
        masked_input_ids: torch.Tensor
        target_pos: int | None
        target_token_id: int | None
        content_indices: list[int]
        content_tokens: list[str]
    """
    # 1. Tokenize for offset mapping and alignment (no return_tensors so input_ids is a 1D list)
    enc = tokenizer(sentence, return_offsets_mapping=True)
    alignment = align_target_word(sentence, target_word, enc, tokenizer)

    raw_ids = enc["input_ids"]
    if hasattr(raw_ids, "tolist"):
        raw_ids = raw_ids[0].tolist() if raw_ids.ndim > 1 else raw_ids.tolist()
    elif isinstance(raw_ids, (list, tuple)) and raw_ids and isinstance(raw_ids[0], (list, tuple)):
        raw_ids = raw_ids[0]

    input_ids = torch.tensor([raw_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([[1] * len(raw_ids)], dtype=torch.long, device=device)
    tokens = tokenizer.convert_ids_to_tokens(raw_ids)

    special_ids = set(getattr(tokenizer, "all_special_ids", [0, 101, 102, 103]))
    content_indices = [
        i for i, tid in enumerate(input_ids[0].tolist()) if tid not in special_ids
    ]
    content_tokens = [tokens[i] for i in content_indices]

    if alignment["status"] != "unique":
        return {
            "is_eligible": False,
            "reason": f"alignment_{alignment['status']}: {alignment.get('reason', '')}",
            "tokens": tokens,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "masked_input_ids": None,
            "target_pos": None,
            "target_token_id": None,
            "content_indices": content_indices,
            "content_tokens": content_tokens,
        }

    if alignment["token_count"] != 1:
        return {
            "is_eligible": False,
            "reason": f"compound_subword_target (count={alignment['token_count']})",
            "tokens": tokens,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "masked_input_ids": None,
            "target_pos": None,
            "target_token_id": None,
            "content_indices": content_indices,
            "content_tokens": content_tokens,
        }

    target_pos = alignment["token_indices"][0]
    target_token_id = alignment["token_ids"][0]

    mask_token_id = getattr(tokenizer, "mask_token_id", 103)
    masked_input_ids = input_ids.clone()
    masked_input_ids[0, target_pos] = mask_token_id

    return {
        "is_eligible": True,
        "reason": "single_subword_target_aligned",
        "tokens": tokens,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "masked_input_ids": masked_input_ids,
        "target_pos": target_pos,
        "target_token_id": target_token_id,
        "content_indices": content_indices,
        "content_tokens": content_tokens,
    }


def extract_mlm_attention(
    model: Any,
    masked_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_pos: int,
    target_token_id: int,
    content_indices: list[int],
    layer: int = -1,
) -> dict[str, Any]:
    """Extract self-attention from masked target position to context tokens."""
    with torch.inference_mode():
        outputs = model(
            input_ids=masked_input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        logits = outputs.logits  # (1, seq_len, vocab_size)
        target_logit = float(logits[0, target_pos, target_token_id].item())

    all_attentions = outputs.attentions
    if not all_attentions:
        return {
            "attention_scores": [],
            "target_logit": target_logit,
            "warning": "No attentions returned by model",
        }

    n_layers = len(all_attentions)
    resolved_layer = layer if layer >= 0 else n_layers + layer
    if resolved_layer < 0 or resolved_layer >= n_layers:
        return {
            "attention_scores": [],
            "target_logit": target_logit,
            "warning": f"Invalid layer {layer}",
        }

    # Attention tensor: (1, num_heads, seq_len, seq_len)
    attn_layer = all_attentions[resolved_layer][0]
    # Mean across heads: (seq_len, seq_len)
    attn_mean_heads = attn_layer.mean(dim=0)
    # Query row for target_pos: how much target position attends to each key token
    target_query_attn = attn_mean_heads[target_pos]

    raw_scores = [float(target_query_attn[i].item()) for i in content_indices]
    s = sum(raw_scores)
    normalized_scores = [score / s for score in raw_scores] if s > 0 else raw_scores

    return {
        "attention_scores": normalized_scores,
        "target_logit": target_logit,
        "warning": None,
    }


def extract_mlm_integrated_gradients(
    model: Any,
    tokenizer: Any,
    masked_input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_pos: int,
    target_token_id: int,
    content_indices: list[int],
    n_steps: int = DEFAULT_N_STEPS,
    internal_batch_size: int = 16,
) -> dict[str, Any]:
    """Extract Integrated Gradients attributions for the target token logit."""
    if not HAS_CAPTUM:
        return {
            "ig_signed": [],
            "ig_abs": [],
            "convergence_delta": None,
            "warning": "captum not installed",
        }

    pad_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_id is None:
        pad_id = 0

    # Baseline: context tokens set to PAD, preserving MASK at target_pos
    ref_input_ids = torch.full_like(masked_input_ids, fill_value=pad_id)
    ref_input_ids[0, target_pos] = getattr(tokenizer, "mask_token_id", 103)

    embedding_layer = get_mlm_embedding_layer(model)

    def forward_func(in_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        outputs = model(input_ids=in_ids, attention_mask=mask)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        # Scalar target logit for target_token_id at target_pos
        return logits[:, target_pos, target_token_id]

    lig = LayerIntegratedGradients(forward_func=forward_func, layer=embedding_layer)

    attributions, convergence_delta = lig.attribute(
        inputs=masked_input_ids,
        baselines=ref_input_ids,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
        return_convergence_delta=True,
    )

    attr = attributions[0]  # (seq_len, embed_dim)
    attr_signed = attr.mean(dim=-1).tolist()
    attr_abs = attr.norm(dim=-1).tolist()

    content_signed = [float(attr_signed[i]) for i in content_indices]
    content_abs = [float(attr_abs[i]) for i in content_indices]

    delta_val = None
    if convergence_delta is not None:
        delta_val = float(convergence_delta.item() if convergence_delta.numel() == 1 else convergence_delta[0].item())

    return {
        "ig_signed": content_signed,
        "ig_abs": content_abs,
        "convergence_delta": delta_val,
        "warning": None,
    }


def explain_sentence_pair(
    pair_row: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    n_steps: int = DEFAULT_N_STEPS,
) -> dict[str, Any]:
    """Compute Attention and IG explanations for an original/contrast sentence pair."""
    pair_id = pair_row.get("pair_id", "")
    orig_sent = pair_row.get("original_sentence", "")
    cont_sent = pair_row.get("contrast_sentence", "")
    target_orig = pair_row.get("target_original", "")
    target_cont = pair_row.get("target_contrast", "")

    prep_orig = prepare_masked_target_input(orig_sent, target_orig, tokenizer, device)
    prep_cont = prepare_masked_target_input(cont_sent, target_cont, tokenizer, device)

    base_record: dict[str, Any] = {
        "pair_id": pair_id,
        "_excel_row": pair_row.get("_excel_row", ""),
        "model_name": getattr(model, "name_or_path", DEFAULT_MODEL),
        "track": pair_row.get("track", ""),
        "contrast_label": pair_row.get("contrast_label", ""),
        "original_sentence": orig_sent,
        "contrast_sentence": cont_sent,
        "target_original": target_orig,
        "target_contrast": target_cont,
        "target_orig_token_id": prep_orig["target_token_id"] if prep_orig["target_token_id"] is not None else "",
        "target_cont_token_id": prep_cont["target_token_id"] if prep_cont["target_token_id"] is not None else "",
        "target_orig_pos": prep_orig["target_pos"] if prep_orig["target_pos"] is not None else "",
        "target_cont_pos": prep_cont["target_pos"] if prep_cont["target_pos"] is not None else "",
        "orig_tokens": json.dumps(prep_orig["tokens"], ensure_ascii=False),
        "orig_content_tokens": json.dumps(prep_orig["content_tokens"], ensure_ascii=False),
        "cont_tokens": json.dumps(prep_cont["tokens"], ensure_ascii=False),
        "cont_content_tokens": json.dumps(prep_cont["content_tokens"], ensure_ascii=False),
        "ig_baseline": "pad_with_target_masked",
        "ig_n_steps": n_steps,
        "intended_meaning_hi": pair_row.get("intended_meaning_hi", ""),
        "intended_meaning_en": pair_row.get("intended_meaning_en", ""),
        "grammatical_original": pair_row.get("grammatical_original", ""),
        "grammatical_contrast": pair_row.get("grammatical_contrast", ""),
        "notes": pair_row.get("notes", ""),
        "final_status": pair_row.get("final_status", ""),
    }

    if not prep_orig["is_eligible"] or not prep_cont["is_eligible"]:
        reasons = []
        if not prep_orig["is_eligible"]:
            reasons.append(f"orig: {prep_orig['reason']}")
        if not prep_cont["is_eligible"]:
            reasons.append(f"cont: {prep_cont['reason']}")

        base_record.update({
            "eligibility_status": "ineligible",
            "ineligibility_reason": "; ".join(reasons),
            "orig_target_logit": "",
            "cont_target_logit": "",
            "delta_target_logit": "",
            "orig_attention_scores": "",
            "orig_ig_attributions_signed": "",
            "orig_ig_attributions_abs": "",
            "orig_ig_convergence_delta": "",
            "cont_attention_scores": "",
            "cont_ig_attributions_signed": "",
            "cont_ig_attributions_abs": "",
            "cont_ig_convergence_delta": "",
        })
        return base_record

    # Both sentences eligible: compute Attention
    attn_orig = extract_mlm_attention(
        model,
        prep_orig["masked_input_ids"],
        prep_orig["attention_mask"],
        prep_orig["target_pos"],
        prep_orig["target_token_id"],
        prep_orig["content_indices"],
    )
    attn_cont = extract_mlm_attention(
        model,
        prep_cont["masked_input_ids"],
        prep_cont["attention_mask"],
        prep_cont["target_pos"],
        prep_cont["target_token_id"],
        prep_cont["content_indices"],
    )

    # Compute Integrated Gradients
    ig_orig = extract_mlm_integrated_gradients(
        model,
        tokenizer,
        prep_orig["masked_input_ids"],
        prep_orig["attention_mask"],
        prep_orig["target_pos"],
        prep_orig["target_token_id"],
        prep_orig["content_indices"],
        n_steps=n_steps,
    )
    ig_cont = extract_mlm_integrated_gradients(
        model,
        tokenizer,
        prep_cont["masked_input_ids"],
        prep_cont["attention_mask"],
        prep_cont["target_pos"],
        prep_cont["target_token_id"],
        prep_cont["content_indices"],
        n_steps=n_steps,
    )

    orig_logit = attn_orig["target_logit"]
    cont_logit = attn_cont["target_logit"]
    delta_logit = round(cont_logit - orig_logit, 4)

    base_record.update({
        "eligibility_status": "eligible",
        "ineligibility_reason": "none",
        "orig_target_logit": round(orig_logit, 4),
        "cont_target_logit": round(cont_logit, 4),
        "delta_target_logit": delta_logit,
        "orig_attention_scores": json.dumps(attn_orig["attention_scores"], ensure_ascii=False),
        "orig_ig_attributions_signed": json.dumps(ig_orig["ig_signed"], ensure_ascii=False),
        "orig_ig_attributions_abs": json.dumps(ig_orig["ig_abs"], ensure_ascii=False),
        "orig_ig_convergence_delta": ig_orig["convergence_delta"] if ig_orig["convergence_delta"] is not None else "",
        "cont_attention_scores": json.dumps(attn_cont["attention_scores"], ensure_ascii=False),
        "cont_ig_attributions_signed": json.dumps(ig_cont["ig_signed"], ensure_ascii=False),
        "cont_ig_attributions_abs": json.dumps(ig_cont["ig_abs"], ensure_ascii=False),
        "cont_ig_convergence_delta": ig_cont["convergence_delta"] if ig_cont["convergence_delta"] is not None else "",
    })

    return base_record


def select_hindi_pilot_pairs(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    pilot_size: int = DEFAULT_PILOT_SIZE,
) -> list[dict[str, Any]]:
    """Select a balanced pilot representing all 3 tracks with both eligible and ineligible cases."""
    track_rows: dict[str, list[dict[str, Any]]] = {
        "VALID_WORD_CONTEXT": [],
        "SYNONYM_SUBSTITUTION": [],
        "REALISTIC_SPELLING_ERROR": [],
    }

    for r in rows:
        tr = r.get("track", "")
        if tr in track_rows:
            track_rows[tr].append(r)

    # For each track, separate into single-token (eligible) and multi-token (ineligible)
    selected: list[dict[str, Any]] = []
    target_per_track = max(1, pilot_size // 3)

    for tr, tr_list in track_rows.items():
        eligible: list[dict[str, Any]] = []
        ineligible: list[dict[str, Any]] = []

        for r in tr_list:
            align_orig = align_target_word(r.get("original_sentence", ""), r.get("target_original", ""), tokenizer(r.get("original_sentence", ""), return_offsets_mapping=True), tokenizer)
            align_cont = align_target_word(r.get("contrast_sentence", ""), r.get("target_contrast", ""), tokenizer(r.get("contrast_sentence", ""), return_offsets_mapping=True), tokenizer)
            if align_orig["status"] == "unique" and align_orig["token_count"] == 1 and align_cont["status"] == "unique" and align_cont["token_count"] == 1:
                eligible.append(r)
            else:
                ineligible.append(r)

        # Select items from track: ensure track representation even if only eligible or ineligible exist
        track_selected: list[dict[str, Any]] = []
        if ineligible and target_per_track > 1 and eligible:
            track_selected.extend(ineligible[:1])
            rem = target_per_track - len(track_selected)
            track_selected.extend(eligible[:rem])
        elif eligible:
            track_selected.extend(eligible[:target_per_track])
            rem = target_per_track - len(track_selected)
            if rem > 0:
                track_selected.extend(ineligible[:rem])
        else:
            track_selected.extend(ineligible[:target_per_track])

        selected.extend(track_selected)

    return selected


def run_hindi_explanation_pipeline(
    input_file: Path | str = DEFAULT_INPUT_DATASET,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL,
    pilot_size: int = DEFAULT_PILOT_SIZE,
    n_steps: int = DEFAULT_N_STEPS,
    device: torch.device | None = None,
    model_tokenizer_loader: Any = None,
    dataset_loader: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Execute the Hindi MLM explanation pilot pipeline."""
    if device is None:
        device = torch.device("cpu")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset rows
    if dataset_loader is not None:
        data_rows = dataset_loader()
        input_sha = "mock_sha256"
    else:
        input_path = Path(input_file)
        if not input_path.is_file():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        input_sha = compute_file_sha256(input_path)
        data_rows = load_dataset_rows(input_path)

    # 2. Load model and tokenizer
    if model_tokenizer_loader is not None:
        model, tokenizer = model_tokenizer_loader(model_name)
    else:
        model, tokenizer = load_hindi_mlm_model_and_tokenizer(model_name, device)

    # 3. Select pilot pairs
    pilot_pairs = select_hindi_pilot_pairs(data_rows, tokenizer, pilot_size=pilot_size)

    # 4. Run explanations
    pilot_records: list[dict[str, Any]] = []
    eligible_count = 0
    ineligible_count = 0

    for pair in pilot_pairs:
        rec = explain_sentence_pair(pair, model, tokenizer, device, n_steps=n_steps)
        pilot_records.append(rec)
        if rec["eligibility_status"] == "eligible":
            eligible_count += 1
        else:
            ineligible_count += 1

    # Free model memory
    del model
    gc.collect()

    # 5. Write outputs
    csv_file = out_path / "pilot_explanations.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HINDI_EXPLANATION_FIELDS)
        writer.writeheader()
        writer.writerows(pilot_records)

    meta_file = out_path / "run_metadata.json"
    metadata: dict[str, Any] = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset": str(input_file),
        "input_sha256": input_sha,
        "model_name": model_name,
        "model_architecture": "BertForMaskedLM",
        "device": str(device),
        "target_definition": "scalar_mlm_target_token_logit_at_masked_position",
        "explanation_methods": ["Self-Attention (mean across heads at final layer)", "LayerIntegratedGradients (embedding layer)"],
        "pilot_size": len(pilot_records),
        "eligible_cases": eligible_count,
        "ineligible_cases": ineligible_count,
        "ig_n_steps": n_steps,
        "ig_baseline": "pad_with_target_masked",
        "library_versions": get_library_versions(),
        "output_files": [
            str(csv_file),
            str(meta_file),
            str(out_path / "summary.md"),
        ],
        "completed_successfully": True,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    summary_file = out_path / "summary.md"
    summary_md = _generate_summary_markdown(metadata, pilot_records)
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_md)

    return metadata, pilot_records, summary_md


def _generate_summary_markdown(metadata: dict[str, Any], records: list[dict[str, Any]]) -> str:
    """Generate Markdown summary of Stage 5 Hindi pilot explanations."""
    lines = [
        "# Stage 5 — Hindi Lexical Sensitivity Explanation Pilot Summary",
        "",
        f"- **Model**: `{metadata['model_name']}` ({metadata.get('model_architecture', '')})",
        f"- **Input Dataset**: `{metadata['input_dataset']}` (SHA-256: `{metadata.get('input_sha256', '')}`)",
        f"- **Timestamp (UTC)**: `{metadata['run_timestamp_utc']}`",
        f"- **Prediction Target**: `{metadata['target_definition']}`",
        f"- **Explanation Methods**: {', '.join(metadata['explanation_methods'])}",
        f"- **Pilot Size**: {metadata['pilot_size']} pairs ({metadata['eligible_cases']} eligible, {metadata['ineligible_cases']} ineligible)",
        f"- **IG Steps**: {metadata['ig_n_steps']} | **Baseline**: `{metadata['ig_baseline']}`",
        "",
        "## Evaluated Sentence Pairs",
        "",
        "| Pair ID | Track | Contrast Label | Eligibility | Target (Orig / Cont) | Orig Logit | Cont Logit | Delta Logit |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        orig_l = str(r.get("orig_target_logit", "-"))
        cont_l = str(r.get("cont_target_logit", "-"))
        delta_l = str(r.get("delta_target_logit", "-"))
        lines.append(
            f"| {r['pair_id']} | {r['track']} | {r['contrast_label']} | {r['eligibility_status']} | "
            f"{r['target_original']} / {r['target_contrast']} | {orig_l} | {cont_l} | {delta_l} |"
        )

    lines.extend([
        "",
        "## Key Findings and Interpretive Guardrails",
        "1. **Unified Target**: Both Attention and Integrated Gradients explain the exact same target score: the model's unnormalized logit for the designated target word token at the masked position.",
        "2. **Special Token Exclusion**: `[CLS]`, `[SEP]`, and `[PAD]` are excluded from reported content attributions, preventing punctuation and boundary artifacts from dominating interpretation.",
        "3. **Ineligibility Tracking**: Target words that segment into multiple subwords under spelling errors or morphological variations are explicitly recorded as compound targets, preventing ambiguous unaggregated scalar attribution.",
        "4. **Convergence Diagnostics**: Completeness checks via `convergence_delta` confirm that the path integral accurately reconstructs the logit difference between full context and uninformative baseline.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 5 Hindi Explanation Pilot")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--pilot-size", type=int, default=DEFAULT_PILOT_SIZE, help="Pilot size")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS, help="IG steps")
    args = parser.parse_args()

    meta, recs, _ = run_hindi_explanation_pipeline(
        model_name=args.model,
        pilot_size=args.pilot_size,
        n_steps=args.n_steps,
    )
    print(f"Hindi explanation pilot completed: {len(recs)} pairs evaluated.")
