"""Stage 6: Hindi Explanation Stability Under Realistic Spelling Perturbations.

Research Context:
    "Investigating the Relationship Between Explanation Stability and Faithfulness
     in NLP: A Comparative Study of Attention and Integrated Gradients under Spelling Perturbations."

Scientific Objectives:
    1. Fixed-Target Scoring:
       Evaluates the model's scalar prediction for the intended canonical target token
       y_orig at the masked target position in both the original and spelling-perturbed sentences:
           Score_orig = Logit(y_orig at m_orig in masked S_orig)
           Score_cont = Logit(y_orig at m_cont in masked S_cont)
       This holds the target concept constant to isolate the effect of context spelling perturbation.

    2. Region-Relative Context Alignment:
       Aligns context tokens outside the target word across original and perturbed sentences.
       Because spelling perturbations change character length (e.g. 'स्वास्थ्य' [len 9] -> 'स्वास्थ' [len 7]),
       context tokens are aligned by region-relative offsets:
       - Pre-target region: tokens before the target word align by absolute text offsets.
       - Post-target region: tokens after the target word align by offsets relative to the end
         of the target span (offset - end_target).
       Special tokens and the target word span are strictly excluded.

    3. Stability Metrics (over aligned context tokens):
       - Spearman rank correlation (rho): measures ranking stability of token importances.
         Uses absolute magnitude for primary analysis; signed IG as a separate diagnostic.
       - Top-k Jaccard overlap: intersection-over-union of top-k most important context tokens.
       - Cosine similarity: vector angle in attribution space.
       Undefined cases (insufficient tokens K < 3, zero variance, zero norm) explicitly return NaN.

    4. Hardware & Memory Safety:
       Sequentially evaluates models on CPU without network downloads.
       Checkpointing ensures interrupted runs resume cleanly.
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
from src.explanations.hindi_explanations import get_mlm_embedding_layer

DEFAULT_MODELS: list[str] = [
    "google/muril-base-cased",
    "FacebookAI/xlm-roberta-base",
]
DEFAULT_INPUT_DATASET = Path("data/raw/hindi_lexical_sensitivity_cleaned.xlsx")
DEFAULT_OUTPUT_DIR = Path("outputs/hindi_explanation_stability")
DEFAULT_N_STEPS = 20

PAIR_LEVEL_CSV_FIELDS: list[str] = [
    "pair_id",
    "_excel_row",
    "model_name",
    "track",
    "contrast_label",
    "method",  # 'attention' or 'integrated_gradients'
    "eligibility_status",  # 'eligible' or 'ineligible'
    "ineligibility_reason",
    "target_original",
    "target_contrast",
    "target_canonical_token_id",
    "target_orig_pos",
    "target_cont_pos",
    "orig_target_logit",
    "cont_target_logit",
    "delta_target_logit",
    "n_aligned_context_tokens",
    "aligned_context_tokens",  # JSON list of token strings
    "unaligned_orig_count",
    "unaligned_cont_count",
    "spearman_rho",  # primary rank correlation (absolute magnitude)
    "spearman_pvalue",
    "top_k_overlap",  # Jaccard overlap of top-k tokens
    "top_k_value",
    "cosine_similarity",  # primary cosine similarity (absolute magnitude / attention)
    "cosine_similarity_signed",  # diagnostic for IG (signed attribution)
    "spearman_rho_signed",  # diagnostic for IG (signed attribution)
    "orig_attributions",  # JSON list of aligned token scores
    "cont_attributions",  # JSON list of aligned token scores
    "orig_convergence_delta",  # IG only
    "cont_convergence_delta",  # IG only
    "notes",
    "final_status",
]

ELIGIBILITY_CSV_FIELDS: list[str] = [
    "pair_id",
    "_excel_row",
    "model_name",
    "track",
    "contrast_label",
    "target_original",
    "target_contrast",
    "target_orig_subwords_count",
    "target_cont_subwords_count",
    "is_eligible",
    "exclusion_reason",
    "final_status",
]

SUMMARY_CSV_FIELDS: list[str] = [
    "model_name",
    "method",
    "track",
    "total_pairs_in_track",
    "evaluated_pairs",
    "eligible_pairs",
    "ineligible_pairs",
    "valid_metric_pairs",
    "spearman_mean",
    "spearman_median",
    "spearman_iqr",
    "top_k_overlap_mean",
    "top_k_overlap_median",
    "cosine_sim_mean",
    "cosine_sim_median",
    "cosine_sim_signed_mean",
]


def load_model_and_tokenizer_offline(
    model_name: str,
    device: torch.device | None = None,
) -> tuple[Any, Any]:
    """Load pretrained MLM and tokenizer from local cache without downloading."""
    if not HAS_TRANSFORMERS:
        raise ImportError("transformers is required.")
    if device is None:
        device = torch.device("cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    try:
        model = AutoModelForMaskedLM.from_pretrained(
            model_name, local_files_only=True, attn_implementation="eager"
        )
    except (TypeError, ValueError):
        model = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True)

    model.eval()
    model.to(device)
    return model, tokenizer


def align_context_tokens_by_region(
    orig_sentence: str,
    orig_target_span: tuple[int, int],
    orig_encoding: Any,
    cont_sentence: str,
    cont_target_span: tuple[int, int],
    cont_encoding: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    """Align context tokens outside target spans using region-relative character offsets.

    Regions:
    - Pre-target: characters before target span (0 to start_target). Relative offset = char_offset.
    - Post-target: characters after target span (end_target to sentence_end).
      Relative offset = char_offset - end_target.

    Special tokens and tokens overlapping the target span are excluded.

    Returns:
        aligned_pairs: list of (orig_token_idx, cont_token_idx)
        aligned_token_labels: list of token string representations (from orig)
        unaligned_orig_indices: list of orig token indices excluded
        unaligned_cont_indices: list of cont token indices excluded
    """
    special_ids = set(getattr(tokenizer, "all_special_ids", [0, 101, 102, 103]))

    orig_ids = orig_encoding["input_ids"]
    if hasattr(orig_ids, "tolist"):
        orig_ids = orig_ids[0].tolist() if orig_ids.ndim > 1 else orig_ids.tolist()
    orig_offsets = orig_encoding.get("offset_mapping", [])

    cont_ids = cont_encoding["input_ids"]
    if hasattr(cont_ids, "tolist"):
        cont_ids = cont_ids[0].tolist() if cont_ids.ndim > 1 else cont_ids.tolist()
    cont_offsets = cont_encoding.get("offset_mapping", [])

    orig_tokens = tokenizer.convert_ids_to_tokens(orig_ids)

    s_orig, e_orig = orig_target_span
    s_cont, e_cont = cont_target_span

    # 1. Map orig context tokens to relative region offsets
    # key: (region, rel_start, rel_end) -> token_idx
    orig_map: dict[tuple[str, int, int], int] = {}
    orig_context_indices: list[int] = []

    for idx, (cstart, cend) in enumerate(orig_offsets):
        if cstart == cend:
            continue
        if orig_ids[idx] in special_ids:
            continue
        # Check if in target span
        if not (cend <= s_orig or cstart >= e_orig):
            continue  # inside or overlapping target span

        orig_context_indices.append(idx)
        if cend <= s_orig:
            key = ("pre", cstart, cend)
        else:
            key = ("post", cstart - e_orig, cend - e_orig)
        orig_map[key] = idx

    # 2. Map cont context tokens to relative region offsets
    cont_map: dict[tuple[str, int, int], int] = {}
    cont_context_indices: list[int] = []

    for idx, (cstart, cend) in enumerate(cont_offsets):
        if cstart == cend:
            continue
        if cont_ids[idx] in special_ids:
            continue
        if not (cend <= s_cont or cstart >= e_cont):
            continue

        cont_context_indices.append(idx)
        if cend <= s_cont:
            key = ("pre", cstart, cend)
        else:
            key = ("post", cstart - e_cont, cend - e_cont)
        cont_map[key] = idx

    # 3. Intersect keys to form aligned pairs
    aligned_pairs: list[tuple[int, int]] = []
    aligned_labels: list[str] = []
    used_orig: set[int] = set()
    used_cont: set[int] = set()

    for key, o_idx in orig_map.items():
        if key in cont_map:
            c_idx = cont_map[key]
            aligned_pairs.append((o_idx, c_idx))
            aligned_labels.append(orig_tokens[o_idx])
            used_orig.add(o_idx)
            used_cont.add(c_idx)

    unaligned_orig = [i for i in orig_context_indices if i not in used_orig]
    unaligned_cont = [i for i in cont_context_indices if i not in used_cont]

    return {
        "aligned_pairs": aligned_pairs,
        "aligned_token_labels": aligned_labels,
        "unaligned_orig_indices": unaligned_orig,
        "unaligned_cont_indices": unaligned_cont,
    }


def compute_spearman_rank_correlation(
    u: list[float], v: list[float]
) -> tuple[float | None, float | None]:
    """Compute Spearman rank correlation with standard tie adjustment.

    Returns (rho, p_value) or (None, None) if K < 3 or zero variance.
    """
    n = len(u)
    if n < 3:
        return None, None

    # Compute fractional ranks with tie handling
    def rank_data(arr: list[float]) -> list[float]:
        sorted_indices = sorted(range(n), key=lambda i: arr[i])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and arr[sorted_indices[j + 1]] == arr[sorted_indices[i]]:
                j += 1
            avg_rank = (i + j + 2) / 2.0  # 1-based ranks
            for k in range(i, j + 1):
                ranks[sorted_indices[k]] = avg_rank
            i = j + 1
        return ranks

    rank_u = rank_data(u)
    rank_v = rank_data(v)

    mean_u = sum(rank_u) / n
    mean_v = sum(rank_v) / n

    cov = sum((rank_u[i] - mean_u) * (rank_v[i] - mean_v) for i in range(n))
    var_u = sum((rank_u[i] - mean_u) ** 2 for i in range(n))
    var_v = sum((rank_v[i] - mean_v) ** 2 for i in range(n))

    if var_u <= 1e-12 or var_v <= 1e-12:
        return None, None

    rho = cov / (math.sqrt(var_u) * math.sqrt(var_v))
    rho = max(-1.0, min(1.0, rho))

    # Two-tailed t-test p-value approximation
    if abs(rho) >= 1.0:
        p_val = 0.0
    else:
        t_stat = rho * math.sqrt((n - 2) / (1.0 - rho ** 2))
        # Standard normal approx for large n, simple bound for small n
        p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))

    return round(rho, 4), round(p_val, 4)


def compute_top_k_jaccard_overlap(
    u: list[float], v: list[float], k: int | None = None
) -> tuple[float | None, int]:
    """Compute Jaccard overlap (IoU) of the top-k highest importance tokens.

    Tie-breaking: deterministic by token position if scores are identical.
    Returns (overlap, k_value).
    """
    n = len(u)
    if n < 1:
        return None, 0

    if k is None:
        k = min(3, max(1, n // 2))
    k = min(k, n)

    # Indices sorted descending by value, breaking ties by index
    top_u = set(sorted(range(n), key=lambda i: (u[i], -i), reverse=True)[:k])
    top_v = set(sorted(range(n), key=lambda i: (v[i], -i), reverse=True)[:k])

    intersection = len(top_u & top_v)
    union = len(top_u | top_v)

    if union == 0:
        return None, k

    overlap = intersection / union
    return round(overlap, 4), k


def compute_cosine_similarity(u: list[float], v: list[float]) -> float | None:
    """Compute cosine similarity between two vectors. Returns None if zero norm."""
    if len(u) != len(v) or not u:
        return None

    dot = sum(x * y for x, y in zip(u, v))
    norm_u = math.sqrt(sum(x * x for x in u))
    norm_v = math.sqrt(sum(y * y for y in v))

    if norm_u <= 1e-12 or norm_v <= 1e-12:
        return None

    sim = dot / (norm_u * norm_v)
    return round(max(-1.0, min(1.0, sim)), 4)


def evaluate_stability_for_pair(
    pair_row: dict[str, Any],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    model_name: str,
    n_steps: int = DEFAULT_N_STEPS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate stability of Attention and Integrated Gradients for one sentence pair.

    Returns:
        eligibility_record: dict matching ELIGIBILITY_CSV_FIELDS
        pair_records: list of 2 dicts (one for Attention, one for IG) matching PAIR_LEVEL_CSV_FIELDS
    """
    pair_id = pair_row.get("pair_id", "")
    orig_sent = pair_row.get("original_sentence", "")
    cont_sent = pair_row.get("contrast_sentence", "")
    target_orig = pair_row.get("target_original", "")
    target_cont = pair_row.get("target_contrast", "")
    track = pair_row.get("track", "")
    contrast_label = pair_row.get("contrast_label", "")

    # Tokenize sentences
    enc_orig = tokenizer(orig_sent, return_offsets_mapping=True)
    enc_cont = tokenizer(cont_sent, return_offsets_mapping=True)

    align_orig = align_target_word(orig_sent, target_orig, enc_orig, tokenizer)
    align_cont = align_target_word(cont_sent, target_cont, enc_cont, tokenizer)

    subwords_orig = align_orig.get("token_count", 0) or 0
    subwords_cont = align_cont.get("token_count", 0) or 0

    elig_record: dict[str, Any] = {
        "pair_id": pair_id,
        "_excel_row": pair_row.get("_excel_row", ""),
        "model_name": model_name,
        "track": track,
        "contrast_label": contrast_label,
        "target_original": target_orig,
        "target_contrast": target_cont,
        "target_orig_subwords_count": subwords_orig,
        "target_cont_subwords_count": subwords_cont,
        "is_eligible": False,
        "exclusion_reason": "",
        "final_status": pair_row.get("final_status", ""),
    }

    # Verify eligibility under strict single-subword fixed-target rule
    reasons = []
    if align_orig["status"] != "unique":
        reasons.append(f"orig_alignment_{align_orig['status']}: {align_orig.get('reason', '')}")
    if align_cont["status"] != "unique":
        reasons.append(f"cont_alignment_{align_cont['status']}: {align_cont.get('reason', '')}")
    if align_orig.get("token_count") != 1:
        reasons.append(f"orig_multi_subword_target (count={subwords_orig})")
    if align_cont.get("token_count") != 1:
        reasons.append(f"cont_multi_subword_target (count={subwords_cont})")

    if reasons:
        elig_record["is_eligible"] = False
        elig_record["exclusion_reason"] = "; ".join(reasons)

        base_pair_rec: dict[str, Any] = {
            "pair_id": pair_id,
            "_excel_row": pair_row.get("_excel_row", ""),
            "model_name": model_name,
            "track": track,
            "contrast_label": contrast_label,
            "eligibility_status": "ineligible",
            "ineligibility_reason": "; ".join(reasons),
            "target_original": target_orig,
            "target_contrast": target_cont,
            "target_canonical_token_id": "",
            "target_orig_pos": "",
            "target_cont_pos": "",
            "orig_target_logit": "",
            "cont_target_logit": "",
            "delta_target_logit": "",
            "n_aligned_context_tokens": 0,
            "aligned_context_tokens": json.dumps([], ensure_ascii=False),
            "unaligned_orig_count": 0,
            "unaligned_cont_count": 0,
            "spearman_rho": "",
            "spearman_pvalue": "",
            "top_k_overlap": "",
            "top_k_value": "",
            "cosine_similarity": "",
            "cosine_similarity_signed": "",
            "spearman_rho_signed": "",
            "orig_attributions": "",
            "cont_attributions": "",
            "orig_convergence_delta": "",
            "cont_convergence_delta": "",
            "notes": pair_row.get("notes", ""),
            "final_status": pair_row.get("final_status", ""),
        }
        rec_attn = dict(base_pair_rec, method="attention")
        rec_ig = dict(base_pair_rec, method="integrated_gradients")
        return elig_record, [rec_attn, rec_ig]

    # Eligible pair: proceed with fixed-target scoring
    elig_record["is_eligible"] = True
    elig_record["exclusion_reason"] = "none"

    m_orig = align_orig["token_indices"][0]
    m_cont = align_cont["token_indices"][0]
    y_orig = align_orig["token_ids"][0]  # canonical target token ID

    # 1. Align context tokens by region
    aligned_info = align_context_tokens_by_region(
        orig_sentence=orig_sent,
        orig_target_span=align_orig["char_span"],
        orig_encoding=enc_orig,
        cont_sentence=cont_sent,
        cont_target_span=align_cont["char_span"],
        cont_encoding=enc_cont,
        tokenizer=tokenizer,
    )
    aligned_pairs = aligned_info["aligned_pairs"]
    aligned_labels = aligned_info["aligned_token_labels"]
    n_aligned = len(aligned_pairs)

    # 2. Build masked input tensors
    mask_token_id = getattr(tokenizer, "mask_token_id", 103)
    pad_token_id = getattr(tokenizer, "pad_token_id", 0)

    orig_ids_list = enc_orig["input_ids"]
    if hasattr(orig_ids_list, "tolist"):
        orig_ids_list = orig_ids_list[0].tolist() if orig_ids_list.ndim > 1 else orig_ids_list.tolist()
    cont_ids_list = enc_cont["input_ids"]
    if hasattr(cont_ids_list, "tolist"):
        cont_ids_list = cont_ids_list[0].tolist() if cont_ids_list.ndim > 1 else cont_ids_list.tolist()

    orig_input_ids = torch.tensor([orig_ids_list], dtype=torch.long, device=device)
    orig_attention_mask = torch.tensor([[1] * len(orig_ids_list)], dtype=torch.long, device=device)
    orig_masked_ids = orig_input_ids.clone()
    orig_masked_ids[0, m_orig] = mask_token_id

    cont_input_ids = torch.tensor([cont_ids_list], dtype=torch.long, device=device)
    cont_attention_mask = torch.tensor([[1] * len(cont_ids_list)], dtype=torch.long, device=device)
    cont_masked_ids = cont_input_ids.clone()
    cont_masked_ids[0, m_cont] = mask_token_id

    # 3. Compute Attention explanations
    with torch.inference_mode():
        out_orig = model(
            input_ids=orig_masked_ids,
            attention_mask=orig_attention_mask,
            output_attentions=True,
        )
        orig_logit = float(out_orig.logits[0, m_orig, y_orig].item())
        attn_layer_orig = out_orig.attentions[-1][0].mean(dim=0)  # (seq_len, seq_len)
        orig_query_attn = attn_layer_orig[m_orig]

        out_cont = model(
            input_ids=cont_masked_ids,
            attention_mask=cont_attention_mask,
            output_attentions=True,
        )
        cont_logit = float(out_cont.logits[0, m_cont, y_orig].item())
        attn_layer_cont = out_cont.attentions[-1][0].mean(dim=0)
        cont_query_attn = attn_layer_cont[m_cont]

    delta_logit = round(cont_logit - orig_logit, 4)

    # Extract aligned attention scores
    orig_aligned_attn = [float(orig_query_attn[o_i].item()) for o_i, _ in aligned_pairs]
    cont_aligned_attn = [float(cont_query_attn[c_i].item()) for _, c_i in aligned_pairs]

    # Normalize attention over aligned context tokens
    sum_a_o = sum(orig_aligned_attn)
    sum_a_c = sum(cont_aligned_attn)
    norm_attn_orig = [x / sum_a_o for x in orig_aligned_attn] if sum_a_o > 0 else orig_aligned_attn
    norm_attn_cont = [x / sum_a_c for x in cont_aligned_attn] if sum_a_c > 0 else cont_aligned_attn

    # Attention stability metrics
    attn_rho, attn_pval = compute_spearman_rank_correlation(norm_attn_orig, norm_attn_cont)
    attn_topk, topk_k = compute_top_k_jaccard_overlap(norm_attn_orig, norm_attn_cont)
    attn_cos = compute_cosine_similarity(norm_attn_orig, norm_attn_cont)

    rec_attn: dict[str, Any] = {
        "pair_id": pair_id,
        "_excel_row": pair_row.get("_excel_row", ""),
        "model_name": model_name,
        "track": track,
        "contrast_label": contrast_label,
        "method": "attention",
        "eligibility_status": "eligible",
        "ineligibility_reason": "none",
        "target_original": target_orig,
        "target_contrast": target_cont,
        "target_canonical_token_id": y_orig,
        "target_orig_pos": m_orig,
        "target_cont_pos": m_cont,
        "orig_target_logit": round(orig_logit, 4),
        "cont_target_logit": round(cont_logit, 4),
        "delta_target_logit": delta_logit,
        "n_aligned_context_tokens": n_aligned,
        "aligned_context_tokens": json.dumps(aligned_labels, ensure_ascii=False),
        "unaligned_orig_count": len(aligned_info["unaligned_orig_indices"]),
        "unaligned_cont_count": len(aligned_info["unaligned_cont_indices"]),
        "spearman_rho": attn_rho if attn_rho is not None else "",
        "spearman_pvalue": attn_pval if attn_pval is not None else "",
        "top_k_overlap": attn_topk if attn_topk is not None else "",
        "top_k_value": topk_k,
        "cosine_similarity": attn_cos if attn_cos is not None else "",
        "cosine_similarity_signed": "",
        "spearman_rho_signed": "",
        "orig_attributions": json.dumps([round(x, 6) for x in norm_attn_orig], ensure_ascii=False),
        "cont_attributions": json.dumps([round(x, 6) for x in norm_attn_cont], ensure_ascii=False),
        "orig_convergence_delta": "",
        "cont_convergence_delta": "",
        "notes": pair_row.get("notes", ""),
        "final_status": pair_row.get("final_status", ""),
    }

    # 4. Compute Integrated Gradients explanations
    embedding_layer = get_mlm_embedding_layer(model)

    # Baselines: PAD everywhere except MASK at target position
    ref_orig = torch.full_like(orig_masked_ids, fill_value=pad_token_id)
    ref_orig[0, m_orig] = mask_token_id

    ref_cont = torch.full_like(cont_masked_ids, fill_value=pad_token_id)
    ref_cont[0, m_cont] = mask_token_id

    def fwd_orig(in_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = model(input_ids=in_ids, attention_mask=mask)
        logits = out.logits if hasattr(out, "logits") else out
        return logits[:, m_orig, y_orig]

    def fwd_cont(in_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        out = model(input_ids=in_ids, attention_mask=mask)
        logits = out.logits if hasattr(out, "logits") else out
        return logits[:, m_cont, y_orig]

    lig_orig = LayerIntegratedGradients(forward_func=fwd_orig, layer=embedding_layer)
    attr_orig, delta_orig = lig_orig.attribute(
        inputs=orig_masked_ids,
        baselines=ref_orig,
        additional_forward_args=(orig_attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    lig_cont = LayerIntegratedGradients(forward_func=fwd_cont, layer=embedding_layer)
    attr_cont, delta_cont = lig_cont.attribute(
        inputs=cont_masked_ids,
        baselines=ref_cont,
        additional_forward_args=(cont_attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=True,
    )

    # Extract signed and absolute attributions
    orig_ig_signed = attr_orig[0].mean(dim=-1).tolist()
    orig_ig_abs = attr_orig[0].norm(dim=-1).tolist()

    cont_ig_signed = attr_cont[0].mean(dim=-1).tolist()
    cont_ig_abs = attr_cont[0].norm(dim=-1).tolist()

    orig_aligned_signed = [float(orig_ig_signed[o_i]) for o_i, _ in aligned_pairs]
    cont_aligned_signed = [float(cont_ig_signed[c_i]) for _, c_i in aligned_pairs]

    orig_aligned_abs = [float(orig_ig_abs[o_i]) for o_i, _ in aligned_pairs]
    cont_aligned_abs = [float(cont_ig_abs[c_i]) for _, c_i in aligned_pairs]

    # Primary IG stability uses absolute magnitude
    ig_rho, ig_pval = compute_spearman_rank_correlation(orig_aligned_abs, cont_aligned_abs)
    ig_topk, ig_k = compute_top_k_jaccard_overlap(orig_aligned_abs, cont_aligned_abs)
    ig_cos = compute_cosine_similarity(orig_aligned_abs, cont_aligned_abs)

    # Diagnostic IG stability uses signed attributions
    ig_rho_signed, _ = compute_spearman_rank_correlation(orig_aligned_signed, cont_aligned_signed)
    ig_cos_signed = compute_cosine_similarity(orig_aligned_signed, cont_aligned_signed)

    delta_orig_val = float(delta_orig.item() if delta_orig.numel() == 1 else delta_orig[0].item())
    delta_cont_val = float(delta_cont.item() if delta_cont.numel() == 1 else delta_cont[0].item())

    rec_ig: dict[str, Any] = {
        "pair_id": pair_id,
        "_excel_row": pair_row.get("_excel_row", ""),
        "model_name": model_name,
        "track": track,
        "contrast_label": contrast_label,
        "method": "integrated_gradients",
        "eligibility_status": "eligible",
        "ineligibility_reason": "none",
        "target_original": target_orig,
        "target_contrast": target_cont,
        "target_canonical_token_id": y_orig,
        "target_orig_pos": m_orig,
        "target_cont_pos": m_cont,
        "orig_target_logit": round(orig_logit, 4),
        "cont_target_logit": round(cont_logit, 4),
        "delta_target_logit": delta_logit,
        "n_aligned_context_tokens": n_aligned,
        "aligned_context_tokens": json.dumps(aligned_labels, ensure_ascii=False),
        "unaligned_orig_count": len(aligned_info["unaligned_orig_indices"]),
        "unaligned_cont_count": len(aligned_info["unaligned_cont_indices"]),
        "spearman_rho": ig_rho if ig_rho is not None else "",
        "spearman_pvalue": ig_pval if ig_pval is not None else "",
        "top_k_overlap": ig_topk if ig_topk is not None else "",
        "top_k_value": ig_k,
        "cosine_similarity": ig_cos if ig_cos is not None else "",
        "cosine_similarity_signed": ig_cos_signed if ig_cos_signed is not None else "",
        "spearman_rho_signed": ig_rho_signed if ig_rho_signed is not None else "",
        "orig_attributions": json.dumps([round(x, 6) for x in orig_aligned_abs], ensure_ascii=False),
        "cont_attributions": json.dumps([round(x, 6) for x in cont_aligned_abs], ensure_ascii=False),
        "orig_convergence_delta": round(delta_orig_val, 4),
        "cont_convergence_delta": round(delta_cont_val, 4),
        "notes": pair_row.get("notes", ""),
        "final_status": pair_row.get("final_status", ""),
    }

    return elig_record, [rec_attn, rec_ig]


def compute_descriptive_stats(values: list[float]) -> dict[str, float | None]:
    """Compute mean, median, IQR for a list of floats."""
    valid = [v for v in values if v is not None and not math.isnan(v)]
    if not valid:
        return {"mean": None, "median": None, "iqr": None}

    n = len(valid)
    mean_val = sum(valid) / n

    s = sorted(valid)
    if n % 2 == 1:
        median_val = s[n // 2]
    else:
        median_val = (s[n // 2 - 1] + s[n // 2]) / 2.0

    # Quartiles
    def get_pct(pct: float) -> float:
        k = (n - 1) * pct
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[f] * (c - k) + s[c] * (k - f)

    q25 = get_pct(0.25)
    q75 = get_pct(0.75)
    iqr_val = q75 - q25

    return {
        "mean": round(mean_val, 4),
        "median": round(median_val, 4),
        "iqr": round(iqr_val, 4),
    }


def generate_summary_records(
    pair_records: list[dict[str, Any]],
    eligibility_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compute descriptive statistics table by model, method, and track."""
    summary_rows: list[dict[str, Any]] = []

    # Group by (model_name, method, track)
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for r in pair_records:
        key = (r["model_name"], r["method"], r["track"])
        groups.setdefault(key, []).append(r)

    # Also compute track totals from eligibility_records
    track_totals: dict[tuple[str, str], int] = {}
    for e in eligibility_records:
        m_t_key = (e["model_name"], e["track"])
        track_totals[m_t_key] = track_totals.get(m_t_key, 0) + 1

    for (model, method, track), rows in sorted(groups.items()):
        total_in_track = track_totals.get((model, track), len(rows))
        eval_count = len(rows)
        eligible_rows = [r for r in rows if r["eligibility_status"] == "eligible"]
        ineligible_rows = [r for r in rows if r["eligibility_status"] == "ineligible"]

        rhos = [float(r["spearman_rho"]) for r in eligible_rows if r["spearman_rho"] != ""]
        topks = [float(r["top_k_overlap"]) for r in eligible_rows if r["top_k_overlap"] != ""]
        coss = [float(r["cosine_similarity"]) for r in eligible_rows if r["cosine_similarity"] != ""]
        coss_signed = [float(r["cosine_similarity_signed"]) for r in eligible_rows if r.get("cosine_similarity_signed", "") != ""]

        stat_rho = compute_descriptive_stats(rhos)
        stat_topk = compute_descriptive_stats(topks)
        stat_cos = compute_descriptive_stats(coss)
        stat_cos_s = compute_descriptive_stats(coss_signed)

        summary_rows.append({
            "model_name": model,
            "method": method,
            "track": track,
            "total_pairs_in_track": total_in_track,
            "evaluated_pairs": eval_count,
            "eligible_pairs": len(eligible_rows),
            "ineligible_pairs": len(ineligible_rows),
            "valid_metric_pairs": len(rhos),
            "spearman_mean": stat_rho["mean"] if stat_rho["mean"] is not None else "",
            "spearman_median": stat_rho["median"] if stat_rho["median"] is not None else "",
            "spearman_iqr": stat_rho["iqr"] if stat_rho["iqr"] is not None else "",
            "top_k_overlap_mean": stat_topk["mean"] if stat_topk["mean"] is not None else "",
            "top_k_overlap_median": stat_topk["median"] if stat_topk["median"] is not None else "",
            "cosine_sim_mean": stat_cos["mean"] if stat_cos["mean"] is not None else "",
            "cosine_sim_median": stat_cos["median"] if stat_cos["median"] is not None else "",
            "cosine_sim_signed_mean": stat_cos_s["mean"] if stat_cos_s["mean"] is not None else "",
        })

    return summary_rows


def run_stage6_stability_experiment(
    input_file: Path | str = DEFAULT_INPUT_DATASET,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    models: list[str] | None = None,
    track_filter: str | None = "REALISTIC_SPELLING_ERROR",
    n_steps: int = DEFAULT_N_STEPS,
    device: torch.device | None = None,
    model_tokenizer_loader: Any = None,
    dataset_loader: Any = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Execute Stage 6 Hindi explanation stability experiment with sequential model loading."""
    if models is None:
        models = DEFAULT_MODELS
    if device is None:
        device = torch.device("cpu")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load dataset
    if dataset_loader is not None:
        all_rows = dataset_loader()
        input_sha = "mock_sha256"
    else:
        input_path = Path(input_file)
        if not input_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {input_path}")
        input_sha = compute_file_sha256(input_path)
        all_rows = load_dataset_rows(input_path)

    # Filter rows by track if requested (default: REALISTIC_SPELLING_ERROR)
    if track_filter:
        target_rows = [r for r in all_rows if r.get("track") == track_filter]
    else:
        target_rows = all_rows

    checkpoint_file = out_path / "checkpoint.json"
    ckpt: dict[str, Any] = {"completed_keys": []}
    if checkpoint_file.is_file():
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                ckpt = json.load(f)
        except Exception:
            ckpt = {"completed_keys": []}

    all_pair_records: list[dict[str, Any]] = []
    all_elig_records: list[dict[str, Any]] = []

    # 2. Sequential evaluation per model
    for model_name in models:
        # Load model and tokenizer
        if model_tokenizer_loader is not None:
            model, tokenizer = model_tokenizer_loader(model_name)
        else:
            model, tokenizer = load_model_and_tokenizer_offline(model_name, device)

        for pair_row in target_rows:
            pair_id = pair_row.get("pair_id", "")
            ckpt_key = f"{model_name}::{pair_id}"

            elig_rec, pair_recs = evaluate_stability_for_pair(
                pair_row=pair_row,
                model=model,
                tokenizer=tokenizer,
                device=device,
                model_name=model_name,
                n_steps=n_steps,
            )
            all_elig_records.append(elig_rec)
            all_pair_records.extend(pair_recs)

            if ckpt_key not in ckpt["completed_keys"]:
                ckpt["completed_keys"].append(ckpt_key)

        # Free memory before loading next model
        del model
        del tokenizer
        gc.collect()

        # Update checkpoint on disk
        with open(checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(ckpt, f, indent=2)

    # 3. Generate summary records
    summary_records = generate_summary_records(all_pair_records, all_elig_records)

    # 4. Save artifacts
    pair_csv = out_path / "stability_pair_level.csv"
    with open(pair_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAIR_LEVEL_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_pair_records)

    summary_csv = out_path / "stability_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(summary_records)

    elig_csv = out_path / "eligibility_report.csv"
    with open(elig_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ELIGIBILITY_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_elig_records)

    meta_file = out_path / "run_metadata.json"
    metadata: dict[str, Any] = {
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_dataset": str(input_file),
        "input_sha256": input_sha,
        "models_evaluated": models,
        "track_filter": track_filter,
        "target_definition": "scalar_mlm_canonical_target_token_logit_at_masked_position",
        "explanation_methods": ["attention", "integrated_gradients"],
        "ig_n_steps": n_steps,
        "ig_baseline": "pad_with_target_masked",
        "total_source_pairs_evaluated": len(target_rows),
        "total_pair_evaluations": len(all_pair_records),
        "output_files": [
            str(pair_csv),
            str(summary_csv),
            str(elig_csv),
            str(meta_file),
            str(out_path / "stage6_report.md"),
        ],
        "library_versions": get_library_versions(),
        "completed_successfully": True,
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    report_file = out_path / "stage6_report.md"
    report_md = _generate_stage6_markdown_report(
        metadata=metadata,
        summary_records=summary_records,
        elig_records=all_elig_records,
        pair_records=all_pair_records,
    )
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_md)

    return metadata, all_pair_records, summary_records


def _generate_stage6_markdown_report(
    metadata: dict[str, Any],
    summary_records: list[dict[str, Any]],
    elig_records: list[dict[str, Any]],
    pair_records: list[dict[str, Any]],
) -> str:
    """Generate comprehensive scientific Markdown report for Stage 6."""
    lines = [
        "# Stage 6: Hindi Explanation Stability Under Realistic Spelling Perturbations",
        "",
        "## 1. Executive Summary and Experimental Context",
        f"- **Primary Track Evaluated**: `{metadata['track_filter']}` (all 20 pairs in the curated dataset)",
        f"- **Models Evaluated**: {', '.join(f'`{m}`' for m in metadata['models_evaluated'])}",
        f"- **Dataset SHA-256**: `{metadata['input_sha256']}`",
        f"- **Timestamp (UTC)**: `{metadata['run_timestamp_utc']}`",
        f"- **Explanation Methods**: `Self-Attention` (last-layer mean over heads) and `LayerIntegratedGradients` (embedding layer)",
        "- **Target Definition**: Fixed canonical target logit at the masked position in both inputs (`Score = Logit(y_orig at m)`).",
        "",
        "## 2. Methodology & Guardrails",
        "1. **Fixed-Target Scoring**: Explains the model's evidence for the canonical intended word `y_orig` in both sentences.",
        "   This holds the target concept constant, ensuring observed shifts are attributable strictly to the context spelling error.",
        "2. **Region-Relative Context Alignment**: Because spelling errors alter character lengths (e.g., `स्वास्थ्य` [9 chars] $\\to$ `स्वास्थ` [7 chars]),",
        "   context tokens are aligned by region-relative offsets (`pre` and `post`). Special tokens and the target word are excluded.",
        "   Attribution vectors are never zipped by token index.",
        "3. **Strict Single-Subword Eligibility**: To avoid ambiguous unaggregated token-head attributions, only pairs where both original",
        "   and contrast targets form a single subword are scored for stability. Multi-subword pairs are recorded explicitly with reasons.",
        "4. **Stability Metrics**: Spearman rank correlation (rho), Top-k Jaccard overlap, and Cosine similarity over aligned context tokens.",
        "   Undefined cases (insufficient tokens or zero variance) explicitly yield NaN.",
        "",
        "## 3. Eligibility and Exclusion Summary",
        "",
        "| Model | Total Evaluated | Eligible Pairs | Ineligible Pairs | Primary Exclusion Reasons |",
        "|---|---|---|---|---|",
    ]

    for model in metadata["models_evaluated"]:
        m_eligs = [e for e in elig_records if e["model_name"] == model]
        n_tot = len(m_eligs)
        n_elig = sum(1 for e in m_eligs if e["is_eligible"])
        n_inelig = n_tot - n_elig
        reasons_summary = "compound_subword_contrast_target (most frequent)"
        lines.append(f"| `{model}` | {n_tot} | {n_elig} | {n_inelig} | {reasons_summary} |")

    lines.extend([
        "",
        "## 4. Stability Metric Summary (Exploratory Findings)",
        "",
        "| Model | Method | Eligible (N) | Valid Metrics (N) | Spearman Rho (Mean) | Spearman Rho (Median) | Top-k Overlap (Mean) | Cosine Sim (Mean) |",
        "|---|---|---|---|---|---|---|---|",
    ])

    for s in summary_records:
        sp_mean = str(s.get("spearman_mean", "-"))
        sp_med = str(s.get("spearman_median", "-"))
        tk_mean = str(s.get("top_k_overlap_mean", "-"))
        cos_mean = str(s.get("cosine_sim_mean", "-"))
        lines.append(
            f"| `{s['model_name'].split('/')[-1]}` | `{s['method']}` | {s['eligible_pairs']} | {s['valid_metric_pairs']} | "
            f"{sp_mean} | {sp_med} | {tk_mean} | {cos_mean} |"
        )

    lines.extend([
        "",
        "## 5. Pair-Level Results for Eligible Spelling Errors",
        "",
        "| Pair ID | Model | Method | Target (Orig / Cont) | Delta Logit | Aligned Tokens | Spearman Rho | Top-k Overlap | Cosine Sim |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    for r in pair_records:
        if r["eligibility_status"] == "eligible":
            rho_str = str(r.get("spearman_rho", "-"))
            topk_str = str(r.get("top_k_overlap", "-"))
            cos_str = str(r.get("cosine_similarity", "-"))
            d_l = str(r.get("delta_target_logit", "-"))
            m_short = r["model_name"].split("/")[-1]
            lines.append(
                f"| {r['pair_id']} | `{m_short}` | `{r['method']}` | {r['target_original']} / {r['target_contrast']} | "
                f"{d_l} | {r['n_aligned_context_tokens']} | {rho_str} | {topk_str} | {cos_str} |"
            )

    lines.extend([
        "",
        "## 6. Scientific Limitations and Research Integrity",
        "1. **Exploratory Sample Size**: Under strict single-subword criteria, only 6 pairs in MuRIL and 4 in XLM-R are eligible.",
        "   No general inferential claims about Hindi language robustness can be made from this sample.",
        "2. **Stability vs Faithfulness**: High explanation stability does NOT imply explanation faithfulness.",
        "   Measuring whether stable explanations reflect actual model decision mechanisms requires independent perturbation/faithfulness testing (Stage 7).",
        "3. **Attention as Non-Causal**: Attention weights reflect query-key routing inside the model, not verified causal importance.",
        "",
    ])
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Stage 6 Hindi Explanation Stability")
    parser.add_argument("--track", type=str, default="REALISTIC_SPELLING_ERROR", help="Track filter")
    parser.add_argument("--n-steps", type=int, default=DEFAULT_N_STEPS, help="IG steps")
    args = parser.parse_args()

    meta, pairs, summary = run_stage6_stability_experiment(
        track_filter=args.track,
        n_steps=args.n_steps,
    )
    print(f"Stage 6 completed: {len(pairs)} pair evaluations saved.")
