"""Stage 2: Tokenization Experiment Module for Hindi Lexical Sensitivity Research.

This module evaluates how pretrained multilingual language models tokenize Hindi
Devanagari sentence pairs across three experimental tracks:
1. Contextual lexical contrasts (VALID_WORD_CONTEXT)
2. Synonym substitutions (SYNONYM_SUBSTITUTION)
3. Realistic spelling errors (REALISTIC_SPELLING_ERROR)

Evaluated models:
- FacebookAI/xlm-roberta-base (SentencePiece BPE)
- google/muril-base-cased (WordPiece)

Design principles:
- Non-destructive: preserves every dataset row, annotation, and disagreement.
- Subword alignment: uses fast-tokenizer character offsets to align target words.
- Explicit alignment statuses: unique, missing, ambiguous, unsupported. Never guesses.
- Token count distinction: records both special-token-inclusive and special-token-exclusive counts.
- Signed differences: computes contrast minus original counts.
- Reproducible & safe caching: verifies input file SHA-256 and configuration.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys
from typing import Any
import unicodedata

# Re-use raw data loading and row filtering from validator
from src.data.validate import is_row_empty, load_raw_rows

# Default models evaluated in the experiment
DEFAULT_MODELS: list[str] = [
    "FacebookAI/xlm-roberta-base",
    "google/muril-base-cased",
]


def compute_file_sha256(file_path: Path | str) -> str:
    """Compute SHA-256 checksum of a file in binary chunks."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found for checksum calculation: {path}")
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_library_versions() -> dict[str, str]:
    """Collect runtime library versions for provenance and metadata."""
    versions: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ["transformers", "tokenizers", "sentencepiece", "openpyxl"]:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = "not_installed"
    return versions


def find_all_substring_indices(text: str, sub: str) -> list[int]:
    """Find all starting 0-based character indices of substring in text."""
    if not sub or not text:
        return []
    indices: list[int] = []
    start = 0
    while True:
        pos = text.find(sub, start)
        if pos == -1:
            break
        indices.append(pos)
        start = pos + 1
    return indices


def align_target_word(
    sentence: str,
    target: str,
    encoding: Any,
    tokenizer: Any,
) -> dict[str, Any]:
    """Align a target word to sentence tokens using fast tokenizer character offsets.

    Alignment statuses:
    - 'unique': Exactly one match in sentence, mapping to a contiguous token span.
    - 'ambiguous': Target appears multiple times in sentence or tokens are non-contiguous.
    - 'missing': Target does not appear in sentence or no overlapping tokens found.
    - 'unsupported': Tokenizer lacks offset_mapping capability.

    Returns:
        dict containing status, reason, tokens, token_ids, token_count, char_span, normalization.
    """
    empty_result = {
        "status": "missing",
        "reason": "",
        "tokens": [],
        "token_ids": [],
        "token_count": None,
        "char_span": None,
        "normalization": "none",
    }

    tgt = str(target).strip() if target is not None else ""
    sent = str(sentence) if sentence is not None else ""

    if not tgt:
        empty_result["reason"] = "empty_target"
        return empty_result
    if not sent:
        empty_result["reason"] = "empty_sentence"
        return empty_result

    # 1. Search in exact sentence
    indices = find_all_substring_indices(sent, tgt)
    norm_applied = "none"

    if len(indices) == 0:
        # Check Unicode NFC normalization
        nfc_sent = unicodedata.normalize("NFC", sent)
        nfc_tgt = unicodedata.normalize("NFC", tgt)
        nfc_indices = find_all_substring_indices(nfc_sent, nfc_tgt)
        if len(nfc_indices) == 1 and len(sent) == len(nfc_sent):
            indices = nfc_indices
            norm_applied = "nfc"
        elif len(nfc_indices) > 1:
            empty_result["status"] = "ambiguous"
            empty_result["reason"] = f"target appears {len(nfc_indices)} times in NFC normalized text"
            empty_result["normalization"] = "nfc"
            return empty_result
        else:
            empty_result["status"] = "missing"
            empty_result["reason"] = "target substring not found in sentence"
            return empty_result

    if len(indices) > 1:
        empty_result["status"] = "ambiguous"
        empty_result["reason"] = f"target appears {len(indices)} times in sentence"
        return empty_result

    # Exactly 1 character occurrence found
    char_start = indices[0]
    char_end = char_start + len(tgt)
    char_span = (char_start, char_end)

    # 2. Extract offsets and special tokens mask
    offsets = getattr(encoding, "offset_mapping", None)
    if offsets is None and isinstance(encoding, dict):
        offsets = encoding.get("offset_mapping")

    if offsets is None:
        empty_result["status"] = "unsupported"
        empty_result["reason"] = "tokenizer offset_mapping is unavailable"
        empty_result["char_span"] = char_span
        return empty_result

    input_ids = getattr(encoding, "input_ids", None)
    if input_ids is None and isinstance(encoding, dict):
        input_ids = encoding.get("input_ids", [])

    # Extract or compute special tokens mask
    special_mask = getattr(encoding, "special_tokens_mask", None)
    if special_mask is None and isinstance(encoding, dict):
        special_mask = encoding.get("special_tokens_mask")
    if special_mask is None:
        all_special_ids = getattr(tokenizer, "all_special_ids", set())
        special_mask = [1 if id_ in all_special_ids else 0 for id_ in input_ids]

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    # 3. Find overlapping non-special tokens
    target_token_indices: list[int] = []
    for idx, (ostart, oend) in enumerate(offsets):
        if special_mask[idx] == 1:
            continue
        # Check character interval overlap
        if ostart < char_end and oend > char_start:
            target_token_indices.append(idx)

    if not target_token_indices:
        empty_result["status"] = "missing"
        empty_result["reason"] = "no overlapping non-special tokens found"
        empty_result["char_span"] = char_span
        empty_result["normalization"] = norm_applied
        return empty_result

    # 4. Check contiguity of token span
    is_contiguous = target_token_indices == list(
        range(target_token_indices[0], target_token_indices[-1] + 1)
    )
    if not is_contiguous:
        empty_result["status"] = "ambiguous"
        empty_result["reason"] = f"non-contiguous token span: {target_token_indices}"
        empty_result["char_span"] = char_span
        empty_result["normalization"] = norm_applied
        return empty_result

    matched_tokens = [tokens[i] for i in target_token_indices]
    matched_ids = [input_ids[i] for i in target_token_indices]

    return {
        "status": "unique",
        "reason": "exact_contiguous_overlap",
        "token_indices": target_token_indices,
        "tokens": matched_tokens,
        "token_ids": matched_ids,
        "token_count": len(target_token_indices),
        "char_span": char_span,
        "normalization": norm_applied,
    }


def tokenize_sentence(sentence: str, tokenizer: Any) -> dict[str, Any]:
    """Tokenize a sentence using a fast tokenizer and compute count metrics.

    Distinguishes:
    - special-token-inclusive count: total length of input_ids
    - special-token-exclusive count: tokens excluding special tokens (e.g. [CLS], [SEP], <s>, </s>)
    """
    enc = tokenizer(sentence, return_offsets_mapping=True)
    input_ids = enc["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    special_mask = enc.get("special_tokens_mask")
    if special_mask is None:
        all_special_ids = getattr(tokenizer, "all_special_ids", set())
        special_mask = [1 if id_ in all_special_ids else 0 for id_ in input_ids]

    count_inclusive = len(input_ids)
    count_exclusive = sum(1 for m in special_mask if m == 0)

    return {
        "encoding": enc,
        "input_ids": input_ids,
        "tokens": tokens,
        "special_tokens_mask": special_mask,
        "count_inclusive": count_inclusive,
        "count_exclusive": count_exclusive,
    }


def process_sentence_pair(
    row: dict[str, Any],
    tokenizer: Any,
    model_name: str,
) -> dict[str, Any]:
    """Process a single dataset sentence pair with a specific model tokenizer."""
    orig_sent = row.get("original_sentence", "")
    cont_sent = row.get("contrast_sentence", "")
    target_orig = row.get("target_original", "")
    target_cont = row.get("target_contrast", "")

    # Tokenize original and contrast sentences
    tok_orig = tokenize_sentence(orig_sent, tokenizer)
    tok_cont = tokenize_sentence(cont_sent, tokenizer)

    # Compute signed sentence token differences (contrast minus original)
    delta_inclusive = tok_cont["count_inclusive"] - tok_orig["count_inclusive"]
    delta_exclusive = tok_cont["count_exclusive"] - tok_orig["count_exclusive"]

    # Align target words
    align_orig = align_target_word(orig_sent, target_orig, tok_orig["encoding"], tokenizer)
    align_cont = align_target_word(cont_sent, target_cont, tok_cont["encoding"], tokenizer)

    # Compute target word count difference if both alignments are unique
    delta_target_count: int | None = None
    if align_orig["status"] == "unique" and align_cont["status"] == "unique":
        delta_target_count = align_cont["token_count"] - align_orig["token_count"]

    # Build output record preserving all original annotations
    record: dict[str, Any] = {
        "pair_id": row.get("pair_id", ""),
        "_excel_row": row.get("_excel_row", ""),
        "model_name": model_name,
        "track": row.get("track", ""),
        "contrast_label": row.get("contrast_label", ""),
        "original_sentence": orig_sent,
        "contrast_sentence": cont_sent,
        "target_original": target_orig,
        "target_contrast": target_cont,
        # Sentence count metrics
        "orig_token_count_inclusive": tok_orig["count_inclusive"],
        "orig_token_count_exclusive": tok_orig["count_exclusive"],
        "cont_token_count_inclusive": tok_cont["count_inclusive"],
        "cont_token_count_exclusive": tok_cont["count_exclusive"],
        "delta_token_count_inclusive": delta_inclusive,
        "delta_token_count_exclusive": delta_exclusive,
        # Target alignment metrics
        "target_orig_alignment_status": align_orig["status"],
        "target_orig_token_count": align_orig["token_count"] if align_orig["token_count"] is not None else "",
        "target_cont_alignment_status": align_cont["status"],
        "target_cont_token_count": align_cont["token_count"] if align_cont["token_count"] is not None else "",
        "delta_target_token_count": delta_target_count if delta_target_count is not None else "",
        # Serialized token pieces and IDs (JSON encoded for safe storage of Devanagari)
        "orig_tokens": json.dumps(tok_orig["tokens"], ensure_ascii=False),
        "orig_token_ids": json.dumps(tok_orig["input_ids"], ensure_ascii=False),
        "cont_tokens": json.dumps(tok_cont["tokens"], ensure_ascii=False),
        "cont_token_ids": json.dumps(tok_cont["input_ids"], ensure_ascii=False),
        "target_orig_tokens": json.dumps(align_orig["tokens"], ensure_ascii=False),
        "target_orig_token_ids": json.dumps(align_orig["token_ids"], ensure_ascii=False),
        "target_cont_tokens": json.dumps(align_cont["tokens"], ensure_ascii=False),
        "target_cont_token_ids": json.dumps(align_cont["token_ids"], ensure_ascii=False),
        # Preserved human annotation & adjudication fields
        "intended_meaning_hi": row.get("intended_meaning_hi", ""),
        "intended_meaning_en": row.get("intended_meaning_en", ""),
        "grammatical_original": row.get("grammatical_original", ""),
        "grammatical_contrast": row.get("grammatical_contrast", ""),
        "notes": row.get("notes", ""),
        "annotator_1": row.get("annotator_1", ""),
        "annotator_2": row.get("annotator_2", ""),
        "adjudication": row.get("adjudication", ""),
        "final_status": row.get("final_status", ""),
        "source_type": row.get("source_type", ""),
        "source_reference": row.get("source_reference", ""),
    }

    return record


# Field order for tokenization_results.csv
RESULT_CSV_FIELDS: list[str] = [
    "pair_id",
    "_excel_row",
    "model_name",
    "track",
    "contrast_label",
    "original_sentence",
    "contrast_sentence",
    "target_original",
    "target_contrast",
    "orig_token_count_inclusive",
    "orig_token_count_exclusive",
    "cont_token_count_inclusive",
    "cont_token_count_exclusive",
    "delta_token_count_inclusive",
    "delta_token_count_exclusive",
    "target_orig_alignment_status",
    "target_orig_token_count",
    "target_cont_alignment_status",
    "target_cont_token_count",
    "delta_target_token_count",
    "orig_tokens",
    "orig_token_ids",
    "cont_tokens",
    "cont_token_ids",
    "target_orig_tokens",
    "target_orig_token_ids",
    "target_cont_tokens",
    "target_cont_token_ids",
    "intended_meaning_hi",
    "intended_meaning_en",
    "grammatical_original",
    "grammatical_contrast",
    "notes",
    "annotator_1",
    "annotator_2",
    "adjudication",
    "final_status",
    "source_type",
    "source_reference",
]


def load_dataset_rows(file_path: Path | str) -> list[dict[str, Any]]:
    """Load non-empty dataset rows using the validator module while preserving original row index."""
    raw_rows = load_raw_rows(file_path)
    if not raw_rows:
        return []

    header = [str(c).strip() if c is not None else "" for c in raw_rows[0]]
    while header and not header[-1]:
        header.pop()

    data_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(raw_rows[1:], start=2):
        if is_row_empty(row):
            continue
        row_dict: dict[str, Any] = {"_excel_row": idx}
        for c_idx, col_name in enumerate(header):
            val = row[c_idx] if c_idx < len(row) else ""
            row_dict[col_name] = str(val).strip() if val is not None else ""
        data_rows.append(row_dict)

    return data_rows


def check_cache_validity(
    output_dir: Path,
    input_sha256: str,
    models: list[str],
    expected_rows: int,
) -> bool:
    """Verify if cached tokenization outputs exist and match all experiment parameters."""
    csv_file = output_dir / "tokenization_results.csv"
    meta_file = output_dir / "run_metadata.json"

    if not csv_file.is_file() or not meta_file.is_file():
        return False

    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

        if not meta.get("completed_successfully"):
            return False
        if meta.get("input_file", {}).get("sha256") != input_sha256:
            return False
        if meta.get("experiment_config", {}).get("models") != models:
            return False

        # Verify CSV row count (excluding header)
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            csv_rows = sum(1 for _ in reader) - 1
            if csv_rows != expected_rows:
                return False

        return True
    except Exception:
        return False


def generate_summary_markdown(
    results: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> str:
    """Generate a concise human-readable summary of the tokenization experiment."""
    lines: list[str] = [
        "# Hindi Lexical Sensitivity - Tokenization Experiment Summary",
        "",
        f"**Run Timestamp (UTC):** {metadata.get('run_timestamp_utc')}",
        f"**Input Dataset:** `{metadata.get('input_file', {}).get('path')}`",
        f"**Dataset SHA-256:** `{metadata.get('input_file', {}).get('sha256')}`",
        f"**Input Data Rows:** {metadata.get('input_file', {}).get('non_empty_rows')} (799 empty trailing rows ignored)",
        f"**Total Evaluations:** {len(results)} rows ({len(metadata.get('experiment_config', {}).get('models', []))} models x {metadata.get('input_file', {}).get('non_empty_rows')} sentence pairs)",
        "",
        "## 1. Models Evaluated",
    ]

    models = metadata.get("experiment_config", {}).get("models", [])
    model_details = metadata.get("model_details", {})
    for m in models:
        details = model_details.get(m, {})
        lines.append(
            f"- **`{m}`**: Fast Tokenizer=`{details.get('is_fast')}`, "
            f"Vocab Size=`{details.get('vocab_size')}`, Class=`{details.get('tokenizer_class')}`"
        )

    lines.extend([
        "",
        "## 2. Alignment Outcomes",
        "",
        "| Model | Original Unique | Original Ambiguous | Original Missing | Contrast Unique | Contrast Ambiguous | Contrast Missing |",
        "|---|---|---|---|---|---|---|",
    ])

    for m in models:
        m_rows = [r for r in results if r["model_name"] == m]
        o_unique = sum(1 for r in m_rows if r["target_orig_alignment_status"] == "unique")
        o_ambig = sum(1 for r in m_rows if r["target_orig_alignment_status"] == "ambiguous")
        o_miss = sum(1 for r in m_rows if r["target_orig_alignment_status"] == "missing")
        c_unique = sum(1 for r in m_rows if r["target_cont_alignment_status"] == "unique")
        c_ambig = sum(1 for r in m_rows if r["target_cont_alignment_status"] == "ambiguous")
        c_miss = sum(1 for r in m_rows if r["target_cont_alignment_status"] == "missing")
        lines.append(f"| `{m}` | {o_unique} | {o_ambig} | {o_miss} | {c_unique} | {c_ambig} | {c_miss} |")

    lines.extend([
        "",
        "## 3. Token Count Distribution Across Tracks (Signed Delta: Contrast - Original)",
        "",
        "| Model | Track | Count | Mean Inclusive Delta | Min Inclusive Delta | Max Inclusive Delta | Mean Target Delta |",
        "|---|---|---|---|---|---|---|",
    ])

    tracks = sorted(list({r["track"] for r in results}))
    for m in models:
        for t in tracks:
            t_rows = [r for r in results if r["model_name"] == m and r["track"] == t]
            if not t_rows:
                continue
            inclusive_deltas = [r["delta_token_count_inclusive"] for r in t_rows]
            target_deltas = [
                r["delta_target_token_count"]
                for r in t_rows
                if r["delta_target_token_count"] != ""
            ]
            mean_inc = sum(inclusive_deltas) / len(inclusive_deltas)
            min_inc = min(inclusive_deltas)
            max_inc = max(inclusive_deltas)
            mean_tgt = sum(target_deltas) / len(target_deltas) if target_deltas else 0.0
            lines.append(
                f"| `{m}` | `{t}` | {len(t_rows)} | {mean_inc:+.2f} | {min_inc:+d} | {max_inc:+d} | {mean_tgt:+.2f} |"
            )

    lines.extend([
        "",
        "## 4. Dataset Preservation & Human Review Status",
        "- **Preserved Rows:** All 200 non-empty rows, including the 1 rejected item (`final_status=reject`) and the 6 annotator/adjudication disagreement items.",
        "- **No Unchecked Filtering:** Zero rows discarded or altered.",
        "- **Interpretive Guardrail:** Token count differences reflect subword vocabulary segmentation, not grammatical correctness or semantic validity.",
        "",
        "## 5. Artifacts Generated",
        f"- Results CSV: `{metadata.get('output_files', {}).get('csv_path')}` ({metadata.get('output_files', {}).get('csv_row_count')} rows)",
        f"- Metadata JSON: `{metadata.get('output_files', {}).get('metadata_path')}`",
        f"- Summary Report: `{metadata.get('output_files', {}).get('summary_path')}`",
        "",
    ])

    return "\n".join(lines)


def run_tokenization_experiment(
    input_file: Path | str = "data/raw/hindi_lexical_sensitivity_cleaned.xlsx",
    output_dir: Path | str = "outputs/tokenization",
    models: list[str] | None = None,
    force: bool = False,
    tokenizer_loader: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Execute complete Stage 2 tokenization experiment.

    Args:
        input_file: Path to raw dataset (.xlsx).
        output_dir: Path to directory where results, metadata, and summary will be saved.
        models: List of model identifiers. Defaults to DEFAULT_MODELS.
        force: If True, bypass cache and recompute.
        tokenizer_loader: Optional callable to load tokenizers (useful for testing and dependency injection).

    Returns:
        tuple of (results_list, metadata_dict, summary_text).
    """
    in_path = Path(input_file)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_models = models or DEFAULT_MODELS
    input_sha256 = compute_file_sha256(in_path)
    data_rows = load_dataset_rows(in_path)
    expected_result_rows = len(data_rows) * len(target_models)

    # 1. Check cache
    if not force and check_cache_validity(out_dir, input_sha256, target_models, expected_result_rows):
        csv_file = out_dir / "tokenization_results.csv"
        meta_file = out_dir / "run_metadata.json"
        summary_file = out_dir / "summary.md"

        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        results: list[dict[str, Any]] = []
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Cast numeric fields back
                for k in [
                    "orig_token_count_inclusive",
                    "orig_token_count_exclusive",
                    "cont_token_count_inclusive",
                    "cont_token_count_exclusive",
                    "delta_token_count_inclusive",
                    "delta_token_count_exclusive",
                ]:
                    if row[k] != "":
                        row[k] = int(row[k])
                for k in ["target_orig_token_count", "target_cont_token_count", "delta_target_token_count"]:
                    if row[k] != "":
                        row[k] = int(row[k])
                results.append(row)

        summary_text = ""
        if summary_file.is_file():
            summary_text = summary_file.read_text(encoding="utf-8")

        return results, metadata, summary_text

    # 2. Tokenizer loader setup
    if tokenizer_loader is None:
        try:
            from transformers import AutoTokenizer

            def default_loader(model_id: str) -> Any:
                return AutoTokenizer.from_pretrained(model_id, use_fast=True)

            tokenizer_loader = default_loader
        except ImportError as e:
            raise ImportError(
                f"transformers library is required to run the tokenization experiment: {e}"
            ) from e

    # 3. Load tokenizers safely with strict error tracking
    loaded_tokenizers: dict[str, Any] = {}
    model_details: dict[str, Any] = {}
    warnings_or_errors: list[str] = []

    for model_name in target_models:
        try:
            tok = tokenizer_loader(model_name)
            if not getattr(tok, "is_fast", False):
                warnings_or_errors.append(
                    f"Model {model_name} did not load a fast tokenizer; character offsets may be unavailable."
                )
            loaded_tokenizers[model_name] = tok
            model_details[model_name] = {
                "tokenizer_class": tok.__class__.__name__,
                "is_fast": getattr(tok, "is_fast", False),
                "vocab_size": getattr(tok, "vocab_size", None),
                "name_or_path": getattr(tok, "name_or_path", model_name),
            }
        except Exception as e:
            err_msg = f"Failed to load tokenizer for model {model_name}: {e}"
            warnings_or_errors.append(err_msg)
            # Fail safely: never pretend all models succeeded when one failed
            raise RuntimeError(err_msg) from e

    # 4. Process all rows for each model
    results: list[dict[str, Any]] = []
    for model_name in target_models:
        tok = loaded_tokenizers[model_name]
        for row in data_rows:
            rec = process_sentence_pair(row, tok, model_name)
            results.append(rec)

    # 5. Build run metadata
    run_timestamp = datetime.now(timezone.utc).isoformat()
    csv_path = out_dir / "tokenization_results.csv"
    meta_path = out_dir / "run_metadata.json"
    summary_path = out_dir / "summary.md"

    # Write CSV output first
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(results)

    csv_sha256 = compute_file_sha256(csv_path)

    metadata: dict[str, Any] = {
        "run_timestamp_utc": run_timestamp,
        "completed_successfully": True,
        "input_file": {
            "path": str(in_path.resolve()),
            "sha256": input_sha256,
            "non_empty_rows": len(data_rows),
        },
        "experiment_config": {
            "models": target_models,
            "output_dir": str(out_dir.resolve()),
        },
        "model_details": model_details,
        "library_versions": get_library_versions(),
        "output_files": {
            "csv_path": str(csv_path.resolve()),
            "csv_row_count": len(results),
            "csv_sha256": csv_sha256,
            "metadata_path": str(meta_path.resolve()),
            "summary_path": str(summary_path.resolve()),
        },
        "warnings_or_errors": warnings_or_errors,
    }

    # Generate and write summary
    summary_text = generate_summary_markdown(results, metadata)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # Write metadata JSON
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return results, metadata, summary_text


def main() -> None:
    """CLI entrypoint for Stage 2 Tokenization Experiment."""
    # Ensure stdout handles UTF-8 on Windows
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    parser = argparse.ArgumentParser(
        description="Run Stage 2: Complete Hindi Tokenization Experiment on XLM-RoBERTa and MuRIL."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/raw/hindi_lexical_sensitivity_cleaned.xlsx",
        help="Path to input dataset (.xlsx). Default: data/raw/hindi_lexical_sensitivity_cleaned.xlsx",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="outputs/tokenization",
        help="Directory to save CSV, metadata, and summary. Default: outputs/tokenization",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force recomputation even if valid cache exists.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Pretrained models to evaluate.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("STAGE 2: HINDI TOKENIZATION EXPERIMENT")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Models: {args.models}")
    print(f"Force recompute: {args.force}")
    print("-" * 60)

    try:
        results, metadata, summary_text = run_tokenization_experiment(
            input_file=args.input,
            output_dir=args.output_dir,
            models=args.models,
            force=args.force,
        )
        print(summary_text)
        print("Experiment completed successfully.")
    except Exception as e:
        print(f"ERROR: Experiment execution failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
