"""Dataset validation module for Hindi Lexical Sensitivity research.

This module validates raw Excel dataset files against expected schema,
counts non-empty vs. trailing empty rows, checks integrity constraints
(duplicate IDs, missing fields, identical sentence pairs, track validity),
computes annotator/adjudication agreement diagnostics for human review,
and runs Devanagari script diagnostics without altering or cleaning the data.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET
import zipfile

# Try importing openpyxl; provide fallback reader if openpyxl is not yet installed.
try:
    import openpyxl

    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


# Canonical expected schema (18 columns in expected order)
EXPECTED_COLUMNS: list[str] = [
    "pair_id",
    "track",
    "original_sentence",
    "contrast_sentence",
    "target_original",
    "target_contrast",
    "intended_meaning_hi",
    "intended_meaning_en",
    "grammatical_original",
    "grammatical_contrast",
    "contrast_label",
    "notes",
    "annotator_1",
    "annotator_2",
    "adjudication",
    "final_status",
    "source_type",
    "source_reference",
]

# Valid track categories defined in the research protocol
VALID_TRACKS: set[str] = {
    "VALID_WORD_CONTEXT",
    "SYNONYM_SUBSTITUTION",
    "REALISTIC_SPELLING_ERROR",
}

# Fields strictly required for each research pair
REQUIRED_FIELDS: list[str] = [
    "pair_id",
    "track",
    "original_sentence",
    "contrast_sentence",
    "target_original",
    "target_contrast",
    "final_status",
]

# Devanagari Unicode block regex
DEVANAGARI_REGEX = re.compile(r"[\u0900-\u097F]")


def contains_devanagari(text: str) -> bool:
    """Return True if text contains at least one Devanagari character."""
    if not text:
        return False
    return bool(DEVANAGARI_REGEX.search(str(text)))


def _read_excel_openpyxl(file_path: Path) -> list[list[Any]]:
    """Read an Excel file using openpyxl in read-only mode."""
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    sheet = wb.active
    rows: list[list[Any]] = []
    for row in sheet.iter_rows(values_only=True):
        rows.append(list(row))
    wb.close()
    return rows


def _read_excel_stdlib(file_path: Path) -> list[list[Any]]:
    """Fallback reader using Python standard library (zipfile + XML).

    Safely reads .xlsx files without third-party dependencies.
    """
    with zipfile.ZipFile(file_path, "r") as z:
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

        # 1. Read shared strings table if present
        sst: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            sst_xml = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in sst_xml.findall(f"{ns}si"):
                t = si.find(f"{ns}t")
                if t is not None and t.text:
                    sst.append(t.text)
                else:
                    parts = [node.text for node in si.findall(f".//{ns}t") if node.text]
                    sst.append("".join(parts))

        # 2. Determine first worksheet
        sheet_path = "xl/worksheets/sheet1.xml"
        if sheet_path not in z.namelist():
            # Find worksheet from workbook.xml
            wb_xml = ET.fromstring(z.read("xl/workbook.xml"))
            sheets = wb_xml.findall(f"{ns}sheets/{ns}sheet")
            if not sheets:
                return []
            sheet_path = "xl/worksheets/sheet1.xml"

        sheet_xml = ET.fromstring(z.read(sheet_path))
        sheet_data = sheet_xml.find(f"{ns}sheetData")
        if sheet_data is None:
            return []

        # Helper to convert Excel column letters to 0-based index (A->0, B->1, ..., AA->26)
        def col2idx(col_str: str) -> int:
            idx = 0
            for char in col_str:
                idx = idx * 26 + (ord(char.upper()) - ord("A") + 1)
            return idx - 1

        rows_map: dict[int, dict[int, Any]] = {}
        max_col = 0

        for r_node in sheet_data.findall(f"{ns}row"):
            r_idx = int(r_node.attrib.get("r", 0))
            col_cells: dict[int, Any] = {}
            for c_node in r_node.findall(f"{ns}c"):
                ref = c_node.attrib.get("r", "")
                col_letters = "".join(filter(str.isalpha, ref))
                c_idx = col2idx(col_letters) if col_letters else 0
                max_col = max(max_col, c_idx)

                cell_type = c_node.attrib.get("t", "")
                val: Any = ""
                if cell_type == "s":
                    v = c_node.find(f"{ns}v")
                    if v is not None and v.text:
                        idx = int(v.text)
                        val = sst[idx] if idx < len(sst) else ""
                elif cell_type == "inlineStr":
                    is_node = c_node.find(f"{ns}is")
                    if is_node is not None:
                        val = "".join(n.text for n in is_node.findall(f".//{ns}t") if n.text)
                else:
                    v = c_node.find(f"{ns}v")
                    if v is not None and v.text is not None:
                        val = v.text

                col_cells[c_idx] = val
            rows_map[r_idx] = col_cells

        if not rows_map:
            return []

        max_row = max(rows_map.keys())
        matrix: list[list[Any]] = []
        for r in range(1, max_row + 1):
            row_dict = rows_map.get(r, {})
            row_list = [row_dict.get(c, "") for c in range(max_col + 1)]
            matrix.append(row_list)

        return matrix


def load_raw_rows(file_path: str | Path) -> list[list[Any]]:
    """Load rows from an Excel file without modifying it.

    Uses openpyxl when installed, otherwise falls back to standard library.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    if HAS_OPENPYXL:
        return _read_excel_openpyxl(path)
    return _read_excel_stdlib(path)


def is_row_empty(row: list[Any]) -> bool:
    """Return True if all values in the row are None or empty/whitespace strings."""
    for cell in row:
        if cell is not None and str(cell).strip() != "":
            return False
    return True


def validate_dataset(file_path: str | Path) -> dict[str, Any]:
    """Validate a Hindi lexical sensitivity Excel dataset.

    Performs read-only validation according to the research requirements:
    - Checks 18 expected columns and identifies missing or unexpected columns.
    - Differentiates populated data rows from fully empty trailing rows.
    - Detects missing required fields, duplicates, empty sentences, identical pairs, unknown tracks.
    - Summarizes final_status and annotator/adjudication disagreements for human review.
    - Runs Devanagari script diagnostics without altering data.

    Returns:
        A structured dictionary containing validation metrics, diagnostics, and findings.
    """
    path = Path(file_path)
    raw_matrix = load_raw_rows(path)

    if not raw_matrix:
        return {
            "file_name": path.name,
            "file_path": str(path.resolve()),
            "is_valid": False,
            "error": "File is empty or contains no readable sheets/rows.",
            "total_rows": 0,
            "non_empty_rows": 0,
            "empty_trailing_rows": 0,
        }

    # Extract header row
    header_raw = raw_matrix[0]
    # Trim trailing completely empty header columns
    actual_columns: list[str] = []
    for col in header_raw:
        col_name = str(col).strip() if col is not None else ""
        actual_columns.append(col_name)

    # Remove trailing empty column names if any
    while actual_columns and not actual_columns[-1]:
        actual_columns.pop()

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in actual_columns]
    unexpected_columns = [c for c in actual_columns if c and c not in EXPECTED_COLUMNS]
    column_order_matches = actual_columns[: len(EXPECTED_COLUMNS)] == EXPECTED_COLUMNS

    # Process data rows
    data_matrix = raw_matrix[1:]
    total_data_rows = len(data_matrix)

    non_empty_rows: list[dict[str, Any]] = []
    empty_trailing_count = 0

    for idx, row in enumerate(data_matrix, start=2):  # 1-indexed Excel row numbers
        if is_row_empty(row):
            empty_trailing_count += 1
            continue

        # Build row dictionary mapping column names to cell contents
        row_dict: dict[str, Any] = {"_excel_row": idx}
        for c_idx, col_name in enumerate(actual_columns):
            val = row[c_idx] if c_idx < len(row) else ""
            row_dict[col_name] = str(val).strip() if val is not None else ""

        non_empty_rows.append(row_dict)

    # Integrity analysis
    pair_id_tracker: dict[str, list[int]] = defaultdict(list)
    missing_required_by_row: list[dict[str, Any]] = []
    empty_sentences: list[dict[str, Any]] = []
    identical_pairs: list[dict[str, Any]] = []
    unknown_tracks: list[dict[str, Any]] = []
    track_counts: Counter[str] = Counter()
    final_status_counts: Counter[str] = Counter()

    # Disagreements for human review
    disagreements: list[dict[str, Any]] = []

    # Devanagari diagnostics
    devanagari_diagnostics = {
        "original_sentence_with_devanagari": 0,
        "contrast_sentence_with_devanagari": 0,
        "target_original_with_devanagari": 0,
        "target_contrast_with_devanagari": 0,
        "intended_meaning_hi_with_devanagari": 0,
        "rows_without_devanagari_sentences": [],
    }

    for row in non_empty_rows:
        excel_row = row["_excel_row"]
        pair_id = row.get("pair_id", "")
        track = row.get("track", "")
        orig_sent = row.get("original_sentence", "")
        cont_sent = row.get("contrast_sentence", "")
        target_orig = row.get("target_original", "")
        target_cont = row.get("target_contrast", "")
        meaning_hi = row.get("intended_meaning_hi", "")
        annotator_1 = row.get("annotator_1", "")
        annotator_2 = row.get("annotator_2", "")
        adjudication = row.get("adjudication", "")
        final_status = row.get("final_status", "")
        notes = row.get("notes", "")

        # 1. Pair ID tracking
        if pair_id:
            pair_id_tracker[pair_id].append(excel_row)
        else:
            pair_id_tracker["[MISSING]"].append(excel_row)

        # 2. Required fields
        missing_fields = [f for f in REQUIRED_FIELDS if not row.get(f)]
        if missing_fields:
            missing_required_by_row.append({
                "excel_row": excel_row,
                "pair_id": pair_id,
                "missing_fields": missing_fields,
            })

        # 3. Track counts & validity
        if track:
            track_counts[track] += 1
            if track not in VALID_TRACKS:
                unknown_tracks.append({
                    "excel_row": excel_row,
                    "pair_id": pair_id,
                    "track_value": track,
                })
        else:
            track_counts["[BLANK]"] += 1
            unknown_tracks.append({
                "excel_row": excel_row,
                "pair_id": pair_id,
                "track_value": "[BLANK]",
            })

        # 4. Sentence pairs
        if not orig_sent or not cont_sent:
            empty_sentences.append({
                "excel_row": excel_row,
                "pair_id": pair_id,
                "missing_original": not orig_sent,
                "missing_contrast": not cont_sent,
            })
        elif orig_sent == cont_sent:
            identical_pairs.append({
                "excel_row": excel_row,
                "pair_id": pair_id,
                "sentence": orig_sent,
            })

        # 5. Status counts
        if final_status:
            final_status_counts[final_status] += 1
        else:
            final_status_counts["[BLANK]"] += 1

        # 6. Disagreement detection (for human review, never auto-resolved)
        has_disagreement = False
        disagreement_reasons: list[str] = []

        if annotator_1 and annotator_2 and annotator_1 != annotator_2:
            has_disagreement = True
            disagreement_reasons.append("annotator_1 != annotator_2")

        if adjudication and final_status and adjudication != final_status:
            has_disagreement = True
            disagreement_reasons.append("adjudication != final_status")

        if annotator_1 and final_status and annotator_1 != final_status:
            has_disagreement = True
            disagreement_reasons.append("annotator_1 != final_status")

        if annotator_2 and final_status and annotator_2 != final_status:
            has_disagreement = True
            disagreement_reasons.append("annotator_2 != final_status")

        if has_disagreement:
            disagreements.append({
                "excel_row": excel_row,
                "pair_id": pair_id,
                "annotator_1": annotator_1,
                "annotator_2": annotator_2,
                "adjudication": adjudication,
                "final_status": final_status,
                "reasons": disagreement_reasons,
                "notes": notes,
            })

        # 7. Devanagari script diagnostics
        orig_has_dev = contains_devanagari(orig_sent)
        cont_has_dev = contains_devanagari(cont_sent)

        if orig_has_dev:
            devanagari_diagnostics["original_sentence_with_devanagari"] += 1
        if cont_has_dev:
            devanagari_diagnostics["contrast_sentence_with_devanagari"] += 1
        if contains_devanagari(target_orig):
            devanagari_diagnostics["target_original_with_devanagari"] += 1
        if contains_devanagari(target_cont):
            devanagari_diagnostics["target_contrast_with_devanagari"] += 1
        if contains_devanagari(meaning_hi):
            devanagari_diagnostics["intended_meaning_hi_with_devanagari"] += 1

        if not orig_has_dev or not cont_has_dev:
            devanagari_diagnostics["rows_without_devanagari_sentences"].append({
                "excel_row": excel_row,
                "pair_id": pair_id,
                "missing_in_original": not orig_has_dev,
                "missing_in_contrast": not cont_has_dev,
            })

    duplicate_pair_ids = {
        pid: rows for pid, rows in pair_id_tracker.items() if len(rows) > 1 and pid != "[MISSING]"
    }

    # Core validity: schema complete, no duplicates, no unknown tracks, no empty/identical sentences
    is_valid = (
        len(missing_columns) == 0
        and len(duplicate_pair_ids) == 0
        and len(empty_sentences) == 0
        and len(identical_pairs) == 0
        and len(unknown_tracks) == 0
        and len(non_empty_rows) > 0
    )

    return {
        "file_name": path.name,
        "file_path": str(path.resolve()),
        "is_valid": is_valid,
        "schema": {
            "expected_column_count": len(EXPECTED_COLUMNS),
            "actual_column_count": len(actual_columns),
            "columns_present": actual_columns,
            "missing_columns": missing_columns,
            "unexpected_columns": unexpected_columns,
            "column_order_matches": column_order_matches,
        },
        "row_counts": {
            "total_raw_rows_in_sheet": len(raw_matrix),
            "total_data_rows": total_data_rows,
            "non_empty_data_rows": len(non_empty_rows),
            "empty_trailing_rows": empty_trailing_count,
        },
        "track_distribution": dict(track_counts),
        "final_status_distribution": dict(final_status_counts),
        "integrity_checks": {
            "duplicate_pair_ids": duplicate_pair_ids,
            "missing_required_fields_count": len(missing_required_by_row),
            "missing_required_by_row": missing_required_by_row,
            "empty_sentences_count": len(empty_sentences),
            "empty_sentences": empty_sentences,
            "identical_pairs_count": len(identical_pairs),
            "identical_pairs": identical_pairs,
            "unknown_tracks_count": len(unknown_tracks),
            "unknown_tracks": unknown_tracks,
        },
        "human_review_disagreements": {
            "total_disagreements_count": len(disagreements),
            "disagreements": disagreements,
            "policy_note": "Disagreements are reported for human review only and are not automatically resolved.",
        },
        "devanagari_diagnostics": {
            "metrics": devanagari_diagnostics,
            "policy_note": "Devanagari presence is a diagnostic indicator, not an automatic rejection rule.",
        },
    }


def format_validation_report(result: dict[str, Any]) -> str:
    """Format validation results into a human-readable text report."""
    lines: list[str] = [
        "=" * 60,
        "HINDI LEXICAL SENSITIVITY - DATASET VALIDATION REPORT",
        "=" * 60,
        f"File: {result.get('file_name')}",
        f"Path: {result.get('file_path')}",
        f"Overall Valid: {'YES' if result.get('is_valid') else 'NO'}",
        "",
        "--- 1. SCHEMA VALIDATION ---",
    ]

    schema = result.get("schema", {})
    lines.append(f"Expected Columns ({schema.get('expected_column_count')}): {EXPECTED_COLUMNS}")
    lines.append(f"Actual Columns ({schema.get('actual_column_count')}): {schema.get('columns_present')}")
    lines.append(f"Missing Columns: {schema.get('missing_columns') or 'None'}")
    lines.append(f"Unexpected Columns: {schema.get('unexpected_columns') or 'None'}")
    lines.append(f"Column Order Matches: {schema.get('column_order_matches')}")

    lines.extend([
        "",
        "--- 2. ROW COUNTS ---",
    ])
    rc = result.get("row_counts", {})
    lines.append(f"Total Rows in Sheet: {rc.get('total_raw_rows_in_sheet')}")
    lines.append(f"Total Data Rows (excl. header): {rc.get('total_data_rows')}")
    lines.append(f"Non-Empty Data Rows: {rc.get('non_empty_data_rows')}")
    lines.append(f"Fully Empty Trailing Rows: {rc.get('empty_trailing_rows')}")

    lines.extend([
        "",
        "--- 3. TRACK & STATUS DISTRIBUTIONS ---",
        f"Tracks: {result.get('track_distribution')}",
        f"Final Status: {result.get('final_status_distribution')}",
        "",
        "--- 4. INTEGRITY CHECKS ---",
    ])
    ic = result.get("integrity_checks", {})
    lines.append(f"Duplicate pair_id Count: {len(ic.get('duplicate_pair_ids', {}))}")
    if ic.get("duplicate_pair_ids"):
        lines.append(f"  Duplicates: {ic.get('duplicate_pair_ids')}")
    lines.append(f"Missing Required Fields Rows: {ic.get('missing_required_fields_count')}")
    lines.append(f"Empty Sentence Rows: {ic.get('empty_sentences_count')}")
    lines.append(f"Identical Sentence Pairs: {ic.get('identical_pairs_count')}")
    lines.append(f"Unknown Track Values: {ic.get('unknown_tracks_count')}")

    lines.extend([
        "",
        "--- 5. HUMAN REVIEW DISAGREEMENTS ---",
    ])
    hr = result.get("human_review_disagreements", {})
    disagreements = hr.get("disagreements", [])
    lines.append(f"Disagreements Found: {len(disagreements)}")
    lines.append(f"Policy Note: {hr.get('policy_note')}")
    if disagreements:
        lines.append("Sample Disagreements:")
        for item in disagreements[:10]:
            lines.append(
                f"  - Row {item['excel_row']} (pair_id: {item['pair_id']}): "
                f"annotator_1={item['annotator_1']}, annotator_2={item['annotator_2']}, "
                f"adjudication={item['adjudication']} -> final_status={item['final_status']} "
                f"[{', '.join(item['reasons'])}]"
            )
        if len(disagreements) > 10:
            lines.append(f"  ... and {len(disagreements) - 10} more.")

    lines.extend([
        "",
        "--- 6. DEVANAGARI SCRIPT DIAGNOSTICS ---",
    ])
    dd = result.get("devanagari_diagnostics", {})
    metrics = dd.get("metrics", {})
    lines.append(f"Original Sentences with Devanagari: {metrics.get('original_sentence_with_devanagari')}")
    lines.append(f"Contrast Sentences with Devanagari: {metrics.get('contrast_sentence_with_devanagari')}")
    lines.append(f"Target Words (Orig) with Devanagari: {metrics.get('target_original_with_devanagari')}")
    lines.append(f"Target Words (Cont) with Devanagari: {metrics.get('target_contrast_with_devanagari')}")
    lines.append(f"Rows Lacking Devanagari Sentences: {len(metrics.get('rows_without_devanagari_sentences', []))}")
    lines.append(f"Policy Note: {dd.get('policy_note')}")

    lines.extend(["=" * 60, ""])
    return "\n".join(lines)


def export_report(result: dict[str, Any], output_path: str | Path, input_path: str | Path) -> Path:
    """Export validation report to a separate path.

    Raises:
        ValueError: If output_path resolves to the same path as input_path.
    """
    in_resolved = Path(input_path).resolve()
    out_path = Path(output_path)
    out_resolved = out_path.resolve()

    if in_resolved == out_resolved:
        raise ValueError(
            f"Safety check failed: output path cannot overwrite input dataset ({out_resolved})"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix.lower() == ".json":
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
    else:
        text_content = format_validation_report(result)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text_content)

    return out_path


def main() -> None:
    """CLI entrypoint for running dataset validation."""
    parser = argparse.ArgumentParser(
        description="Validate raw Hindi lexical sensitivity Excel dataset."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Path to raw dataset (.xlsx)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Optional separate path to export validation report (.json or .txt/.md). Never overwrites input.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON output to stdout.",
    )

    args = parser.parse_args()

    result = validate_dataset(args.input_path)

    if args.output:
        saved_path = export_report(result, args.output, args.input_path)
        print(f"Validation report exported to: {saved_path}")

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(format_validation_report(result))


if __name__ == "__main__":
    main()
