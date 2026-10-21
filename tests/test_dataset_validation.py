"""Unit tests for the Hindi lexical sensitivity dataset validator.

All tests create and run against synthetic temporary Excel files.
Tests do not depend on or modify the real research dataset.
Compatible with pytest and standard library unittest.
"""

from __future__ import annotations

import io
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile

from src.data.validate import (
    EXPECTED_COLUMNS,
    contains_devanagari,
    export_report,
    is_row_empty,
    validate_dataset,
)

try:
    import openpyxl

    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False


def create_synthetic_xlsx(path: Path, rows: list[list[any]]) -> None:
    """Create a minimal .xlsx file for testing purposes.

    Uses openpyxl if available, otherwise creates a valid minimal OpenXML zip archive.
    """
    if HAS_OPENPYXL:
        wb = openpyxl.Workbook()
        ws = wb.active
        for row in rows:
            ws.append(row)
        wb.save(path)
        wb.close()
        return

    # Fallback minimal xlsx creator using Python standard library zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>""",
        )
        z.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
        )
        z.writestr(
            "xl/workbook.xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Sheet1" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>""",
        )

        sheet_rows: list[str] = []
        for r_idx, row in enumerate(rows, start=1):
            cell_strs: list[str] = []
            for c_idx, val in enumerate(row, start=1):
                col_letter = chr(64 + c_idx) if c_idx <= 26 else f"A{chr(64 + c_idx - 26)}"
                if val is not None and str(val) != "":
                    esc = (
                        str(val)
                        .replace("&", "&amp;")
                        .replace("<", "&lt;")
                        .replace(">", "&gt;")
                        .replace('"', "&quot;")
                    )
                    cell_strs.append(
                        f'<c r="{col_letter}{r_idx}" t="inlineStr"><is><t>{esc}</t></is></c>'
                    )
            sheet_rows.append(f'<row r="{r_idx}">{"".join(cell_strs)}</row>')

        sheet_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">\n'
            f"<sheetData>{''.join(sheet_rows)}</sheetData>\n"
            "</worksheet>"
        )
        z.writestr("xl/worksheets/sheet1.xml", sheet_xml)

    with open(path, "wb") as f:
        f.write(buf.getvalue())


def get_sample_valid_row(pair_id: str = "SYNTH_001", track: str = "VALID_WORD_CONTEXT") -> list[str]:
    """Return a single synthetic row matching all 18 expected columns."""
    return [
        pair_id,  # pair_id
        track,  # track
        "यह एक सुंदर घर है।",  # original_sentence
        "यह एक सुंदर मकान है।",  # contrast_sentence
        "घर",  # target_original
        "मकान",  # target_contrast
        "निवास स्थान",  # intended_meaning_hi
        "living place",  # intended_meaning_en
        "yes",  # grammatical_original
        "yes",  # grammatical_contrast
        "meaning_preserving_variant",  # contrast_label
        "synthetic test note",  # notes
        "accept",  # annotator_1
        "accept",  # annotator_2
        "accept",  # adjudication
        "accept",  # final_status
        "synthetic",  # source_type
        "synthetic_test",  # source_reference
    ]


class TestDatasetValidation(unittest.TestCase):
    """Test suite for dataset validation logic using synthetic temporary files."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_devanagari_detector(self) -> None:
        """Test Devanagari character detection."""
        self.assertTrue(contains_devanagari("नमस्ते"))
        self.assertTrue(contains_devanagari("घर 123"))
        self.assertFalse(contains_devanagari("Hello world"))
        self.assertFalse(contains_devanagari("12345"))
        self.assertFalse(contains_devanagari(""))

    def test_empty_row_detector(self) -> None:
        """Test is_row_empty helper."""
        self.assertTrue(is_row_empty([]))
        self.assertTrue(is_row_empty([None, "", "   "]))
        self.assertFalse(is_row_empty([None, "data", ""]))

    def test_valid_synthetic_dataset(self) -> None:
        """Test that a complete, valid dataset passes validation."""
        xlsx_path = self.test_dir / "valid_test.xlsx"
        rows = [
            EXPECTED_COLUMNS,
            get_sample_valid_row("TEST_001", "VALID_WORD_CONTEXT"),
            get_sample_valid_row("TEST_002", "SYNONYM_SUBSTITUTION"),
            get_sample_valid_row("TEST_003", "REALISTIC_SPELLING_ERROR"),
        ]
        create_synthetic_xlsx(xlsx_path, rows)

        result = validate_dataset(xlsx_path)

        self.assertTrue(result["is_valid"])
        self.assertEqual(result["schema"]["missing_columns"], [])
        self.assertEqual(result["schema"]["unexpected_columns"], [])
        self.assertEqual(result["row_counts"]["non_empty_data_rows"], 3)
        self.assertEqual(result["row_counts"]["empty_trailing_rows"], 0)
        self.assertEqual(len(result["integrity_checks"]["duplicate_pair_ids"]), 0)
        self.assertEqual(result["integrity_checks"]["missing_required_fields_count"], 0)
        self.assertEqual(result["integrity_checks"]["empty_sentences_count"], 0)
        self.assertEqual(result["integrity_checks"]["identical_pairs_count"], 0)
        self.assertEqual(result["integrity_checks"]["unknown_tracks_count"], 0)

    def test_trailing_empty_rows_separated(self) -> None:
        """Test that trailing empty rows are counted separately from data rows."""
        xlsx_path = self.test_dir / "trailing_empty.xlsx"
        empty_row = ["" for _ in EXPECTED_COLUMNS]
        rows = [
            EXPECTED_COLUMNS,
            get_sample_valid_row("TEST_001"),
            get_sample_valid_row("TEST_002"),
            empty_row,
            empty_row,
            empty_row,
        ]
        create_synthetic_xlsx(xlsx_path, rows)

        result = validate_dataset(xlsx_path)

        self.assertEqual(result["row_counts"]["total_data_rows"], 5)
        self.assertEqual(result["row_counts"]["non_empty_data_rows"], 2)
        self.assertEqual(result["row_counts"]["empty_trailing_rows"], 3)
        self.assertTrue(result["is_valid"])

    def test_missing_and_unexpected_columns(self) -> None:
        """Test detection of missing and unexpected columns."""
        xlsx_path = self.test_dir / "bad_columns.xlsx"
        # Remove 'track' and add 'extra_custom_col'
        corrupted_cols = [c for c in EXPECTED_COLUMNS if c != "track"] + ["extra_custom_col"]
        sample_row = get_sample_valid_row("TEST_001")
        # Remove index 1 (track) and add an extra value
        corrupted_row = sample_row[:1] + sample_row[2:] + ["extra_val"]
        rows = [corrupted_cols, corrupted_row]
        create_synthetic_xlsx(xlsx_path, rows)

        result = validate_dataset(xlsx_path)

        self.assertFalse(result["is_valid"])
        self.assertIn("track", result["schema"]["missing_columns"])
        self.assertIn("extra_custom_col", result["schema"]["unexpected_columns"])

    def test_duplicate_pair_id_detected(self) -> None:
        """Test detection of duplicate pair_id."""
        xlsx_path = self.test_dir / "duplicate_id.xlsx"
        rows = [
            EXPECTED_COLUMNS,
            get_sample_valid_row("DUPLICATE_ID"),
            get_sample_valid_row("DUPLICATE_ID"),
        ]
        create_synthetic_xlsx(xlsx_path, rows)

        result = validate_dataset(xlsx_path)

        self.assertFalse(result["is_valid"])
        self.assertIn("DUPLICATE_ID", result["integrity_checks"]["duplicate_pair_ids"])
        self.assertEqual(len(result["integrity_checks"]["duplicate_pair_ids"]["DUPLICATE_ID"]), 2)

    def test_empty_and_identical_sentences_detected(self) -> None:
        """Test empty sentence detection and identical sentence pair detection."""
        xlsx_path = self.test_dir / "sentence_issues.xlsx"
        # Row 1 has empty contrast sentence
        row_empty_sent = get_sample_valid_row("EMPTY_SENT")
        row_empty_sent[3] = ""  # contrast_sentence
        # Row 2 has identical sentences
        row_identical = get_sample_valid_row("IDENTICAL")
        row_identical[3] = row_identical[2]  # contrast = original

        rows = [EXPECTED_COLUMNS, row_empty_sent, row_identical]
        create_synthetic_xlsx(xlsx_path, rows)

        result = validate_dataset(xlsx_path)

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["integrity_checks"]["empty_sentences_count"], 1)
        self.assertEqual(result["integrity_checks"]["identical_pairs_count"], 1)
        self.assertEqual(result["integrity_checks"]["identical_pairs"][0]["pair_id"], "IDENTICAL")

    def test_unknown_track_detected(self) -> None:
        """Test detection of invalid track category."""
        xlsx_path = self.test_dir / "bad_track.xlsx"
        row_bad_track = get_sample_valid_row("BAD_TRACK", track="UNSUPPORTED_TRACK")
        rows = [EXPECTED_COLUMNS, row_bad_track]
        create_synthetic_xlsx(xlsx_path, rows)

        result = validate_dataset(xlsx_path)

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["integrity_checks"]["unknown_tracks_count"], 1)
        self.assertEqual(
            result["integrity_checks"]["unknown_tracks"][0]["track_value"],
            "UNSUPPORTED_TRACK",
        )

    def test_human_review_disagreements_reported(self) -> None:
        """Test reporting of annotator/adjudication disagreements without modifying data."""
        xlsx_path = self.test_dir / "disagreement.xlsx"
        row_disagree = get_sample_valid_row("DISAGREE_001")
        row_disagree[12] = "reject"  # annotator_1
        row_disagree[13] = "reject"  # annotator_2
        row_disagree[14] = "reject"  # adjudication
        row_disagree[15] = "accept"  # final_status differs from adjudication!

        rows = [EXPECTED_COLUMNS, row_disagree]
        create_synthetic_xlsx(xlsx_path, rows)

        result = validate_dataset(xlsx_path)

        disagreements = result["human_review_disagreements"]["disagreements"]
        self.assertEqual(len(disagreements), 1)
        self.assertEqual(disagreements[0]["pair_id"], "DISAGREE_001")
        self.assertIn("adjudication != final_status", disagreements[0]["reasons"])
        # Original values preserved
        self.assertEqual(disagreements[0]["final_status"], "accept")

    def test_devanagari_diagnostics_reported(self) -> None:
        """Test that Devanagari text diagnostics are reported without failing validation."""
        xlsx_path = self.test_dir / "script_diag.xlsx"
        # Row with English words in sentence fields
        row_non_dev = get_sample_valid_row("NON_DEV_001")
        row_non_dev[2] = "This is an English sentence."
        row_non_dev[3] = "This is a contrast sentence."

        rows = [EXPECTED_COLUMNS, row_non_dev]
        create_synthetic_xlsx(xlsx_path, rows)

        result = validate_dataset(xlsx_path)

        dd = result["devanagari_diagnostics"]["metrics"]
        self.assertEqual(dd["original_sentence_with_devanagari"], 0)
        self.assertEqual(len(dd["rows_without_devanagari_sentences"]), 1)
        self.assertEqual(
            dd["rows_without_devanagari_sentences"][0]["pair_id"],
            "NON_DEV_001",
        )

    def test_export_safety_never_overwrites_input(self) -> None:
        """Test that export_report raises ValueError if output path matches input path."""
        xlsx_path = self.test_dir / "input.xlsx"
        create_synthetic_xlsx(xlsx_path, [EXPECTED_COLUMNS, get_sample_valid_row()])

        result = validate_dataset(xlsx_path)

        # Attempting to export to the exact same file path must raise ValueError
        with self.assertRaises(ValueError):
            export_report(result, xlsx_path, xlsx_path)

        # Exporting to a separate file must succeed
        report_out = self.test_dir / "validation_report.json"
        saved = export_report(result, report_out, xlsx_path)
        self.assertTrue(saved.is_file())


if __name__ == "__main__":
    unittest.main()
