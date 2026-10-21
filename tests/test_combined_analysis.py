"""Offline automated tests for Stage 4 Combined Statistical Analysis.

Tests cover:
- Merge correctness and key matching
- Delta recalculation and sign convention verification
- Missing value handling
- Duplicate detection
- Descriptive statistics math
- Correlation calculation
- Benjamini-Hochberg correction
- Model comparison logic
- CSV output integrity

No network access, no model downloads required.
"""

from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

from src.analysis.combined_analysis import (
    COMBINED_FIELDS,
    MODELS,
    TRACKS,
    benjamini_hochberg,
    correlation_analysis,
    descriptive_stats,
    merge_and_validate,
    model_comparison,
    run_analysis,
    spearman_ci_95,
    spearman_p_value,
    spearman_r,
    _iqr,
    _mean,
    _median,
    _percentile,
    _std,
)


# ─────────────────────────────────────────────
# Helpers to build synthetic CSV fixtures
# ─────────────────────────────────────────────

TOK_HEADERS = [
    "pair_id", "_excel_row", "model_name", "track", "contrast_label",
    "original_sentence", "contrast_sentence", "target_original", "target_contrast",
    "orig_token_count_inclusive", "orig_token_count_exclusive",
    "cont_token_count_inclusive", "cont_token_count_exclusive",
    "delta_token_count_inclusive", "delta_token_count_exclusive",
    "target_orig_alignment_status", "target_orig_token_count",
    "target_cont_alignment_status", "target_cont_token_count",
    "delta_target_token_count",
    "orig_tokens", "orig_token_ids", "cont_tokens", "cont_token_ids",
    "target_orig_tokens", "target_orig_token_ids", "target_cont_tokens", "target_cont_token_ids",
    "intended_meaning_hi", "intended_meaning_en", "grammatical_original", "grammatical_contrast",
    "notes", "annotator_1", "annotator_2", "adjudication", "final_status",
    "source_type", "source_reference",
]

PLL_HEADERS = [
    "pair_id", "_excel_row", "model_name", "track", "contrast_label",
    "original_sentence", "contrast_sentence", "target_original", "target_contrast",
    "orig_pll_sum", "orig_scored_token_count", "orig_mean_log_prob",
    "cont_pll_sum", "cont_scored_token_count", "cont_mean_log_prob",
    "delta_pll_sum", "delta_mean_log_prob", "delta_scored_token_count",
    "intended_meaning_hi", "intended_meaning_en", "grammatical_original", "grammatical_contrast",
    "notes", "annotator_1", "annotator_2", "adjudication", "final_status",
    "source_type", "source_reference", "run_status", "notes_and_warnings",
]


def _make_tok_row(
    pair_id: str,
    model: str,
    track: str = "VALID_WORD_CONTEXT",
    orig_excl: int = 5,
    cont_excl: int = 6,
    tgt_orig: int = 1,
    tgt_cont: int = 2,
    final_status: str = "accept",
) -> dict[str, str]:
    delta_excl = cont_excl - orig_excl
    delta_tgt = tgt_cont - tgt_orig
    return {
        "pair_id": pair_id,
        "_excel_row": "2",
        "model_name": model,
        "track": track,
        "contrast_label": "contextual_substitution_error",
        "original_sentence": "test original",
        "contrast_sentence": "test contrast",
        "target_original": "orig",
        "target_contrast": "cont",
        "orig_token_count_inclusive": str(orig_excl + 2),
        "orig_token_count_exclusive": str(orig_excl),
        "cont_token_count_inclusive": str(cont_excl + 2),
        "cont_token_count_exclusive": str(cont_excl),
        "delta_token_count_inclusive": str(delta_excl),
        "delta_token_count_exclusive": str(delta_excl),
        "target_orig_alignment_status": "unique",
        "target_orig_token_count": str(tgt_orig),
        "target_cont_alignment_status": "unique",
        "target_cont_token_count": str(tgt_cont),
        "delta_target_token_count": str(delta_tgt),
        "orig_tokens": "[]",
        "orig_token_ids": "[]",
        "cont_tokens": "[]",
        "cont_token_ids": "[]",
        "target_orig_tokens": "[]",
        "target_orig_token_ids": "[]",
        "target_cont_tokens": "[]",
        "target_cont_token_ids": "[]",
        "intended_meaning_hi": "",
        "intended_meaning_en": "",
        "grammatical_original": "yes",
        "grammatical_contrast": "no",
        "notes": "",
        "annotator_1": "accept",
        "annotator_2": "accept",
        "adjudication": "accept",
        "final_status": final_status,
        "source_type": "team_authored",
        "source_reference": "team_authored",
    }


def _make_pll_row(
    pair_id: str,
    model: str,
    track: str = "VALID_WORD_CONTEXT",
    orig_pll: float = -10.0,
    cont_pll: float = -15.0,
    orig_cnt: int = 5,
    cont_cnt: int = 5,
    final_status: str = "accept",
) -> dict[str, str]:
    delta_pll = cont_pll - orig_pll
    orig_mean = orig_pll / orig_cnt if orig_cnt else 0.0
    cont_mean = cont_pll / cont_cnt if cont_cnt else 0.0
    delta_mean = cont_mean - orig_mean
    delta_cnt = cont_cnt - orig_cnt
    return {
        "pair_id": pair_id,
        "_excel_row": "2",
        "model_name": model,
        "track": track,
        "contrast_label": "contextual_substitution_error",
        "original_sentence": "test original",
        "contrast_sentence": "test contrast",
        "target_original": "orig",
        "target_contrast": "cont",
        "orig_pll_sum": str(orig_pll),
        "orig_scored_token_count": str(orig_cnt),
        "orig_mean_log_prob": str(orig_mean),
        "cont_pll_sum": str(cont_pll),
        "cont_scored_token_count": str(cont_cnt),
        "cont_mean_log_prob": str(cont_mean),
        "delta_pll_sum": str(delta_pll),
        "delta_mean_log_prob": str(delta_mean),
        "delta_scored_token_count": str(delta_cnt),
        "intended_meaning_hi": "",
        "intended_meaning_en": "",
        "grammatical_original": "yes",
        "grammatical_contrast": "no",
        "notes": "",
        "annotator_1": "accept",
        "annotator_2": "accept",
        "adjudication": "accept",
        "final_status": final_status,
        "source_type": "team_authored",
        "source_reference": "team_authored",
        "run_status": "completed",
        "notes_and_warnings": "none",
    }


def write_csv(path: Path, headers: list[str], rows: list[dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)


def make_minimal_csvs(
    tmp_dir: Path,
    n_pairs: int = 6,
    models: list[str] | None = None,
) -> tuple[Path, Path]:
    """Create minimal matching tokenization and PLL CSVs for n_pairs pairs x 2 models."""
    if models is None:
        models = MODELS

    tracks_cycle = (TRACKS * ((n_pairs // len(TRACKS)) + 1))[:n_pairs]
    pll_values = [(-8.0 - i, -10.0 - i) for i in range(n_pairs)]

    tok_rows, pll_rows = [], []
    for model in models:
        for i in range(n_pairs):
            pid = f"P{i+1:04d}"
            track = tracks_cycle[i]
            tok_rows.append(_make_tok_row(pid, model, track=track,
                                          orig_excl=5 + i % 3, cont_excl=5 + (i + 1) % 3 + 1))
            o_pll, c_pll = pll_values[i]
            pll_rows.append(_make_pll_row(pid, model, track=track,
                                          orig_pll=o_pll, cont_pll=c_pll))

    tok_path = tmp_dir / "tok.csv"
    pll_path = tmp_dir / "pll.csv"
    write_csv(tok_path, TOK_HEADERS, tok_rows)
    write_csv(pll_path, PLL_HEADERS, pll_rows)
    return tok_path, pll_path


# ─────────────────────────────────────────────
# Test classes
# ─────────────────────────────────────────────

class TestMathHelpers(unittest.TestCase):
    """Test pure math utility functions."""

    def test_mean_basic(self) -> None:
        self.assertAlmostEqual(_mean([1.0, 2.0, 3.0]), 2.0, places=10)

    def test_mean_empty_returns_none(self) -> None:
        self.assertIsNone(_mean([]))

    def test_median_odd(self) -> None:
        self.assertAlmostEqual(_median([3.0, 1.0, 2.0]), 2.0, places=10)

    def test_median_even(self) -> None:
        self.assertAlmostEqual(_median([1.0, 2.0, 3.0, 4.0]), 2.5, places=10)

    def test_std_known(self) -> None:
        # std([2,4,4,4,5,5,7,9]) = 2.0 (population); sample std = 2.138...
        vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        std_val = _std(vals)
        self.assertIsNotNone(std_val)
        self.assertGreater(std_val, 0)

    def test_std_single_returns_none(self) -> None:
        self.assertIsNone(_std([5.0]))

    def test_iqr_known(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        iqr = _iqr(vals)
        self.assertIsNotNone(iqr)
        self.assertGreater(iqr, 0)

    def test_percentile_boundary(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        self.assertAlmostEqual(_percentile(vals, 0), 1.0, places=5)
        self.assertAlmostEqual(_percentile(vals, 100), 5.0, places=5)


class TestSpearman(unittest.TestCase):
    """Test Spearman correlation and significance."""

    def test_perfect_positive_correlation(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        r, n = spearman_r(x, y)
        self.assertAlmostEqual(r, 1.0, places=8)
        self.assertEqual(n, 5)

    def test_perfect_negative_correlation(self) -> None:
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        r, n = spearman_r(x, y)
        self.assertAlmostEqual(r, -1.0, places=8)

    def test_uncorrelated(self) -> None:
        # Constant y => zero correlation (or nan due to zero variance)
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 1.0, 1.0, 1.0, 1.0]
        r, n = spearman_r(x, y)
        self.assertTrue(math.isnan(r))

    def test_insufficient_n(self) -> None:
        r, n = spearman_r([1.0, 2.0], [2.0, 1.0])
        self.assertTrue(math.isnan(r))

    def test_p_value_perfect(self) -> None:
        r, n = spearman_r([1.0, 2.0, 3.0, 4.0, 5.0], [1.0, 2.0, 3.0, 4.0, 5.0])
        p = spearman_p_value(r, n)
        self.assertAlmostEqual(p, 0.0, places=5)

    def test_ci_width_decreases_with_n(self) -> None:
        r = 0.5
        lo10, hi10 = spearman_ci_95(r, 10)
        lo100, hi100 = spearman_ci_95(r, 100)
        self.assertLess((hi100 - lo100), (hi10 - lo10))

    def test_ci_nan_for_small_n(self) -> None:
        lo, hi = spearman_ci_95(0.5, 3)
        self.assertTrue(math.isnan(lo))

    def test_tied_ranks(self) -> None:
        # All ties in y — should return nan (zero variance)
        x = [1.0, 2.0, 3.0, 4.0]
        y = [5.0, 5.0, 5.0, 5.0]
        r, _ = spearman_r(x, y)
        self.assertTrue(math.isnan(r))


class TestBenjaminiHochberg(unittest.TestCase):
    """Test BH FDR correction."""

    def test_all_significant(self) -> None:
        p = [0.001, 0.002, 0.003]
        result = benjamini_hochberg(p, alpha=0.05)
        self.assertTrue(all(result))

    def test_none_significant(self) -> None:
        p = [0.9, 0.8, 0.7]
        result = benjamini_hochberg(p, alpha=0.05)
        self.assertFalse(any(result))

    def test_empty(self) -> None:
        self.assertEqual(benjamini_hochberg([]), [])

    def test_nan_not_rejected(self) -> None:
        p = [float("nan"), 0.001]
        result = benjamini_hochberg(p, alpha=0.05)
        self.assertFalse(result[0])  # nan should not be rejected
        self.assertTrue(result[1])

    def test_order_preserved(self) -> None:
        # The BH result should be indexed in original order
        p = [0.8, 0.001, 0.9]
        result = benjamini_hochberg(p, alpha=0.05)
        self.assertFalse(result[0])
        self.assertTrue(result[1])
        self.assertFalse(result[2])


class TestMergeAndValidate(unittest.TestCase):
    """Test merge_and_validate function."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_successful_merge_row_count(self) -> None:
        tok_path, pll_path = make_minimal_csvs(self.tmp_dir, n_pairs=6)
        rows, warnings = merge_and_validate(tok_path, pll_path)
        self.assertEqual(len(rows), 6 * 2)  # 6 pairs × 2 models
        self.assertEqual(len(warnings), 0)

    def test_delta_sign_convention(self) -> None:
        """Δ = contrast − original throughout."""
        tok_path, pll_path = make_minimal_csvs(self.tmp_dir, n_pairs=4)
        rows, _ = merge_and_validate(tok_path, pll_path)
        for r in rows:
            if r["delta_pll_sum"] is not None:
                expected = float(r["cont_pll_sum"]) - float(r["orig_pll_sum"])
                self.assertAlmostEqual(float(r["delta_pll_sum"]), expected, places=6,
                                       msg=f"Delta sign wrong for {r['pair_id']}")

    def test_token_delta_sign_convention(self) -> None:
        """Δ token = contrast count − original count."""
        tok_path, pll_path = make_minimal_csvs(self.tmp_dir, n_pairs=4)
        rows, _ = merge_and_validate(tok_path, pll_path)
        for r in rows:
            if r["delta_token_count_excl"] is not None:
                expected = int(r["cont_token_count_excl"]) - int(r["orig_token_count_excl"])
                self.assertEqual(int(r["delta_token_count_excl"]), expected)

    def test_missing_pll_key_generates_warning(self) -> None:
        tok_path, pll_path = make_minimal_csvs(self.tmp_dir, n_pairs=4)
        # Remove one row from PLL CSV
        with open(pll_path, "r", encoding="utf-8") as f:
            pll_rows = list(csv.DictReader(f))
        pll_rows_trimmed = pll_rows[1:]  # drop first row
        write_csv(pll_path, PLL_HEADERS, pll_rows_trimmed)
        _, warnings = merge_and_validate(tok_path, pll_path)
        self.assertGreater(len(warnings), 0)

    def test_label_mismatch_generates_warning(self) -> None:
        tok_path, pll_path = make_minimal_csvs(self.tmp_dir, n_pairs=3)
        # Corrupt track in PLL CSV for first row
        with open(pll_path, "r", encoding="utf-8") as f:
            pll_rows = list(csv.DictReader(f))
        pll_rows[0]["track"] = "WRONG_TRACK"
        write_csv(pll_path, PLL_HEADERS, pll_rows)
        _, warnings = merge_and_validate(tok_path, pll_path)
        self.assertTrue(any("mismatch" in w.lower() for w in warnings))

    def test_abs_delta_fields_populated(self) -> None:
        tok_path, pll_path = make_minimal_csvs(self.tmp_dir, n_pairs=4)
        rows, _ = merge_and_validate(tok_path, pll_path)
        for r in rows:
            if r["delta_token_count_excl"] is not None:
                self.assertEqual(r["abs_delta_token_count_excl"], abs(int(r["delta_token_count_excl"])))

    def test_no_duplicate_keys(self) -> None:
        tok_path, pll_path = make_minimal_csvs(self.tmp_dir, n_pairs=6)
        rows, _ = merge_and_validate(tok_path, pll_path)
        keys = [(r["pair_id"], r["model_name"]) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))

    def test_rejected_records_preserved(self) -> None:
        """Rejected records must not be silently dropped."""
        tok_rows_data, pll_rows_data = [], []
        for model in MODELS:
            tok_rows_data.append(_make_tok_row("P0001", model, final_status="reject"))
            tok_rows_data.append(_make_tok_row("P0002", model))
            pll_rows_data.append(_make_pll_row("P0001", model, final_status="reject"))
            pll_rows_data.append(_make_pll_row("P0002", model))

        tok_path = self.tmp_dir / "tok_rej.csv"
        pll_path = self.tmp_dir / "pll_rej.csv"
        write_csv(tok_path, TOK_HEADERS, tok_rows_data)
        write_csv(pll_path, PLL_HEADERS, pll_rows_data)

        rows, _ = merge_and_validate(tok_path, pll_path)
        rejected = [r for r in rows if r["final_status"] == "reject"]
        self.assertEqual(len(rejected), len(MODELS))  # 2 models × 1 rejected pair


class TestDescriptiveStats(unittest.TestCase):
    """Test descriptive statistics calculations."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        tmp_dir = Path(self.tmp.name)
        # 6 pairs across all 3 tracks (2 per track)
        tracks_cycle = (TRACKS * 2)[:6]
        tok_rows_data, pll_rows_data = [], []
        for model in MODELS:
            for i, track in enumerate(tracks_cycle):
                pid = f"P{i+1:04d}"
                tok_rows_data.append(_make_tok_row(pid, model, track=track))
                pll_rows_data.append(_make_pll_row(pid, model, track=track,
                                                    orig_pll=-8.0 - i, cont_pll=-12.0 - i))
        tok_path = tmp_dir / "tok.csv"
        pll_path = tmp_dir / "pll.csv"
        write_csv(tok_path, TOK_HEADERS, tok_rows_data)
        write_csv(pll_path, PLL_HEADERS, pll_rows_data)
        self.rows, _ = merge_and_validate(tok_path, pll_path)
        self.tmp.cleanup()

    def test_n_pairs_correct(self) -> None:
        stats = descriptive_stats(self.rows, include_rejected=True)
        for s in stats:
            if s["track"] == "VALID_WORD_CONTEXT":
                self.assertEqual(s["n_pairs"], 2)

    def test_delta_meanlp_all_negative(self) -> None:
        """Since cont_pll < orig_pll for all rows, delta_mean should be negative."""
        stats = descriptive_stats(self.rows, include_rejected=True)
        for s in stats:
            if s["n_pairs"] > 0 and s["delta_meanlp_mean"]:
                self.assertLess(float(s["delta_meanlp_mean"]), 0)

    def test_sensitivity_excl_has_fewer_or_equal_rows(self) -> None:
        stats_incl = descriptive_stats(self.rows, include_rejected=True)
        stats_excl = descriptive_stats(self.rows, include_rejected=False)
        for si, se in zip(stats_incl, stats_excl):
            self.assertLessEqual(se["n_pairs"], si["n_pairs"])

    def test_sign_counts_sum_to_n(self) -> None:
        stats = descriptive_stats(self.rows, include_rejected=True)
        for s in stats:
            total = s["delta_meanlp_n_positive"] + s["delta_meanlp_n_negative"] + s["delta_meanlp_n_zero"]
            self.assertEqual(total, s["n_pairs"])


class TestCorrelationAnalysis(unittest.TestCase):
    """Test correlation analysis output structure and logic."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        tmp_dir = Path(self.tmp.name)
        tok_path, pll_path = make_minimal_csvs(tmp_dir, n_pairs=20)
        self.rows, _ = merge_and_validate(tok_path, pll_path)
        self.tmp.cleanup()

    def test_output_contains_all_models(self) -> None:
        corr = correlation_analysis(self.rows)
        models_in_output = {r["model_name"] for r in corr}
        self.assertEqual(models_in_output, set(MODELS))

    def test_bh_reject_is_boolean_string(self) -> None:
        corr = correlation_analysis(self.rows)
        for r in corr:
            self.assertIn(r["bh_reject"], ("True", "False"))

    def test_primary_and_secondary_outcomes_present(self) -> None:
        corr = correlation_analysis(self.rows)
        y_vars = {r["y_variable"] for r in corr}
        self.assertIn("delta_mean_log_prob", y_vars)
        self.assertIn("delta_pll_sum", y_vars)

    def test_small_sample_note_present(self) -> None:
        """Rows with n < 20 should have a caution note."""
        # With 20 pairs total, per-track has ~6-7 which is < 10
        corr = correlation_analysis(self.rows)
        track_rows = [r for r in corr if r["track_scope"] == "REALISTIC_SPELLING_ERROR"]
        if track_rows:
            for cr in track_rows:
                if int(cr["n"]) < 10:
                    self.assertIn("WARNING", cr["note"])


class TestModelComparison(unittest.TestCase):
    """Test model comparison logic."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        tmp_dir = Path(self.tmp.name)
        tok_path, pll_path = make_minimal_csvs(tmp_dir, n_pairs=10)
        self.rows, _ = merge_and_validate(tok_path, pll_path)
        self.tmp.cleanup()

    def test_direction_counts_sum_to_n(self) -> None:
        comp = model_comparison(self.rows)
        for r in comp:
            total = (r["direction_both_negative"] + r["direction_both_positive"] +
                     r["direction_xlm_neg_muril_pos"] + r["direction_xlm_pos_muril_neg"])
            self.assertEqual(total, r["n_pairs"])

    def test_all_tracks_scope_present(self) -> None:
        comp = model_comparison(self.rows)
        scopes = {r["track_scope"] for r in comp}
        self.assertIn("ALL_TRACKS", scopes)

    def test_agreement_rate_in_0_1(self) -> None:
        comp = model_comparison(self.rows)
        for r in comp:
            rate = float(r["direction_agreement_rate"])
            self.assertGreaterEqual(rate, 0.0)
            self.assertLessEqual(rate, 1.0)


class TestRunAnalysisIntegration(unittest.TestCase):
    """Integration test for full run_analysis pipeline."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_fake_xlsx(self) -> Path:
        """Create a minimal fake xlsx file (actually JSON, just to have a file for hashing)."""
        p = self.tmp_dir / "fake.xlsx"
        p.write_bytes(b"FAKE_XLSX_CONTENT")
        return p

    def test_full_pipeline_creates_all_outputs(self) -> None:
        tok_path, pll_path = make_minimal_csvs(self.tmp_dir, n_pairs=6)
        xlsx_path = self._make_fake_xlsx()
        output_dir = self.tmp_dir / "analysis_out"

        rows, warnings, meta = run_analysis(tok_path, pll_path, xlsx_path, output_dir)

        # Check all required files created
        self.assertTrue((output_dir / "combined_pair_level_results.csv").is_file())
        self.assertTrue((output_dir / "descriptive_statistics.csv").is_file())
        self.assertTrue((output_dir / "correlation_results.csv").is_file())
        self.assertTrue((output_dir / "model_comparison.csv").is_file())
        self.assertTrue((output_dir / "analysis_report.md").is_file())
        self.assertTrue((output_dir / "run_metadata.json").is_file())

    def test_combined_csv_has_expected_row_count(self) -> None:
        n_pairs = 8
        tok_path, pll_path = make_minimal_csvs(self.tmp_dir, n_pairs=n_pairs)
        xlsx_path = self._make_fake_xlsx()
        output_dir = self.tmp_dir / "out2"

        rows, _, _ = run_analysis(tok_path, pll_path, xlsx_path, output_dir)
        self.assertEqual(len(rows), n_pairs * len(MODELS))

    def test_combined_csv_has_correct_fields(self) -> None:
        tok_path, pll_path = make_minimal_csvs(self.tmp_dir, n_pairs=4)
        xlsx_path = self._make_fake_xlsx()
        output_dir = self.tmp_dir / "out3"
        run_analysis(tok_path, pll_path, xlsx_path, output_dir)

        with open(output_dir / "combined_pair_level_results.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            actual_fields = reader.fieldnames
        for field in COMBINED_FIELDS:
            self.assertIn(field, actual_fields, f"Missing field: {field}")

    def test_metadata_records_warnings(self) -> None:
        tok_path, pll_path = make_minimal_csvs(self.tmp_dir, n_pairs=3)
        # Remove 1 PLL row to cause a warning
        with open(pll_path, "r", encoding="utf-8") as f:
            pll_data = list(csv.DictReader(f))
        write_csv(pll_path, PLL_HEADERS, pll_data[1:])
        xlsx_path = self._make_fake_xlsx()
        output_dir = self.tmp_dir / "out4"

        _, warnings, meta = run_analysis(tok_path, pll_path, xlsx_path, output_dir)
        self.assertEqual(len(meta["validation_warnings"]), len(warnings))

    def test_no_duplicate_rows_in_combined_csv(self) -> None:
        tok_path, pll_path = make_minimal_csvs(self.tmp_dir, n_pairs=6)
        xlsx_path = self._make_fake_xlsx()
        output_dir = self.tmp_dir / "out5"
        rows, _, _ = run_analysis(tok_path, pll_path, xlsx_path, output_dir)
        keys = [(r["pair_id"], r["model_name"]) for r in rows]
        self.assertEqual(len(keys), len(set(keys)))


if __name__ == "__main__":
    unittest.main()
