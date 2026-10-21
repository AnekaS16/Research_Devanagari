"""Offline unit tests for Stage 6 Hindi Explanation Stability Module.

Tests use synthetic Devanagari data, mocked tokenizers, and mock MLMs.
Zero model downloads, network access, or GPU required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
from typing import Any
import unittest

import torch
import torch.nn as nn

from tests.test_hindi_explanations import (
    MockHindiMLM,
    MockHindiMLMOutput,
    MockHindiTokenizer,
)


class TestRegionRelativeContextAlignment(unittest.TestCase):
    """Test context token alignment across target length shifts and asymmetric tokenization."""

    def setUp(self) -> None:
        self.tokenizer = MockHindiTokenizer()

    def test_post_target_alignment_when_target_length_changes(self) -> None:
        """Verify that when target word length changes, post-target context tokens align accurately.

        Example:
        Orig: 'हमें स्वास्थ्य का ध्यान रखना चाहिए।' (target: 'स्वास्थ्य', length 9)
        Cont: 'हमें स्वास्थ का ध्यान रखना चाहिए।' (target: 'स्वास्थ', length 7)
        The word 'का' starts at index 14 in Orig and index 12 in Cont.
        Region-relative alignment must match 'का' with relative offset 0 in post region!
        """
        from src.analysis.explanation_stability import align_context_tokens_by_region

        orig_sent = "हमें स्वास्थ्य का ध्यान रखना चाहिए।"
        cont_sent = "हमें स्वास्थ का ध्यान रखना चाहिए।"

        # In MockHindiTokenizer:
        enc_orig = {
            "input_ids": [101, 2020, 2021, 2022, 2023, 2024, 2025, 102],
            "offset_mapping": [
                (0, 0),    # [CLS]
                (0, 4),    # हमें (pre-target)
                (5, 14),   # स्वास्थ्य (target span)
                (15, 17),  # का (post-target, rel: 15-14=1, 17-14=3)
                (18, 23),  # ध्यान (post-target, rel: 18-14=4, 23-14=9)
                (24, 28),  # रखना
                (29, 34),  # चाहिए
                (0, 0),    # [SEP]
            ],
        }
        enc_cont = {
            "input_ids": [101, 2020, 2026, 2022, 2023, 2024, 2025, 102],
            "offset_mapping": [
                (0, 0),    # [CLS]
                (0, 4),    # हमें (pre-target)
                (5, 12),   # स्वास्थ (target span, length 7 instead of 9!)
                (13, 15),  # का (post-target, rel: 13-12=1, 15-12=3 -> exact match with orig!)
                (16, 21),  # ध्यान (post-target, rel: 16-12=4, 21-12=9 -> exact match!)
                (22, 26),  # रखना
                (27, 32),  # चाहिए
                (0, 0),    # [SEP]
            ],
        }

        res = align_context_tokens_by_region(
            orig_sentence=orig_sent,
            orig_target_span=(5, 14),
            orig_encoding=enc_orig,
            cont_sentence=cont_sent,
            cont_target_span=(5, 12),
            cont_encoding=enc_cont,
            tokenizer=self.tokenizer,
        )

        aligned_pairs = res["aligned_pairs"]
        # Must align: हमें (1, 1), का (3, 3), ध्यान (4, 4), रखना (5, 5), चाहिए (6, 6)
        self.assertEqual(len(aligned_pairs), 5)
        self.assertIn((1, 1), aligned_pairs)  # pre-target
        self.assertIn((3, 3), aligned_pairs)  # post-target
        self.assertIn((4, 4), aligned_pairs)  # post-target

        # Target tokens (2) must NOT be in aligned pairs
        orig_indices = [o for o, _ in aligned_pairs]
        cont_indices = [c for _, c in aligned_pairs]
        self.assertNotIn(2, orig_indices)
        self.assertNotIn(2, cont_indices)
        self.assertEqual(len(res["unaligned_orig_indices"]), 0)

    def test_special_tokens_excluded_from_alignment(self) -> None:
        from src.analysis.explanation_stability import align_context_tokens_by_region

        enc_orig = {
            "input_ids": [101, 2001, 2002, 102],
            "offset_mapping": [(0, 0), (0, 3), (4, 7), (0, 0)],
        }
        enc_cont = {
            "input_ids": [101, 2001, 2003, 102],
            "offset_mapping": [(0, 0), (0, 3), (4, 7), (0, 0)],
        }
        res = align_context_tokens_by_region(
            orig_sentence="राम और",
            orig_target_span=(4, 7),
            orig_encoding=enc_orig,
            cont_sentence="राम ओर",
            cont_target_span=(4, 7),
            cont_encoding=enc_cont,
            tokenizer=self.tokenizer,
        )
        for o_i, c_i in res["aligned_pairs"]:
            self.assertNotEqual(o_i, 0)   # not [CLS]
            self.assertNotEqual(o_i, 3)   # not [SEP]
            self.assertNotEqual(c_i, 0)
            self.assertNotEqual(c_i, 3)

    def test_unaligned_context_tokens_recorded_when_tokenization_differs(self) -> None:
        """When a word splits into 2 tokens in orig but 1 in cont, it should be marked unaligned."""
        from src.analysis.explanation_stability import align_context_tokens_by_region

        enc_orig = {
            "input_ids": [101, 2001, 2011, 2012, 102],
            "offset_mapping": [(0, 0), (0, 3), (4, 6), (6, 8), (0, 0)],  # word split into 2
        }
        enc_cont = {
            "input_ids": [101, 2001, 2010, 102],
            "offset_mapping": [(0, 0), (0, 3), (4, 8), (0, 0)],           # word single token
        }
        res = align_context_tokens_by_region(
            orig_sentence="राम पानी",
            orig_target_span=(0, 3),  # target is 'राम'
            orig_encoding=enc_orig,
            cont_sentence="राम पानी",
            cont_target_span=(0, 3),
            cont_encoding=enc_cont,
            tokenizer=self.tokenizer,
        )
        self.assertEqual(len(res["aligned_pairs"]), 0)
        self.assertEqual(len(res["unaligned_orig_indices"]), 2)
        self.assertEqual(len(res["unaligned_cont_indices"]), 1)


class TestStabilityMetrics(unittest.TestCase):
    """Test Spearman rho, Top-k Jaccard, and Cosine similarity edge cases."""

    def test_spearman_perfect_positive_and_negative(self) -> None:
        from src.analysis.explanation_stability import compute_spearman_rank_correlation

        u = [1.0, 2.0, 3.0, 4.0, 5.0]
        v = [10.0, 20.0, 30.0, 40.0, 50.0]
        rho, _ = compute_spearman_rank_correlation(u, v)
        self.assertEqual(rho, 1.0)

        v_neg = [50.0, 40.0, 30.0, 20.0, 10.0]
        rho_neg, _ = compute_spearman_rank_correlation(u, v_neg)
        self.assertEqual(rho_neg, -1.0)

    def test_spearman_undefined_when_k_less_than_3(self) -> None:
        from src.analysis.explanation_stability import compute_spearman_rank_correlation

        rho, pval = compute_spearman_rank_correlation([1.0, 2.0], [2.0, 1.0])
        self.assertIsNone(rho)
        self.assertIsNone(pval)

    def test_spearman_undefined_when_zero_variance(self) -> None:
        from src.analysis.explanation_stability import compute_spearman_rank_correlation

        rho, pval = compute_spearman_rank_correlation([3.0, 3.0, 3.0, 3.0], [1.0, 2.0, 3.0, 4.0])
        self.assertIsNone(rho)
        self.assertIsNone(pval)

    def test_top_k_jaccard_overlap_basic_and_perfect(self) -> None:
        from src.analysis.explanation_stability import compute_top_k_jaccard_overlap

        u = [5.0, 4.0, 1.0, 2.0]
        v = [50.0, 40.0, 10.0, 20.0]
        overlap, k = compute_top_k_jaccard_overlap(u, v, k=2)
        self.assertEqual(overlap, 1.0)
        self.assertEqual(k, 2)

        v_disjoint = [10.0, 20.0, 50.0, 40.0]
        overlap_disjoint, _ = compute_top_k_jaccard_overlap(u, v_disjoint, k=2)
        self.assertEqual(overlap_disjoint, 0.0)

    def test_top_k_jaccard_empty(self) -> None:
        from src.analysis.explanation_stability import compute_top_k_jaccard_overlap

        overlap, k = compute_top_k_jaccard_overlap([], [])
        self.assertIsNone(overlap)
        self.assertEqual(k, 0)

    def test_cosine_similarity_perfect_orthogonal_and_zero_norm(self) -> None:
        from src.analysis.explanation_stability import compute_cosine_similarity

        u = [1.0, 2.0, 3.0]
        v = [2.0, 4.0, 6.0]
        self.assertAlmostEqual(compute_cosine_similarity(u, v), 1.0, places=4)

        v_ortho = [-2.0, 1.0, 0.0]
        self.assertAlmostEqual(compute_cosine_similarity(u, v_ortho), 0.0, places=4)

        # Zero vector -> undefined (None)
        self.assertIsNone(compute_cosine_similarity([0.0, 0.0], [1.0, 2.0]))


class TestFixedTargetEvaluationAndEligibility(unittest.TestCase):
    """Test fixed-target scoring and eligibility decision rules."""

    def setUp(self) -> None:
        self.tokenizer = MockHindiTokenizer()
        self.model = MockHindiMLM()
        self.model.eval()
        self.device = torch.device("cpu")

    def test_eligible_single_subword_pair_evaluated_with_both_methods(self) -> None:
        from src.analysis.explanation_stability import evaluate_stability_for_pair

        pair = {
            "pair_id": "HC_0077",
            "_excel_row": 78,
            "track": "REALISTIC_SPELLING_ERROR",
            "contrast_label": "realistic_spelling_error",
            "original_sentence": "मेरी परीक्षा कल है।",
            "contrast_sentence": "मेरी परिक्षा कल है।",
            "target_original": "परीक्षा",
            "target_contrast": "परिक्षा",
            "final_status": "accept",
            "notes": "test",
        }

        elig, recs = evaluate_stability_for_pair(
            pair_row=pair,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            model_name="mock_mlm",
            n_steps=5,
        )

        self.assertTrue(elig["is_eligible"])
        self.assertEqual(elig["exclusion_reason"], "none")
        self.assertEqual(len(recs), 2)  # attention and IG

        # Check Attention record
        attn_rec = [r for r in recs if r["method"] == "attention"][0]
        self.assertEqual(attn_rec["eligibility_status"], "eligible")
        self.assertGreater(attn_rec["n_aligned_context_tokens"], 0)
        self.assertNotEqual(attn_rec["spearman_rho"], "")
        self.assertNotEqual(attn_rec["cosine_similarity"], "")

        # Check IG record
        ig_rec = [r for r in recs if r["method"] == "integrated_gradients"][0]
        self.assertEqual(ig_rec["eligibility_status"], "eligible")
        self.assertNotEqual(ig_rec["spearman_rho"], "")
        self.assertNotEqual(ig_rec["cosine_similarity"], "")
        self.assertNotEqual(ig_rec["orig_convergence_delta"], "")

    def test_ineligible_multi_subword_pair_recorded_with_reason(self) -> None:
        from src.analysis.explanation_stability import evaluate_stability_for_pair

        # "पानि" splits into 2 subwords in MockHindiTokenizer
        pair = {
            "pair_id": "HC_0005",
            "_excel_row": 6,
            "track": "REALISTIC_SPELLING_ERROR",
            "contrast_label": "realistic_spelling_error",
            "original_sentence": "मुझे पानी चाहिए।",
            "contrast_sentence": "मुझे पानि चाहिए।",
            "target_original": "पानी",
            "target_contrast": "पानि",
            "final_status": "accept",
            "notes": "spelling error",
        }

        elig, recs = evaluate_stability_for_pair(
            pair_row=pair,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            model_name="mock_mlm",
            n_steps=5,
        )

        self.assertFalse(elig["is_eligible"])
        self.assertIn("cont_multi_subword_target", elig["exclusion_reason"])
        for r in recs:
            self.assertEqual(r["eligibility_status"], "ineligible")
            self.assertEqual(r["spearman_rho"], "")
            self.assertEqual(r["cosine_similarity"], "")


class TestStage6PipelineIntegration(unittest.TestCase):
    """Test full pipeline execution, checkpointing, and output file schemas."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name) / "stability"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _mock_dataset(self) -> list[dict[str, Any]]:
        return [
            {
                "pair_id": "HC_0077",
                "_excel_row": 78,
                "track": "REALISTIC_SPELLING_ERROR",
                "contrast_label": "realistic_spelling_error",
                "original_sentence": "मेरी परीक्षा कल है।",
                "contrast_sentence": "मेरी परिक्षा कल है।",
                "target_original": "परीक्षा",
                "target_contrast": "परिक्षा",
                "final_status": "accept",
                "notes": "eligible",
            },
            {
                "pair_id": "HC_0005",
                "_excel_row": 6,
                "track": "REALISTIC_SPELLING_ERROR",
                "contrast_label": "realistic_spelling_error",
                "original_sentence": "मुझे पानी चाहिए।",
                "contrast_sentence": "मुझे पानि चाहिए।",
                "target_original": "पानी",
                "target_contrast": "पानि",
                "final_status": "accept",
                "notes": "ineligible",
            },
        ]

    def test_pipeline_generates_all_expected_artifacts(self) -> None:
        from src.analysis.explanation_stability import run_stage6_stability_experiment

        meta, pairs, summary = run_stage6_stability_experiment(
            output_dir=self.out_dir,
            models=["mock_model"],
            track_filter="REALISTIC_SPELLING_ERROR",
            n_steps=5,
            model_tokenizer_loader=lambda m: (MockHindiMLM(), MockHindiTokenizer()),
            dataset_loader=self._mock_dataset,
        )

        self.assertTrue((self.out_dir / "stability_pair_level.csv").is_file())
        self.assertTrue((self.out_dir / "stability_summary.csv").is_file())
        self.assertTrue((self.out_dir / "eligibility_report.csv").is_file())
        self.assertTrue((self.out_dir / "run_metadata.json").is_file())
        self.assertTrue((self.out_dir / "stage6_report.md").is_file())
        self.assertTrue((self.out_dir / "checkpoint.json").is_file())

        self.assertTrue(meta["completed_successfully"])
        self.assertEqual(len(pairs), 4)  # 2 pairs x 2 methods (attention, IG)
        self.assertEqual(len(summary), 2)  # 2 methods for mock_model

    def test_checkpoint_resumption_no_duplicates(self) -> None:
        from src.analysis.explanation_stability import run_stage6_stability_experiment

        # Run 1
        run_stage6_stability_experiment(
            output_dir=self.out_dir,
            models=["mock_model"],
            track_filter="REALISTIC_SPELLING_ERROR",
            n_steps=5,
            model_tokenizer_loader=lambda m: (MockHindiMLM(), MockHindiTokenizer()),
            dataset_loader=self._mock_dataset,
        )

        # Run 2 (resuming)
        meta2, pairs2, _ = run_stage6_stability_experiment(
            output_dir=self.out_dir,
            models=["mock_model"],
            track_filter="REALISTIC_SPELLING_ERROR",
            n_steps=5,
            model_tokenizer_loader=lambda m: (MockHindiMLM(), MockHindiTokenizer()),
            dataset_loader=self._mock_dataset,
        )
        self.assertEqual(len(pairs2), 4)  # No duplicate rows created


if __name__ == "__main__":
    unittest.main()
