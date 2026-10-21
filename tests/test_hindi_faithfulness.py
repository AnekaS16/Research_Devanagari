"""Tests for Stage 7 Hindi Faithfulness Evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from typing import Any

import torch
import torch.nn as nn

from src.analysis.hindi_faithfulness import (
    calculate_visible_target_score,
    compute_k_for_fraction,
    evaluate_faithfulness_interventions,
    evaluate_pair_faithfulness,
    extract_visible_target_attention,
    extract_visible_target_integrated_gradients,
    prepare_visible_target_input,
    run_stage7_faithfulness_experiment,
)
from tests.test_hindi_explanations import MockHindiMLM, MockHindiTokenizer


class TestVisibleTargetPreparationAndScoring(unittest.TestCase):
    """Test input preparation and visible-target scoring."""

    def setUp(self) -> None:
        self.tokenizer = MockHindiTokenizer()
        self.model = MockHindiMLM()
        self.model.eval()
        self.device = torch.device("cpu")

    def test_single_subword_visible_target_input(self) -> None:
        prep = prepare_visible_target_input(
            sentence="मेरी परीक्षा कल है।",
            target_word="परीक्षा",
            tokenizer=self.tokenizer,
            device=self.device,
        )
        self.assertTrue(prep["is_scorable"])
        self.assertTrue(prep["is_single_subword"])
        self.assertEqual(prep["target_subwords_count"], 1)
        self.assertEqual(len(prep["target_indices"]), 1)
        # Verify target is NOT masked in input_ids
        target_pos = prep["target_indices"][0]
        actual_id = prep["input_ids"][0, target_pos].item()
        self.assertNotEqual(actual_id, self.tokenizer.mask_token_id)
        self.assertEqual(actual_id, prep["target_token_ids"][0])
        # Verify context tokens do not include target
        self.assertNotIn(target_pos, prep["context_indices"])

    def test_multi_subword_visible_target_input(self) -> None:
        # In MockHindiTokenizer, 'पानि' splits into 'पा' and 'नि'
        prep = prepare_visible_target_input(
            sentence="मुझे पानि दो।",
            target_word="पानि",
            tokenizer=self.tokenizer,
            device=self.device,
        )
        self.assertTrue(prep["is_scorable"])
        self.assertFalse(prep["is_single_subword"])
        self.assertEqual(prep["target_subwords_count"], 2)
        self.assertEqual(len(prep["target_indices"]), 2)
        for p in prep["target_indices"]:
            self.assertNotIn(p, prep["context_indices"])

    def test_visible_target_score_calculation(self) -> None:
        prep = prepare_visible_target_input(
            sentence="मेरी परीक्षा कल है।",
            target_word="परीक्षा",
            tokenizer=self.tokenizer,
            device=self.device,
        )
        scores = calculate_visible_target_score(
            model=self.model,
            input_ids=prep["input_ids"],
            attention_mask=prep["attention_mask"],
            target_indices=prep["target_indices"],
            target_token_ids=prep["target_token_ids"],
        )
        self.assertIn("mean_logit", scores)
        self.assertIn("total_logit", scores)
        self.assertIn("mean_logp", scores)
        self.assertIsInstance(scores["mean_logit"], float)
        self.assertIsInstance(scores["mean_logp"], float)


class TestInterventionMechanics(unittest.TestCase):
    """Test fraction rounding policy and context masking interventions."""

    def setUp(self) -> None:
        self.tokenizer = MockHindiTokenizer()
        self.model = MockHindiMLM()
        self.model.eval()
        self.device = torch.device("cpu")

    def test_compute_k_for_fraction_rounding(self) -> None:
        # Rounding policy: k = max(1, round(fraction * n_context))
        self.assertEqual(compute_k_for_fraction(10, 0.10), 1)
        self.assertEqual(compute_k_for_fraction(10, 0.20), 2)
        self.assertEqual(compute_k_for_fraction(10, 0.30), 3)

        # Clamping at minimum 1 and maximum n_context
        self.assertEqual(compute_k_for_fraction(4, 0.10), 1)  # round(0.4) = 0 -> clamped to 1
        self.assertEqual(compute_k_for_fraction(1, 0.10), 1)
        self.assertEqual(compute_k_for_fraction(0, 0.10), 0)

    def test_target_preservation_during_interventions(self) -> None:
        """Verify that target tokens are never masked during faithfulness interventions."""
        prep = prepare_visible_target_input(
            sentence="मेरी परीक्षा कल है।",
            target_word="परीक्षा",
            tokenizer=self.tokenizer,
            device=self.device,
        )
        fake_importance = [0.1, 0.5, 0.2, 0.8]  # length equals len(context_indices)

        interventions = evaluate_faithfulness_interventions(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=prep["input_ids"],
            attention_mask=prep["attention_mask"],
            target_indices=prep["target_indices"],
            target_token_ids=prep["target_token_ids"],
            context_indices=prep["context_indices"],
            importance_scores=fake_importance,
            pair_id="HC_0077",
            sentence_type="original",
            method="attention",
            fractions=[0.20],
        )

        self.assertGreater(len(interventions), 0)
        target_pos = prep["target_indices"][0]

        for int_rec in interventions:
            masked_indices = json.loads(int_rec["masked_token_indices"])
            # Target position MUST NEVER be in masked_indices
            self.assertNotIn(target_pos, masked_indices)
            self.assertIn("logit_drop", int_rec)
            self.assertIn("abs_logit_drop", int_rec)
            self.assertIn("logp_drop", int_rec)


class TestExplanationExtractionMethods(unittest.TestCase):
    """Test Attention and Integrated Gradients extraction on visible target."""

    def setUp(self) -> None:
        self.tokenizer = MockHindiTokenizer()
        self.model = MockHindiMLM()
        self.model.eval()
        self.device = torch.device("cpu")

    def test_extract_visible_target_attention(self) -> None:
        prep = prepare_visible_target_input(
            sentence="मेरी परीक्षा कल है।",
            target_word="परीक्षा",
            tokenizer=self.tokenizer,
            device=self.device,
        )
        attn = extract_visible_target_attention(
            model=self.model,
            input_ids=prep["input_ids"],
            attention_mask=prep["attention_mask"],
            target_indices=prep["target_indices"],
            context_indices=prep["context_indices"],
        )
        self.assertEqual(len(attn["normalized_attentions"]), len(prep["context_indices"]))
        self.assertAlmostEqual(sum(attn["normalized_attentions"]), 1.0, places=4)

    def test_extract_visible_target_integrated_gradients(self) -> None:
        prep = prepare_visible_target_input(
            sentence="मेरी परीक्षा कल है।",
            target_word="परीक्षा",
            tokenizer=self.tokenizer,
            device=self.device,
        )
        ig = extract_visible_target_integrated_gradients(
            model=self.model,
            tokenizer=self.tokenizer,
            input_ids=prep["input_ids"],
            attention_mask=prep["attention_mask"],
            target_indices=prep["target_indices"],
            target_token_ids=prep["target_token_ids"],
            context_indices=prep["context_indices"],
            n_steps=5,
        )
        self.assertEqual(len(ig["ig_abs"]), len(prep["context_indices"]))
        self.assertEqual(len(ig["ig_signed"]), len(prep["context_indices"]))
        self.assertIsInstance(ig["convergence_delta"], float)


class TestStage7PipelineIntegration(unittest.TestCase):
    """Test end-to-end evaluation of pair and experiment pipeline."""

    def setUp(self) -> None:
        self.tokenizer = MockHindiTokenizer()
        self.model = MockHindiMLM()
        self.model.eval()
        self.device = torch.device("cpu")

    def test_evaluate_pair_faithfulness_end_to_end(self) -> None:
        pair_row = {
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

        elig, recs, stab = evaluate_pair_faithfulness(
            pair_row=pair_row,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            model_name="mock_mlm",
            fractions=[0.20],
            n_steps=5,
        )

        self.assertTrue(elig["is_scorable_visible"])
        self.assertGreater(len(recs), 0)
        # Check intervention record fields
        sample = recs[0]
        self.assertEqual(sample["pair_id"], "HC_0077")
        self.assertIn(sample["intervention_type"], ["top_k", "bottom_k", "random_k"])
        self.assertIn("baseline_target_logit_mean", sample)
        self.assertIn("intervened_target_logit_mean", sample)

    def test_summary_aggregation_and_win_rate(self) -> None:
        from src.analysis.hindi_faithfulness import aggregate_faithfulness_summary

        elig = [{
            "pair_id": "HC_0077",
            "model_name": "mock_mlm",
            "is_single_subword_strict": True,
        }]
        recs = [
            # original sentence
            {"pair_id": "HC_0077", "model_name": "mock_mlm", "track": "REALISTIC_SPELLING_ERROR", "method": "attention", "fraction": 0.20, "sentence_type": "original", "intervention_type": "top_k", "logit_drop": 2.5, "k_tokens": 1, "eligibility_status": "eligible"},
            {"pair_id": "HC_0077", "model_name": "mock_mlm", "track": "REALISTIC_SPELLING_ERROR", "method": "attention", "fraction": 0.20, "sentence_type": "original", "intervention_type": "bottom_k", "logit_drop": 0.5, "k_tokens": 1, "eligibility_status": "eligible"},
            {"pair_id": "HC_0077", "model_name": "mock_mlm", "track": "REALISTIC_SPELLING_ERROR", "method": "attention", "fraction": 0.20, "sentence_type": "original", "intervention_type": "random_k", "logit_drop": 1.0, "k_tokens": 1, "eligibility_status": "eligible"},
        ]

        summary = aggregate_faithfulness_summary(recs, elig)
        self.assertGreater(len(summary), 0)
        s = summary[0]
        self.assertEqual(s["win_rate_vs_bottom"], 1.0)
        self.assertEqual(s["win_rate_vs_random"], 1.0)
        self.assertAlmostEqual(s["faithfulness_gap_mean"], 2.0, places=4)

    def test_build_stability_faithfulness_table(self) -> None:
        from src.analysis.hindi_faithfulness import build_stability_faithfulness_table

        elig = [{
            "pair_id": "HC_0077",
            "model_name": "mock_mlm",
            "is_single_subword_strict": True,
        }]
        recs = [
            {"pair_id": "HC_0077", "model_name": "mock_mlm", "track": "REALISTIC_SPELLING_ERROR", "method": "attention", "fraction": 0.20, "sentence_type": "original", "intervention_type": "top_k", "logit_drop": 3.0, "k_tokens": 1, "eligibility_status": "eligible"},
            {"pair_id": "HC_0077", "model_name": "mock_mlm", "track": "REALISTIC_SPELLING_ERROR", "method": "attention", "fraction": 0.20, "sentence_type": "original", "intervention_type": "bottom_k", "logit_drop": 1.0, "k_tokens": 1, "eligibility_status": "eligible"},
        ]
        stab = {
            ("HC_0077", "mock_mlm", "attention"): {
                "attention_rho": 0.85,
                "attention_cosine": 0.92,
            }
        }
        rel = build_stability_faithfulness_table(recs, stab, elig)
        self.assertEqual(len(rel), 1)
        row = rel[0]
        self.assertEqual(row["pair_id"], "HC_0077")
        self.assertEqual(row["visible_target_stability_rho"], 0.85)
        self.assertEqual(row["orig_faithfulness_gap_f20"], 2.0)
