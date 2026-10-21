"""Offline unit tests for Stage 3 Pretrained Model PLL Scoring.

Uses synthetic inputs, mocked MaskedLM models, and mocked tokenizers.
Does not require network access, model weight downloads, or GPUs.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.validate import EXPECTED_COLUMNS
from src.models.pll_scoring import (
    load_checkpoint,
    run_pll_scoring_experiment,
    save_checkpoint,
    score_pair_for_model,
    score_sentence_pll,
    select_pilot_pairs,
)
from tests.test_dataset_validation import create_synthetic_xlsx, get_sample_valid_row


class MockModelOutput:
    """Container mimicking Hugging Face MaskedLMOutput."""

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits


class MockMaskedLM(nn.Module):
    """Mock MaskedLM model returning deterministic synthetic logits."""

    def __init__(self, vocab_size: int = 1000) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> MockModelOutput:
        # Batch size B, sequence length L
        b, l = input_ids.shape
        # Create uniform logits with a slight boost at target ID for numerical stability
        logits = torch.zeros((b, l, self.vocab_size), dtype=torch.float32)
        # For each batch element, give a specific score
        for i in range(b):
            for j in range(l):
                # Set a known logit distribution
                logits[i, j, :] = -2.0
                tok_id = input_ids[i, j].item()
                if 0 <= tok_id < self.vocab_size:
                    logits[i, j, tok_id] = 2.0
        return MockModelOutput(logits=logits)


class MockTokenizerForPLL:
    """Mock tokenizer simulating BERT and RoBERTa tokenizers for PLL scoring."""

    def __init__(
        self,
        model_type: str = "bert",
        vocab_size: int = 1000,
    ) -> None:
        self.model_type = model_type
        self.vocab_size = vocab_size
        self.name_or_path = f"mock-{model_type}"

        if model_type == "roberta":
            self.cls_token_id = 0
            self.sep_token_id = 2
            self.pad_token_id = 1
            self.mask_token_id = 500
            self.all_special_ids = {0, 1, 2, 500}
        else:  # bert
            self.cls_token_id = 101
            self.sep_token_id = 102
            self.pad_token_id = 0
            self.mask_token_id = 103
            self.all_special_ids = {0, 101, 102, 103}

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"tok_{token_id}"

    def __call__(
        self,
        text: str,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        words = text.split()
        # Simple mapping: word length + 10 -> token_id
        content_ids = [min(len(w) * 10 + idx + 10, self.vocab_size - 1) for idx, w in enumerate(words)]

        input_ids = [self.cls_token_id] + content_ids + [self.sep_token_id]
        special_mask = [1] + [0] * len(content_ids) + [1]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor([input_ids], dtype=torch.long),
                "special_tokens_mask": torch.tensor([special_mask], dtype=torch.long),
                "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
            }
        return {
            "input_ids": input_ids,
            "special_tokens_mask": special_mask,
            "attention_mask": [1] * len(input_ids),
        }


class TestPLLScoring(unittest.TestCase):
    """Test suite for pseudo-log-likelihood scoring logic."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.device = torch.device("cpu")
        self.model = MockMaskedLM(vocab_size=1000)
        self.model.eval()
        self.tok_bert = MockTokenizerForPLL("bert")
        self.tok_roberta = MockTokenizerForPLL("roberta")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_pll_calculation_known_probabilities(self) -> None:
        """Test PLL calculation produces mathematically sound log-probabilities."""
        sentence = "यह एक सुंदर घर है।"
        res = score_sentence_pll(
            sentence=sentence,
            model=self.model,
            tokenizer=self.tok_bert,
            device=self.device,
            batch_size=4,
        )

        # 5 words -> 5 ordinary scored tokens
        self.assertEqual(res["scored_token_count"], 5)
        self.assertIsInstance(res["pll_sum"], float)
        self.assertLess(res["pll_sum"], 0.0)  # Log-probabilities must be negative
        self.assertAlmostEqual(res["mean_log_prob"], res["pll_sum"] / 5, places=5)
        self.assertEqual(len(res["token_scores"]), 5)

    def test_special_tokens_and_padding_excluded(self) -> None:
        """Test that special tokens ([CLS], [SEP]) are excluded from scoring."""
        sentence = "शब्द"
        res = score_sentence_pll(
            sentence=sentence,
            model=self.model,
            tokenizer=self.tok_bert,
            device=self.device,
        )

        # Exactly 1 ordinary token
        self.assertEqual(res["scored_token_count"], 1)
        self.assertEqual(len(res["token_scores"]), 1)

    def test_empty_input_handling(self) -> None:
        """Test graceful handling of empty or blank input sentences."""
        res = score_sentence_pll(
            sentence="",
            model=self.model,
            tokenizer=self.tok_bert,
            device=self.device,
        )
        self.assertEqual(res["pll_sum"], 0.0)
        self.assertEqual(res["scored_token_count"], 0)
        self.assertIsNone(res["mean_log_prob"])
        self.assertEqual(res["warning"], "empty_sentence")

    def test_roberta_tokenizer_configuration(self) -> None:
        """Test scoring works correctly with RoBERTa-style tokenizer."""
        sentence = "परीक्षण वाक्य"
        res = score_sentence_pll(
            sentence=sentence,
            model=self.model,
            tokenizer=self.tok_roberta,
            device=self.device,
        )
        self.assertEqual(res["scored_token_count"], 2)
        self.assertLess(res["pll_sum"], 0.0)

    def test_signed_differences_pair_scoring(self) -> None:
        """Test that delta PLL and delta mean log-prob are contrast minus original."""
        row = {
            "pair_id": "P01",
            "_excel_row": 2,
            "track": "VALID_WORD_CONTEXT",
            "contrast_label": "contextual_substitution_error",
            "original_sentence": "घर सुंदर है।",
            "contrast_sentence": "बड़ा मकान अधिक सुंदर है।",
            "target_original": "घर",
            "target_contrast": "मकान",
            "final_status": "accept",
        }

        rec = score_pair_for_model(
            row=row,
            model=self.model,
            tokenizer=self.tok_bert,
            model_name="mock_model",
            device=self.device,
        )

        # Delta = Contrast - Original
        expected_delta_pll = rec["cont_pll_sum"] - rec["orig_pll_sum"]
        self.assertAlmostEqual(rec["delta_pll_sum"], expected_delta_pll, places=5)
        expected_delta_mean = rec["cont_mean_log_prob"] - rec["orig_mean_log_prob"]
        self.assertAlmostEqual(rec["delta_mean_log_prob"], expected_delta_mean, places=5)
        self.assertEqual(rec["delta_scored_token_count"], rec["cont_scored_token_count"] - rec["orig_scored_token_count"])

    def test_pilot_pair_selection(self) -> None:
        """Test pilot selection balances across all tracks and includes rejections."""
        rows = [
            {"pair_id": "1", "track": "VALID_WORD_CONTEXT", "final_status": "accept"},
            {"pair_id": "2", "track": "VALID_WORD_CONTEXT", "final_status": "accept"},
            {"pair_id": "3", "track": "SYNONYM_SUBSTITUTION", "final_status": "accept"},
            {"pair_id": "4", "track": "SYNONYM_SUBSTITUTION", "final_status": "accept"},
            {"pair_id": "5", "track": "REALISTIC_SPELLING_ERROR", "final_status": "accept"},
            {"pair_id": "6", "track": "REALISTIC_SPELLING_ERROR", "final_status": "reject"},  # rejected row
        ]

        pilot = select_pilot_pairs(rows, count_per_track=1)
        tracks_in_pilot = {r["track"] for r in pilot}
        self.assertEqual(len(tracks_in_pilot), 3)
        self.assertTrue(any(r["final_status"] == "reject" for r in pilot))

    def test_checkpoint_save_and_resume(self) -> None:
        """Test saving and loading checkpoints."""
        ckpt_path = self.test_dir / "checkpoint.json"
        data = {"P01::model1": {"pair_id": "P01", "orig_pll_sum": -10.5}}

        save_checkpoint(ckpt_path, data)
        loaded = load_checkpoint(ckpt_path)

        self.assertEqual(loaded, data)

    def test_full_experiment_mock_run(self) -> None:
        """Test complete PLL experiment execution with synthetic files and mocked loader."""
        xlsx_path = self.test_dir / "synthetic_dataset.xlsx"
        out_dir = self.test_dir / "scoring_out"

        rows = [
            EXPECTED_COLUMNS,
            get_sample_valid_row("P1", "VALID_WORD_CONTEXT"),
            get_sample_valid_row("P2", "SYNONYM_SUBSTITUTION"),
        ]
        create_synthetic_xlsx(xlsx_path, rows)

        def mock_loader(m_name: str) -> tuple[Any, Any]:
            return MockMaskedLM(), MockTokenizerForPLL()

        results, meta, summary = run_pll_scoring_experiment(
            input_file=xlsx_path,
            output_dir=out_dir,
            models=["mock_model"],
            pilot=False,
            force=False,
            model_loader=mock_loader,
        )

        self.assertEqual(len(results), 2)
        self.assertTrue((out_dir / "pll_scoring_results.csv").is_file())
        self.assertTrue((out_dir / "run_metadata.json").is_file())
        self.assertTrue((out_dir / "summary.md").is_file())
        self.assertTrue(meta["completed_successfully"])


if __name__ == "__main__":
    unittest.main()
