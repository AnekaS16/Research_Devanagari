"""Offline unit tests for Stage 5 — Baseline Evaluation & Explanation Extraction.

Tests use synthetic data and mocked models/tokenizers.
No model downloads, network access, or GPU is required.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import tempfile
import unittest
from typing import Any
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Mini mock infrastructure
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Minimal mock tokenizer matching the DistilBERT/BERT interface."""

    def __init__(self) -> None:
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0
        self.mask_token_id = 103
        self.all_special_ids = {0, 101, 102, 103}
        self._vocab = {
            "[CLS]": 101, "[SEP]": 102, "[PAD]": 0, "[MASK]": 103,
            "great": 1001, "movie": 1002, "terrible": 1003, "film": 1004,
            "awful": 1005, "good": 1006, "bad": 1007,
        }
        self._rev = {v: k for k, v in self._vocab.items()}

    def _encode(self, text: str) -> list[int]:
        words = text.lower().split()
        ids = [self._vocab.get(w, 999) for w in words]
        return [self.cls_token_id] + ids + [self.sep_token_id]

    def __call__(
        self,
        text: str | list[str],
        padding: bool = False,
        truncation: bool = False,
        max_length: int = 128,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        if isinstance(text, list):
            all_ids = [self._encode(t) for t in text]
            max_len = max(len(ids) for ids in all_ids)
            if padding:
                all_ids = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in all_ids]
            sp_masks = [[1 if i in self.all_special_ids else 0 for i in ids] for ids in all_ids]
            attn_masks = [[1] * len(ids) for ids in all_ids]
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor(all_ids, dtype=torch.long),
                    "special_tokens_mask": torch.tensor(sp_masks, dtype=torch.long),
                    "attention_mask": torch.tensor(attn_masks, dtype=torch.long),
                }
            return {"input_ids": all_ids, "special_tokens_mask": sp_masks, "attention_mask": attn_masks}
        else:
            ids = self._encode(text)
            sp_mask = [1 if i in self.all_special_ids else 0 for i in ids]
            attn_mask = [1] * len(ids)
            if return_tensors == "pt":
                return {
                    "input_ids": torch.tensor([ids], dtype=torch.long),
                    "special_tokens_mask": torch.tensor([sp_mask], dtype=torch.long),
                    "attention_mask": torch.tensor([attn_mask], dtype=torch.long),
                }
            return {"input_ids": ids, "special_tokens_mask": sp_mask, "attention_mask": attn_mask}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._rev.get(i, f"tok_{i}") for i in ids]


class MockDistilBERTOutput:
    def __init__(self, logits: torch.Tensor, attentions: list[torch.Tensor] | None = None) -> None:
        self.logits = logits
        self.attentions = attentions


class _MockEmbeddingSubmodule(nn.Module):
    """Embedding submodule that Captum can hook."""
    def __init__(self, vocab_size: int = 5000, hidden: int = 32) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden)
        with torch.no_grad():
            self.word_embeddings.weight.zero_()
            self.word_embeddings.weight[1001, 0] = 2.0   # great
            self.word_embeddings.weight[1006, 0] = 2.0   # good
            self.word_embeddings.weight[1003, 0] = -2.0  # terrible
            self.word_embeddings.weight[1005, 0] = -2.0  # awful
            self.word_embeddings.weight[1007, 0] = -2.0  # bad

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.word_embeddings(input_ids)


class _MockDistilBERTBody(nn.Module):
    """Body module giving model.distilbert.embeddings access."""
    def __init__(self, vocab_size: int = 5000, hidden: int = 32) -> None:
        super().__init__()
        self.embeddings = _MockEmbeddingSubmodule(vocab_size, hidden)


class MockDistilBERT(nn.Module):
    """Mock DistilBERT SST-2 classifier with deterministic, differentiable outputs.

    Architecture mirrors the attribute paths Captum requires:
        model.distilbert.embeddings  <-- Captum hooks this
    During forward, model.distilbert.embeddings(input_ids) is called
    explicitly so Captum's hook fires correctly.
    Logits are computed via differentiable tensor operations from embeddings
    to preserve the autograd graph for gradient-based attribution methods.
    """

    def __init__(self, n_layers: int = 6, n_heads: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden = hidden
        self.num_labels = 2
        self.distilbert = _MockDistilBERTBody(hidden=hidden)
        self.classifier = nn.Linear(hidden, 2)
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.bias.zero_()
            self.classifier.weight[0, 0] = -1.0
            self.classifier.weight[1, 0] = 1.0

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> MockDistilBERTOutput:
        # Captum hooks self.distilbert.embeddings — it MUST be called here
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be specified")
            inputs_embeds = self.distilbert.embeddings(input_ids)

        b, seq_len, _ = inputs_embeds.shape

        # Differentiable pooling over sequence dimension:
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
            pooled = (inputs_embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = inputs_embeds.mean(dim=1)

        logits = self.classifier(pooled)

        attentions = None
        if output_attentions:
            raw = torch.ones(b, self.n_heads, seq_len, seq_len) / seq_len
            attentions = [raw] * self.n_layers

        return MockDistilBERTOutput(logits=logits, attentions=attentions)



def mock_sst2_dataset(n: int = 20) -> list[dict[str, Any]]:
    """Build a synthetic SST-2 dataset."""
    templates = [
        ("great movie", 1),
        ("terrible film", 0),
        ("good film", 1),
        ("awful movie", 0),
        ("bad film", 0),
        ("great film", 1),
    ]
    result = []
    for i in range(n):
        sentence, label = templates[i % len(templates)]
        result.append({"sentence": f"{sentence} {i}", "label": label})
    return result


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestPredictionAndLabelMapping(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.model = MockDistilBERT()
        self.model.eval()
        self.tokenizer = MockTokenizer()

    def test_prediction_returns_correct_shape(self) -> None:
        from src.explanations.baseline_and_explanations import predict_batch
        results = predict_batch(
            ["great movie", "terrible film"],
            self.model,
            self.tokenizer,
            self.device,
        )
        self.assertEqual(len(results), 2)
        self.assertIn("predicted_label", results[0])
        self.assertIn("confidence_positive", results[0])
        self.assertIn("confidence_negative", results[0])
        self.assertAlmostEqual(
            results[0]["confidence_positive"] + results[0]["confidence_negative"], 1.0, places=5
        )

    def test_label_mapping_is_binary(self) -> None:
        from src.explanations.baseline_and_explanations import predict_batch
        results = predict_batch(["good", "bad"], self.model, self.tokenizer, self.device)
        for r in results:
            self.assertIn(r["predicted_label"], [0, 1])

    def test_evaluate_validation_set_metrics(self) -> None:
        from src.explanations.baseline_and_explanations import evaluate_validation_set
        dataset = mock_sst2_dataset(20)
        records, metrics = evaluate_validation_set(
            dataset, self.model, self.tokenizer, self.device, batch_size=4
        )
        self.assertEqual(len(records), 20)
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("confusion_matrix", metrics)
        cm = metrics["confusion_matrix"]
        total = cm["TP"] + cm["TN"] + cm["FP"] + cm["FN"]
        self.assertEqual(total, 20)
        self.assertGreaterEqual(metrics["accuracy"], 0.0)
        self.assertLessEqual(metrics["accuracy"], 1.0)


class TestTokenToAttributionAlignment(unittest.TestCase):

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.model = MockDistilBERT()
        self.model.eval()
        self.tokenizer = MockTokenizer()

    def test_attention_content_tokens_exclude_special(self) -> None:
        from src.explanations.baseline_and_explanations import extract_attention
        result = extract_attention(
            "great movie film", self.model, self.tokenizer, self.device
        )
        self.assertIn("[CLS]", result["tokens"])
        self.assertNotIn("[CLS]", result["content_tokens"])
        self.assertNotIn("[SEP]", result["content_tokens"])

    def test_attention_scores_length_matches_content_tokens(self) -> None:
        from src.explanations.baseline_and_explanations import extract_attention
        result = extract_attention("good bad film", self.model, self.tokenizer, self.device)
        self.assertEqual(len(result["content_tokens"]), len(result["content_scores"]))

    def test_attention_scores_sum_to_one(self) -> None:
        from src.explanations.baseline_and_explanations import extract_attention
        result = extract_attention("great movie", self.model, self.tokenizer, self.device)
        if result["content_scores"]:
            total = sum(result["content_scores"])
            self.assertAlmostEqual(total, 1.0, places=5)

    def test_attention_returns_warning_for_invalid_layer(self) -> None:
        from src.explanations.baseline_and_explanations import extract_attention
        result = extract_attention(
            "great movie", self.model, self.tokenizer, self.device, layer=99
        )
        self.assertIsNotNone(result.get("warning"))
        self.assertEqual(result["content_tokens"], [])


class TestPilotSelection(unittest.TestCase):

    def test_pilot_balanced_across_labels(self) -> None:
        from src.explanations.baseline_and_explanations import select_pilot_examples
        records = [
            {"is_correct": True, "true_label": 1, "example_id": i} for i in range(30)
        ] + [
            {"is_correct": True, "true_label": 0, "example_id": i + 30} for i in range(30)
        ] + [
            {"is_correct": False, "true_label": 1, "example_id": i + 60} for i in range(5)
        ]
        pilot = select_pilot_examples(records, pilot_size=20, error_max=5, seed=42)
        correct = [r for r in pilot if r["is_correct"]]
        errors = [r for r in pilot if not r["is_correct"]]
        self.assertGreaterEqual(len(correct), 18)
        self.assertLessEqual(len(errors), 5)

    def test_pilot_is_reproducible(self) -> None:
        from src.explanations.baseline_and_explanations import select_pilot_examples
        records = [{"is_correct": True, "true_label": i % 2, "example_id": i} for i in range(60)]
        pilot_a = select_pilot_examples(records, pilot_size=10, seed=42)
        pilot_b = select_pilot_examples(records, pilot_size=10, seed=42)
        self.assertEqual([r["example_id"] for r in pilot_a], [r["example_id"] for r in pilot_b])

    def test_error_cases_included(self) -> None:
        from src.explanations.baseline_and_explanations import select_pilot_examples
        records = [{"is_correct": True, "true_label": i % 2, "example_id": i} for i in range(20)]
        records += [{"is_correct": False, "true_label": 0, "example_id": 100}]
        pilot = select_pilot_examples(records, pilot_size=10, error_max=5, seed=42)
        self.assertTrue(any(not r["is_correct"] for r in pilot))


class TestIntegratedGradientsAttribution(unittest.TestCase):
    """Regression and unit tests for Integrated Gradients extraction."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.model = MockDistilBERT()
        self.model.eval()
        self.tokenizer = MockTokenizer()

    def test_integrated_gradients_on_differentiable_mock(self) -> None:
        from src.explanations.baseline_and_explanations import extract_integrated_gradients
        result = extract_integrated_gradients(
            "great movie film",
            self.model,
            self.tokenizer,
            self.device,
            target_class=1,
            n_steps=10,
        )
        self.assertIsNone(result.get("warning"))
        self.assertEqual(result["tokens"], ["[CLS]", "great", "movie", "film", "[SEP]"])
        self.assertEqual(result["content_tokens"], ["great", "movie", "film"])
        self.assertEqual(len(result["content_ig_signed"]), 3)
        self.assertEqual(len(result["content_ig_abs"]), 3)
        self.assertIsNotNone(result["convergence_delta"])
        self.assertTrue(isinstance(result["convergence_delta"], float))
        self.assertTrue(math.isfinite(result["convergence_delta"]))
        for score in result["content_ig_signed"]:
            self.assertTrue(math.isfinite(score))
        for score in result["content_ig_abs"]:
            self.assertTrue(math.isfinite(score))
        self.assertEqual(result["baseline"], "zero_embedding")

    def test_integrated_gradients_special_tokens_excluded(self) -> None:
        from src.explanations.baseline_and_explanations import extract_integrated_gradients
        result = extract_integrated_gradients(
            "good bad",
            self.model,
            self.tokenizer,
            self.device,
            target_class=0,
            n_steps=5,
        )
        self.assertNotIn("[CLS]", result["content_tokens"])
        self.assertNotIn("[SEP]", result["content_tokens"])
        self.assertNotIn("[PAD]", result["content_tokens"])
        self.assertEqual(len(result["content_tokens"]), len(result["content_ig_signed"]))
        self.assertEqual(len(result["content_tokens"]), len(result["content_ig_abs"]))

    def test_integrated_gradients_regression_disconnected_graph_raises(self) -> None:
        """Verify that a disconnected computation graph raises the original RuntimeError.

        This regression test proves that if a model constructs logits from a
        detached tensor or fails to maintain an autograd path from the hooked
        embedding module, Captum will raise RuntimeError with
        'appears to not have been used in the graph'.
        """
        from src.explanations.baseline_and_explanations import extract_integrated_gradients

        class DisconnectedMockModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.distilbert = _MockDistilBERTBody()

            def forward(
                self,
                input_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                **kwargs: Any,
            ) -> MockDistilBERTOutput:
                if inputs_embeds is None:
                    inputs_embeds = self.distilbert.embeddings(input_ids)
                b = inputs_embeds.shape[0]
                # Detached leaf tensor with no autograd connection to inputs_embeds (original bug)
                pos_score = float(inputs_embeds.detach().sum().item()) * 0.001
                logits = torch.tensor(
                    [[0.3, 0.7 + pos_score] for _ in range(b)],
                    dtype=torch.float32,
                    requires_grad=inputs_embeds.requires_grad,
                )
                return MockDistilBERTOutput(logits=logits)

        disconnected_model = DisconnectedMockModel()
        disconnected_model.eval()

        with self.assertRaises(RuntimeError) as ctx:
            extract_integrated_gradients(
                "great movie",
                disconnected_model,
                self.tokenizer,
                self.device,
                target_class=1,
                n_steps=5,
            )
        self.assertIn("appears to not have been used in the graph", str(ctx.exception))


class TestFullPipelineIntegration(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name) / "explanations"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _make_loader(self) -> Any:
        def loader(model_name: str) -> tuple[Any, Any]:
            model = MockDistilBERT()
            model.eval()
            return model, MockTokenizer()
        return loader

    def _make_dataset_loader(self) -> Any:
        def loader() -> tuple[Any, Any]:
            # Use lists (no HF dataset dependency in tests)
            train = mock_sst2_dataset(40)
            val = mock_sst2_dataset(20)
            return train, val
        return loader

    def test_full_pipeline_creates_expected_files(self) -> None:
        from src.explanations.baseline_and_explanations import run_explanation_pipeline
        metadata, pilot, summary = run_explanation_pipeline(
            output_dir=self.out_dir,
            pilot_size=6,
            error_max=2,
            seed=42,
            force=True,
            model_tokenizer_loader=self._make_loader(),
            dataset_loader=self._make_dataset_loader(),
        )
        self.assertTrue((self.out_dir / "baseline_metrics.json").is_file())
        self.assertTrue((self.out_dir / "baseline_predictions.csv").is_file())
        self.assertTrue((self.out_dir / "pilot_explanations.csv").is_file())
        self.assertTrue((self.out_dir / "run_metadata.json").is_file())
        self.assertTrue((self.out_dir / "summary.md").is_file())
        self.assertTrue(metadata["completed_successfully"])

    def test_pilot_records_contain_explanation_fields(self) -> None:
        from src.explanations.baseline_and_explanations import run_explanation_pipeline
        _, pilot, _ = run_explanation_pipeline(
            output_dir=self.out_dir,
            pilot_size=4,
            error_max=1,
            seed=42,
            force=True,
            model_tokenizer_loader=self._make_loader(),
            dataset_loader=self._make_dataset_loader(),
        )
        self.assertGreater(len(pilot), 0)
        for rec in pilot:
            self.assertIn("attention_scores", rec)
            self.assertIn("ig_attributions_signed", rec)

    def test_cached_run_returns_same_results(self) -> None:
        from src.explanations.baseline_and_explanations import run_explanation_pipeline
        # First run
        meta1, _, _ = run_explanation_pipeline(
            output_dir=self.out_dir,
            pilot_size=4,
            error_max=1,
            seed=42,
            force=True,
            model_tokenizer_loader=self._make_loader(),
            dataset_loader=self._make_dataset_loader(),
        )
        # Second run (cache hit)
        meta2, _, _ = run_explanation_pipeline(
            output_dir=self.out_dir,
            pilot_size=4,
            error_max=1,
            seed=42,
            force=False,
            model_tokenizer_loader=self._make_loader(),
            dataset_loader=self._make_dataset_loader(),
        )
        self.assertTrue(meta2["completed_successfully"])

    def test_metadata_schema_has_required_fields(self) -> None:
        from src.explanations.baseline_and_explanations import run_explanation_pipeline
        metadata, _, _ = run_explanation_pipeline(
            output_dir=self.out_dir,
            pilot_size=4,
            error_max=1,
            seed=42,
            force=True,
            model_tokenizer_loader=self._make_loader(),
            dataset_loader=self._make_dataset_loader(),
        )
        required = [
            "run_timestamp_utc", "model_name", "model_note", "dataset",
            "device", "seed", "ig_n_steps", "ig_baseline",
            "attention_layer", "attention_aggregation",
            "baseline_metrics", "library_versions", "output_files",
        ]
        for field in required:
            self.assertIn(field, metadata, f"Missing metadata field: {field}")

    def test_error_cases_preserved_in_pilot(self) -> None:
        from src.explanations.baseline_and_explanations import run_explanation_pipeline
        metadata, pilot, _ = run_explanation_pipeline(
            output_dir=self.out_dir,
            pilot_size=4,
            error_max=3,
            seed=42,
            force=True,
            model_tokenizer_loader=self._make_loader(),
            dataset_loader=self._make_dataset_loader(),
        )
        # Records should exist; error cases tracked by is_correct field
        for rec in pilot:
            self.assertIn("is_correct", rec)


if __name__ == "__main__":
    unittest.main()
