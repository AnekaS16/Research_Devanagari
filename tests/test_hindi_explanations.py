"""Offline unit tests for Stage 5 Hindi Explanation Pipeline.

Tests use synthetic Devanagari data and mocked models/tokenizers.
No network access, GPU, or large model downloads are required.
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


class MockHindiTokenizer:
    """Mock fast tokenizer for Hindi Devanagari matching BertTokenizerFast interface."""

    def __init__(self) -> None:
        self.cls_token_id = 101
        self.sep_token_id = 102
        self.pad_token_id = 0
        self.mask_token_id = 103
        self.all_special_ids = {0, 101, 102, 103}
        self.vocab = {
            "[PAD]": 0, "[CLS]": 101, "[SEP]": 102, "[MASK]": 103,
            "राम": 2001, "और": 2002, "ओर": 2003, "सीता": 2004, "साथ": 2005, "हैं": 2006, "।": 2007,
            "स्कूल": 2008, "विद्यालय": 2009, "पानी": 2010, "पा": 2011, "नि": 2012,
            "परीक्षा": 2013, "परिक्षा": 2014,
        }
        self.rev_vocab = {v: k for k, v in self.vocab.items()}

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = False,
        return_tensors: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        words = text.split()
        input_ids = [self.cls_token_id]
        offsets: list[tuple[int, int]] = [(0, 0)]

        for w in words:
            # Punctuation separation
            clean_w = w.rstrip("।")
            punct = "।" if w.endswith("।") else ""

            # Check if multi-subword
            if clean_w == "पानि":
                # splits into "पा" and "नि"
                pos = text.find("पानि")
                input_ids.append(self.vocab["पा"])
                offsets.append((pos, pos + 2))
                input_ids.append(self.vocab["नि"])
                offsets.append((pos + 2, pos + 4))
            else:
                pos = text.find(clean_w) if clean_w else 0
                tid = self.vocab.get(clean_w, 9999)
                input_ids.append(tid)
                offsets.append((pos, pos + len(clean_w)))

            if punct:
                pos = text.find("।")
                input_ids.append(self.vocab["।"])
                offsets.append((pos, pos + 1))

        input_ids.append(self.sep_token_id)
        offsets.append((0, 0))

        attention_mask = [1] * len(input_ids)
        special_mask = [1 if i in self.all_special_ids else 0 for i in input_ids]
        res: dict[str, Any] = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long) if return_tensors == "pt" else input_ids,
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long) if return_tensors == "pt" else attention_mask,
            "special_tokens_mask": special_mask,
        }
        if return_offsets_mapping:
            res["offset_mapping"] = offsets

        return res

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self.rev_vocab.get(i, f"tok_{i}") for i in ids]

    def convert_tokens_to_ids(self, tok: str) -> int:
        return self.vocab.get(tok, 9999)


class _MockBertEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 10000, hidden: int = 32) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.word_embeddings(input_ids)


class _MockBertBody(nn.Module):
    def __init__(self, vocab_size: int = 10000, hidden: int = 32) -> None:
        super().__init__()
        self.embeddings = _MockBertEmbedding(vocab_size, hidden)


class MockHindiMLMOutput:
    def __init__(self, logits: torch.Tensor, attentions: list[torch.Tensor] | None = None) -> None:
        self.logits = logits
        self.attentions = attentions


class MockHindiMLM(nn.Module):
    """Mock Masked Language Model with differentiable MLM prediction head."""

    def __init__(self, vocab_size: int = 10000, hidden: int = 32, n_layers: int = 4, n_heads: int = 4) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.bert = _MockBertBody(vocab_size, hidden)
        # MLM prediction head: projects hidden representation to vocabulary logits
        self.mlm_head = nn.Linear(hidden, vocab_size)
        with torch.no_grad():
            nn.init.normal_(self.mlm_head.weight, mean=0.1, std=0.05)
            self.mlm_head.bias.zero_()
            # Give target tokens positive bias
            self.mlm_head.bias[2002] = 5.0  # और
            self.mlm_head.bias[2003] = 4.0  # ओर

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> MockHindiMLMOutput:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("input_ids or inputs_embeds required")
            inputs_embeds = self.bert.embeddings(input_ids)

        b, seq_len, _ = inputs_embeds.shape
        # Context mixing to simulate transformer attention over sequence tokens
        context_summary = inputs_embeds.mean(dim=1, keepdim=True).expand(-1, seq_len, -1)
        pos_mod = torch.linspace(1.0, 2.0, seq_len, device=inputs_embeds.device).view(1, seq_len, 1)
        mixed = 0.5 * inputs_embeds + 0.5 * (context_summary * pos_mod)
        logits = self.mlm_head(mixed)  # (b, seq_len, vocab_size)

        attentions = None
        if output_attentions:
            # Varying weights across positions so attention distribution has non-zero variance
            base_weights = torch.linspace(0.1, 1.0, seq_len)
            raw = base_weights.unsqueeze(0).repeat(seq_len, 1)
            raw = raw / raw.sum(dim=-1, keepdim=True)
            raw = raw.unsqueeze(0).unsqueeze(0).repeat(b, self.n_heads, 1, 1)
            attentions = [raw] * self.n_layers

        return MockHindiMLMOutput(logits=logits, attentions=attentions)


class TestHindiTargetAlignmentAndMasking(unittest.TestCase):

    def setUp(self) -> None:
        self.tokenizer = MockHindiTokenizer()
        self.device = torch.device("cpu")

    def test_single_subword_target_is_eligible(self) -> None:
        from src.explanations.hindi_explanations import prepare_masked_target_input

        res = prepare_masked_target_input("राम और सीता साथ हैं।", "और", self.tokenizer, self.device)
        self.assertTrue(res["is_eligible"])
        self.assertEqual(res["target_token_id"], 2002)
        self.assertIsNotNone(res["target_pos"])
        # Check target position is replaced with [MASK] (103)
        self.assertEqual(res["masked_input_ids"][0, res["target_pos"]].item(), 103)
        # Check original input_ids had the real token
        self.assertEqual(res["input_ids"][0, res["target_pos"]].item(), 2002)

    def test_multi_subword_target_is_ineligible(self) -> None:
        from src.explanations.hindi_explanations import prepare_masked_target_input

        # "पानि" splits into 2 tokens ("पा", "नि") in MockHindiTokenizer
        res = prepare_masked_target_input("मुझे पानि चाहिए।", "पानि", self.tokenizer, self.device)
        self.assertFalse(res["is_eligible"])
        self.assertIn("compound_subword_target", res["reason"])
        self.assertIsNone(res["masked_input_ids"])

    def test_missing_target_is_ineligible(self) -> None:
        from src.explanations.hindi_explanations import prepare_masked_target_input

        res = prepare_masked_target_input("राम और सीता साथ हैं।", "विद्यालय", self.tokenizer, self.device)
        self.assertFalse(res["is_eligible"])
        self.assertIn("missing", res["reason"])

    def test_special_tokens_excluded_from_content(self) -> None:
        from src.explanations.hindi_explanations import prepare_masked_target_input

        res = prepare_masked_target_input("राम और सीता साथ हैं।", "और", self.tokenizer, self.device)
        self.assertNotIn("[CLS]", res["content_tokens"])
        self.assertNotIn("[SEP]", res["content_tokens"])
        self.assertNotIn("[PAD]", res["content_tokens"])
        self.assertIn("राम", res["content_tokens"])


class TestHindiExplanationMethods(unittest.TestCase):

    def setUp(self) -> None:
        self.tokenizer = MockHindiTokenizer()
        self.model = MockHindiMLM()
        self.model.eval()
        self.device = torch.device("cpu")

    def test_attention_extraction_matches_content_tokens(self) -> None:
        from src.explanations.hindi_explanations import (
            extract_mlm_attention,
            prepare_masked_target_input,
        )

        prep = prepare_masked_target_input("राम और सीता साथ हैं।", "और", self.tokenizer, self.device)
        attn = extract_mlm_attention(
            self.model,
            prep["masked_input_ids"],
            prep["attention_mask"],
            prep["target_pos"],
            prep["target_token_id"],
            prep["content_indices"],
        )
        self.assertIsNone(attn["warning"])
        self.assertEqual(len(attn["attention_scores"]), len(prep["content_tokens"]))
        self.assertAlmostEqual(sum(attn["attention_scores"]), 1.0, places=5)
        self.assertTrue(math.isfinite(attn["target_logit"]))

    def test_integrated_gradients_differentiable_mock(self) -> None:
        from src.explanations.hindi_explanations import (
            extract_mlm_integrated_gradients,
            prepare_masked_target_input,
        )

        prep = prepare_masked_target_input("राम और सीता साथ हैं।", "और", self.tokenizer, self.device)
        ig = extract_mlm_integrated_gradients(
            self.model,
            self.tokenizer,
            prep["masked_input_ids"],
            prep["attention_mask"],
            prep["target_pos"],
            prep["target_token_id"],
            prep["content_indices"],
            n_steps=10,
        )
        self.assertIsNone(ig["warning"])
        self.assertEqual(len(ig["ig_signed"]), len(prep["content_tokens"]))
        self.assertEqual(len(ig["ig_abs"]), len(prep["content_tokens"]))
        self.assertIsNotNone(ig["convergence_delta"])
        self.assertTrue(math.isfinite(ig["convergence_delta"]))
        for s in ig["ig_signed"]:
            self.assertTrue(math.isfinite(s))
        for a in ig["ig_abs"]:
            self.assertTrue(math.isfinite(a))

    def test_integrated_gradients_regression_disconnected_raises(self) -> None:
        """Verify that disconnecting logits from embeddings triggers RuntimeError."""
        from src.explanations.hindi_explanations import (
            extract_mlm_integrated_gradients,
            prepare_masked_target_input,
        )

        class DisconnectedMLM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.bert = _MockBertBody()

            def forward(
                self,
                input_ids: torch.Tensor | None = None,
                attention_mask: torch.Tensor | None = None,
                inputs_embeds: torch.Tensor | None = None,
                **kwargs: Any,
            ) -> MockHindiMLMOutput:
                if inputs_embeds is None:
                    inputs_embeds = self.bert.embeddings(input_ids)
                b, s, _ = inputs_embeds.shape
                # Disconnect graph via float item
                pos_score = float(inputs_embeds.detach().sum().item()) * 0.001
                logits = torch.tensor(
                    [[[0.1 + pos_score] * 10000 for _ in range(s)] for _ in range(b)],
                    dtype=torch.float32,
                    requires_grad=inputs_embeds.requires_grad,
                )
                return MockHindiMLMOutput(logits=logits)

        disc_model = DisconnectedMLM()
        disc_model.eval()
        prep = prepare_masked_target_input("राम और सीता साथ हैं।", "और", self.tokenizer, self.device)

        with self.assertRaises(RuntimeError) as ctx:
            extract_mlm_integrated_gradients(
                disc_model,
                self.tokenizer,
                prep["masked_input_ids"],
                prep["attention_mask"],
                prep["target_pos"],
                prep["target_token_id"],
                prep["content_indices"],
                n_steps=5,
            )
        self.assertIn("appears to not have been used in the graph", str(ctx.exception))


class TestHindiPilotPipelineIntegration(unittest.TestCase):

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.temp_dir.name) / "hindi_explanations"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _mock_dataset(self) -> list[dict[str, Any]]:
        return [
            {
                "pair_id": "HC_0001",
                "_excel_row": 2,
                "track": "VALID_WORD_CONTEXT",
                "contrast_label": "contextual_substitution_error",
                "original_sentence": "राम और सीता साथ हैं।",
                "contrast_sentence": "राम ओर सीता साथ हैं।",
                "target_original": "और",
                "target_contrast": "ओर",
                "intended_meaning_hi": "साथ हैं",
                "intended_meaning_en": "together",
                "grammatical_original": "yes",
                "grammatical_contrast": "no",
                "notes": "परीक्षण",
                "final_status": "accept",
            },
            {
                "pair_id": "HC_0004",
                "_excel_row": 5,
                "track": "SYNONYM_SUBSTITUTION",
                "contrast_label": "meaning_preserving_variant",
                "original_sentence": "राम स्कूल गया।",
                "contrast_sentence": "राम विद्यालय गया।",
                "target_original": "स्कूल",
                "target_contrast": "विद्यालय",
                "intended_meaning_hi": "शिक्षण",
                "intended_meaning_en": "school",
                "grammatical_original": "yes",
                "grammatical_contrast": "yes",
                "notes": "समानार्थी",
                "final_status": "accept",
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
                "intended_meaning_hi": "जल",
                "intended_meaning_en": "water",
                "grammatical_original": "yes",
                "grammatical_contrast": "no",
                "notes": "वर्तनी",
                "final_status": "accept",
            },
        ]

    def test_pipeline_creates_all_expected_files(self) -> None:
        from src.explanations.hindi_explanations import run_hindi_explanation_pipeline

        meta, records, summary = run_hindi_explanation_pipeline(
            output_dir=self.out_dir,
            pilot_size=3,
            n_steps=5,
            model_tokenizer_loader=lambda m: (MockHindiMLM(), MockHindiTokenizer()),
            dataset_loader=self._mock_dataset,
        )

        self.assertTrue((self.out_dir / "pilot_explanations.csv").is_file())
        self.assertTrue((self.out_dir / "run_metadata.json").is_file())
        self.assertTrue((self.out_dir / "summary.md").is_file())
        self.assertTrue(meta["completed_successfully"])
        self.assertEqual(len(records), 3)

    def test_pipeline_preserves_sentence_pair_fields(self) -> None:
        from src.explanations.hindi_explanations import run_hindi_explanation_pipeline

        _, records, _ = run_hindi_explanation_pipeline(
            output_dir=self.out_dir,
            pilot_size=3,
            n_steps=5,
            model_tokenizer_loader=lambda m: (MockHindiMLM(), MockHindiTokenizer()),
            dataset_loader=self._mock_dataset,
        )

        r0 = records[0]
        self.assertEqual(r0["pair_id"], "HC_0001")
        self.assertEqual(r0["track"], "VALID_WORD_CONTEXT")
        self.assertEqual(r0["eligibility_status"], "eligible")
        self.assertIn("orig_attention_scores", r0)
        self.assertIn("orig_ig_attributions_signed", r0)
        self.assertIn("orig_ig_attributions_abs", r0)

    def test_ineligible_compound_target_recorded_with_reason(self) -> None:
        from src.explanations.hindi_explanations import run_hindi_explanation_pipeline

        _, records, _ = run_hindi_explanation_pipeline(
            output_dir=self.out_dir,
            pilot_size=3,
            n_steps=5,
            model_tokenizer_loader=lambda m: (MockHindiMLM(), MockHindiTokenizer()),
            dataset_loader=self._mock_dataset,
        )

        # HC_0005 has contrast "पानि" which splits into 2 subwords
        r2 = [r for r in records if r["pair_id"] == "HC_0005"][0]
        self.assertEqual(r2["eligibility_status"], "ineligible")
        self.assertIn("compound_subword_target", r2["ineligibility_reason"])
        self.assertEqual(r2["orig_attention_scores"], "")


if __name__ == "__main__":
    unittest.main()
