"""Offline unit tests for Stage 2 Hindi Tokenization Experiment.

Uses synthetic Hindi text, temporary directories, and mocked fast tokenizers.
Does not require network access, model downloads, or PyTorch.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from typing import Any

from src.data.tokenization_experiment import (
    align_target_word,
    check_cache_validity,
    compute_file_sha256,
    process_sentence_pair,
    run_tokenization_experiment,
    tokenize_sentence,
)
from src.data.validate import EXPECTED_COLUMNS
from tests.test_dataset_validation import create_synthetic_xlsx, get_sample_valid_row


class MockFastTokenizer:
    """Mock HuggingFace Fast Tokenizer for offline unit testing."""

    def __init__(
        self,
        name: str = "mock-fast-tokenizer",
        split_map: dict[str, list[str]] | None = None,
    ) -> None:
        self.name_or_path = name
        self.is_fast = True
        self.vocab_size = 30000
        self.all_special_ids = {101, 102}  # [CLS]=101, [SEP]=102
        self.split_map = split_map or {}
        # Simple id table
        self._vocab: dict[str, int] = {"[CLS]": 101, "[SEP]": 102}
        self._rev_vocab: dict[int, str] = {101: "[CLS]", 102: "[SEP]"}
        self._next_id = 1000

    def _get_id(self, token: str) -> int:
        if token not in self._vocab:
            self._vocab[token] = self._next_id
            self._rev_vocab[self._next_id] = token
            self._next_id += 1
        return self._vocab[token]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [self._rev_vocab.get(i, f"[UNK_{i}]") for i in ids]

    def __call__(
        self,
        text: str,
        return_offsets_mapping: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        text_str = str(text)

        # Word-level tokenization with character offset tracking and optional subword splitting
        tokens: list[str] = ["[CLS]"]
        input_ids: list[int] = [101]
        offsets: list[tuple[int, int]] = [(0, 0)]
        special_mask: list[int] = [1]

        # Scan words
        import re

        for match in re.finditer(r"\S+", text_str):
            word = match.group()
            w_start, w_end = match.start(), match.end()

            # Check if custom subword split is defined for this word
            if word in self.split_map:
                subpieces = self.split_map[word]
                curr_start = w_start
                for piece in subpieces:
                    curr_end = curr_start + len(piece)
                    tokens.append(piece)
                    input_ids.append(self._get_id(piece))
                    offsets.append((curr_start, curr_end))
                    special_mask.append(0)
                    curr_start = curr_end
            else:
                tokens.append(word)
                input_ids.append(self._get_id(word))
                offsets.append((w_start, w_end))
                special_mask.append(0)

        # Append [SEP]
        tokens.append("[SEP]")
        input_ids.append(102)
        offsets.append((0, 0))
        special_mask.append(1)

        result: dict[str, Any] = {
            "input_ids": input_ids,
            "special_tokens_mask": special_mask,
        }
        if return_offsets_mapping:
            result["offset_mapping"] = offsets
        return result


class TestTokenizationExperiment(unittest.TestCase):
    """Test suite for tokenization experiment functions and workflows."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        self.mock_tok = MockFastTokenizer()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_token_counts_inclusive_and_exclusive(self) -> None:
        """Test special-token-inclusive and exclusive counts."""
        sentence = "यह एक सुंदर घर है।"
        tok_res = tokenize_sentence(sentence, self.mock_tok)

        # Sentence has 5 words + 2 special tokens ([CLS], [SEP]) = 7 inclusive
        self.assertEqual(tok_res["count_inclusive"], 7)
        # Exclusive of special tokens = 5
        self.assertEqual(tok_res["count_exclusive"], 5)
        self.assertEqual(tok_res["special_tokens_mask"][0], 1)
        self.assertEqual(tok_res["special_tokens_mask"][-1], 1)
        self.assertEqual(sum(tok_res["special_tokens_mask"][1:-1]), 0)

    def test_signed_differences(self) -> None:
        """Test that signed differences are contrast minus original."""
        row = {
            "pair_id": "PAIR_01",
            "_excel_row": 2,
            "track": "VALID_WORD_CONTEXT",
            "contrast_label": "contextual_substitution_error",
            "original_sentence": "घर सुंदर है।",  # 3 words -> 5 inclusive
            "contrast_sentence": "मकान बहुत अधिक सुंदर है।",  # 5 words -> 7 inclusive
            "target_original": "घर",
            "target_contrast": "मकान",
            "final_status": "accept",
        }
        rec = process_sentence_pair(row, self.mock_tok, "mock_model")

        # Contrast (7) - Original (5) = +2
        self.assertEqual(rec["delta_token_count_inclusive"], 2)
        self.assertEqual(rec["delta_token_count_exclusive"], 2)
        # Both targets are 1 token -> delta target count = 0
        self.assertEqual(rec["delta_target_token_count"], 0)

    def test_unique_target_alignment(self) -> None:
        """Test unique alignment when target appears exactly once."""
        sentence = "यह एक सुंदर घर है।"
        target = "घर"
        tok_res = tokenize_sentence(sentence, self.mock_tok)

        align = align_target_word(sentence, target, tok_res["encoding"], self.mock_tok)

        self.assertEqual(align["status"], "unique")
        self.assertEqual(align["token_count"], 1)
        self.assertEqual(align["tokens"], ["घर"])
        self.assertEqual(align["char_span"], (12, 14))

    def test_subword_target_alignment(self) -> None:
        """Test unique alignment when target word splits into multiple subwords."""
        # Setup mock tokenizer with subword split: 'पाठशाला' -> ['पाठ', 'शाला']
        tok_subword = MockFastTokenizer(split_map={"पाठशाला": ["पाठ", "शाला"]})
        sentence = "बच्चा पाठशाला जा रहा है।"
        target = "पाठशाला"
        tok_res = tokenize_sentence(sentence, tok_subword)

        align = align_target_word(sentence, target, tok_res["encoding"], tok_subword)

        self.assertEqual(align["status"], "unique")
        self.assertEqual(align["token_count"], 2)
        self.assertEqual(align["tokens"], ["पाठ", "शाला"])

    def test_ambiguous_target_alignment_repeated(self) -> None:
        """Test that multiple occurrences in sentence yield ambiguous status."""
        sentence = "घर से घर तक जाओ।"
        target = "घर"
        tok_res = tokenize_sentence(sentence, self.mock_tok)

        align = align_target_word(sentence, target, tok_res["encoding"], self.mock_tok)

        self.assertEqual(align["status"], "ambiguous")
        self.assertIn("appears 2 times", align["reason"])
        self.assertIsNone(align["token_count"])
        self.assertEqual(align["tokens"], [])

    def test_missing_target_alignment(self) -> None:
        """Test missing status when target is not present in sentence."""
        sentence = "यह एक सुंदर घर है।"
        target = "दुकान"
        tok_res = tokenize_sentence(sentence, self.mock_tok)

        align = align_target_word(sentence, target, tok_res["encoding"], self.mock_tok)

        self.assertEqual(align["status"], "missing")
        self.assertEqual(align["reason"], "target substring not found in sentence")
        self.assertIsNone(align["token_count"])

    def test_preservation_of_all_rows_and_labels(self) -> None:
        """Test that all original metadata, rejections, and disagreements are preserved."""
        row = {
            "pair_id": "REJECT_001",
            "_excel_row": 88,
            "track": "VALID_WORD_CONTEXT",
            "contrast_label": "contextual_substitution_error",
            "original_sentence": "यह एक वाक्य है।",
            "contrast_sentence": "यह एक दूसरा वाक्य है।",
            "target_original": "वाक्य",
            "target_contrast": "दूसरा",
            "intended_meaning_hi": "अर्थ",
            "intended_meaning_en": "meaning",
            "grammatical_original": "yes",
            "grammatical_contrast": "no",
            "notes": "special review note",
            "annotator_1": "reject",
            "annotator_2": "reject",
            "adjudication": "reject",
            "final_status": "accept",  # Known disagreement
            "source_type": "team_authored",
            "source_reference": "ref_123",
        }

        rec = process_sentence_pair(row, self.mock_tok, "mock_model")

        self.assertEqual(rec["pair_id"], "REJECT_001")
        self.assertEqual(rec["annotator_1"], "reject")
        self.assertEqual(rec["annotator_2"], "reject")
        self.assertEqual(rec["adjudication"], "reject")
        self.assertEqual(rec["final_status"], "accept")
        self.assertEqual(rec["notes"], "special review note")
        self.assertEqual(rec["_excel_row"], 88)

    def test_sha256_computation(self) -> None:
        """Test binary SHA-256 computation."""
        sample_file = self.test_dir / "sample.txt"
        sample_file.write_text("Hindi NLP Research 2026", encoding="utf-8")
        digest = compute_file_sha256(sample_file)
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64)

    def test_full_experiment_run_and_caching(self) -> None:
        """Test full experiment execution, caching, and cache invalidation."""
        xlsx_path = self.test_dir / "test_dataset.xlsx"
        out_dir = self.test_dir / "output_test"

        rows = [
            EXPECTED_COLUMNS,
            get_sample_valid_row("TEST_01", "VALID_WORD_CONTEXT"),
            get_sample_valid_row("TEST_02", "SYNONYM_SUBSTITUTION"),
        ]
        create_synthetic_xlsx(xlsx_path, rows)

        def mock_loader(m_name: str) -> Any:
            return MockFastTokenizer(name=m_name)

        # 1. First run: recomputes and creates outputs
        results, meta, summary = run_tokenization_experiment(
            input_file=xlsx_path,
            output_dir=out_dir,
            models=["mock_model_1", "mock_model_2"],
            force=False,
            tokenizer_loader=mock_loader,
        )

        self.assertEqual(len(results), 4)  # 2 rows x 2 models
        self.assertTrue((out_dir / "tokenization_results.csv").is_file())
        self.assertTrue((out_dir / "run_metadata.json").is_file())
        self.assertTrue((out_dir / "summary.md").is_file())
        self.assertTrue(meta["completed_successfully"])

        # 2. Second run: cache hit
        cache_valid = check_cache_validity(
            output_dir=out_dir,
            input_sha256=meta["input_file"]["sha256"],
            models=["mock_model_1", "mock_model_2"],
            expected_rows=4,
        )
        self.assertTrue(cache_valid)

        results_cached, meta_cached, _ = run_tokenization_experiment(
            input_file=xlsx_path,
            output_dir=out_dir,
            models=["mock_model_1", "mock_model_2"],
            force=False,
            tokenizer_loader=mock_loader,
        )
        self.assertEqual(len(results_cached), 4)

        # 3. Modify input file -> cache invalidation
        rows.append(get_sample_valid_row("TEST_03", "REALISTIC_SPELLING_ERROR"))
        create_synthetic_xlsx(xlsx_path, rows)
        new_sha256 = compute_file_sha256(xlsx_path)

        cache_valid_after_change = check_cache_validity(
            output_dir=out_dir,
            input_sha256=new_sha256,
            models=["mock_model_1", "mock_model_2"],
            expected_rows=6,
        )
        self.assertFalse(cache_valid_after_change)

    def test_graceful_handling_of_tokenizer_errors(self) -> None:
        """Test that failure to load a tokenizer raises RuntimeError and does not falsely claim success."""
        xlsx_path = self.test_dir / "test_dataset.xlsx"
        out_dir = self.test_dir / "output_test_error"

        rows = [EXPECTED_COLUMNS, get_sample_valid_row("TEST_01")]
        create_synthetic_xlsx(xlsx_path, rows)

        def failing_loader(m_name: str) -> Any:
            raise ConnectionError(f"Mocked network failure downloading {m_name}")

        with self.assertRaises(RuntimeError) as ctx:
            run_tokenization_experiment(
                input_file=xlsx_path,
                output_dir=out_dir,
                models=["failing_model"],
                force=True,
                tokenizer_loader=failing_loader,
            )
        self.assertIn("Failed to load tokenizer for model failing_model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
