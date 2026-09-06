# SPDX-License-Identifier: Apache-2.0
"""CPU-only regression checks; also runnable directly with Python/unittest."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[3] / "benchmarks/scripts/check_ep_lora_accuracy.py"
SPEC = importlib.util.spec_from_file_location("ep_lora_accuracy", SCRIPT)
accuracy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(accuracy)


def response(token, logprob=-0.5):
    return {"logprobs": [{"token": token, "logprob": logprob, "top_logprobs": [{"token": token, "logprob": logprob}]}]}


class TestAccuracyComparison(unittest.TestCase):
    def test_invalid_answer_is_not_false(self):
        rows = [{"answer": False, "models": {"ds": response("invalid")}}]
        summary = accuracy.summarize(rows)["ds"]
        self.assertEqual(summary["valid"], 0)
        self.assertEqual(summary["correct"], 0)
        self.assertEqual(summary["accuracy"], 0.0)

    def test_first_token_labels(self):
        for token in ("T", "TRUE", " true "):
            self.assertIs(accuracy.normalized_answer(token), True)
        for token in ("F", "FALSE", " false "):
            self.assertIs(accuracy.normalized_answer(token), False)
        self.assertIsNone(accuracy.normalized_answer("Maybe"))

    def test_same_token_probability_delta(self):
        result = accuracy.pair_metrics([(response("T", -0.5), response("T", -0.75))])
        self.assertEqual(result["same_first_token"], 1)
        self.assertEqual(result["same_token_logprob_max_abs"], 0.25)

    def test_different_tokens_not_compared_as_same_probability(self):
        result = accuracy.pair_metrics([(response("T"), response("F"))])
        self.assertEqual(result["same_first_token"], 0)
        self.assertIsNone(result["same_token_logprob_max_abs"])

    def test_decode_detects_divergence_after_first_token(self):
        left = response("A")
        right = response("A")
        left["logprobs"].append(response("B")["logprobs"][0])
        right["logprobs"].append(response("C")["logprobs"][0])
        result = accuracy.pair_metrics([(left, right)])
        self.assertEqual(result["same_first_token"], 1)
        self.assertEqual(result["same_token_sequence"], 0)
        self.assertEqual(result["common_prefix_tokens_mean"], 1)

    def test_decode_is_not_scored_as_binary_accuracy(self):
        rows = [{"answer": None, "models": {"ds": response("Hello")}}]
        self.assertEqual(accuracy.summarize(rows, "none")["ds"], {"total": 1, "output_tokens": 1})

    def test_decode_probability_drift_after_equal_first_token(self):
        left, right = response("A"), response("A")
        left["logprobs"].append(response("B", -0.5)["logprobs"][0])
        right["logprobs"].append(response("B", -0.75)["logprobs"][0])
        result = accuracy.pair_metrics([(left, right)])
        self.assertEqual(result["same_token_logprob_max_abs"], 0)
        self.assertEqual(result["common_prefix_logprob_max_abs"], 0.25)

    def test_compare_rejects_different_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("left.json", "right.json")]
            for i, path in enumerate(paths):
                path.write_text(json.dumps({"config": {"question_sha256": str(i)}}))
            with self.assertRaisesRegex(ValueError, "question_sha256"):
                accuracy.compare_reports(paths, Path(directory) / "out.json")

    def test_compare_rejects_misaligned_questions(self):
        config = dict.fromkeys(
            ("question_sha256", "system_prompt", "max_tokens", "workers", "model_specs", "score_mode"), "same"
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("left.json", "right.json")]
            for i, path in enumerate(paths):
                row = {"question": str(i), "answer": True, "models": {"ds": response("T")}}
                path.write_text(json.dumps({"config": config, "rows": [row]}))
            with self.assertRaisesRegex(ValueError, "alignment"):
                accuracy.compare_reports(paths, Path(directory) / "out.json")


if __name__ == "__main__":
    unittest.main()
