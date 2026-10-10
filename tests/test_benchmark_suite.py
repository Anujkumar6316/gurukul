"""Unit tests for Gurukul Standalone DeepEval Benchmark Suite."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from benchmarks.run_benchmark import (
    format_html_report,
    format_markdown_report,
    parse_artifact_data,
    resolve_input_artifact,
    run,
)



class TestBenchmarkSuite(unittest.TestCase):
    """Tests artifact resolution, parsing, test case generation, and reporting."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

        # Sample batch results data
        self.sample_batch = {
            "total_queries": 3,
            "matched_count": 2,
            "abstention_count": 1,
            "match_rate_pct": 66.7,
            "avg_latency_ms": 12.5,
            "results": [
                {
                    "line_number": 1,
                    "query": "who was shivaji maharaj",
                    "language": "en",
                    "intent": "new_query",
                    "card_id": "HIS-04-01-001",
                    "card_topic": "Shivaji Maharaj",
                    "confidence": 0.95,
                    "is_abstention": False,
                    "answer_preview": "Chhatrapati Shivaji Maharaj was the founder of the Maratha Empire.",
                    "citation": "Balbharati History Std 4, Ch 1",
                    "follow_up_offer": "Would you like to know about his coronation at Raigad?",
                    "offered_card_id": "HIS-04-01-002",
                    "follow_up_chips": ["Coronation"],
                    "latency_ms": 11.2,
                },
                {
                    "line_number": 2,
                    "query": "प्रकाशसंश्लेषण म्हणजे काय",
                    "language": "mr",
                    "intent": "new_query",
                    "card_id": "SCI-07-02-005",
                    "card_topic": "Photosynthesis",
                    "confidence": 0.92,
                    "is_abstention": False,
                    "answer_preview": "वनस्पती सूर्यप्रकाशात अन्न तयार करण्याच्या प्रक्रियेला प्रकाशसंश्लेषण म्हणतात.",
                    "citation": "Balbharati Science Std 7, Ch 2",
                    "follow_up_offer": "याबद्दल अधिक जाणून घ्यायला आवडेल का?",
                    "offered_card_id": "SCI-07-02-006",
                    "follow_up_chips": ["क्लोरोफिल"],
                    "latency_ms": 14.1,
                },
                {
                    "line_number": 3,
                    "query": "how to build a rocket to mars",
                    "language": "en",
                    "intent": "abstention",
                    "card_id": None,
                    "card_topic": None,
                    "confidence": 0.2,
                    "is_abstention": True,
                    "answer_preview": "I don't have information on this topic in the Maharashtra State Board syllabus.",
                    "citation": None,
                    "follow_up_offer": None,
                    "offered_card_id": None,
                    "follow_up_chips": [],
                    "latency_ms": 8.0,
                },
            ],
        }

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_resolve_input_artifact_explicit(self):
        sample_file = self.tmp_path / "test_queries.json"
        sample_file.write_text(json.dumps(self.sample_batch), encoding="utf-8")

        resolved = resolve_input_artifact(str(sample_file), artifact_dir=self.tmp_path)
        self.assertEqual(resolved, sample_file.resolve())

    def test_resolve_input_artifact_from_dir(self):
        sample_file = self.tmp_path / "auto_found.json"
        sample_file.write_text(json.dumps(self.sample_batch), encoding="utf-8")

        # Passing only filename
        resolved = resolve_input_artifact("auto_found.json", artifact_dir=self.tmp_path)
        self.assertEqual(resolved, sample_file.resolve())

        # Passing None (auto-detect newest)
        resolved_auto = resolve_input_artifact(None, artifact_dir=self.tmp_path)
        self.assertEqual(resolved_auto, sample_file.resolve())

    def test_parse_json_artifact(self):
        json_file = self.tmp_path / "batch_out.json"
        json_file.write_text(json.dumps(self.sample_batch), encoding="utf-8")

        records = parse_artifact_data(json_file)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["card_id"], "HIS-04-01-001")
        self.assertTrue(records[2]["is_abstention"])

    def test_parse_txt_artifact_mocked(self):
        txt_file = self.tmp_path / "queries.txt"
        txt_file.write_text("who was shivaji maharaj\nप्रकाशसंश्लेषण म्हणजे काय\n", encoding="utf-8")

        with patch("gurukul.engine.OfflineFactBot") as mock_bot_cls:
            mock_bot = MagicMock()

            def mock_run_batch(chunk, **kwargs):
                resp = MagicMock()
                results = []
                for q in chunk:
                    res = MagicMock()
                    res.model_dump.return_value = {"query": q}
                    results.append(res)
                resp.results = results
                return resp

            mock_bot.run_batch_debug.side_effect = mock_run_batch
            mock_bot_cls.return_value = mock_bot

            records = parse_artifact_data(txt_file)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["query"], "who was shivaji maharaj")


    def test_format_markdown_report(self):
        records = self.sample_batch["results"]
        md = format_markdown_report([], records, "gemini-1.5-flash", self.tmp_path / "batch.json")
        self.assertIn("# Gurukul LLM-as-a-Judge Benchmark Report", md)
        self.assertIn("Evaluated Queries**: 3", md)
        self.assertIn("HIS-04-01-001", md)

    def test_format_html_report(self):
        records = self.sample_batch["results"]
        html_out = format_html_report([], records, "gemini-1.5-flash", self.tmp_path / "batch.json")
        self.assertIn("<!DOCTYPE html>", html_out)
        self.assertIn("Gurukul LLM-as-a-Judge Benchmark Report", html_out)
        self.assertIn("who was shivaji maharaj", html_out)
        self.assertIn("HIS-04-01-001", html_out)
        self.assertIn("filterStatus", html_out)

    def test_dry_run_end_to_end(self):
        input_file = self.tmp_path / "test_input.json"
        input_file.write_text(json.dumps(self.sample_batch), encoding="utf-8")

        html_report_path = run(
            input_path=str(input_file),
            judge_model="gemini-1.5-flash",
            output_dir=self.tmp_path,
            dry_run=True,
        )

        self.assertTrue(html_report_path.exists())
        self.assertEqual(html_report_path.name, "benchmark_report.html")
        html_content = html_report_path.read_text(encoding="utf-8")
        self.assertIn("Gurukul LLM-as-a-Judge Benchmark Report", html_content)

        md_file = self.tmp_path / "benchmark_report.md"
        self.assertTrue(md_file.exists())
        md_content = md_file.read_text(encoding="utf-8")
        self.assertIn("Gurukul LLM-as-a-Judge Benchmark Report", md_content)


    def test_benchmark_settings_isolated(self):
        from benchmarks.settings import BenchmarkSettings, settings

        self.assertIsNotNone(settings.artifact_dir)
        self.assertIsNotNone(settings.project_root)
        self.assertEqual(settings.language_consistency_threshold, 0.8)
        self.assertEqual(settings.curriculum_grounding_threshold, 0.7)
        self.assertEqual(settings.default_report_name, "benchmark_report.md")

        # Custom instantiated settings with OpenAI-compatible endpoint
        custom = BenchmarkSettings(
            openai_api_key="sk-test-key-123",
            openai_base_url="http://localhost:8000/v1",
            openai_model_id="meta-llama/Llama-3.1-8B-Instruct",
            language_consistency_threshold=0.9,
        )
        self.assertEqual(custom.openai_api_key, "sk-test-key-123")
        self.assertEqual(custom.openai_base_url, "http://localhost:8000/v1")
        self.assertEqual(custom.effective_judge, "meta-llama/Llama-3.1-8B-Instruct")

        self.assertEqual(custom.language_consistency_threshold, 0.9)

        # Test environment synchronization
        custom.apply_environment()
        import os
        self.assertEqual(os.environ.get("OPENAI_BASE_URL"), "http://localhost:8000/v1")


if __name__ == "__main__":
    unittest.main()


