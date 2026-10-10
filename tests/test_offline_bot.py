from __future__ import annotations

import unittest
from pathlib import Path

from app.offline.engine import OfflineFactBot
from app.offline.index import FactCardIndex
from app.offline.normalizer import (
    clean_text,
    detect_script,
    infer_preferred_language,
)
from fastapi.testclient import TestClient
from main import build_studio, create_api_app


class TestNormalizer(unittest.TestCase):
    def test_unicode_cleaning(self):
        # Devanagari text with symbols and punctuation
        raw = "अधिवास म्हणजे काय??!! @#$"
        cleaned = clean_text(raw)
        self.assertEqual(cleaned, "अधिवास म्हणजे काय")

    def test_script_detection(self):
        self.assertEqual(detect_script("शिवाजी महाराज"), "devanagari")
        self.assertEqual(detect_script("who was shivaji maharaj"), "latin")
        self.assertEqual(detect_script("shivaji maharaj kon hote"), "latin")

    def test_language_inference(self):
        # Statistical language inference for mr, hi, en
        self.assertEqual(infer_preferred_language("अधिवास म्हणजे काय"), "mr")
        self.assertEqual(infer_preferred_language("छत्रपती शिवाजी महाराज कोण होते"), "mr")
        self.assertEqual(infer_preferred_language("what is a habitat"), "en")
        self.assertEqual(infer_preferred_language("speed of light in vacuum"), "en")
        self.assertEqual(infer_preferred_language("भारत का राष्ट्रगान किसने लिखा"), "hi")
        self.assertEqual(infer_preferred_language("प्रकाश का वेग कितना है"), "hi")


class TestFactCardIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = FactCardIndex()

    def test_index_size(self):
        self.assertGreaterEqual(len(self.index.cards_by_id), 952)
        self.assertIsNotNone(self.index.vectors)
        self.assertGreater(len(self.index.indexed_card_ids), 3000)

    def test_semantic_retrieval(self):
        # Exact alias
        results = self.index.search("अधिवास म्हणजे काय")
        self.assertTrue(len(results) > 0)
        top_card, score = results[0]
        self.assertEqual(top_card.id, "concept_habitat")
        self.assertGreater(score, 0.75)

    def test_follow_up_graph(self):
        chips = self.index.get_follow_up_chips("concept_habitat", lang="mr")
        self.assertTrue(len(chips) > 0)
        for chip in chips:
            self.assertIsNotNone(chip.card_id)
            self.assertIsNotNone(chip.title)
            self.assertEqual(chip.language, "mr")
            self.assertIn(chip.card_id, self.index.cards_by_id)

    def test_curriculum_tree(self):
        tree = self.index.get_curriculum_tree()
        self.assertTrue(len(tree) > 0)
        self.assertIn("Grade 6", tree)


class TestOfflineFactBot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = OfflineFactBot()

    def test_marathi_resolution(self):
        resp = self.bot.resolve_query("छत्रपती शिवाजी महाराज कोण होते", lang="mr")
        self.assertFalse(resp.is_abstention)
        self.assertEqual(resp.card_id, "person_shivaji_maharaj")
        self.assertIn("स्वराज्याचे संस्थापक", resp.answer)
        self.assertIsNotNone(resp.citation)
        self.assertTrue(len(resp.follow_up_chips) > 0)
        for chip in resp.follow_up_chips:
            self.assertEqual(chip.language, "mr")

    def test_english_resolution(self):
        resp = self.bot.resolve_query("who wrote national anthem", lang="en")
        self.assertFalse(resp.is_abstention)
        self.assertEqual(resp.card_id, "national_anthem_author")
        self.assertIn("Rabindranath Tagore", resp.answer)
        for chip in resp.follow_up_chips:
            self.assertEqual(chip.language, "en")

    def test_hindi_resolution(self):
        resp = self.bot.resolve_query("भारत का राष्ट्रगान किसने लिखा", lang="hi")
        self.assertFalse(resp.is_abstention)
        self.assertEqual(resp.card_id, "national_anthem_author")
        self.assertIn("रवींद्रनाथ टैगोर", resp.answer)
        for chip in resp.follow_up_chips:
            self.assertEqual(chip.language, "hi")

    def test_language_consistency_pipeline(self):
        """Validates that query -> response -> follow up offer questions all maintain identical language."""
        # 1. English Pipeline
        en_resp = self.bot.resolve_query("what is photosynthesis")
        self.assertEqual(en_resp.language, "en")
        self.assertTrue(any(c.isalpha() for c in en_resp.answer))
        self.assertTrue(len(en_resp.follow_up_chips) > 0)
        for chip in en_resp.follow_up_chips:
            self.assertEqual(chip.language, "en")
            self.assertEqual(detect_script(chip.title), "latin")

        # 2. Marathi Pipeline
        mr_resp = self.bot.resolve_query("अधिवास म्हणजे काय")
        self.assertEqual(mr_resp.language, "mr")
        self.assertEqual(detect_script(mr_resp.answer), "devanagari")
        self.assertTrue(len(mr_resp.follow_up_chips) > 0)
        for chip in mr_resp.follow_up_chips:
            self.assertEqual(chip.language, "mr")
            self.assertEqual(detect_script(chip.title), "devanagari")

        # 3. Hindi Pipeline
        hi_resp = self.bot.resolve_query("भारत का राष्ट्रगान किसने लिखा")
        self.assertEqual(hi_resp.language, "hi")
        self.assertEqual(detect_script(hi_resp.answer), "devanagari")
        self.assertTrue(len(hi_resp.follow_up_chips) > 0)
        for chip in hi_resp.follow_up_chips:
            self.assertEqual(chip.language, "hi")
            self.assertEqual(detect_script(chip.title), "devanagari")

    def test_paraphrase_semantic_generalization(self):
        # Query with zero exact alias match - relies on deep semantic vector similarity
        resp = self.bot.resolve_query("प्राणी कुठे राहतात आणि त्यांचे घर काय असते", lang="mr")
        self.assertFalse(resp.is_abstention)
        self.assertEqual(resp.card_id, "concept_habitat")
        self.assertGreater(resp.confidence, 0.70)

    def test_english_paraphrase_generalization(self):
        # English paraphrase
        resp = self.bot.resolve_query("speed at which light travels across the universe", lang="en")
        self.assertFalse(resp.is_abstention)
        self.assertEqual(resp.card_id, "science_speed_of_light")
        self.assertGreater(resp.confidence, 0.70)

    def test_chip_navigation(self):
        resp = self.bot.resolve_query("शिवाजी महाराज", lang="mr")
        self.assertFalse(resp.is_abstention)
        self.assertTrue(len(resp.follow_up_chips) > 0)

        # Tap first chip
        first_chip = resp.follow_up_chips[0]
        chip_resp = self.bot.get_card_by_id(first_chip.card_id, lang="mr")
        self.assertIsNotNone(chip_resp)
        self.assertEqual(chip_resp.card_id, first_chip.card_id)
        self.assertFalse(chip_resp.is_abstention)
        self.assertEqual(chip_resp.confidence, 1.0)
        self.assertEqual(chip_resp.language, "mr")

    def test_out_of_syllabus_abstention(self):
        initial_queue_len = len(self.bot.get_sync_queue())
        resp = self.bot.resolve_query("crypto bitcoin blockchain trading wallet")
        self.assertTrue(resp.is_abstention)
        self.assertIsNone(resp.card_id)
        self.assertEqual(resp.language, "en")
        self.assertTrue(len(resp.suggested_topics) > 0)
        self.assertIn("Chhatrapati Shivaji Maharaj", resp.suggested_topics[0])

        # Verify query was appended to offline sync queue
        queue = self.bot.get_sync_queue()
        self.assertEqual(len(queue), initial_queue_len + 1)
        self.assertEqual(queue[-1].query, "crypto bitcoin blockchain trading wallet")
        self.assertTrue(queue[-1].is_abstention)

    def test_marathi_abstention_language(self):
        resp = self.bot.resolve_query("क्रिप्टोकरन्सी बिटकॉइन ट्रेडिंग कसे करावे")
        self.assertTrue(resp.is_abstention)
        self.assertEqual(resp.language, "mr")
        self.assertIn("माफ करा", resp.answer)
        self.assertIn("छत्रपती शिवाजी महाराज", resp.suggested_topics[0])

    def test_random_follow_up_offer_generation(self):
        """Verifies that follow-up offer questions are generated naturally from a random KB ID."""
        card = self.bot.index.get_card("national_anthem_author")
        self.assertIsNotNone(card)
        self.assertTrue(len(card.follow_up_cards) > 0)

        # Run query and verify offer is present and natural
        resp = self.bot.resolve_query("who wrote national anthem", lang="en")
        self.assertFalse(resp.is_abstention)
        self.assertIsNotNone(resp.follow_up_offer)
        self.assertTrue(resp.follow_up_offer.startswith("Would you like to know"))
        self.assertTrue(len(resp.follow_up_chips) > 0)

        # Top chip should correspond to the featured offer question
        top_chip = resp.follow_up_chips[0]
        self.assertIn(top_chip.card_id, card.follow_up_cards)
        self.assertEqual(top_chip.language, "en")
        self.assertTrue(top_chip.title.endswith("?"))

        # Verify Marathi offer generation
        mr_resp = self.bot.resolve_query("राष्ट्रगीत कोणी लिहिले", lang="mr")
        self.assertFalse(mr_resp.is_abstention)
        self.assertIsNotNone(mr_resp.follow_up_offer)
        self.assertIn("जाणून घ्यायला आवडेल का", mr_resp.follow_up_offer)
        self.assertEqual(mr_resp.language, "mr")
        self.assertEqual(mr_resp.follow_up_chips[0].language, "mr")

        # Verify Hindi offer generation
        hi_resp = self.bot.resolve_query("भारत का राष्ट्रगान किसने लिखा", lang="hi")
        self.assertFalse(hi_resp.is_abstention)
        self.assertIsNotNone(hi_resp.follow_up_offer)
        self.assertIn("जानना चाहेंगे", hi_resp.follow_up_offer)
        self.assertEqual(hi_resp.language, "hi")

    def test_follow_up_affirmation_english(self):
        """Student accepts the follow-up offer in English ('yes', 'tell me')."""
        session_id = "test-session-en-affirm"
        resp1 = self.bot.resolve_query("who wrote national anthem", session_id=session_id, lang="en")
        self.assertIsNotNone(resp1.offered_card_id)
        offered_id = resp1.offered_card_id

        # Student responds 'yes' to the offer
        resp2 = self.bot.resolve_query("yes", session_id=session_id)
        self.assertFalse(resp2.is_abstention)
        self.assertEqual(resp2.intent, "affirm")
        self.assertEqual(resp2.card_id, offered_id)
        self.assertEqual(resp2.language, "en")
        self.assertIn("Certainly!", resp2.answer)
        self.assertIsNotNone(resp2.follow_up_offer)

    def test_follow_up_affirmation_marathi(self):
        """Student accepts the follow-up offer in Marathi ('हो सांगा', 'हो')."""
        session_id = "test-session-mr-affirm"
        resp1 = self.bot.resolve_query("छत्रपती शिवाजी महाराज कोण होते", session_id=session_id, lang="mr")
        self.assertIsNotNone(resp1.offered_card_id)
        offered_id = resp1.offered_card_id

        # Student responds 'हो सांगा'
        resp2 = self.bot.resolve_query("हो सांगा", session_id=session_id)
        self.assertFalse(resp2.is_abstention)
        self.assertEqual(resp2.intent, "affirm")
        self.assertEqual(resp2.card_id, offered_id)
        self.assertEqual(resp2.language, "mr")
        self.assertIn("नक्कीच!", resp2.answer)

    def test_follow_up_affirmation_hindi(self):
        """Student accepts the follow-up offer in Hindi ('हाँ बताओ', 'ज़रूर')."""
        session_id = "test-session-hi-affirm"
        resp1 = self.bot.resolve_query("भारत का राष्ट्रगान किसने लिखा", session_id=session_id, lang="hi")
        self.assertIsNotNone(resp1.offered_card_id)
        offered_id = resp1.offered_card_id

        # Student responds 'हाँ बताओ'
        resp2 = self.bot.resolve_query("हाँ बताओ", session_id=session_id)
        self.assertFalse(resp2.is_abstention)
        self.assertEqual(resp2.intent, "affirm")
        self.assertEqual(resp2.card_id, offered_id)
        self.assertEqual(resp2.language, "hi")
        self.assertIn("ज़रूर!", resp2.answer)

    def test_follow_up_affirmation_transliterated(self):
        """Student responds with transliterated Hinglish/MarEnglish ('batao na', 'ho sanga')."""
        session_id = "test-session-transliterated"
        resp1 = self.bot.resolve_query("who wrote national anthem", session_id=session_id, lang="en")
        offered_id = resp1.offered_card_id

        resp2 = self.bot.resolve_query("batao na", session_id=session_id)
        self.assertEqual(resp2.intent, "affirm")
        self.assertEqual(resp2.card_id, offered_id)

    def test_follow_up_negation(self):
        """Student politely declines the follow-up offer ('no', 'नको', 'नहीं')."""
        session_id = "test-session-negation"
        self.bot.resolve_query("who was shivaji maharaj", session_id=session_id, lang="en")

        resp = self.bot.resolve_query("no thanks", session_id=session_id)
        self.assertFalse(resp.is_abstention)
        self.assertEqual(resp.intent, "deny")
        self.assertIn("No problem at all!", resp.answer)
        self.assertIsNone(resp.follow_up_offer)
        self.assertTrue(len(resp.suggested_topics) > 0)

    def test_follow_up_chip_numeric_selection(self):
        """Student types '1' or '2' to pick an offered follow-up chip."""
        session_id = "test-session-chips"
        resp1 = self.bot.resolve_query("who was shivaji maharaj", session_id=session_id, lang="en")
        target_chip = resp1.follow_up_chips[0]

        resp2 = self.bot.resolve_query("1", session_id=session_id)
        self.assertEqual(resp2.intent, "select_option")
        self.assertEqual(resp2.card_id, target_chip.card_id)

    def test_follow_up_new_topic_override(self):
        """Student asks an unrelated curriculum question instead of replying to the offer."""
        session_id = "test-session-new-topic"
        self.bot.resolve_query("who was shivaji maharaj", session_id=session_id, lang="en")

        resp = self.bot.resolve_query("what is the speed of light", session_id=session_id)
        self.assertEqual(resp.intent, "new_query")
        self.assertEqual(resp.card_id, "science_speed_of_light")

    def test_session_isolation(self):
        """Validates that distinct session IDs maintain independent dialogue states."""
        session_a = "sess-isolation-A"
        session_b = "sess-isolation-B"

        resp_a1 = self.bot.resolve_query("who was shivaji maharaj", session_id=session_a, lang="en")
        resp_b1 = self.bot.resolve_query("what is the speed of light", session_id=session_b, lang="en")

        self.assertNotEqual(resp_a1.card_id, resp_b1.card_id)

        # Affirm session A -> resolves Shivaji Maharaj's follow-up
        resp_a2 = self.bot.resolve_query("yes", session_id=session_a)
        self.assertEqual(resp_a2.card_id, resp_a1.offered_card_id)

        # Affirm session B -> resolves Speed of light's follow-up
        resp_b2 = self.bot.resolve_query("yes", session_id=session_b)
        self.assertEqual(resp_b2.card_id, resp_b1.offered_card_id)

    def test_batch_debug_chained_flow(self):
        """Tests batch debugging simulating a multi-turn conversation flow down consecutive lines."""
        queries = [
            "who wrote national anthem",
            "yes",
            "what is photosynthesis",
            "no thanks",
        ]
        batch_resp = self.bot.run_batch_debug(queries, lang="en", conversational=True)

        self.assertEqual(batch_resp.total_queries, 4)
        self.assertEqual(len(batch_resp.results), 4)

        # Query 1: Initial curriculum query
        self.assertEqual(batch_resp.results[0].intent, "new_query")
        self.assertEqual(batch_resp.results[0].card_id, "national_anthem_author")
        self.assertIsNotNone(batch_resp.results[0].offered_card_id)
        offered_card = batch_resp.results[0].offered_card_id

        # Query 2: Affirmation should resolve the offered card
        self.assertEqual(batch_resp.results[1].intent, "affirm")
        self.assertEqual(batch_resp.results[1].card_id, offered_card)

        # Query 3: New curriculum query
        self.assertEqual(batch_resp.results[2].intent, "new_query")
        self.assertEqual(batch_resp.results[2].card_id, "kb_25d82af7")

        # Query 4: Negation
        self.assertEqual(batch_resp.results[3].intent, "deny")

    def test_batch_debug_independent_mode(self):
        """Tests batch debugging with independent per-query evaluations."""
        queries = [
            "who was shivaji maharaj",
            "what is the speed of light",
        ]
        batch_resp = self.bot.run_batch_debug(queries, lang="en", conversational=False)

        self.assertEqual(batch_resp.total_queries, 2)
        self.assertEqual(batch_resp.matched_count, 2)
        self.assertEqual(batch_resp.results[0].card_id, "person_shivaji_maharaj")
        self.assertEqual(batch_resp.results[1].card_id, "science_speed_of_light")

    def test_batch_debug_api_endpoint(self):
        """Tests the FastAPI POST /api/debug/batch file upload and JSON endpoints."""
        app = create_api_app(self.bot)
        client = TestClient(app)

        # Test multipart file upload endpoint
        file_content = "who was shivaji maharaj\nyes\n"
        response = client.post(
            "/api/debug/batch",
            files={"file": ("queries.txt", file_content.encode("utf-8"), "text/plain")},
            data={"conversational": "true"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_queries"], 2)
        self.assertEqual(data["results"][0]["card_id"], "person_shivaji_maharaj")
        self.assertEqual(data["results"][1]["intent"], "affirm")

        # Test JSON endpoint
        json_resp = client.post(
            "/api/debug/batch-json",
            json={"queries": ["who was shivaji maharaj", "yes"], "conversational": True},
        )
        self.assertEqual(json_resp.status_code, 200)
        json_data = json_resp.json()
        self.assertEqual(json_data["total_queries"], 2)
        self.assertEqual(json_data["results"][0]["card_id"], "person_shivaji_maharaj")

    def test_card_search_api_endpoint(self):
        """Tests the FastAPI GET /api/cards/search endpoint for looking up card IDs."""
        app = create_api_app(self.bot)
        client = TestClient(app)

        resp = client.get("/api/cards/search?q=person_shivaji")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertGreater(data["total_matches"], 0)
        matched_ids = [c["id"] for c in data["cards"]]
        self.assertIn("person_shivaji_maharaj", matched_ids)


class TestStudioUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bot = OfflineFactBot()
        cls.studio = build_studio(cls.bot)

    def test_studio_title_english(self):
        self.assertEqual(self.studio.title, "Gurukul — Studio")

    def test_studio_language_defaults_to_english(self):
        dropdowns = [
            comp for comp in self.studio.blocks.values()
            if hasattr(comp, "label") and comp.label == "Language"
        ]
        self.assertEqual(len(dropdowns), 1)
        lang_dropdown = dropdowns[0]
        self.assertEqual(lang_dropdown.value, "en")

    def test_studio_has_benchmark_tab(self):
        tabs = [
            comp for comp in self.studio.blocks.values()
            if comp.__class__.__name__ == "Tab" and hasattr(comp, "label")
        ]
        tab_labels = [t.label for t in tabs]
        self.assertIn("🏆 Benchmark Suite", tab_labels)

    def test_studio_benchmark_controls_present(self):
        buttons = [
            comp.value for comp in self.studio.blocks.values()
            if hasattr(comp, "value") and isinstance(comp.value, str)
        ]
        self.assertIn("🚀 Run Benchmark", buttons)
        self.assertIn("📂 Load Latest Report", buttons)

        judge_dropdowns = [
            comp for comp in self.studio.blocks.values()
            if hasattr(comp, "label") and comp.label == "Judge Model"
        ]
        self.assertEqual(len(judge_dropdowns), 1)
        self.assertEqual(judge_dropdowns[0].value, "default")
        choices_values = [c[1] if isinstance(c, tuple) else c for c in judge_dropdowns[0].choices]
        self.assertIn("default", choices_values)

    def test_load_existing_benchmark_report(self):
        from main import load_existing_benchmark_report

        status_text, iframe_html, md_text, scorecard, dl_path = load_existing_benchmark_report()
        self.assertIn("Benchmark Report", status_text)
        self.assertIn("<iframe", iframe_html)
        self.assertIn("srcdoc=", iframe_html)
        self.assertTrue(len(md_text) > 0)

    def test_get_available_benchmark_artifacts(self):
        from main import get_available_benchmark_artifacts

        artifacts = get_available_benchmark_artifacts()
        self.assertGreater(len(artifacts), 0)
        self.assertEqual(artifacts[0][0], "Auto-discover latest artifact")

    def test_benchmark_progress_callback(self):
        from gurukul.benchmarks.run_benchmark import run

        progress_events = []

        def track_progress(ratio: float, desc: str):
            progress_events.append((ratio, desc))

        res = run(
            input_path="sample_batch_input.json",
            dry_run=True,
            output_dir=Path("artifacts"),
            progress_callback=track_progress,
        )
        self.assertTrue(res.exists())
        self.assertGreater(len(progress_events), 0)
        for ratio, desc in progress_events:
            self.assertGreaterEqual(ratio, 0.0)
            self.assertLessEqual(ratio, 1.0)
            self.assertTrue(isinstance(desc, str))

    def test_nlu_aspect_classification(self):
        from gurukul.nlu_aspects import QueryAspect, QueryModality, classify_query

        c_res = classify_query("अभ्यासात एकाग्रता कमी होण्याची कारणे कोणती?")
        self.assertEqual(c_res.aspect, QueryAspect.CAUSE)
        self.assertEqual(c_res.modality, QueryModality.GENERAL)

        r_res = classify_query("ऑनलाइन शिकताना लक्ष कसे केंद्रित करावे?")
        self.assertEqual(r_res.aspect, QueryAspect.REMEDY)
        self.assertEqual(r_res.modality, QueryModality.ONLINE)

        rd_res = classify_query("पुस्तक वाचताना लक्ष कसे केंद्रित करावे?")
        self.assertEqual(rd_res.modality, QueryModality.READING)

        ls_res = classify_query("सलग अभ्यास करताना लक्ष कसे टिकवावे?")
        self.assertEqual(ls_res.modality, QueryModality.LONG_SESSION)

    def test_all_cards_devanagari_purity(self):
        from gurukul.normalizer import detect_script

        cards = list(self.bot.index.cards_by_id.values())
        self.assertGreaterEqual(len(cards), 957)
        for c in cards:
            q_mr = c.get_offer_question("mr")
            self.assertEqual(detect_script(q_mr), "devanagari", f"MR non-devanagari for {c.id}: {q_mr}")
            q_hi = c.get_offer_question("hi")
            self.assertEqual(detect_script(q_hi), "devanagari", f"HI non-devanagari for {c.id}: {q_hi}")

    def test_study_specialized_card_routing(self):
        # Cause query
        resp_cause = self.bot.resolve_query("अभ्यासात एकाग्रता कमी होण्याची कारणे कोणती?", lang="mr")
        self.assertEqual(resp_cause.card_id, "kb_study_causes_of_distraction")
        self.assertIn("अपुरी झोप", resp_cause.answer)

        # Online study query
        resp_online = self.bot.resolve_query("ऑनलाइन शिकताना लक्ष कसे केंद्रित करावे?", lang="mr")
        self.assertEqual(resp_online.card_id, "kb_study_online_learning_focus")
        self.assertIn("नोटिफिकेशन्स", resp_online.answer)


if __name__ == "__main__":
    unittest.main()

