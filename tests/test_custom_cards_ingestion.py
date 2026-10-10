from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from gurukul import FactCard, FactCardIndex, OfflineFactBot


class TestCustomCardsIngestion(unittest.TestCase):
    def setUp(self):
        self.sample_cards = [
            {
                "id": "solar_panel",
                "topic": "Solar Panels",
                "question_aliases": [
                    "What are solar panels?",
                    "How do solar panels work?",
                    "सौर पॅनेल्स काय आहेत?",
                    "सोलर पैनल कैसे काम करते हैं?",
                ],
                "answers": {
                    "en": "Solar panels convert sunlight into electrical energy using photovoltaic cells.",
                    "hi": "सोलर पैनल फोटोवोल्टिक कोशिकाओं का उपयोग करके सूर्य के प्रकाश को विद्युत ऊर्जा में बदलते हैं।",
                    "mr": "सौर पॅनेल्स फोटोव्होल्टेइक पेशींचा वापर करून सूर्यप्रकाशाचे विजेमध्ये रूपांतर करतात.",
                },
                "follow_up_cards": ["inverter_01"],
            },
            {
                "id": "inverter_01",
                "topic": "Solar Inverters",
                "question_aliases": [
                    "What is a solar inverter?",
                    "सोलर इन्व्हर्टर म्हणजे काय?",
                    "सोलर इन्वर्टर क्या है?",
                ],
                "answers": {
                    "en": "A solar inverter converts direct current (DC) from solar panels into alternating current (AC).",
                    "hi": "सोलर इन्वर्टर सोलर पैनल से आने वाले डायरेक्ट करंट (DC) को अल्टरनेटिंग करंट (AC) में बदलता है।",
                    "mr": "सोलर इन्व्हर्टर सौर पॅनेलमधून येणाऱ्या डीसी विजेचे एसी विजेमध्ये रूपांतर करतो.",
                },
                "follow_up_cards": ["solar_panel"],
            },
        ]

    def test_in_memory_cards_ingestion(self):
        bot = OfflineFactBot(cards=self.sample_cards)
        self.assertEqual(len(bot.index.cards_by_id), 2)
        self.assertIn("solar_panel", bot.index.cards_by_id)

        # English query
        resp_en = bot.resolve_query("How does a solar panel generate electricity?")
        self.assertFalse(resp_en.is_abstention)
        self.assertEqual(resp_en.card_id, "solar_panel")
        self.assertIn("photovoltaic", resp_en.answer.lower())

        # Marathi query
        resp_mr = bot.resolve_query("सौर पॅनेल कसे काम करतात?")
        self.assertFalse(resp_mr.is_abstention)
        self.assertEqual(resp_mr.card_id, "solar_panel")
        self.assertEqual(resp_mr.language, "mr")

        # Abstention on out of scope query
        resp_ood = bot.resolve_query("How to make chocolate cake?")
        self.assertTrue(resp_ood.is_abstention)

    def test_json_file_cards_ingestion(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(self.sample_cards, f)
            temp_path = f.name

        try:
            bot = OfflineFactBot(cards=temp_path)
            self.assertEqual(len(bot.index.cards_by_id), 2)
            resp = bot.resolve_query("What does a solar inverter do?")
            self.assertFalse(resp.is_abstention)
            self.assertEqual(resp.card_id, "inverter_01")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_flexible_schema_tolerance(self):
        # Card with single question, single string answer, and omitted id
        minimal_card_dict = {
            "topic": "Python Lang",
            "question": "What is Python?",
            "answer": "Python is a modern high-level programming language.",
        }
        card = FactCard.model_validate(minimal_card_dict)
        self.assertTrue(bool(card.id))
        self.assertEqual(card.question_aliases, ["What is Python?"])
        self.assertIn("en", card.answer_templates)
        self.assertIn("Python is a modern", card.get_answer("en"))

    def test_multiturn_dialogue_with_custom_cards(self):
        bot = OfflineFactBot(cards=self.sample_cards)
        session_id = "test-session-custom"

        # Turn 1
        t1 = bot.resolve_query("Tell me about solar panels", session_id=session_id)
        self.assertEqual(t1.card_id, "solar_panel")
        self.assertIsNotNone(t1.follow_up_offer)

        # Turn 2: Affirmation ("yes please")
        t2 = bot.resolve_query("yes please tell me", session_id=session_id)
        self.assertEqual(t2.intent, "affirm")
        self.assertEqual(t2.card_id, "inverter_01")
        self.assertIn("alternating current", t2.answer.lower())


if __name__ == "__main__":
    unittest.main()
