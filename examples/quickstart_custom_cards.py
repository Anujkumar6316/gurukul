"""Gurukul Quickstart: Build a 100% Offline Multilingual Q&A Bot with Custom Knowledge Cards.

Demonstrates:
1. Ingesting custom fact cards (Python dicts or JSON file).
2. Semantic search with Google EmbeddingGemma 2 / FastEmbed ONNX.
3. Trilingual responses (English, Hindi, Marathi).
4. Zero-hallucination abstention on out-of-domain queries.
5. Multi-turn conversational follow-up turns.
"""

from gurukul import OfflineFactBot

# 1. Define custom knowledge base as Python dicts
CUSTOM_FACTS = [
    {
        "id": "wifi_setup",
        "topic": "Office Wi-Fi Setup",
        "question_aliases": [
            "How do I connect to office Wi-Fi?",
            "What is the office Wi-Fi password?",
            "Connect to internet",
            "वायफाय कसे कनेक्ट करायचे?",
            "वाईफाई कैसे कनेक्ट करें?",
        ],
        "answers": {
            "en": "Connect to SSID 'Office-Guest'. Open your browser and enter your employee ID to authenticate.",
            "hi": "एसएसआईडी 'Office-Guest' से कनेक्ट करें। अपना ब्राउज़र खोलें और प्रमाणीकरण के लिए अपनी कर्मचारी आईडी दर्ज करें।",
            "mr": "'Office-Guest' या वायफायशी कनेक्ट करा. ब्राउझर उघडून तुमचा कर्मचारी क्रमांक प्रविष्ट करा.",
        },
        "follow_up_cards": ["vpn_access"],
    },
    {
        "id": "vpn_access",
        "topic": "Remote VPN Access",
        "question_aliases": [
            "How to connect to VPN from home?",
            "Remote work VPN setup",
            "घरून व्हीपीएन कसे वापरायचे?",
            "घर से वीपीएन कैसे जोड़ें?",
        ],
        "answers": {
            "en": "Download GlobalProtect VPN client, enter gateway 'vpn.company.com', and sign in with MFA token.",
            "hi": "ग्लोबलप्रोटेक्ट वीपीएन क्लाइंट डाउनलोड करें, गेटवे 'vpn.company.com' दर्ज करें और एमएफए टोकन से साइन इन करें।",
            "mr": "ग्लोबलप्रोटेक्ट व्हीपीएन ॲप डाउनलोड करा, 'vpn.company.com' गेटवे टाका आणि एमएफए टोकनने साइन इन करा.",
        },
        "follow_up_cards": ["wifi_setup"],
    },
]

def main() -> None:
    print("=" * 70)
    print("🚀 Initializing OfflineFactBot with custom in-memory cards...")
    print("=" * 70)

    # 2. Instantiate OfflineFactBot directly on custom facts
    bot = OfflineFactBot(cards=CUSTOM_FACTS)
    print(f"Loaded {len(bot.index.cards_by_id)} cards into offline semantic index.\n")

    # 3. Query in English
    q_en = "How do I get Wi-Fi access in the office?"
    resp_en = bot.resolve_query(q_en)
    print(f"Q (English): {q_en}")
    print(f"Match: [{resp_en.card_id}] (Confidence: {resp_en.confidence:.2f})")
    print(f"A: {resp_en.answer}\n")

    # 4. Query in Marathi
    q_mr = "ऑफिसचे वायफाय कसे कनेक्ट करू?"
    resp_mr = bot.resolve_query(q_mr)
    print(f"Q (Marathi): {q_mr}")
    print(f"Match: [{resp_mr.card_id}] (Confidence: {resp_mr.confidence:.2f})")
    print(f"A: {resp_mr.answer}\n")

    # 5. Query in Hindi
    q_hi = "घर से वीपीएन कैसे कनेक्ट करना है?"
    resp_hi = bot.resolve_query(q_hi)
    print(f"Q (Hindi): {q_hi}")
    print(f"Match: [{resp_hi.card_id}] (Confidence: {resp_hi.confidence:.2f})")
    print(f"A: {resp_hi.answer}\n")

    # 6. Out-of-Domain Query -> Polite Zero-Hallucination Abstention
    q_ood = "What is the price of Bitcoin today?"
    resp_ood = bot.resolve_query(q_ood)
    print(f"Q (Out of Domain): {q_ood}")
    print(f"Abstained: {resp_ood.is_abstention} (Confidence: {resp_ood.confidence:.2f})")
    print(f"A: {resp_ood.answer}\n")

    # 7. Multi-Turn Dialogue
    session_id = "user-123"
    print("--- Multi-Turn Conversation Simulation ---")
    turn_1 = bot.resolve_query("How to connect to wifi?", session_id=session_id)
    print(f"Turn 1 Q: How to connect to wifi?")
    print(f"Turn 1 A: {turn_1.answer}")
    if turn_1.follow_up_offer:
        print(f"Turn 1 Offer: {turn_1.follow_up_offer}")

    turn_2 = bot.resolve_query("yes please tell me", session_id=session_id)
    print(f"\nTurn 2 Q: yes please tell me (Affirmation intent)")
    print(f"Turn 2 A: {turn_2.answer}")
    print("=" * 70)


if __name__ == "__main__":
    main()
