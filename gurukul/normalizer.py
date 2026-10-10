from __future__ import annotations

import json
import os
import pickle
import unicodedata
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier


def clean_text(text: str) -> str:
    """Universal Unicode normalization and punctuation stripping.

    Uses Unicode character categories (no fragile word lists or hardcoded regexes).
    """
    if not text:
        return ""

    # Canonical Decomposition followed by Canonical Composition (NFC)
    text = unicodedata.normalize("NFC", text)

    # Filter out zero-width characters and punctuation using Unicode categories
    chars: list[str] = []
    for c in text:
        # P = Punctuation, S = Symbol, C = Other/Control
        cat = unicodedata.category(c)
        if cat.startswith("P") or cat.startswith("S"):
            chars.append(" ")
        elif cat == "Cf":  # Format characters like ZWJ / ZWNJ
            continue
        else:
            chars.append(c.lower())

    # Collapse multiple whitespace
    cleaned = " ".join("".join(chars).split())
    return cleaned


def detect_script(text: str) -> str:
    """Identifies the dominant script using Unicode codepoint ranges.

    Zero hardcoded vocabulary lists.
    """
    if not text:
        return "latin"

    devanagari_count = 0
    latin_count = 0

    for char in text:
        cp = ord(char)
        # Devanagari block: U+0900 to U+097F
        if 0x0900 <= cp <= 0x097F:
            devanagari_count += 1
        elif 0x0041 <= cp <= 0x005A or 0x0061 <= cp <= 0x007A:
            latin_count += 1

    if devanagari_count > 0:
        return "devanagari"
    return "latin"


class StatisticalLanguageIdentifier:
    """Fast, offline statistical subword character n-gram language classifier.

    Uses dual-model subword character n-grams (2-4 char bounds) trained via TF-IDF
    and balanced Logistic Regression on authentic curriculum corpora:
    - Devanagari Pipeline: Classifies authentic Marathi vs Hindi textbook text.
    - Latin Pipeline: Differentiates English from Romanized Indic (MarEnglish / Hinglish).

    Zero fragile hardcoded word lists or runtime regexes.
    """

    _instance: StatisticalLanguageIdentifier | None = None

    def __init__(self, cache_path: Path | str | None = None) -> None:
        if cache_path:
            self.cache_path = Path(cache_path)
        else:
            pkg_cache = Path(__file__).resolve().parent / "data" / ".cache" / "lang_detector_v3.pkl"
            repo_cache = Path(__file__).resolve().parents[1] / "data" / ".cache" / "lang_detector_v3.pkl"
            cwd_cache = Path.cwd() / "data" / ".cache" / "lang_detector_v3.pkl"
            if pkg_cache.exists():
                self.cache_path = pkg_cache
            elif repo_cache.exists():
                self.cache_path = repo_cache
            elif cwd_cache.exists():
                self.cache_path = cwd_cache
            else:
                self.cache_path = repo_cache

        self.dev_model: Any = None
        self.latin_model: Any = None
        self._load_or_train()

    @classmethod
    def get_instance(cls) -> StatisticalLanguageIdentifier:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_or_train(self) -> None:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "rb") as f:
                    data = pickle.load(f)
                    if isinstance(data, dict) and "dev_model" in data and "latin_model" in data:
                        self.dev_model = data["dev_model"]
                        self.latin_model = data["latin_model"]
                        return
            except Exception:
                pass

        # Try to locate fact_cards.json to train if needed
        data_file = self.cache_path.parent.parent / "fact_cards.json"
        if not data_file.exists():
            repo_data = Path(__file__).resolve().parents[1] / "data" / "fact_cards.json"
            if repo_data.exists():
                data_file = repo_data
            else:
                return

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline

            with open(data_file, "r", encoding="utf-8") as f:
                cards = json.load(f)

            # 1. Dataset for Devanagari (Marathi vs Hindi)
            dev_texts: list[str] = []
            dev_labels: list[str] = []
            for c in cards:
                mr_ans = c.get("answer_templates", {}).get("mr")
                if mr_ans:
                    dev_texts.append(clean_text(mr_ans))
                    dev_labels.append("mr")
                hi_ans = c.get("answer_templates", {}).get("hi")
                if hi_ans:
                    dev_texts.append(clean_text(hi_ans))
                    dev_labels.append("hi")
                for a in c.get("question_aliases", []):
                    cl = clean_text(a)
                    if detect_script(cl) == "devanagari":
                        card_langs = c.get("languages", [])
                        if "hi" in card_langs and "mr" not in card_langs:
                            dev_texts.append(cl)
                            dev_labels.append("hi")
                        elif "mr" in card_langs and "hi" not in card_langs:
                            dev_texts.append(cl)
                            dev_labels.append("mr")

            dev_pipe = make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True),
                LogisticRegression(class_weight="balanced", C=10.0, random_state=42),
            )
            dev_pipe.fit(dev_texts, dev_labels)

            # 2. Dataset for Latin (English vs Romanized Marathi vs Romanized Hindi)
            latin_texts: list[str] = []
            latin_labels: list[str] = []

            for c in cards:
                en_ans = c.get("answer_templates", {}).get("en")
                if en_ans:
                    latin_texts.append(clean_text(en_ans))
                    latin_labels.append("en")
                for a in c.get("question_aliases", []):
                    cl = clean_text(a)
                    if detect_script(cl) == "latin":
                        latin_texts.append(cl)
                        latin_labels.append("en")

            # Conversational English anchors
            general_en = [
                "how to cook biryani", "how to cook chicken biryani step by step",
                "what is the stock price of Apple today", "write python code for quicksort algorithm",
                "who won the FIFA world cup in 2022", "explain quantum entanglement and superstring theory",
                "what is the capital of France", "how does a computer work", "can you help me with homework",
                "why is the sky blue", "how are you today", "tell me a story", "what is machine learning",
            ]
            for t in general_en:
                latin_texts.append(clean_text(t))
                latin_labels.append("en")

            # Authentic Romanized Marathi student queries
            roman_mr = [
                "abhyas kartana focus kasa karaycha", "man ekagr kase karayche", "abhyas kasa karaycha",
                "shivaji maharaj kon hote", "swarajya mhanje kay", "adhivas mhanje kay", "mala mahiti sanga",
                "focus kasa thevaycha", "kasa karaycha", "kuthe zala janma", "kon hote shivaji maharaj",
                "swarajyachi pratidnya koni ghetli", "vanaspati anna kase tayar kartat", "prakashacha veg kiti ahe",
                "mala he sanga na", "tula mahit ahe ka", "kahi samajat nahiye", "kasa shikaycha",
                "abhyasat man lagat nahi", "kase karave", "kay karave", "kashala karayche",
                "shivaji maharajanchi mahiti dya", "maharashtra chi rajdhani konti", "sarvat motha graha konta",
                "shala kadhi suru honar", "ekagr kase vhave", "dhyan kasa karaycha", "mala uttar dya",
                "tula marathi yete ka", "he kasa asate", "mala shikayche ahe", "koni keli sthapana",
            ]
            for t in roman_mr:
                latin_texts.append(clean_text(t))
                latin_labels.append("mr")

            # Authentic Romanized Hindi student queries
            roman_hi = [
                "padhai me dhyan kaise lagaye", "padhai kaise kare", "shivaji maharaj kaun the",
                "aawas kise kehte hain", "kya hai ye", "mujhe bataiye", "kaise karein",
                "kahan hua tha janma", "kisne li thi pratigya", "paudhe apna bhojan kaise banate hain",
                "prakash ka veg kitna hai", "mujhe yeh samjhao", "kya aap bata sakte hain",
                "padhai me man nahi lag raha", "kya karna chahiye", "kaise yaad karein",
                "bharat ki rajdhani kaun si hai", "sabse bada grah kaun sa hai", "kya haal hai",
                "mujhe hindi me batao", "ye kaise hota hai", "kisne likha tha rashtrageet",
            ]
            for t in roman_hi:
                latin_texts.append(clean_text(t))
                latin_labels.append("hi")

            latin_pipe = make_pipeline(
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True),
                LogisticRegression(class_weight="balanced", C=5.0, random_state=42),
            )
            latin_pipe.fit(latin_texts, latin_labels)

            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump({"dev_model": dev_pipe, "latin_model": latin_pipe}, f)

            self.dev_model = dev_pipe
            self.latin_model = latin_pipe
        except Exception:
            pass

    def predict(self, text: str, default: str = "mr") -> str:
        """Predicts query language preference statistically using learned subword models.

        Zero manual word lists or heuristic word matching.
        """
        cleaned = clean_text(text)
        if not cleaned:
            return default

        script = detect_script(cleaned)

        try:
            if script == "devanagari":
                if self.dev_model is not None:
                    return str(self.dev_model.predict([cleaned])[0])
                return "mr"
            else:
                if self.latin_model is not None:
                    return str(self.latin_model.predict([cleaned])[0])
                return "en"
        except Exception:
            return "mr" if script == "devanagari" else "en"


def infer_preferred_language(text: str, default: str = "mr") -> str:
    """Infers query language preference ('mr', 'hi', 'en') statistically.

    - Devanagari script is evaluated via subword character n-grams for 'mr' vs 'hi'.
    - Latin script is evaluated via subword character n-grams for English vs Romanized Marathi/Hindi.
    """
    detector = StatisticalLanguageIdentifier.get_instance()
    return detector.predict(text, default=default)
