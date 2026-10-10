# 🎓 Gurukul (गुरुकुल)

> **Zero-LLM Multilingual Offline FactBot Library & DeepEval Benchmarking Engine**  
> Build 100% offline, zero-hallucination, multilingual Q&A chatbots and assistants powered by custom JSON fact cards or school curricula.  
> 100% Offline · Zero Cloud LLM Cost · Pure Statistical ML Language Classifier · Google EmbeddingGemma 2 · Trilingual (Marathi, Hindi, English).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Embedding: Google EmbeddingGemma 2](https://img.shields.io/badge/Embedding-EmbeddingGemma%202%20(300M)-orange.svg)](https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/)
[![Benchmark: DeepEval 100%](https://img.shields.io/badge/DeepEval%20Score-100%25%20Pass%20Rate-brightgreen.svg)](#-deepeval-llm-as-a-judge-benchmarking-suite)

---

## 🌟 What is Gurukul?

**Gurukul** is an offline-first conversational Q&A library that replaces unpredictable, hallucination-prone cloud LLMs with **dense multilingual neural retrieval (FastEmbed ONNX)**, **Okapi BM25 lexical anchoring**, **RapidFuzz verbatim reranking**, and **statistical dialogue intent classification**.

Originally built for rural Maharashtra schools grounded in the **Balbharati** state textbook curriculum, Gurukul is now packaged as a general-purpose library that allows **any developer to supply their own custom JSON fact cards** (for customer support, product FAQs, documentation, internal SOPs, or study guides) and deploy a fast, conversational, multilingual chatbot with:
- **Zero API bills**: Runs 100% locally on standard CPU hardware.
- **Zero Hallucination**: Only answers from verified fact cards; politely abstains when asked out-of-domain queries.
- **True Trilingual Support**: Understands and answers in authentic Marathi, Hindi, English, and Romanized script (Hinglish/MarEnglish).
- **Multi-turn Follow-up Exploration**: Guides users through follow-up topic chips and conversation chains.
- **Integrated DeepEval Benchmark Suite**: Built-in LLM-as-a-judge evaluation suite to test custom cards on factual accuracy, language consistency, and hallucination-free abstention.

---

## ⚡ 60-Second Quickstart

### 1. Installation

Install via pip:
```bash
pip install gurukul
```

Or install with optional API and UI web services:
```bash
pip install "gurukul[all]"
```

### 2. Ingest Custom Cards in 10 Lines of Python

Create `bot.py`:
```python
from gurukul import OfflineFactBot

# Supply custom facts as a Python dictionary list (or path to a .json file)
cards = [
    {
        "id": "wifi_setup",
        "topic": "Office Wi-Fi",
        "question_aliases": ["How do I connect to office Wi-Fi?", "वायफाय कसे कनेक्ट करायचे?", "वाईफाई कैसे कनेक्ट करें?"],
        "answers": {
            "en": "Connect to SSID 'Company-Guest' and log in with your employee ID.",
            "mr": "'Company-Guest' या वायफायशी कनेक्ट करा आणि तुमचा कर्मचारी आयडी टाका.",
            "hi": "'Company-Guest' वाईफाई से कनेक्ट करें और अपनी कर्मचारी आईडी दर्ज करें."
        },
        "follow_up_cards": ["vpn_access"]
    },
    {
        "id": "vpn_access",
        "topic": "Remote VPN",
        "question_aliases": ["How to connect to VPN from home?", "घरून व्हीपीएन कसे वापरायचे?", "घर से वीपीएन कैसे जोड़ें?"],
        "answers": {
            "en": "Download GlobalProtect VPN and connect using gateway 'vpn.company.com'.",
            "mr": "GlobalProtect ॲप डाउनलोड करा आणि 'vpn.company.com' द्वारे कनेक्ट करा.",
            "hi": "GlobalProtect ऐप डाउनलोड करें और 'vpn.company.com' के माध्यम से कनेक्ट करें."
        }
    }
]

# Initialize 100% offline bot
bot = OfflineFactBot(cards=cards)

# Resolve queries in any language
response = bot.resolve_query("How do I get on the internet at work?")
print(response.answer)
# -> "Connect to SSID 'Company-Guest' and log in with your employee ID."

# Automatically abstains if asked out-of-domain questions
ood = bot.resolve_query("What is the price of Bitcoin?")
print(ood.is_abstention, ood.answer)
# -> True, "Sorry, this question is outside the currently loaded fact cards..."
```

---

## 📖 Custom Fact Cards JSON Specification

Users can load custom fact cards from any `.json` file by passing `OfflineFactBot(cards="path/to/my_facts.json")` or `--cards path/to/my_facts.json`.

### Minimal Schema
You can provide an ultra-simple JSON list:
```json
[
  {
    "topic": "Solar Energy",
    "question": "What is solar energy?",
    "answer": "Solar energy is light and heat radiated from the Sun converted into electricity."
  }
]
```

### Full-Featured Production Schema
For rich multi-lingual applications with dialogue graphs:
```json
[
  {
    "id": "return_policy",
    "topic": "Return Policy",
    "category": "Customer Support",
    "question_aliases": [
      "What is your return policy?",
      "How do I return an item?",
      "Can I return my purchase?",
      "सामान वापस कैसे करें?",
      "वस्तू परत कशी करायची?"
    ],
    "answers": {
      "en": "You can return any unused item within 30 days of purchase for a full refund.",
      "hi": "आप खरीदारी के 30 दिनों के भीतर किसी भी अप्रयुक्त वस्तु को पूर्ण रिफंड के लिए वापस कर सकते हैं।",
      "mr": "तुम्ही खरेदी केल्यापासून ३० दिवसांच्या आत कोणतीही न वापरलेली वस्तू संपूर्ण परताव्यासाठी परत करू शकता."
    },
    "follow_up_cards": ["refund_timeline", "shipping_methods"],
    "source": {
      "book_id": "Store Policies Handbook",
      "page": 12
    }
  }
]
```

### Field Reference

| Field | Type | Required? | Description |
|---|---|---|---|
| `id` | `str` | Optional | Unique card identifier (e.g. `ret_01`). Auto-generated from topic/question if omitted. |
| `topic` | `str` | Optional | Human-readable title or subject area. |
| `question_aliases` / `question` | `list[str]` or `str` | **Required** | Variations of questions students/users might ask in any supported language. |
| `answers` / `answer_templates` / `answer` | `dict[str, str]` or `str` | **Required** | The verified factual answer. Supports language keys (`"en"`, `"hi"`, `"mr"`) or a single string. |
| `follow_up_cards` | `list[str]` | Optional | List of target `id`s for related cards to offer as follow-up chips. |
| `source` | `object` | Optional | Citation information (`{"book_id": "...", "page": 42}`). |
| `grade` / `grades` | `int` or `list[int]` | Optional | Target audience level (1-12). |

### Automatic Content-Addressed Caching
When custom cards are loaded, Gurukul calculates a **SHA-256 checksum** of your card content and saves the ONNX embeddings into `.cache/fact_embeddings_{model}_{hash}.npz`:
- **First Load**: Embeds custom cards into dense vectors using ONNX runtime (~1-2 seconds).
- **Subsequent Runs**: Loads precomputed vectors from disk in **< 50 milliseconds**.
- If you edit your JSON file, Gurukul detects the change and rebuilds the cache automatically!

---

## 💻 Unified Command-Line Interface (CLI)

The `gurukul` CLI command is installed automatically with the package.

### 1. Interactive Terminal Chatbot
Chat with your custom cards in an interactive terminal REPL:
```bash
# Chat with custom cards
gurukul chat --cards examples/customer_support_kb.json

# Chat with specific language or grade filter
gurukul chat --cards examples/customer_support_kb.json --lang mr
```

Inside the CLI, try:
- Typing queries in English, Hindi, or Marathi
- Typing numbers (`1`, `2`) to select suggested follow-up chips
- Typing `/lang mr`, `/lang hi`, or `/lang en` to switch languages
- Typing `/tree` to view the knowledge graph
- Typing `/reset` to clear dialogue history

### 2. Pre-download Model for 100% Offline / Air-Gapped Use
When you run Gurukul for the first time on an internet-connected machine, FastEmbed automatically downloads the ONNX weights. To explicitly download and warm up the weights ahead of time (e.g. in Docker builds or before shipping to an offline device):
```bash
# Download default Google EmbeddingGemma 2 weights
gurukul download-model

# Download specific model to a custom cache directory
gurukul download-model --model google/embeddinggemma-300m --cache-dir /path/to/cache
```

### 3. Precompute Vector Embeddings
Pre-generate the vector cache before deploying to production:
```bash
gurukul index --cards examples/customer_support_kb.json
```

### 4. Launch Web API and Studio UI
Run a FastAPI server or Gradio Studio interface over your custom cards:
```bash
# Launch unified REST API & Studio on port 8000
gurukul serve --cards examples/customer_support_kb.json --port 8000

# Launch Gradio Studio directly on port 7860
gurukul serve --cards examples/customer_support_kb.json --studio --port 7860
```

---

## 🔬 DeepEval LLM-as-a-Judge Benchmarking Suite

Gurukul includes a complete **DeepEval LLM-as-a-judge benchmarking suite** that allows you to rigorously evaluate your knowledge base using G-Eval rubrics.

### The 4 Custom Evaluation Rubrics

1. **Language Consistency**: Ensures the query, answer, and follow-up suggestions strictly preserve the user's language (no accidental English answers to Marathi/Hindi queries).
2. **Follow-up Offer Quality**: Verifies that follow-up suggestions are polite invitations to explore rather than unsolicited interrogation or pop quizzes.
3. **Curriculum Grounding**: Evaluates factual faithfulness to the retrieval card, strictly punishing hallucinations.
4. **Abstention Accuracy**: Ensures the bot politely and courteously abstains on out-of-domain questions while answering valid queries.

### Running Benchmarks via CLI

```bash
# Dry-run validation (no LLM judge API costs, tests pipeline locally)
gurukul benchmark --cards examples/customer_support_kb.json --input queries.txt --dry-run

# Run with Gemini judge
gurukul benchmark --cards examples/customer_support_kb.json --input queries.txt --judge gemini-1.5-flash

# Run with OpenAI judge
gurukul benchmark --cards examples/customer_support_kb.json --input queries.txt --judge gpt-4o-mini

# Run 100% offline using a local Ollama model
gurukul benchmark --cards examples/customer_support_kb.json --input queries.txt --judge ollama/llama3.1:8b
```

### Running Benchmarks via Python API

```python
from gurukul.benchmarks import run_benchmark

report_path = run_benchmark(
    input_path="test_queries.txt",
    cards="my_knowledgebase.json",
    judge_model="gemini-1.5-flash",  # or "gpt-4o-mini", "ollama/llama3.1:8b"
    dry_run=False
)
print(f"Interactive HTML Report generated at: {report_path}")
```

### Report Outputs
Each benchmark run automatically outputs:
- **`benchmark_report.html`**: An interactive, color-coded HTML dashboard with pass rates, metric radar, and query-by-query breakdown.
- **`benchmark_report.md`**: Clean GitHub Markdown scorecard.
- **`benchmark_scorecard.json`**: Machine-readable metrics.

### 🎨 View & Run Directly in Gurukul Studio (Tab 5)
You can run benchmarks and inspect interactive HTML dashboards directly in the browser:
- Start the studio: `poe studio` (or `python main.py --service studio`)
- Navigate to **Tab 5: 🏆 Benchmark Suite**
- Select existing artifacts, upload files, or paste queries to evaluate with zero terminal switching.

---

## 🏛️ Technical Architecture & Guarantees

```
[User Query: Marathi / Hindi / English]
                   │
                   ▼
  [Universal Unicode Category Normalization]
 (NFC Normalization + Zero Brittle Word Lists)
                   │
                   ▼
  [Subword Char n-gram Language Classifier]
 (TF-IDF + Logistic Regression on 2-4 char n-grams)
                   │
                   ▼
  [ONNX Multilingual Sentence Transformer]
 (google/embeddinggemma-2 · 768 dimensions)
                   │
                   ▼
  [Hybrid Neural Retrieval + Okapi BM25 + RapidFuzz]
 (Dense Dot Product + Lexical Anchoring + Verbatim Sort)
                   │
                   ▼
   [Calibrated Confidence Threshold (>= 0.50)]
        ┌──────────┴──────────┐
        ▼                     ▼
[Ground Answer + Chips]   [Polite Abstention + Topics]
```

### Key Performance Characteristics
- **Embedding Foundation**: Google's official text embedding model `google/embeddinggemma-2` via ONNX Runtime running on CPU.
- **Inference Latency**: **< 15 milliseconds** per query on modern laptop CPUs.
- **Vector Search**: Dot-product cosine matrix multiplication in pure NumPy (`np.dot(matrix, query_vec)`).
- **Zero Word Lists**: Language detection uses a trained statistical model (`lang_detector_v3.pkl`) trained on subword character n-grams ($n \in [2, 4]$), completely eliminating fragile hardcoded keyword lists.
- **Backward Compatibility**: Fully compatible with existing Balbharati tests, legacy `app.offline` imports, and Poe tasks (`poe test`, `poe api`, `poe studio`).

---

## 🧪 Testing

Run the automated test suite covering script detection, Unicode normalization, custom cards ingestion, multi-turn follow-ups, and benchmark runners:
```bash
poe test
# OR
python -m unittest discover -s tests -v
```
All **43 unit tests** execute locally in under 30 seconds with 100% pass rate.

---

## 📁 Repository Structure

```
gurukul/
├── gurukul/                      # Core installable Python package
│   ├── __init__.py               # Top-level exports: OfflineFactBot, FactCard, run_benchmark
│   ├── engine.py                 # Multi-turn conversational dialogue engine
│   ├── index.py                  # Hybrid neural + BM25 + RapidFuzz search index with SHA-256 caching
│   ├── models.py                 # Pydantic schemas (FactCard, FollowUpChip, Session, Responses)
│   ├── normalizer.py             # Script detection and statistical subword char n-gram language classifier
│   ├── nlu.py                    # Conversational intent classifier (affirmation, negation, chip clicks)
│   ├── cli.py                    # Unified CLI (`gurukul chat`, `serve`, `benchmark`, `index`)
│   ├── data/                     # Bundled assets (fact_cards.json, pre-trained ML classifier)
│   └── benchmarks/               # Bundled DeepEval benchmarking suite
├── examples/                     # Ready-to-run examples and sample knowledge bases
│   ├── quickstart_custom_cards.py
│   ├── customer_support_kb.json
│   └── benchmark_custom_cards.py
├── tests/                        # Automated unit tests
│   ├── test_custom_cards_ingestion.py
│   ├── test_offline_bot.py
│   └── test_benchmark_suite.py
├── main.py                       # Unified service runner (REST API + Gradio Studio)
└── pyproject.toml                # Standard PEP 517/621 packaging configuration
```

---

## 📄 License

MIT License. Designed with ❤️ for offline-first educational empowerment and zero-cloud-cost factual chatbots.
