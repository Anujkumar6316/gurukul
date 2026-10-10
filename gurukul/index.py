from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from fastembed import TextEmbedding
from rapidfuzz import fuzz

from gurukul.models import FactCard, FollowUpChip
from gurukul.normalizer import clean_text, detect_script
from gurukul.nlu_aspects import classify_query, QueryAspect, QueryModality


class BM25Index:
    """Lightweight, offline Okapi BM25 inverted index for exact lexical scoring."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_len: dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.doc_count: int = 0
        self.inverted_index: dict[str, dict[str, int]] = {}
        self.idf: dict[str, float] = {}

    def index_documents(self, documents: dict[str, list[str]]) -> None:
        """Indexes tokenized documents where key is card_id and value is list of tokens."""
        self.doc_count = len(documents)
        if self.doc_count == 0:
            return

        total_len = 0
        doc_freqs: dict[str, int] = {}

        for doc_id, tokens in documents.items():
            length = len(tokens)
            self.doc_len[doc_id] = length
            total_len += length

            counts = Counter(tokens)
            for term, count in counts.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_id] = count
                doc_freqs[term] = doc_freqs.get(term, 0) + 1

        self.avg_doc_len = total_len / float(self.doc_count)

        for term, df in doc_freqs.items():
            self.idf[term] = math.log(1.0 + (self.doc_count - df + 0.5) / (df + 0.5))

    def score(self, query_tokens: list[str]) -> dict[str, float]:
        """Scores all matching documents for the given query tokens."""
        scores: dict[str, float] = {}
        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            idf_val = self.idf[token]
            postings = self.inverted_index[token]
            for doc_id, freq in postings.items():
                dl = self.doc_len[doc_id]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (dl / self.avg_doc_len))
                term_score = idf_val * (numerator / denominator)
                scores[doc_id] = scores.get(doc_id, 0.0) + term_score
        return scores


class EmbeddingGemma2ONNX:
    """Zero-LLM on-device ONNX runtime encoder for Google EmbeddingGemma 2 (google/embeddinggemma-2)."""

    def __init__(
        self,
        model_name: str = "google/embeddinggemma-2",
        cache_dir: Path | str | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = str(cache_dir) if cache_dir else None
        repo_id = "onnx-community/embeddinggemma-2-ONNX"

        try:
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer
            import onnxruntime as ort
        except ImportError as e:
            raise ImportError(
                f"EmbeddingGemma 2 requires huggingface_hub, tokenizers, and onnxruntime: {e}"
            )

        self.model_path = hf_hub_download(repo_id, "onnx/model_q4.onnx", cache_dir=self.cache_dir)
        _ = hf_hub_download(repo_id, "onnx/model_q4.onnx_data", cache_dir=self.cache_dir)
        tok_path = hf_hub_download(repo_id, "tokenizer.json", cache_dir=self.cache_dir)

        self.tokenizer = Tokenizer.from_file(tok_path)
        self.tokenizer.enable_padding()
        self.tokenizer.enable_truncation(max_length=512)

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._empty_feat = np.zeros((0, 512), dtype=np.float32)

    def embed(
        self,
        texts: list[str] | Iterable[str],
        batch_size: int = 64,
    ) -> Iterable[np.ndarray]:
        text_list = list(texts)
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i : i + batch_size]
            encodings = self.tokenizer.encode_batch(batch)
            input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
            attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
            feed = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "image_features": self._empty_feat,
                "video_features": self._empty_feat,
                "audio_features": self._empty_feat,
            }
            out = self.session.run(["sentence_embedding"], feed)
            emb = out[0]
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normed = (emb / norms).astype(np.float32)
            for vec in normed:
                yield vec


class FactCardIndex:
    """Production-grade neural semantic search index for curriculum and custom knowledge cards.

    Uses an ONNX multilingual sentence transformer (EmbeddingGemma 2 or MiniLM)
    paired with an in-memory BM25 lexical index to map user queries into
    grounded fact cards, eliminating fragile word lists and hallucinated matches.
    """

    MODEL_NAME = os.getenv("GURUKUL_EMBEDDING_MODEL", os.getenv("FACTBOT_EMBEDDING_MODEL", "google/embeddinggemma-2"))

    def __init__(
        self,
        cards: Path | str | list[dict] | list[FactCard] | None = None,
        data_path: Path | str | None = None,
        cache_dir: Path | str | None = None,
        model_name: str | None = None,
    ) -> None:
        self.model_name = model_name or self.MODEL_NAME
        model_slug = self.model_name.split("/")[-1].replace("-", "_")

        # Resolve card source
        raw_cards_data: list[dict | FactCard] | None = None
        content_hash: str | None = None

        target_input = cards if cards is not None else data_path

        if isinstance(target_input, list):
            raw_cards_data = target_input
            raw_bytes = json.dumps([c.model_dump() if isinstance(c, FactCard) else c for c in target_input], sort_keys=True).encode("utf-8")
            content_hash = hashlib.sha256(raw_bytes).hexdigest()[:12]
            resolved_cache_dir = Path(cache_dir or (Path.cwd() / ".cache"))
        elif isinstance(target_input, (str, Path)):
            resolved_path = Path(target_input)
            if not resolved_path.exists():
                raise FileNotFoundError(f"Fact cards file not found at {resolved_path}")
            with open(resolved_path, "rb") as f:
                content_bytes = f.read()
            raw_cards_data = json.loads(content_bytes.decode("utf-8"))
            content_hash = hashlib.sha256(content_bytes).hexdigest()[:12]
            resolved_cache_dir = Path(cache_dir or (resolved_path.parent / ".cache"))
        else:
            # Default fallback search for bundled fact_cards.json
            pkg_data = Path(__file__).resolve().parent / "data" / "fact_cards.json"
            repo_data = Path(__file__).resolve().parents[1] / "data" / "fact_cards.json"
            cwd_data = Path.cwd() / "data" / "fact_cards.json"

            if pkg_data.exists():
                resolved_path = pkg_data
            elif repo_data.exists():
                resolved_path = repo_data
            elif cwd_data.exists():
                resolved_path = cwd_data
            else:
                resolved_path = repo_data

            if not resolved_path.exists():
                raise FileNotFoundError(f"Default fact cards file not found at {resolved_path}")

            with open(resolved_path, "r", encoding="utf-8") as f:
                raw_cards_data = json.load(f)
            resolved_cache_dir = Path(cache_dir or (resolved_path.parent / ".cache"))

        self.cache_dir = resolved_cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if content_hash:
            self.cache_file = self.cache_dir / f"fact_embeddings_{model_slug}_{content_hash}.npz"
        else:
            self.cache_file = self.cache_dir / f"fact_embeddings_{model_slug}.npz"
            if not self.cache_file.exists():
                repo_cache = Path(__file__).resolve().parents[1] / "data" / ".cache" / f"fact_embeddings_{model_slug}.npz"
                if repo_cache.exists():
                    self.cache_file = repo_cache
            if not self.cache_file.exists() and "minilm" in model_slug.lower():
                legacy_file = self.cache_dir / "fact_embeddings.npz"
                if not legacy_file.exists():
                    legacy_file = Path(__file__).resolve().parents[1] / "data" / ".cache" / "fact_embeddings.npz"
                if legacy_file.exists():
                    self.cache_file = legacy_file

        self.cards_by_id: dict[str, FactCard] = {}
        self.bm25: BM25Index | None = None
        self.encoder: TextEmbedding | None = None

        self.vectors: np.ndarray | None = None
        self.indexed_card_ids: list[str] = []
        self.indexed_texts: list[str] = []

        self._load_cards(raw_cards_data or [])
        self._init_encoder()
        self._load_or_build_embeddings()

    def _load_cards(self, raw_data: list[Any]) -> None:
        # Handle dict wrapper like {"cards": [...]} or {"facts": [...]}
        if isinstance(raw_data, dict):
            raw_data = raw_data.get("cards") or raw_data.get("facts") or []

        doc_tokens: dict[str, list[str]] = {}

        for item in raw_data:
            if isinstance(item, FactCard):
                card = item
            else:
                card = FactCard.model_validate(item)

            self.cards_by_id[card.id] = card

            tokens: list[str] = []
            for alias in card.question_aliases:
                tokens.extend(clean_text(alias).split())
            if card.topic:
                tokens.extend(clean_text(card.topic.replace("_", " ")).split())
            for entity in card.entities:
                tokens.extend(clean_text(entity).split())

            mr_ans = clean_text(card.get_answer("mr"))
            if mr_ans:
                tokens.extend(mr_ans[:100].split())
            en_ans = clean_text(card.get_answer("en"))
            if en_ans:
                tokens.extend(en_ans[:100].split())
            hi_ans = clean_text(card.get_answer("hi"))
            if hi_ans:
                tokens.extend(hi_ans[:100].split())

            doc_tokens[card.id] = [t for t in tokens if len(t) > 1]

        self.bm25 = BM25Index()
        self.bm25.index_documents(doc_tokens)

    def _init_encoder(self) -> None:
        """Initializes the offline ONNX multilingual text encoder."""
        m_lower = self.model_name.lower()
        if "embeddinggemma-2" in m_lower or "embeddinggemma_2" in m_lower:
            model_cache = os.getenv("GURUKUL_MODEL_CACHE") or os.getenv("FACTBOT_MODEL_CACHE")
            self.encoder = EmbeddingGemma2ONNX(model_name=self.model_name, cache_dir=model_cache)
        else:
            self.encoder = TextEmbedding(model_name=self.model_name)

    def _load_or_build_embeddings(self) -> None:
        """Loads pre-computed semantic embeddings from disk cache or builds them."""
        if self.cache_file.exists():
            data = np.load(self.cache_file)
            self.vectors = data["vectors"]
            self.indexed_card_ids = list(data["card_ids"])
            self.indexed_texts = list(data["texts"])
            self._calibrate_dynamic_range()
            return

        texts: list[str] = []
        card_ids: list[str] = []

        for card in self.cards_by_id.values():
            for alias in card.question_aliases:
                cleaned = clean_text(alias)
                if cleaned:
                    texts.append(cleaned)
                    card_ids.append(card.id)

            if card.topic:
                t = clean_text(card.topic.replace("_", " "))
                if t:
                    texts.append(t)
                    card_ids.append(card.id)

            for entity in card.entities:
                e = clean_text(entity)
                if e:
                    texts.append(e)
                    card_ids.append(card.id)

            mr_ans = clean_text(card.get_answer("mr"))
            if mr_ans:
                texts.append(mr_ans[:200])
                card_ids.append(card.id)

            en_ans = clean_text(card.get_answer("en"))
            if en_ans:
                texts.append(en_ans[:200])
                card_ids.append(card.id)

            hi_ans = clean_text(card.get_answer("hi"))
            if hi_ans:
                texts.append(hi_ans[:200])
                card_ids.append(card.id)

        if not texts:
            self.vectors = np.empty((0, 384), dtype=np.float32)
            self.indexed_card_ids = []
            self.indexed_texts = []
            return

        assert self.encoder is not None

        if "embeddinggemma" in self.model_name.lower():
            embed_inputs = [f"title: none | text: {t}" for t in texts]
        else:
            embed_inputs = texts

        raw_vecs = np.array(list(self.encoder.embed(embed_inputs, batch_size=256)))
        norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        norm_vecs = (raw_vecs / norms).astype(np.float32)

        self.vectors = norm_vecs
        self.indexed_card_ids = card_ids
        self.indexed_texts = texts

        try:
            np.savez_compressed(
                self.cache_file,
                vectors=norm_vecs,
                card_ids=card_ids,
                texts=texts,
            )
        except Exception:
            pass

        self._calibrate_dynamic_range()

    def _calibrate_dynamic_range(self) -> None:
        """Data-driven calibration of cosine similarity dynamic range from indexed vectors."""
        if hasattr(self, "vectors") and self.vectors is not None and len(self.vectors) > 20:
            sample_size = min(300, len(self.vectors))
            rng = np.random.default_rng(42)
            i1 = rng.integers(0, len(self.vectors), size=sample_size)
            i2 = rng.integers(0, len(self.vectors), size=sample_size)
            dots = np.sum(self.vectors[i1] * self.vectors[i2], axis=1)
            self.cosine_floor = float(np.percentile(dots, 5))
            self.cosine_ceiling = float(np.percentile(dots, 99))
        else:
            self.cosine_floor = 0.58 if "embeddinggemma" in self.model_name.lower() else 0.20
            self.cosine_ceiling = 0.86 if "embeddinggemma" in self.model_name.lower() else 0.90

    def calibrate_semantic_score(self, sem_score: float) -> float:
        """Min-max dynamic normalization calibrated to active embedding model geometry."""
        floor = getattr(self, "cosine_floor", 0.58 if "embeddinggemma" in self.model_name.lower() else 0.20)
        ceil = getattr(self, "cosine_ceiling", 0.86 if "embeddinggemma" in self.model_name.lower() else 0.90)
        return min(1.0, max(0.0, (sem_score - floor) / max(1e-6, ceil - floor)))

    def get_card(self, card_id: str) -> FactCard | None:
        """Fetch card by ID in O(1) time."""
        return self.cards_by_id.get(card_id)

    def search(
        self,
        query: str,
        grade: int | None = None,
        top_k: int = 5,
        aspect: str | None = None,
        modality: str | None = None,
    ) -> list[tuple[FactCard, float]]:
        """Dense neural semantic retrieval with BM25 lexical anchoring, aspect/modality reranking, and RapidFuzz scoring."""
        cleaned_query = clean_text(query)
        if not cleaned_query or self.vectors is None or len(self.vectors) == 0 or self.encoder is None:
            return []

        # Analyze query aspect and modality with zero-shot prototype support
        if aspect is None or modality is None:
            analysis = classify_query(query, encoder=self.encoder)
            q_aspect = aspect or analysis.aspect.value
            q_modality = modality or analysis.modality.value
        else:
            q_aspect = aspect
            q_modality = modality

        q_text = (
            f"task: search result | query: {cleaned_query}"
            if "embeddinggemma" in self.model_name.lower()
            else cleaned_query
        )
        q_vec = np.array(list(self.encoder.embed([q_text]))[0])
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm

        semantic_scores = np.dot(self.vectors, q_vec)

        q_tokens = [t for t in cleaned_query.split() if len(t) > 1]
        bm25_scores = self.bm25.score(q_tokens) if self.bm25 else {}
        max_bm25 = max(bm25_scores.values()) if bm25_scores else 0.0

        limit_cand = min(40, len(semantic_scores))
        top_indices = np.argsort(semantic_scores)[::-1][:limit_cand]

        card_scores: dict[str, float] = {}

        for idx in top_indices:
            cid = str(self.indexed_card_ids[idx])
            sem_score = float(semantic_scores[idx])

            indexed_str = str(self.indexed_texts[idx])
            fuzz_ratio = fuzz.token_sort_ratio(cleaned_query, indexed_str) / 100.0

            bm25_norm = (bm25_scores.get(cid, 0.0) / max_bm25) if max_bm25 > 0 else 0.0

            sem_calibrated = self.calibrate_semantic_score(sem_score)

            if (bm25_norm > 0 and sem_score >= 0.70) or fuzz_ratio >= 0.70:
                lexical_signal = max(bm25_norm, fuzz_ratio)
                composite_score = sem_calibrated * 0.75 + lexical_signal * 0.25
                final_score = max(sem_calibrated, composite_score)
            else:
                final_score = sem_calibrated

            card_obj = self.cards_by_id.get(cid)
            if card_obj and card_obj.is_curriculum and any(not getattr(k, "is_curriculum", True) for k in self.cards_by_id.values()):
                final_score += 0.02
            if card_obj:
                c_aspect = card_obj.get_aspect()
                c_modality = card_obj.get_modality()

                # Modality alignment
                if q_modality != "general":
                    if c_modality == q_modality:
                        final_score += 0.05
                    elif c_modality != "general":
                        final_score -= 0.05

                # Aspect alignment
                if q_aspect == "cause":
                    if c_aspect == "cause":
                        final_score += 0.05
                    elif c_aspect == "remedy":
                        final_score -= 0.05
                elif q_aspect in ("definition", "schedule", "symptom"):
                    if c_aspect == q_aspect:
                        final_score += 0.05

            if cid not in card_scores or final_score > card_scores[cid]:
                card_scores[cid] = final_score

        results: list[tuple[FactCard, float]] = []
        for cid, score in card_scores.items():
            card = self.cards_by_id.get(str(cid))
            if not card:
                continue
            final_score = score
            if grade is not None and card.grades and grade in card.grades:
                final_score += 0.05
            results.append((card, final_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [(c, min(1.0, max(0.0, s))) for c, s in results[:top_k]]

    def get_random_follow_up_offer(
        self,
        card_id: str,
        lang: str = "mr",
        preferred_aspect: str | None = None,
    ) -> tuple[str | None, str | None, str | None]:
        """Selects a related follow-up card ID, generates a natural sounding offer question,

        and returns (offer_prompt, natural_question, chosen_card_id).
        """
        import random
        from gurukul.models import format_offer_prompt
        from gurukul.normalizer import detect_script, infer_preferred_language

        card = self.cards_by_id.get(card_id)
        if not card:
            return None, None, None

        # 1. First priority: explicit follow_up_cards
        candidates = [fid for fid in card.follow_up_cards if fid in self.cards_by_id and fid != card_id]

        # 2. Second priority: same category cards
        if not candidates and card.category:
            candidates = [
                c.id for c in self.cards_by_id.values()
                if c.id != card_id and c.category == card.category
            ]

        # 3. Third priority: same subject cards
        if not candidates and card.subject and card.subject != "general":
            candidates = [
                c.id for c in self.cards_by_id.values()
                if c.id != card_id and c.subject == card.subject
            ]

        # Filter candidates for script and language purity
        valid_candidates = []
        for cid in candidates:
            c = self.cards_by_id[cid]
            card_langs = c.languages or list(c.answer_templates.keys())
            if lang not in card_langs:
                continue
            q = c.get_offer_question(lang)
            if lang in ("mr", "hi"):
                if detect_script(q) != "devanagari":
                    continue
                if infer_preferred_language(q, default=lang) != lang:
                    continue
            elif lang == "en" and detect_script(q) != "latin":
                continue
            valid_candidates.append(cid)

        if not valid_candidates:
            # Fallback across knowledgebase cards that satisfy script and language purity
            for c in self.cards_by_id.values():
                card_langs = c.languages or list(c.answer_templates.keys())
                if c.id == card_id or lang not in card_langs:
                    continue
                q = c.get_offer_question(lang)
                if lang in ("mr", "hi"):
                    if detect_script(q) == "devanagari" and infer_preferred_language(q, default=lang) == lang:
                        valid_candidates.append(c.id)
                elif lang == "en" and detect_script(q) == "latin":
                    valid_candidates.append(c.id)
                if len(valid_candidates) >= 20:
                    break

        if not valid_candidates:
            return None, None, None

        if preferred_aspect:
            aspect_matches = [
                cid for cid in valid_candidates
                if self.cards_by_id[cid].get_aspect() == preferred_aspect
            ]
            if not aspect_matches:
                # Also check across knowledgebase for preferred aspect cards
                for c in self.cards_by_id.values():
                    card_langs = c.languages or list(c.answer_templates.keys())
                    if c.id != card_id and c.get_aspect() == preferred_aspect and lang in card_langs:
                        q = c.get_offer_question(lang)
                        if lang in ("mr", "hi"):
                            if detect_script(q) == "devanagari" and infer_preferred_language(q, default=lang) == lang:
                                aspect_matches.append(c.id)
                        elif lang == "en" and detect_script(q) == "latin":
                            aspect_matches.append(c.id)
            if aspect_matches:
                chosen_id = random.choice(aspect_matches)
            else:
                chosen_id = random.choice(valid_candidates)
        else:
            chosen_id = random.choice(valid_candidates)

        chosen_card = self.cards_by_id[chosen_id]

        offer_q = chosen_card.get_offer_question(lang)
        offer_prompt = format_offer_prompt(offer_q, lang=lang)
        return offer_prompt, offer_q, chosen_id

    def get_follow_up_chips(
        self,
        card_id: str,
        lang: str = "mr",
        max_chips: int = 4,
        featured_id: str | None = None,
        randomize: bool = True,
    ) -> list[FollowUpChip]:
        """Resolves follow-up card IDs into clean, clickable FollowUpChip navigation objects."""
        import random
        from gurukul.normalizer import detect_script

        card = self.cards_by_id.get(card_id)
        if not card:
            return []

        fu_ids = [fid for fid in card.follow_up_cards if fid in self.cards_by_id]
        if not fu_ids and card.category:
            fu_ids = [c.id for c in self.cards_by_id.values() if c.id != card_id and c.category == card.category]

        # Filter out chips that fail script check
        valid_fu_ids = []
        for fid in fu_ids:
            fc = self.cards_by_id.get(fid)
            if not fc:
                continue
            title = fc.get_title(lang)
            if lang in ("mr", "hi") and detect_script(title) != "devanagari":
                continue
            if lang == "en" and detect_script(title) != "latin":
                continue
            valid_fu_ids.append(fid)

        if randomize:
            other_ids = [fid for fid in valid_fu_ids if fid != featured_id]
            random.shuffle(other_ids)
            selected_ids = ([featured_id] if featured_id and featured_id in valid_fu_ids else []) + other_ids
        else:
            selected_ids = valid_fu_ids

        chips: list[FollowUpChip] = []
        for fu_id in selected_ids[:max_chips]:
            fu_card = self.cards_by_id.get(fu_id)
            if fu_card:
                chips.append(
                    FollowUpChip(
                        card_id=fu_card.id,
                        title=fu_card.get_title(lang),
                        topic=fu_card.topic,
                        language=lang,
                    )
                )
        return chips

    def get_suggested_topics(self, lang: str = "mr", count: int = 4) -> list[str]:
        """Return a curated set of topics for exploration in the requested language."""
        # If custom non-curriculum cards have topic titles, offer those
        custom_topics = [c.topic for c in self.cards_by_id.values() if c.topic and not c.is_curriculum]
        if len(custom_topics) >= count:
            return list(dict.fromkeys(custom_topics))[:count]

        topics_file = Path(__file__).resolve().parent / "data" / "suggested_topics.json"
        if topics_file.exists():
            try:
                with open(topics_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                curated = data.get(lang) or data.get("mr", [])
                if curated:
                    return curated[:count]
            except Exception:
                pass

        # Dynamic fallback: extract from indexed cards
        from gurukul.normalizer import detect_script, infer_preferred_language

        target_script = "devanagari" if lang in ("mr", "hi") else "latin"
        extracted: list[str] = []

        for card in self.cards_by_id.values():
            cand: str | None = None
            for e in card.entities:
                if detect_script(e) == target_script:
                    cand = e
                    break
            if not cand:
                for a in card.question_aliases:
                    clean = a.strip("?|।॥ ").strip()
                    if detect_script(clean) == target_script and infer_preferred_language(clean, default=lang) == lang and len(clean.split()) <= 4:
                        cand = clean
                        break
            if not cand:
                for a in card.question_aliases:
                    clean = a.strip("?|।॥ ").strip()
                    if detect_script(clean) == target_script and len(clean.split()) <= 4:
                        cand = clean
                        break
            if not cand and card.topic:
                cand = card.topic.replace("_", " ").title()

            if cand and cand not in extracted:
                extracted.append(cand)
                if len(extracted) >= count:
                    break

        return extracted[:count]

    def get_curriculum_tree(self) -> dict[str, Any]:
        """Builds a hierarchical view of the knowledgebase: Grade -> Subject -> Topics."""
        tree: dict[str, dict[str, list[dict[str, str]]]] = {}

        for card in self.cards_by_id.values():
            grades = card.grades or ([card.grade] if card.grade else [0])
            subject = (card.subject or "general").title()
            title = card.get_title()

            for g in grades:
                grade_key = f"Grade {g}" if g > 0 else "All Grades"
                if grade_key not in tree:
                    tree[grade_key] = {}
                if subject not in tree[grade_key]:
                    tree[grade_key][subject] = []

                tree[grade_key][subject].append(
                    {"card_id": card.id, "title": title, "topic": card.topic or ""}
                )

        return tree
