"""
End-to-end RAG pipeline.

Query flow:
  1. Embed question (embedder — always resident)
  2. Retrieve top-20 from ChromaDB
  3. Rerank → top-5 (reranker — always resident)
  4. [Optional] If image_path provided:
       4a. Load VLM (unloads LLM if resident)
       4b. Describe image → JSON findings
       4c. Build enriched query from findings + original question
       4d. Re-retrieve + re-rank with enriched query
       4e. Unload VLM
  5. Load LLM (unloads VLM if resident)
  6. Generate grounded answer with inline citations
  7. Unload LLM (optional — leave resident for follow-up queries)
  8. Return {answer, sources, latency_breakdown}

Latency notes (Jetson Orin Nano 8GB INT4):
  - Embed + retrieve + rerank: ~100-200ms
  - VLM image description:    ~1-3s
  - LLM generation (512 tok): ~15-25s
  - Total (image query):      ~20-30s
  - Total (text-only query):  ~15-25s
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from models.model_manager import ModelManager, ModelName
from rag.prompts import (
    build_rag_prompt,
    build_vlm_rag_prompt,
    OUT_OF_SCOPE_RESPONSE,
    FALLBACK_MESSAGE_MULTILANG,
    INSUFFICIENT_MATCH_TOKEN,
)
from rag.consistency_check import check_grounding
from retrieval.embedder import Embedder
from retrieval.reranker import Reranker
from retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Minimum rerank score to include a chunk (below this → likely irrelevant)
_MIN_RERANK_SCORE = -5.0  # cross-encoder logit; very permissive by default


@dataclass
class RAGResult:
    answer: str
    sources: List[Dict[str, Any]]
    latency: Dict[str, float] = field(default_factory=dict)
    retrieval_count: int = 0
    has_image: bool = False


class RAGPipeline:
    """
    Orchestrates the full Avera RAG pipeline.

    The embedder and reranker are kept resident. LLM and VLM are loaded
    and unloaded sequentially via ModelManager.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        self.cfg = config
        self.embedder = Embedder(config)
        self.store = VectorStore(config)
        self.reranker = Reranker(config)
        self.manager = ModelManager(config)
        self.top_k: int = config["retrieval"]["top_k"]
        self.top_n: int = config["retrieval"]["rerank_top_n"]

    # ── Public API ────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        image_path: Optional[str | Path] = None,
        keep_llm_loaded: bool = False,
        language: str = "en",
        query_en: Optional[str] = None,
    ) -> RAGResult:
        """
        Run a full RAG query, optionally with an image.

        Parameters
        ----------
        question : str
            User's natural-language question.
        image_path : str or Path, optional
            Path to a medical image. If provided, VLM stage is activated.
        keep_llm_loaded : bool
            If True, LLM stays in memory after generation (useful for
            consecutive text queries). Defaults to False (unload after each).
        language : str
            Target language code (e.g. 'en', 'ta', 'hi', 'te', 'kn').
        query_en : str, optional
            English translation of query for searching the English vector store.

        Returns
        -------
        RAGResult
            answer, sources, per-stage latency breakdown.
        """
        timings: Dict[str, float] = {}
        vlm_findings: Optional[Dict] = None

        # Normalize language code (e.g. "Voice:ka", "ka", "voice:ta")
        if language:
            lang_clean = str(language).lower().replace("voice:", "").strip()
            language = "kn" if lang_clean in ("ka", "kn") else lang_clean

        # If question is non-English, translate to English clinical search terms
        if language != "en" and not query_en:
            fast_q = self._match_fast_clinical_query(question, language)
            if fast_q:
                search_query = fast_q
            else:
                self.manager.require(ModelName.LLM)
                llm = self.manager.get_llm()
                search_query = llm.translate_to_clinical_query(question, language=language)
            logger.info(f"Clinical search query: '{question}' → '{search_query}'")
        else:
            search_query = query_en if query_en else question

        # ── Stage 1: Embed query ──────────────────────────────────────────────
        t = time.perf_counter()
        query_embedding = self.embedder.embed_query(search_query)
        timings["embed_ms"] = round((time.perf_counter() - t) * 1000)

        # ── Stage 2: Retrieve ─────────────────────────────────────────────────
        t = time.perf_counter()

        # If the patient did NOT mention a child/infant, exclude IMCI chunks
        # (IMCI = Integrated Management of Childhood Illness — pediatric only).
        # This lets adult-appropriate MedlinePlus chunks surface instead.
        is_pediatric = self._is_pediatric_query(question)
        where_filter = None if is_pediatric else {"doc_type": {"$ne": "jsonl"}}

        raw_results = self.store.query(query_embedding, top_k=self.top_k, where=where_filter)

        # Fallback: if filtering left nothing, retry without the filter
        if not raw_results and where_filter is not None:
            logger.info(
                "Non-pediatric filter returned no results — retrying without filter."
            )
            raw_results = self.store.query(query_embedding, top_k=self.top_k)

        timings["retrieve_ms"] = round((time.perf_counter() - t) * 1000)

        # ── Post-retrieval: strip Spanish MedlinePlus topics ──────────────────
        # MedlinePlus XML contains both English and Spanish health topics.
        # Detect Spanish via: accented chars, Spanish morphology, or known words.
        # IMPORTANT: avoid short strings that are substrings of English words
        # (e.g. "del" matches "model", "del" in "bladder"; "dad" in "granddad").
        import re as _re
        _SPANISH_ACCENT_RE = _re.compile(r"[áéíóúñü]", _re.IGNORECASE)
        _SPANISH_ENDINGS = ("sivos", "ción", "ciones", "dades", "resivos", "presivos")
        _SPANISH_WORDS = frozenset([
            "enfermedades", "acupuntura", "antidepresivos", "embarazo",
            "ansiedad", "medicamentos", "conmoción", "infecciones",
            "enfermedad", "vacunas",
        ])

        def _is_spanish_source(result: Dict) -> bool:
            src = result.get("metadata", {}).get("source", "").lower()
            url = result.get("metadata", {}).get("url", "").lower()
            # Layer 1: accented Spanish characters
            if _SPANISH_ACCENT_RE.search(src):
                return True
            # Layer 2: Spanish-specific word endings (safe — >4 chars, not in English)
            if any(src.endswith(e) or (f" {e}" in src) for e in _SPANISH_ENDINGS):
                return True
            # Layer 3: known unambiguous Spanish MedlinePlus topic words
            if any(w in src for w in _SPANISH_WORDS):
                return True
            # Layer 4: Spanish MedlinePlus URL pattern
            if "es.medlineplus.gov" in url or "/spanish/" in url:
                return True
            return False

        filtered_results = [r for r in raw_results if not _is_spanish_source(r)]
        if filtered_results:
            raw_results = filtered_results
        else:
            logger.info("Spanish filter removed all results — retaining originals.")

        if not raw_results:
            logger.warning("No results returned from vector store.")
            return RAGResult(
                answer=OUT_OF_SCOPE_RESPONSE,
                sources=[],
                latency=timings,
            )

        # ── Stage 3: Rerank ───────────────────────────────────────────────────
        t = time.perf_counter()
        top_chunks = self.reranker.rerank(search_query, raw_results, top_n=self.top_n)
        timings["rerank_ms"] = round((time.perf_counter() - t) * 1000)

        # ── Stage 4: VLM (optional) ───────────────────────────────────────────
        if image_path is not None:
            image_path = Path(image_path)
            t = time.perf_counter()

            self.manager.require(ModelName.VLM)
            vlm = self.manager.get_vlm()
            vlm_findings = vlm.describe_image(image_path)
            timings["vlm_ms"] = round((time.perf_counter() - t) * 1000)
            logger.info(f"VLM findings: {vlm_findings.get('observations', [])}")

            # Enrich query with VLM observations and re-retrieve
            enriched_query = self._build_enriched_query(search_query, vlm_findings)
            t = time.perf_counter()
            enriched_embedding = self.embedder.embed_query(enriched_query)
            enriched_results = self.store.query(enriched_embedding, top_k=self.top_k)
            top_chunks = self.reranker.rerank(enriched_query, enriched_results, top_n=self.top_n)
            timings["vlm_retrieve_ms"] = round((time.perf_counter() - t) * 1000)

        # ── Stage 5: Generate ─────────────────────────────────────────────────
        t = time.perf_counter()
        self.manager.require(ModelName.LLM)
        llm = self.manager.get_llm()

        # Generate grounded clinical answer in English to prevent 1.5B token collapse
        # raw_query_en is the verbatim translated symptom (no static suffixes)
        raw_query_en = query_en if query_en else search_query
        messages = (
            build_rag_prompt(search_query, top_chunks, language="en", raw_query=raw_query_en)
            if not vlm_findings
            else build_vlm_rag_prompt(search_query, vlm_findings, top_chunks)
        )
        answer = llm.generate_chat(messages)
        timings["llm_ms"] = round((time.perf_counter() - t) * 1000)

        # ── Phase 5: Grounding consistency check ──────────────────────────────
        context_text = "\n".join(c["text"] for c in top_chunks)

        # Check for INSUFFICIENT_MATCH as SUBSTRING (LLM sometimes appends it
        # after a partial answer instead of returning it alone).
        answer_clean = answer.strip()
        has_sentinel = INSUFFICIENT_MATCH_TOKEN in answer_clean.upper()

        # If sentinel is embedded mid-answer, strip it and treat the remainder
        # as the actual answer (then re-run the grounding check on that).
        if has_sentinel:
            # Remove the sentinel token and any surrounding whitespace/newlines
            import re as _re
            answer_clean = _re.sub(
                r"(?i)\bINSUFFICIENT[_\s]MATCH\b",
                "",
                answer_clean,
            ).strip()
            # If nothing meaningful remains → serve fallback
            if len(answer_clean) < 20:
                has_sentinel = True  # force fallback
            else:
                has_sentinel = False  # partial answer survived — re-check grounding
                answer = answer_clean

        # Strip trailing prompt echoes if any
        import re as _re
        answer_clean = _re.sub(r"(?i)\n*If (?:no|the) relevant clinical information.*$", "", answer_clean).strip()
        answer = answer_clean

        if has_sentinel or not check_grounding(
            raw_query=raw_query_en, answer=answer, context=context_text
        ):
            logger.warning(
                f"Grounding check FAILED for query='{raw_query_en[:60]}' — "
                f"serving language fallback instead of LLM answer."
            )
            fallback = FALLBACK_MESSAGE_MULTILANG.get(language, FALLBACK_MESSAGE_MULTILANG["en"])
            if not keep_llm_loaded:
                self.manager.unload_all()
            return RAGResult(
                answer=fallback,
                sources=[],
                latency=timings,
                retrieval_count=len(raw_results),
                has_image=image_path is not None,
            )

        # Translate clinical answer to user's selected language
        if language and language != "en" and answer:
            t_trans = time.perf_counter()
            target_map = {
                "ta": "ta-IN",
                "hi": "hi-IN",
                "te": "te-IN",
                "kn": "kn-IN",
                "ka": "kn-IN",
            }
            target_code = target_map.get(language, language)
            try:
                from deep_translator import MyMemoryTranslator
                from concurrent.futures import ThreadPoolExecutor
                tr = MyMemoryTranslator(source="en-US", target=target_code)

                if len(answer) <= 450:
                    answer = tr.translate(answer)
                else:
                    chunks_to_trans = []
                    current_chunk = []
                    current_len = 0
                    for line in answer.splitlines():
                        line_len = len(line) + 1
                        if current_len + line_len <= 450:
                            current_chunk.append(line)
                            current_len += line_len
                        else:
                            if current_chunk:
                                chunks_to_trans.append("\n".join(current_chunk))
                            current_chunk = [line]
                            current_len = line_len
                    if current_chunk:
                        chunks_to_trans.append("\n".join(current_chunk))

                    with ThreadPoolExecutor(max_workers=min(4, max(1, len(chunks_to_trans)))) as executor:
                        translated_pieces = list(executor.map(tr.translate, chunks_to_trans))

                    # Deduplicate consecutive identical chunks (MyMemory artifact
                    # when source contains non-ASCII chars or hits quota limit)
                    deduped = []
                    for piece in translated_pieces:
                        cleaned = piece.strip()
                        # Strip MyMemory quota warning lines
                        cleaned = "\n".join(
                            l for l in cleaned.splitlines()
                            if "MYMEMORY WARNING" not in l.upper()
                        ).strip()
                        if not deduped or cleaned != deduped[-1]:
                            deduped.append(cleaned)
                    answer = "\n".join(deduped)

                timings["translate_ms"] = round((time.perf_counter() - t_trans) * 1000)
                logger.info(f"Translated clinical response to {language} in {timings['translate_ms']}ms")
            except Exception as e:
                logger.warning(f"Translation to {language} failed ({e}), using raw response.")

        if not keep_llm_loaded:
            self.manager.unload_all()

        # ── Build result ──────────────────────────────────────────────────────
        sources = [
            {
                "source": c.get("metadata", {}).get("source", "Unknown"),
                "page": c.get("metadata", {}).get("page", "?"),
                "section": c.get("metadata", {}).get("section", ""),
                "score": round(c.get("rerank_score", c.get("score", 0.0)), 3),
                "snippet": c["text"][:200] + "…" if len(c["text"]) > 200 else c["text"],
            }
            for c in top_chunks
        ]

        timings["total_ms"] = sum(timings.values())
        logger.info(
            f"RAG query complete — "
            f"embed={timings['embed_ms']}ms "
            f"retrieve={timings['retrieve_ms']}ms "
            f"rerank={timings['rerank_ms']}ms "
            f"llm={timings['llm_ms']}ms "
            f"total={timings['total_ms']}ms"
        )

        return RAGResult(
            answer=answer,
            sources=sources,
            latency=timings,
            retrieval_count=len(raw_results),
            has_image=image_path is not None,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _is_pediatric_query(text: str) -> bool:
        """
        Return True if the patient's query explicitly mentions a child/infant.
        Used to decide whether to include IMCI (pediatric) chunks in retrieval.
        """
        t = text.lower()
        pediatric_keywords = [
            # English
            "child", "infant", "baby", "toddler", "newborn", "pediatric",
            "paediatric", "neonatal",
            # Tamil
            "குழந்தை", "குழந்தைகள்", "சிசு", "பச்சிளம்",
            # Hindi
            "बच्चा", "बच्चे", "शिशु", "नवजात",
            # Telugu
            "శిశువు", "పిల్లలు", "బాబు",
            # Kannada
            "ಮಗು", "ಮಕ್ಕಳು", "ಶಿಶು",
        ]
        return any(k in t for k in pediatric_keywords)

    @staticmethod
    def _build_enriched_query(question: str, findings: Dict) -> str:
        """Combine VLM observation keywords with the original question."""
        obs_terms = " ".join(findings.get("observations", []))[:400]
        region = findings.get("region", "")
        parts = [p for p in [region, obs_terms, question] if p.strip()]
        return " | ".join(parts)

    @staticmethod
    def _match_fast_clinical_query(text: str, language: str) -> Optional[str]:
        """
        Fast, zero-latency clinical term resolution for common outpatient symptoms.
        Matches symptoms across Tamil, Hindi, Telugu, and Kannada with whitespace normalization.
        """
        t = text.lower().strip()
        t_norm = "".join(t.split()).replace("…", "").replace(".", "")
        matches = []

        def _has(kw_list: list[str]) -> bool:
            return any(k in t or "".join(k.split()) in t_norm for k in kw_list)

        # Fever / Temperature
        if _has(["காய்ச்சல்", "கைச்சல்", "ஜுரம்", "சூடு", "बुखार", "ताप", "జ్వరం", "ಜ್ವರ"]):
            matches.append("fever body temperature")
        # Cough / Cold / Throat
        if _has(["இருமல்", "சளி", "தொண்டை", "மூக்கடைப்பு", "खांसी", "जुकाम", "सर्दी", "गला", "దగ్గు", "జలుబు", "గొంతు", "ಕೆಮ್ಮು", "ನೆಗಡಿ", "ಗಂಟಲು"]):
            matches.append("cough cold respiratory infection")
        # Breathing / Pneumonia
        if _has(["மூச்சு", "இளைப்பு", "ஆஸ்துமா", "திணறல்", "सांस", "दमा", "శ్వాస", "ఆయాసం", "ಉಸಿರಾಟ"]):
            matches.append("difficulty breathing fast breathing pneumonia")
        # Diarrhea / Dehydration / Loose motion
        if _has(["வயிற்றுப்போக்கு", "வயிற்றுப் போக்கு", "பேதி", "சீதபேதி", "லூஸ் மோஷன்", "दस्त", "पेचिश", "विरेचనాలు", "భేది", "ಭೇದಿ", "ಅತಿಸಾರ"]):
            matches.append("diarrhea dehydration ORS fluid replacement")
        # Vomiting / Nausea
        if _has(["வாந்தி", "உல்டி", "उल्टी", "వాంతులు", "వాంతి", "ವಾಂತಿ"]):
            matches.append("vomiting nausea management")
        # Headache
        if _has(["தலைவலி", "தலையிடி", "सिरदर्द", "सिर दर्द", "తలనొప్పి", "ತಲೆನೋವು"]):
            matches.append("headache assessment and causes")
        # Abdominal pain
        if _has(["வயிறு வலி", "வயிற்று வலி", "வயித்துவலி", "வயித்து வலி", "पेट दर्द", "కడుపు నొప్పి", "ಹೊಟ್ಟೆ ನೋವು"]):
            matches.append("abdominal pain management")
        # Knee / Joint pain
        if _has(["முழங்கால்", "முட்டி", "மூட்டு", "மூட்டுவலி", "घुटने", "घुटनों", "जोड़ों का दर्द", "మోకాలి", "కీళ్ల నొప్పి", "ಮಂಡಿ ನೋವು", "ಕೀಲು ನೋವು"]):
            matches.append("knee pain joint disorder arthritis")
        # Chest pain
        if _has(["நெஞ்சு வலி", "மார்பு வலி", "மார்பு", "सीने में दर्द", "छाती में दर्द", "ఛాతీ నొప్పి", "ಎದೆ ನೋವು"]):
            matches.append("chest pain urgent warning signs cardiovascular")
        # Back pain
        if _has(["முதுகு வலி", "இடுப்பு வலி", "पीठ दर्द", "कमर दर्द", "వెన్ను నొప్పి", "నడుము నొప్పి", "ಬೆನ್ನು ನೋವು"]):
            matches.append("back pain causes management")
        # Convulsions / Seizures
        if _has(["வலிப்பு", "இழுப்பு", "फिट्स", "दौरा", "मिर्गी", "ఫిట్స్", "మూర్ఛ", "ಫಿಟ್ಸ್", "ಮೂರ್ಛೆ"]):
            matches.append("convulsions general danger signs immediate referral")
        # Skin rash / Allergy
        if _has(["தடிப்பு", "அரிப்பு", "தட்டம்மை", "दाने", "खुजली", "खसरा", "దద్దుర్లు", "దురద", "ಗುಳ್ಳೆ", "ತುರಿಕೆ"]):
            matches.append("skin rash measles allergy")
        # Ear infection / Ear pain
        if _has(["காது வலி", "காது", "சீழ்", "कान दर्द", "पीप", "చెవి నొప్పి", "చీము", "ಕಿವಿ ನೋವು", "ಕೀವು"]):
            matches.append("ear pain acute ear infection")

        if matches:
            return " ".join(matches)
        return Noneppend("diarrhea dehydration ORS fluid replacement")
        # Vomiting
        if any(k in t for k in ["வாந்தி", "உல்டி", "उल्टी", "వాంతులు", "వాంతి", "ವಾಂತಿ"]):
            # NOTE: Do NOT add "child" here — vomiting is not exclusively pediatric.
            matches.append("vomiting nausea management")
        # Headache
        if any(k in t for k in ["தலைவலி", "தலையிடி", "सिरदर्द", "सिर दर्द", "తలనొప్పి", "ತಲೆನೋವು"]):
            matches.append("headache assessment and causes")
        # Abdominal pain
        if any(k in t for k in ["வயிறு வலி", "வயிற்று வலி", "வயித்துவலி", "पेट दर्द", "కడుపు నొప్పి", "ಹೊಟ್ಟೆ ನೋವು"]):
            matches.append("abdominal pain management")
        # Convulsions / Seizures
        if any(k in t for k in ["வலிப்பு", "இழுப்பு", "फिट्स", "दौरा", "मिर्गी", "ఫిట్స్", "మూర్ఛ", "ಫಿಟ್ಸ್", "ಮೂರ್ಛೆ"]):
            matches.append("convulsions general danger signs immediate referral")
        # Skin rash
        if any(k in t for k in ["தடிப்பு", "அரிப்பு", "தட்டம்மை", "दाने", "खुजली", "खसरा", "దద్దుర్లు", "దురద", "ಗುಳ್ಳೆ", "ತುರಿಕೆ"]):
            matches.append("skin rash measles allergy")
        # Ear infection
        if any(k in t for k in ["காது வலி", "காது", "சீழ்", "कान दर्द", "पीप", "చెవి నొప్పి", "చీము", "ಕಿವಿ ನೋವು", "ಕೀವು"]):
            matches.append("ear pain acute ear infection")

        if matches:
            # Return symptom keywords only — no demographic or severity suffix injected here.
            # The LLM generation stage (via system prompt + context) will provide clinical framing.
            return " ".join(matches)
        return None
