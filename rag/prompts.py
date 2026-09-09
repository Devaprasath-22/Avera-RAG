"""
Prompt templates for the Avera RAG system.

All prompts are constants here — change wording in one place, not scattered
across the pipeline. The RAG template enforces source citation inline.

Design principles (for medical safety):
  1. System always forbids diagnosis and prescribing.
  2. Every factual claim must be tied to a [SOURCE n] citation.
  3. If context is absent or insufficient, the model must say so explicitly.
  4. Confidence and uncertainty must be acknowledged.
"""

from string import Template
from typing import Optional

# ─────────────────────────────────────────────────────────────────────────────
# System-level medical safety preamble
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_MEDICAL_SAFETY = (
    "You are Avera, a clinical medical information assistant running offline on an "
    "embedded device. You provide accurate, grounded medical information based on reference literature.\n\n"
    "STRICT RULES — you must ALWAYS follow these:\n"
    "1. You are NOT a doctor and cannot diagnose, prescribe, or replace clinical judgment.\n"
    "2. Base every statement ONLY on the provided [CONTEXT] chunks. Do not invent facts.\n"
    "3. Cite every factual claim inline as [SOURCE n] where n matches the chunk number in [CONTEXT].\n"
    "4. If the context does not contain enough information to answer, respond with exactly: INSUFFICIENT_MATCH\n"
    "5. Do NOT introduce unstated age groups, demographics, or unrelated symptoms. Match the patient's presentation.\n"
    "6. Acknowledge uncertainty — use phrases like \"according to the source\", \"the document states\".\n"
    "7. Keep the answer concise and clinically actionable (≤ 250 words).\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# RAG answer template  (text-only query)
# ─────────────────────────────────────────────────────────────────────────────
# Variables: {context_block}, {question}

RAG_USER_TEMPLATE = Template(
    "[PATIENT QUERY — verbatim translated, use ONLY these symptoms]\n"
    "$raw_query\n\n"
    "[CONTEXT]\n"
    "$context_block\n\n"
    "[PATIENT QUERY OR SYMPTOM (search form)]\n"
    "$question\n\n"
    "[INSTRUCTIONS]\n"
    "Based ONLY on the retrieved context above, write a concise clinical response (under 75 words):\n"
    "1. Likely condition or cause cited as [SOURCE n].\n"
    "2. Key warning or danger signs to monitor.\n"
    "3. Recommended next action or when to seek urgent medical care.\n"
    "Write your clinical answer directly without repeating the instruction bullet points.\n"
    "If the context does not contain relevant clinical information for the symptoms, respond exactly: INSUFFICIENT_MATCH"
)


def build_context_block(chunks: list[dict]) -> str:
    """
    Format retrieval results into a numbered context block.

    Parameters
    ----------
    chunks : list of result dicts (from Reranker.rerank)
        Each dict must have 'text' and 'metadata' keys.

    Returns
    -------
    str
        Formatted context block ready to insert into the prompt.
    """
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("source", "Unknown")
        page = meta.get("page", "?")
        section = meta.get("section", "")
        header = f"[SOURCE {i}] {source}, p.{page}"
        if section:
            header += f" — {section}"
        lines.append(f"{header}\n{chunk['text']}")
    return "\n\n---\n\n".join(lines)


LANGUAGE_NAMES = {
    "en": "English",
    "ta": "Tamil (தமிழ்)",
    "hi": "Hindi (हिन्दी)",
    "te": "Telugu (తెలుగు)",
    "kn": "Kannada (ಕನ್ನಡ)",
}


def build_rag_prompt(
    question: str,
    chunks: list[dict],
    language: str = "en",
    raw_query: Optional[str] = None,
) -> list[dict]:
    """
    Build the full chat message list for a RAG query.

    Parameters
    ----------
    question : str
        The user's question (English search form, may include expanded keywords).
    chunks : list[dict]
        Retrieved context chunks.
    language : str
        Target language code (e.g. 'en', 'ta', 'hi', 'te', 'kn').
    raw_query : str, optional
        Verbatim English translation of the patient's exact words. When provided,
        this is surfaced separately in the prompt so the LLM knows what the patient
        actually said (no static suffixes, no demographic injection).

    Returns
    -------
    List[dict]
        Messages in OpenAI/llama.cpp chat format.
    """
    context_block = build_context_block(chunks)
    # Use raw_query (verbatim patient words) when available; fall back to question.
    verbatim = raw_query if raw_query else question
    user_content = RAG_USER_TEMPLATE.substitute(
        raw_query=verbatim,
        context_block=context_block,
        question=question,
    )

    return [
        {"role": "system", "content": SYSTEM_MEDICAL_SAFETY},
        {"role": "user", "content": user_content},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# VLM + RAG synthesis template  (image + text query)
# ─────────────────────────────────────────────────────────────────────────────
# Variables: {vlm_findings_json}, {context_block}, {question}

VLM_RAG_USER_TEMPLATE = Template(
    "[IMAGE FINDINGS — AI generated, not a diagnosis]\n"
    "$vlm_findings\n\n"
    "[CONTEXT FROM MEDICAL DOCUMENTS]\n"
    "$context_block\n\n"
    "[QUESTION]\n"
    "$question\n\n"
    "[ANSWER]\n"
    "Using the image findings and the context above, answer the question. "
    "Cite every factual claim as [SOURCE n]. "
    "Clearly distinguish what comes from the image versus the documents."
)


def format_vlm_findings(findings: dict) -> str:
    """Convert VLM findings dict to a human-readable string for the prompt."""
    obs = findings.get("observations", [])
    region = findings.get("region", "unspecified")
    notes = findings.get("confidence_notes", "")

    lines = [f"Region/image type: {region}"]
    if obs:
        lines.append("Observations:")
        for o in obs:
            lines.append(f"  - {o}")
    if notes:
        lines.append(f"Confidence notes: {notes}")
    return "\n".join(lines)


def build_vlm_rag_prompt(
    question: str,
    vlm_findings: dict,
    chunks: list[dict],
) -> list[dict]:
    """
    Build the full chat message list for an image + RAG query.

    Returns
    -------
    List[dict]
        Messages in OpenAI/llama.cpp chat format.
    """
    context_block = build_context_block(chunks)
    findings_str = format_vlm_findings(vlm_findings)
    user_content = VLM_RAG_USER_TEMPLATE.substitute(
        vlm_findings=findings_str,
        context_block=context_block,
        question=question,
    )
    return [
        {"role": "system", "content": SYSTEM_MEDICAL_SAFETY},
        {"role": "user", "content": user_content},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Out-of-scope refusal + multilingual fallback messages
# ─────────────────────────────────────────────────────────────────────────────

# Sentinel token the LLM emits when context does not match the query.
# Keep uppercase and unique so it's easy to detect programmatically.
INSUFFICIENT_MATCH_TOKEN = "INSUFFICIENT_MATCH"

OUT_OF_SCOPE_RESPONSE = (
    "This question appears to be outside the scope of the available medical "
    "documents. I can only answer questions that can be grounded in the "
    "indexed reference material. Please consult a qualified healthcare "
    "provider for clinical decisions."
)

# Intentional, safety-critical hardcode — these are fallback *messages*, not
# clinical content. They do NOT travel through the query pipeline.
FALLBACK_MESSAGE_MULTILANG: dict[str, str] = {
    "ta": (
        "இந்த அறிகுறிகளுக்கு எனது தரவுத்தளத்தில் குறிப்பிட்ட தகவல் இல்லை. "
        "தயவுஸெய்து மருத்துவரை அணுகவும்."
    ),
    "hi": (
        "इन लक्षणों के लिए मेरे डेटाबेस में विशिष्ट जानकारी नहीं है. "
        "कृपया डॉक्टर से मिलें."
    ),
    "te": (
        "ఈ లక్షణాలకు నా డేటాబేస్‌లో నిర్దిష్టమైన సమాచారం లేదు. "
        "దయచేసి వైద్యుడిని సంప్రదించండి."
    ),
    "kn": (
        "ಈ ಲಕ್ಷಣಗಳಿಗೆ ನನ್ನ ಡೇಟಾಬೇಸ್‌ನಲ್ಲಿ ನಿರ್ದಿಷ್ಟ ಮಾಹಿತಿ ಇಲ್ಲ. "
        "ದಯವಿಟ್ಟು ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ."
    ),
    "en": OUT_OF_SCOPE_RESPONSE,
}
