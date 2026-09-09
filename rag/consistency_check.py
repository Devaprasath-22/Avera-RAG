"""
Grounding consistency check (Phase 5).

Verifies that the LLM-generated answer does not introduce entities,
demographics, or symptom categories absent from both the patient query
and the retrieved context.

Design decision: heuristic (no second LLM call).
- A second LLM inference on Jetson Orin Nano 8GB adds ~15-25s latency.
- A fast token-overlap heuristic catches the most common failure mode
  (demographic injection -- "child", "infant") in <1ms.
- The system prompt already instructs the LLM to emit INSUFFICIENT_MATCH;
  this module is defence-in-depth for when the model ignores that instruction.

Usage:
    from rag.consistency_check import check_grounding
    if not check_grounding(raw_query, answer, context):
        return fallback_message
"""

import logging
import re
from typing import FrozenSet, List

logger = logging.getLogger(__name__)

# Demographic guard terms.
# If any of these appear in the ANSWER but NOT in the patient's raw_query
# or the retrieved context, the answer is considered ungrounded.
# Extend this list as new failure modes are discovered.
_DEMOGRAPHIC_GUARD_TERMS: List[str] = [
    # English
    "child", "children", "infant", "baby", "babies", "toddler", "newborn",
    "pediatric", "paediatric", "neonatal", "neonate",
    # Tamil
    "குழந்தை", "குழந்தைகள்", "சிசு", "பச்சிளங்குழந்தை",
    # Hindi
    "बच्चा", "बच्चे", "शिशु", "नवजात",
    # Telugu
    "శిశువు", "పిల్లలు", "బాబు",
    # Kannada
    "ಮಗು", "ಮಕ್ಕಳು", "ಶಿಶು",
]

# Pre-compile regex patterns for each term (word-boundary aware for English)
_GUARD_PATTERNS = [
    re.compile(r"\b" + re.escape(t) + r"\b", re.IGNORECASE)
    if t.isascii()
    else re.compile(re.escape(t))
    for t in _DEMOGRAPHIC_GUARD_TERMS
]


def _tokenize(text: str) -> FrozenSet[str]:
    """Lower-case word tokens from text."""
    return frozenset(re.findall(r"\w+", text.lower()))


def check_grounding(raw_query: str, answer: str, context: str) -> bool:
    """
    Return True if the answer appears grounded in the patient's query.
    Return False if the answer injects demographic terms the patient never stated.

    IMPORTANT — why we check raw_query ONLY (not context) for demographic terms:
    The KB is IMCI-heavy (Integrated Management of *Childhood* Illness), so
    retrieved chunks almost always contain "child". If we allow "child" through
    whenever it appears in context, an adult patient saying "I have vomiting"
    will always receive a paediatric answer — exactly the bug this guards against.

    Clinical safety rule: if the PATIENT did not say "child/infant/paediatric",
    the ANSWER must not say it, regardless of what the retrieved context contains.

    Parameters
    ----------
    raw_query : str
        Verbatim English translation of the patient's symptoms — the clean
        signal of what the patient actually said (no static suffixes added).
    answer : str
        LLM-generated response text.
    context : str
        Concatenated retrieved chunk texts (used only for future heuristics,
        NOT for the demographic guard decision).

    Returns
    -------
    bool
        True  → answer is grounded (safe to serve).
        False → answer introduces ungrounded demographic content → serve fallback.
    """
    if not answer or not answer.strip():
        logger.debug("Grounding check: empty answer — treating as ungrounded.")
        return False

    # Quick pass: if answer IS the sentinel token, caller already handles it
    if answer.strip().upper() == "INSUFFICIENT_MATCH":
        return False

    # Demographic guard: check raw_query ONLY.
    # Do NOT include context here — IMCI chunks always contain "child",
    # and we must not let that bleed through to adult patient answers.
    query_lower = raw_query.lower()

    injections: List[str] = []
    for term, pattern in zip(_DEMOGRAPHIC_GUARD_TERMS, _GUARD_PATTERNS):
        # Term appears in the answer…
        if pattern.search(answer):
            # …but NOT in the patient's own words → demographic injection detected
            if term.lower() not in query_lower:
                injections.append(term)

    if injections:
        logger.warning(
            f"Grounding check FAILED: answer introduces demographic term(s) "
            f"{injections!r} absent from patient query='{raw_query[:80]}'. "
            f"Answer snippet: '{answer[:100]}'"
        )
        return False

    return True
