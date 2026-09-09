"""
Regression test suite for Avera RAG pipeline (Phase 7).

Covers:
  - Adult symptoms that must NOT retrieve pediatric chunks
  - Symptoms absent from KB (must trigger fallback)
  - Symptoms clearly in KB (must retrieve correct chunk + cited answer)
  - Consistency check: answer must not introduce ungrounded demographics

Run:
    pytest avera_rag/eval/test_regression.py -v

Or without pytest:
    python avera_rag/eval/test_regression.py
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.consistency_check import check_grounding

# ─────────────────────────────────────────────────────────────────────────────
# Test data
# ─────────────────────────────────────────────────────────────────────────────

# Each case: query (Tamil/Hindi/etc.), optional flags
TEST_CASES: List[Dict[str, Any]] = [
    # ── Adult symptoms: must NOT produce pediatric output ─────────────────────
    {
        "id": "TC01",
        "desc": "Adult Tamil vomiting -- must not mention child",
        "query": "எனக்கு வாந்தி உள்ளது",
        "language": "ta",
        "forbidden_in_answer": ["child", "children", "infant", "குழந்தை", "pediatric"],
        "expect_fallback": False,
    },
    {
        "id": "TC02",
        "desc": "Adult Tamil headache -- must not mention child",
        "query": "எனக்கு தலைவலி உள்ளது",
        "language": "ta",
        "forbidden_in_answer": ["child", "infant", "குழந்தை"],
        "expect_fallback": False,
    },
    {
        "id": "TC03",
        "desc": "Adult Tamil fever -- must not mention child",
        "query": "எனக்கு காய்ச்சல் உள்ளது",
        "language": "ta",
        "forbidden_in_answer": ["child", "infant", "குழந்தை"],
        "expect_fallback": False,
    },
    {
        "id": "TC04",
        "desc": "Adult Tamil cough -- must not mention child",
        "query": "எனக்கு இருமல் வருகிறது",
        "language": "ta",
        "forbidden_in_answer": ["child", "infant", "குழந்தை"],
        "expect_fallback": False,
    },
    {
        "id": "TC05",
        "desc": "Adult Hindi diarrhea -- must not mention child",
        "query": "मुझे दस्त हो रहे हैं",
        "language": "hi",
        "forbidden_in_answer": ["child", "children", "infant", "बच्चा"],
        "expect_fallback": False,
    },
    {
        "id": "TC06",
        "desc": "Adult Telugu abdominal pain -- must not mention child",
        "query": "నాకు కడుపు నొప్పిగా ఉంది",
        "language": "te",
        "forbidden_in_answer": ["child", "infant", "శిశువు", "పిల్లలు"],
        "expect_fallback": False,
    },
    {
        "id": "TC07",
        "desc": "Adult Kannada breathing difficulty -- must not mention child",
        "query": "ನನಗೆ ಉಸಿರಾಟ ಕಷ್ಟವಾಗುತ್ತಿದೆ",
        "language": "kn",
        "forbidden_in_answer": ["child", "infant", "ಮಗು", "ಮಕ್ಕಳು"],
        "expect_fallback": False,
    },
    # ── Explicit pediatric query: CAN mention child ───────────────────────────
    {
        "id": "TC08",
        "desc": "Explicit child query -- child mention is allowed",
        "query": "குழந்தைக்கு காய்ச்சல் மற்றும் இருமல்",
        "language": "ta",
        "forbidden_in_answer": [],   # child mention is fine here
        "expect_fallback": False,
    },
    # ── Symptoms absent from KB: must trigger fallback ────────────────────────
    {
        "id": "TC09",
        "desc": "Completely unrelated query (airplane repair) -- must fallback",
        "query": "How do I fix an airplane engine carburetor",
        "language": "en",
        "forbidden_in_answer": [],
        "expect_fallback": True,
    },
    {
        "id": "TC10",
        "desc": "Stock market question -- must fallback",
        "query": "What is the current stock price of Infosys",
        "language": "en",
        "forbidden_in_answer": [],
        "expect_fallback": True,
    },
    # ── Symptoms clearly in KB: must retrieve + cite ──────────────────────────
    {
        "id": "TC11",
        "desc": "English fever query -- should get sourced answer",
        "query": "fever symptoms and treatment",
        "language": "en",
        "forbidden_in_answer": [],
        "expect_fallback": False,
        "require_citation": True,   # answer must contain [SOURCE n]
    },
    {
        "id": "TC12",
        "desc": "English diarrhea dehydration query -- should get sourced answer",
        "query": "diarrhea treatment and hydration",
        "language": "en",
        "forbidden_in_answer": [],
        "expect_fallback": False,
        "require_citation": True,
    },
    {
        "id": "TC13",
        "desc": "English cough respiratory query -- should get sourced answer",
        "query": "cough cold respiratory infection",
        "language": "en",
        "forbidden_in_answer": [],
        "expect_fallback": False,
        "require_citation": True,
    },
    # ── Consistency check unit tests (offline, no pipeline needed) ────────────
    {
        "id": "TC14",
        "desc": "Consistency check: answer introduces 'child' not in query -- FAIL",
        "consistency_only": True,
        "raw_query": "vomiting nausea management",
        "answer": "Child with vomiting should be assessed for dehydration [SOURCE 1].",
        "context": "Vomiting in adults: assess fluid loss and electrolyte balance.",
        "expected_grounded": False,
    },
    {
        "id": "TC15",
        "desc": "Consistency check: answer stays on topic -- PASS",
        "consistency_only": True,
        "raw_query": "vomiting nausea management",
        "answer": "Vomiting may indicate gastroenteritis. Monitor fluid intake [SOURCE 1].",
        "context": "Vomiting in adults: assess fluid loss and electrolyte balance.",
        "expected_grounded": True,
    },
    {
        "id": "TC16",
        "desc": "Consistency check: 'child' in context only, NOT in raw_query -- FAIL (patient is adult)",
        "consistency_only": True,
        "raw_query": "vomiting nausea management",   # patient did NOT say child
        "answer": "Child with vomiting: assess hydration status [SOURCE 1].",
        "context": "Fever in children: monitor temperature and hydration.",  # IMCI chunk
        "expected_grounded": False,  # patient didn't say child → must not appear in answer
    },
    {
        "id": "TC17",
        "desc": "Consistency check: INSUFFICIENT_MATCH sentinel -- FAIL",
        "consistency_only": True,
        "raw_query": "airplane engine repair",
        "answer": "INSUFFICIENT_MATCH",
        "context": "Fever symptoms and treatment.",
        "expected_grounded": False,
    },
    {
        "id": "TC18",
        "desc": "Consistency check: 'infant' injected not in query -- FAIL",
        "consistency_only": True,
        "raw_query": "headache causes",
        "answer": "Infant headache is uncommon. For adults, assess for migraine [SOURCE 2].",
        "context": "Headache: common causes include tension, migraine, dehydration.",
        "expected_grounded": False,
    },
    {
        "id": "TC19",
        "desc": "Tamil fallback message must not contain clinical content",
        "consistency_only": True,
        "raw_query": "some unknown symptom",
        "answer": "இந்த அறிகுறிகளுக்கு எனது தரவுத்தளத்தில் குறிப்பிட்ட தகவல் இல்லை. தயவுஸெய்து மருத்துவரை அணுகவும்.",
        "context": "",
        "expected_grounded": True,  # Fallback itself has no demographic injection
    },
    {
        "id": "TC20",
        "desc": "Consistency check: empty answer -- FAIL",
        "consistency_only": True,
        "raw_query": "fever",
        "answer": "",
        "context": "Fever treatment protocols.",
        "expected_grounded": False,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Runners
# ─────────────────────────────────────────────────────────────────────────────

CITATION_PATTERN = re.compile(r"\[SOURCE\s*\d+\]", re.IGNORECASE)


def run_consistency_only(case: Dict) -> Dict[str, Any]:
    """Run offline consistency check tests (no pipeline needed)."""
    result = check_grounding(
        raw_query=case["raw_query"],
        answer=case["answer"],
        context=case["context"],
    )
    passed = (result == case["expected_grounded"])
    return {
        "id": case["id"],
        "desc": case["desc"],
        "passed": passed,
        "expected": case["expected_grounded"],
        "got": result,
        "type": "consistency_unit",
    }


def run_pipeline_case(pipeline, case: Dict) -> Dict[str, Any]:
    """Run a full pipeline query test."""
    from rag.prompts import FALLBACK_MESSAGE_MULTILANG

    result = pipeline.query(
        question=case["query"],
        language=case.get("language", "en"),
    )
    answer = result.answer
    lang = case.get("language", "en")
    fallback_msg = FALLBACK_MESSAGE_MULTILANG.get(lang, FALLBACK_MESSAGE_MULTILANG["en"])

    failures = []

    # Check forbidden terms
    for term in case.get("forbidden_in_answer", []):
        pattern = (
            re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            if term.isascii()
            else re.compile(re.escape(term))
        )
        if pattern.search(answer):
            failures.append(f"Forbidden term found in answer: '{term}'")

    # Fallback expectation
    if case.get("expect_fallback"):
        if answer.strip() not in (fallback_msg.strip(), ""):
            # Lenient: check if it at least contains the fallback or OUT_OF_SCOPE text
            if "insufficient" not in answer.lower() and "scope" not in answer.lower() \
               and "இல்லை" not in answer and "नहीं" not in answer:
                failures.append(f"Expected fallback but got: '{answer[:100]}'")

    # Citation check
    if case.get("require_citation") and not CITATION_PATTERN.search(answer):
        failures.append("Expected [SOURCE n] citation in answer but found none.")

    return {
        "id": case["id"],
        "desc": case["desc"],
        "passed": len(failures) == 0,
        "failures": failures,
        "answer_snippet": answer[:120],
        "type": "pipeline",
    }


# ─────────────────────────────────────────────────────────────────────────────
# pytest-compatible test functions
# ─────────────────────────────────────────────────────────────────────────────

def test_consistency_checks():
    """Runs all offline consistency-only tests."""
    consistency_cases = [c for c in TEST_CASES if c.get("consistency_only")]
    for case in consistency_cases:
        r = run_consistency_only(case)
        assert r["passed"], (
            f"[{r['id']}] {r['desc']}: expected grounded={r['expected']}, got {r['got']}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner (no pytest)
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  Avera RAG Regression Suite  --  offline consistency checks")
    print("=" * 70)

    consistency_cases = [c for c in TEST_CASES if c.get("consistency_only")]
    passed = 0
    failed = 0

    for case in consistency_cases:
        r = run_consistency_only(case)
        status = "PASS" if r["passed"] else "FAIL"
        symbol = "[+]" if r["passed"] else "[-]"
        print(f"  [{r['id']}] {symbol} {status}  {r['desc']}")
        if not r["passed"]:
            print(f"         Expected grounded={r['expected']}, got {r['got']}")
            failed += 1
        else:
            passed += 1

    print("=" * 70)
    print(f"  Results: {passed} passed, {failed} failed  (consistency-only tests)")

    if failed:
        print("  SOME TESTS FAILED. Run full pipeline tests with a live pipeline.\n")
        sys.exit(1)
    else:
        print("  All offline consistency tests passed.\n")
        print("  NOTE: Pipeline tests (TC01-TC13) require a running RAGPipeline.")
        print("  Run: python main.py --mode eval   to test the full pipeline.\n")


if __name__ == "__main__":
    main()
