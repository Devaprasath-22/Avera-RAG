"""
Benchmarking and evaluation suite for Avera RAG.

Metrics implemented:
  - recall_at_k       : does the gold chunk appear in top-k retrieved results?
  - answer_latency    : per-stage and end-to-end latency profiling
  - faithfulness      : heuristic check that cited sources cover answer claims
  - collection_stats  : vector DB health stats

Usage:
    python main.py --mode eval
    # or directly:
    from eval.benchmarks import run_eval
    run_eval(pipeline, qa_pairs_path="eval/qa_pairs.jsonl", k=5)
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_qa_pairs(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load Q&A evaluation pairs from a JSONL file.

    Expected format per line:
        {
            "question": "...",
            "answer": "...",          # reference answer (for human review)
            "gold_source": "...",     # source name that should appear in retrieved chunks
            "gold_text_fragment": "..." # substring that should appear in a retrieved chunk
        }
    """
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping malformed QA pair on line {i}: {e}")
    return pairs


# ── Recall@k ──────────────────────────────────────────────────────────────────

def recall_at_k(
    pipeline,  # RAGPipeline
    qa_pairs: List[Dict],
    k: int = 5,
) -> Dict[str, Any]:
    """
    Evaluate retrieval quality: does the gold chunk appear in the top-k results?

    A result is a "hit" if either:
    - The chunk's `source` metadata contains `gold_source`, OR
    - The chunk's text contains `gold_text_fragment` (case-insensitive)

    Returns a dict with hit_rate, total, hits, and per-question details.
    """
    hits = 0
    details = []

    for i, qa in enumerate(qa_pairs):
        question = qa.get("question", "")
        gold_source = qa.get("gold_source", "").lower()
        gold_fragment = qa.get("gold_text_fragment", "").lower()

        if not question:
            continue

        # Retrieve (bypass reranker for pure retrieval eval)
        query_emb = pipeline.embedder.embed_query(question)
        raw_results = pipeline.store.query(query_emb, top_k=k)

        hit = False
        for r in raw_results:
            source = r.get("metadata", {}).get("source", "").lower()
            text = r.get("text", "").lower()
            if (gold_source and gold_source in source) or \
               (gold_fragment and gold_fragment in text):
                hit = True
                break

        hits += int(hit)
        details.append({
            "question": question[:80],
            "hit": hit,
            "gold_source": gold_source,
        })

    total = len(details)
    hit_rate = hits / total if total else 0.0

    return {
        "recall_at_k": round(hit_rate, 3),
        "k": k,
        "hits": hits,
        "total": total,
        "details": details,
    }


# ── Latency profiling ─────────────────────────────────────────────────────────

def latency_profile(
    pipeline,  # RAGPipeline
    questions: List[str],
    include_image: bool = False,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run N text queries through the full pipeline and report per-stage latencies.

    Returns p50/p95/min/max for each pipeline stage.
    """
    stage_times: Dict[str, List[float]] = {}

    for q in questions:
        result = pipeline.query(question=q)
        for stage, ms in result.latency.items():
            stage_times.setdefault(stage, []).append(ms)

    def percentile(data, p):
        s = sorted(data)
        idx = int(len(s) * p / 100)
        return s[min(idx, len(s) - 1)]

    summary = {}
    for stage, times in stage_times.items():
        summary[stage] = {
            "p50_ms": round(percentile(times, 50)),
            "p95_ms": round(percentile(times, 95)),
            "min_ms": round(min(times)),
            "max_ms": round(max(times)),
            "n": len(times),
        }

    return summary


# ── Faithfulness check ────────────────────────────────────────────────────────

def faithfulness_check(answers: List[str], sources_list: List[List[Dict]]) -> Dict[str, Any]:
    """
    Heuristic faithfulness check:
    What % of answers contain at least one [SOURCE n] citation?

    This is a weak signal — it checks citation *presence*, not *accuracy*.
    For full faithfulness checking, use an LLM-as-judge during development.
    """
    import re
    pattern = re.compile(r"\[SOURCE\s*\d+\]", re.IGNORECASE)

    cited = sum(1 for a in answers if pattern.search(a))
    total = len(answers)

    return {
        "faithfulness_proxy": round(cited / total, 3) if total else 0.0,
        "cited": cited,
        "total": total,
        "note": "Heuristic: % of answers containing at least one [SOURCE n] citation.",
    }


# ── Full eval run ─────────────────────────────────────────────────────────────

def run_eval(
    pipeline,  # RAGPipeline
    qa_pairs_path: str | Path,
    k: int = 5,
) -> None:
    """
    Run the full evaluation suite and print a rich results table.
    """
    pairs = load_qa_pairs(qa_pairs_path)
    if not pairs:
        console.print("[red]No Q&A pairs found. Cannot run eval.[/red]")
        return

    console.print(f"\n[bold cyan]Running evaluation on {len(pairs)} Q&A pairs (k={k})…[/bold cyan]")

    # ── Recall@k ──────────────────────────────────────────────────────────────
    recall_result = recall_at_k(pipeline, pairs, k=k)

    # ── Latency (first 5 questions only to keep eval fast) ───────────────────
    questions = [p["question"] for p in pairs[:5]]
    latency_result = latency_profile(pipeline, questions)

    # ── Quick RAG answers for faithfulness ───────────────────────────────────
    answers = []
    sources_all = []
    for p in pairs[:5]:
        r = pipeline.query(p["question"])
        answers.append(r.answer)
        sources_all.append(r.sources)

    faith_result = faithfulness_check(answers, sources_all)

    # ── Print results ─────────────────────────────────────────────────────────
    table = Table(title="Avera RAG Evaluation Results", show_lines=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    table.add_row(f"Recall@{k}", f"{recall_result['recall_at_k']:.1%}  ({recall_result['hits']}/{recall_result['total']} hits)")
    table.add_row("Faithfulness (citation proxy)", f"{faith_result['faithfulness_proxy']:.1%}")

    if "total_ms" in latency_result:
        lat = latency_result["total_ms"]
        table.add_row("Latency p50 (total)", f"{lat['p50_ms']}ms")
        table.add_row("Latency p95 (total)", f"{lat['p95_ms']}ms")

    console.print(table)

    # Per-stage latency breakdown
    lat_table = Table(title="Per-Stage Latency (p50 / p95)", show_lines=True)
    lat_table.add_column("Stage")
    lat_table.add_column("p50 (ms)")
    lat_table.add_column("p95 (ms)")
    for stage, stats in latency_result.items():
        lat_table.add_row(stage, str(stats["p50_ms"]), str(stats["p95_ms"]))
    console.print(lat_table)

    # Per-question recall detail
    detail_table = Table(title=f"Recall@{k} Per Question", show_lines=True)
    detail_table.add_column("Q#")
    detail_table.add_column("Hit", style="green")
    detail_table.add_column("Question")
    for i, d in enumerate(recall_result["details"], 1):
        hit_str = "✓" if d["hit"] else "✗"
        hit_style = "green" if d["hit"] else "red"
        detail_table.add_row(str(i), f"[{hit_style}]{hit_str}[/{hit_style}]", d["question"])
    console.print(detail_table)
