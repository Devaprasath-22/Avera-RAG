"""
Heading-aware document chunker for medical text.

Strategy (in order of preference):
  1. Split on heading boundaries (Markdown ##, ALL-CAPS, bold, numbered)
  2. Within each section, split on blank-line paragraph boundaries
  3. Within an oversized paragraph, split on sentence boundaries
  4. Apply token-overlap between adjacent chunks to avoid cutting dosage
     qualifications from their context.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
# Approximation: English medical text averages ~4 chars/token
CHARS_PER_TOKEN: int = 4


# ── Heading detection patterns ────────────────────────────────────────────────
_HEADING_PATTERNS: list[re.Pattern] = [
    re.compile(r"^#{1,4}\s+.+$", re.MULTILINE),            # ## Markdown heading
    re.compile(r"^[A-Z][A-Z\s\-]{5,}$", re.MULTILINE),     # ALL CAPS heading
    re.compile(r"^\*\*[^*\n]{3,}\*\*\s*$", re.MULTILINE),  # **Bold heading**
    re.compile(r"^\d{1,2}\.\s+[A-Z].{4,}$", re.MULTILINE), # "1. Section Name"
    re.compile(r"^[A-Z][^.!?\n]{4,}:\s*$", re.MULTILINE),  # "Label:"
]

# ── Sentence boundary ─────────────────────────────────────────────────────────
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return len(self.text) // CHARS_PER_TOKEN

    def __repr__(self) -> str:
        return f"Chunk(tokens≈{self.token_estimate}, source={self.metadata.get('source','?')})"


class MedicalChunker:
    """
    Heading-aware chunker targeting 500-700 tokens per chunk with 80-token overlap.

    Parameters
    ----------
    chunk_size_tokens : int
        Target maximum tokens per chunk (default 600).
    overlap_tokens : int
        Overlap between adjacent chunks in tokens (default 80).
    """

    def __init__(
        self,
        chunk_size_tokens: int = 600,
        overlap_tokens: int = 80,
    ) -> None:
        self.max_chars = chunk_size_tokens * CHARS_PER_TOKEN
        self.overlap_chars = overlap_tokens * CHARS_PER_TOKEN

    # ── Public API ────────────────────────────────────────────────────────────

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Chunk a single document record.

        Returns a list of Chunk objects with updated metadata including
        `chunk_index` and `section` fields.
        """
        text = text.strip()
        if not text:
            return []

        # Fast path — whole text fits
        if len(text) <= self.max_chars:
            return [Chunk(text=text, metadata={**metadata, "chunk_index": 0, "section": ""})]

        # Split into (section_text, heading) pairs
        sections = self._split_by_headings(text)

        chunks: List[Chunk] = []
        for section_text, heading in sections:
            section_meta = {**metadata, "section": heading}
            sub = self._split_section(section_text.strip(), section_meta)
            chunks.extend(sub)

        # Assign global chunk index
        for i, c in enumerate(chunks):
            c.metadata = {**c.metadata, "chunk_index": i}

        return chunks

    # ── Private helpers ───────────────────────────────────────────────────────

    def _split_by_headings(self, text: str) -> List[tuple[str, str]]:
        """
        Detect heading positions and split text into (section_text, heading) tuples.
        Headings are included at the start of their own section.
        """
        positions: list[tuple[int, int, str]] = []
        for pattern in _HEADING_PATTERNS:
            for m in pattern.finditer(text):
                positions.append((m.start(), m.end(), m.group().strip()))

        if not positions:
            return [(text, "")]

        # Sort and de-duplicate overlapping matches
        positions.sort(key=lambda x: x[0])
        deduped: list[tuple[int, int, str]] = []
        last_end = -1
        for start, end, heading in positions:
            if start >= last_end:
                deduped.append((start, end, heading))
                last_end = end

        # Build sections
        sections: List[tuple[str, str]] = []
        prev_start = 0
        prev_heading = ""

        for start, end, heading in deduped:
            section_text = text[prev_start:start].strip()
            if section_text:
                sections.append((section_text, prev_heading))
            prev_start = start
            prev_heading = heading

        # Last section
        last = text[prev_start:].strip()
        if last:
            sections.append((last, prev_heading))

        return sections if sections else [(text, "")]

    def _split_section(self, text: str, metadata: Dict) -> List[Chunk]:
        """Recursively split an oversized section."""
        if len(text) <= self.max_chars:
            return [Chunk(text=text, metadata=metadata)]

        paragraphs = [p.strip() for p in re.split(r"\n\n+", text) if p.strip()]

        if len(paragraphs) <= 1:
            # Single huge paragraph — use sentence splitting
            return self._split_by_sentences(text, metadata)

        return self._accumulate(paragraphs, separator="\n\n", metadata=metadata)

    def _split_by_sentences(self, text: str, metadata: Dict) -> List[Chunk]:
        """Split on sentence boundaries as a last resort."""
        sentences = _SENTENCE_SPLIT.split(text)
        return self._accumulate(sentences, separator=" ", metadata=metadata)

    def _accumulate(
        self,
        parts: List[str],
        separator: str,
        metadata: Dict,
    ) -> List[Chunk]:
        """
        Accumulate parts into chunks up to max_chars,
        carrying overlap_chars of the previous chunk into the next.
        """
        chunks: List[Chunk] = []
        current: List[str] = []
        current_len = 0

        for part in parts:
            part_len = len(part) + len(separator)

            # Flush if adding this part would overflow
            if current_len + part_len > self.max_chars and current:
                chunk_text = separator.join(current)
                chunks.append(Chunk(text=chunk_text, metadata=metadata))

                # Carry overlap — keep trailing parts that fit within overlap budget
                overlap: List[str] = []
                overlap_len = 0
                for p in reversed(current):
                    cost = len(p) + len(separator)
                    if overlap_len + cost > self.overlap_chars:
                        break
                    overlap.insert(0, p)
                    overlap_len += cost
                current = overlap
                current_len = overlap_len

            current.append(part)
            current_len += part_len

        if current:
            chunks.append(Chunk(text=separator.join(current), metadata=metadata))

        return chunks
