"""
Document loaders for different file formats.
Each loader is an iterator that yields: {"text": str, "metadata": dict}

Supported formats:
- JSONL  — IMCI-style {source, page, text} records
- XML    — MedlinePlus health-topic XML
- PDF    — PyMuPDF (fitz), preserves page numbers
- DOCX   — python-docx
- TXT/MD — plain text
"""

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterator

logger = logging.getLogger(__name__)

Record = Dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# JSONL Loader
# ──────────────────────────────────────────────────────────────────────────────

class JSONLLoader:
    """Load IMCI-style JSONL files where each line is {source, page, text}."""

    def load(self, path: str | Path) -> Iterator[Record]:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    text = record.get("text", "").strip()
                    if not text:
                        continue
                    yield {
                        "text": text,
                        "metadata": {
                            "source": record.get("source", path.stem),
                            "page": record.get("page", i + 1),
                            "doc_type": "jsonl",
                            "file_path": str(path),
                        },
                    }
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSONL line {i} in {path.name}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# XML Loader  (MedlinePlus format)
# ──────────────────────────────────────────────────────────────────────────────

class XMLLoader:
    """
    Load MedlinePlus-style XML.
    Extracts <health-topic title="..."> elements with their <full-summary> text.
    Falls back to generic text extraction if structure is not recognised.
    """

    def load(self, path: str | Path) -> Iterator[Record]:
        path = Path(path)
        logger.info(f"Parsing XML: {path.name} ({path.stat().st_size // 1024} KB)")

        try:
            # Use iterparse for memory efficiency on large files
            context = ET.iterparse(str(path), events=("start", "end"))
            for event, elem in context:
                if event == "end" and elem.tag in ("health-topic", "{http://www.nlm.nih.gov/medlineplus/topic}health-topic"):
                    record = self._parse_topic(elem, path)
                    if record:
                        yield record
                    elem.clear()  # free memory
        except ET.ParseError:
            logger.warning(f"Standard XML parse failed on {path.name}; trying lxml...")
            yield from self._parse_with_lxml(path)

    def _parse_topic(self, elem: ET.Element, path: Path) -> Record | None:
        title = elem.get("title", "")
        url = elem.get("url", "")

        # Skip Spanish MedlinePlus topics — their URL contains '/spanish/'
        # or the element has language="Spanish". Keeping only English topics
        # ensures the embedding space is not polluted by Spanish topics that
        # would otherwise push relevant English chunks out of the cosine top-k.
        lang_attr = elem.get("language", "").lower()
        if "/spanish/" in url.lower() or lang_attr == "spanish":
            return None

        parts = []
        if title:
            parts.append(f"# {title}")

        # full-summary (strip HTML-like tags)
        for tag in ("full-summary", "{http://www.nlm.nih.gov/medlineplus/topic}full-summary"):
            fs = elem.find(tag)
            if fs is not None:
                text = "".join(fs.itertext()).strip()
                if text:
                    parts.append(text)
                break

        # also-called / synonyms
        for tag in ("also-called", "{http://www.nlm.nih.gov/medlineplus/topic}also-called"):
            for ac in elem.findall(tag):
                if ac.text and ac.text.strip():
                    parts.append(f"Also called: {ac.text.strip()}")

        if not parts:
            return None

        return {
            "text": "\n\n".join(parts),
            "metadata": {
                "source": f"MedlinePlus: {title}",
                "title": title,
                "url": url,
                "page": 1,
                "doc_type": "xml",
                "lang": "en",
                "file_path": str(path),
            },
        }

    def _parse_with_lxml(self, path: Path) -> Iterator[Record]:
        try:
            from lxml import etree
        except ImportError:
            logger.error("lxml not installed — cannot parse malformed XML.")
            return

        tree = etree.parse(str(path), etree.XMLParser(recover=True))
        for elem in tree.findall("//{*}health-topic"):
            title = elem.get("title", "Unknown")
            texts = list(elem.itertext())
            combined = " ".join(t.strip() for t in texts if t.strip())
            if combined:
                yield {
                    "text": f"# {title}\n\n{combined}",
                    "metadata": {
                        "source": f"MedlinePlus: {title}",
                        "title": title,
                        "page": 1,
                        "doc_type": "xml",
                        "file_path": str(path),
                    },
                }


# ──────────────────────────────────────────────────────────────────────────────
# PDF Loader
# ──────────────────────────────────────────────────────────────────────────────

class PDFLoader:
    """Load PDFs via PyMuPDF (fitz), preserving per-page metadata."""

    def load(self, path: str | Path) -> Iterator[Record]:
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF not installed. Run: pip install PyMuPDF")

        path = Path(path)
        doc = fitz.open(str(path))
        logger.info(f"Loading PDF: {path.name} ({len(doc)} pages)")

        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if text:
                yield {
                    "text": text,
                    "metadata": {
                        "source": path.name,
                        "page": page_num,
                        "doc_type": "pdf",
                        "file_path": str(path),
                    },
                }
        doc.close()


# ──────────────────────────────────────────────────────────────────────────────
# DOCX Loader
# ──────────────────────────────────────────────────────────────────────────────

class DOCXLoader:
    """Load DOCX files, grouping paragraphs into page-sized chunks."""

    PARAS_PER_PAGE = 20  # approximate

    def load(self, path: str | Path) -> Iterator[Record]:
        try:
            from docx import Document
        except ImportError:
            raise ImportError("python-docx not installed. Run: pip install python-docx")

        path = Path(path)
        doc = Document(str(path))

        page_buffer: list[str] = []
        page_num = 1

        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                page_buffer.append(text)

            if len(page_buffer) >= self.PARAS_PER_PAGE:
                yield {
                    "text": "\n".join(page_buffer),
                    "metadata": {
                        "source": path.name,
                        "page": page_num,
                        "doc_type": "docx",
                        "file_path": str(path),
                    },
                }
                page_buffer = []
                page_num += 1

        if page_buffer:
            yield {
                "text": "\n".join(page_buffer),
                "metadata": {
                    "source": path.name,
                    "page": page_num,
                    "doc_type": "docx",
                    "file_path": str(path),
                },
            }


# ──────────────────────────────────────────────────────────────────────────────
# TXT / Markdown Loader
# ──────────────────────────────────────────────────────────────────────────────

class TXTLoader:
    """Load plain text or Markdown files as a single record."""

    def load(self, path: str | Path) -> Iterator[Record]:
        path = Path(path)
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            yield {
                "text": text,
                "metadata": {
                    "source": path.name,
                    "page": 1,
                    "doc_type": "txt",
                    "file_path": str(path),
                },
            }


# ──────────────────────────────────────────────────────────────────────────────
# Router
# ──────────────────────────────────────────────────────────────────────────────

_LOADER_MAP = {
    ".jsonl": JSONLLoader,
    ".xml": XMLLoader,
    ".pdf": PDFLoader,
    ".docx": DOCXLoader,
    ".txt": TXTLoader,
    ".md": TXTLoader,
}

SUPPORTED_EXTENSIONS = list(_LOADER_MAP.keys())


def get_loader(path: str | Path):
    """Return the appropriate loader instance for a file's extension."""
    ext = Path(path).suffix.lower()
    cls = _LOADER_MAP.get(ext)
    if cls is None:
        raise ValueError(
            f"No loader for extension '{ext}'. Supported: {SUPPORTED_EXTENSIONS}"
        )
    return cls()
