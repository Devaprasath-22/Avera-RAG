"""
Avera Medical RAG — CLI entry point.

Modes:
  ingest   — Ingest documents into the vector store
  serve    — Start the FastAPI server
  eval     — Run the benchmark evaluation suite
  voice    — Voice-in → RAG → voice-out interactive loop
  query    — Single text query from the command line

Usage:
  python main.py --mode ingest --path /path/to/docs
  python main.py --mode ingest --path data/imci_chart_booklet.jsonl
  python main.py --mode serve
  python main.py --mode eval
  python main.py --mode query --question "What are general danger signs?"
  python main.py --mode voice
"""

import argparse
import sys
import os

# On Windows, ensure UTF-8 output for Indian language scripts & load CUDA DLLs
if os.name == "nt":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    try:
        import site
        for pkg_dir in site.getsitepackages():
            torch_lib = os.path.join(pkg_dir, "torch", "lib")
            if os.path.exists(torch_lib):
                os.add_dll_directory(torch_lib)
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore")

import logging
from pathlib import Path

import yaml
from rich.console import Console
from rich.logging import RichHandler

console = Console()


# ── Logging setup ─────────────────────────────────────────────────────────────

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
    )
    # Quieten noisy third-party loggers
    for lib in ("chromadb", "sentence_transformers", "transformers", "urllib3", "httpx"):
        logging.getLogger(lib).setLevel(logging.WARNING)


# ── Config loading ────────────────────────────────────────────────────────────

def _load_config(config_path: str = None) -> dict:
    if config_path:
        path = Path(config_path)
    else:
        path = Path(__file__).parent / "config.yaml"
    if not path.exists():
        console.print(f"[red]Config not found: {path}[/red]")
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


# ── Mode handlers ─────────────────────────────────────────────────────────────

def mode_ingest(config: dict, path: str, force: bool) -> None:
    from ingestion.pipeline import IngestionPipeline

    pipeline = IngestionPipeline(config)
    target = Path(path)

    if not target.exists():
        console.print(f"[red]Path not found: {path}[/red]")
        sys.exit(1)

    if target.is_dir():
        summary = pipeline.ingest_directory(target, force=force)
    else:
        n = pipeline.ingest_file(target, force=force)
        summary = {"chunks_added": n, "collection_size": pipeline.store.count()}

    console.print(f"\n[bold green]Ingestion summary:[/bold green] {summary}")


def mode_serve(config: dict) -> None:
    import uvicorn

    api_cfg = config.get("api", {})
    host = api_cfg.get("host", "0.0.0.0")
    port = api_cfg.get("port", 8000)

    console.print(f"\n[bold cyan]Starting Avera RAG API on http://{host}:{port}[/bold cyan]")
    console.print("[dim]Press Ctrl+C to stop.[/dim]\n")

    uvicorn.run(
        "api.server:app",
        host=host,
        port=port,
        workers=1,  # MUST be 1 — models are not thread-safe
        log_level="info",
    )


def mode_eval(config: dict, qa_path: str, k: int) -> None:
    from eval.benchmarks import run_eval
    from rag.pipeline import RAGPipeline

    pipeline = RAGPipeline(config)
    run_eval(pipeline, qa_pairs_path=qa_path, k=k)


SUPPORTED_LANGUAGES = {
    "1": {"code": "en", "voice_tag": "Voice:en", "name": "English", "native": "English"},
    "2": {"code": "ta", "voice_tag": "Voice:ta", "name": "Tamil", "native": "தமிழ்"},
    "3": {"code": "hi", "voice_tag": "Voice:hi", "name": "Hindi", "native": "हिन्दी"},
    "4": {"code": "te", "voice_tag": "Voice:te", "name": "Telugu", "native": "తెలుగు"},
    "5": {"code": "kn", "voice_tag": "Voice:ka", "name": "Kannada", "native": "ಕನ್ನಡ"},
}


def resolve_language(lang_input: str | None) -> dict:
    """
    Resolve user input (e.g. '1', 'Voice:en', 'voice:ta', 'Voice:hi', 'Voice:te', 'Voice:ka',
    'en', 'ta', 'hi', 'te', 'ka', 'kn', 'tamil') to a canonical language dict.
    """
    if not lang_input:
        return SUPPORTED_LANGUAGES["1"]

    cleaned = str(lang_input).strip().lower()
    if cleaned in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[cleaned]

    # Strip common prefixes like 'voice:', 'voice :', 'voice-', 'voice_'
    if cleaned.startswith("voice:"):
        cleaned = cleaned[6:].strip()
    elif cleaned.startswith("voice :"):
        cleaned = cleaned[7:].strip()
    elif cleaned.startswith("voice-") or cleaned.startswith("voice_"):
        cleaned = cleaned[6:].strip()

    alias_map = {
        "en": "1", "english": "1",
        "ta": "2", "tamil": "2", "தமிழ்": "2",
        "hi": "3", "hindi": "3", "हिन्दी": "3",
        "te": "4", "telugu": "4", "తెలుగు": "4",
        "kn": "5", "ka": "5", "kannada": "5", "ಕನ್ನಡ": "5",
    }
    key = alias_map.get(cleaned)
    if key and key in SUPPORTED_LANGUAGES:
        return SUPPORTED_LANGUAGES[key]

    return SUPPORTED_LANGUAGES["1"]


def mode_query(config: dict, question: str, language: str = "en") -> None:
    from rag.pipeline import RAGPipeline
    from rich.panel import Panel
    from rich.markdown import Markdown

    lang_info = resolve_language(language)
    selected_lang = lang_info["code"]
    lang_native = lang_info["native"]
    voice_tag = lang_info["voice_tag"]

    pipeline = RAGPipeline(config)
    console.print(f"\n[bold]Query ([cyan]{voice_tag}[/cyan] — {lang_native}):[/bold] {question}\n")

    result = pipeline.query(question, keep_llm_loaded=False, language=selected_lang)

    console.print(Panel(Markdown(result.answer), title=f"[bold green]Answer ({voice_tag})[/bold green]"))
    console.print(f"\n[dim]Sources:[/dim]")
    for i, s in enumerate(result.sources, 1):
        console.print(f"  [{i}] {s['source']} p.{s['page']}  (score: {s['score']})")
    console.print(f"\n[dim]Latency: {result.latency}[/dim]")


def mode_voice(config: dict, language: str = "") -> None:
    """Interactive voice loop: mic → ASR → RAG → TTS → speaker."""
    from voice.asr import ASRBackend
    from voice.tts import TTSBackend
    from rag.pipeline import RAGPipeline
    from rich.panel import Panel

    if not language:
        console.print("\n[bold cyan]choose Avera Available lang:[/bold cyan] [bold green]Voice:en[/bold green], [bold green]voice:ta[/bold green], [bold green]Voice:hi[/bold green], [bold green]Voice:te[/bold green], [bold green]Voice:ka[/bold green]\n")
        console.print("  [bold white][1][/bold white] [bold green]Voice:en[/bold green] — English")
        console.print("  [bold white][2][/bold white] [bold green]Voice:ta[/bold green] — Tamil (தமிழ்)")
        console.print("  [bold white][3][/bold white] [bold green]Voice:hi[/bold green] — Hindi (हिन्दी)")
        console.print("  [bold white][4][/bold white] [bold green]Voice:te[/bold green] — Telugu (తెలుగు)")
        console.print("  [bold white][5][/bold white] [bold green]Voice:ka[/bold green] — Kannada (ಕನ್ನಡ)")
        try:
            choice = input("\nchoose Avera Available lang (Voice:en, voice:ta, Voice:hi, Voice:te, Voice:ka or 1-5) [default Voice:en]: ").strip()
        except EOFError:
            choice = "1"
        lang_info = resolve_language(choice)
    else:
        lang_info = resolve_language(language)

    selected_lang = lang_info["code"]
    lang_native = lang_info["native"]
    voice_tag = lang_info["voice_tag"]

    console.print(f"\n[bold green]✓ Active Voice Language:[/bold green] [bold cyan]{voice_tag} — {lang_native} ({selected_lang})[/bold cyan]\n")

    asr = ASRBackend(config)
    asr.load()
    tts = TTSBackend(config)
    pipeline = RAGPipeline(config)

    console.print(
        f"\n[bold cyan]Voice mode active.[/bold cyan] "
        f"[dim]Speak in {lang_native} ({voice_tag}). Type or say another 'Voice:xx' to switch language. Say 'exit' or 'quit' to stop.[/dim]\n"
    )

    while True:
        try:
            console.print(f"[bold yellow]● Listening ({voice_tag} — {lang_native})…[/bold yellow]")
            transcript = asr.transcribe_mic(language=selected_lang)
            question = transcript["text"].strip()
            query_en = transcript.get("query_en", None)

            if not question:
                console.print("[dim]No speech detected. Try again.[/dim]")
                continue

            console.print(f"[bold]You ({voice_tag} — {lang_native}):[/bold] {question}")

            q_clean = question.lower().strip(" .?!")
            if q_clean in ("exit", "quit", "stop", "முடி", "நில்", "நிறுத்து", "बंद", "रुको", "चुप", "ఆపు", "ఆగు", "ನಿಲ್ಲಿಸು", "ನಿಲ್ಲು"):
                farewell = {
                    "ta": "வணக்கம். பாதுகாப்பாக இருங்கள்.",
                    "hi": "नमस्ते। सुरक्षित रहें।",
                    "te": "నమస్కారం. జాగ్రత్తగా ఉండండి.",
                    "kn": "ನಮಸ್ಕಾರ. ಸುರಕ್ಷಿತವಾಗಿರಿ.",
                    "en": "Goodbye. Stay safe.",
                }.get(selected_lang, "Goodbye. Stay safe.")
                console.print(f"[bold cyan]{farewell}[/bold cyan]")
                tts.speak(farewell, language=selected_lang)
                break

            # On-the-fly language switch support: e.g. "Voice:ta", "voice:hi", "voice:ka", "switch to tamil"
            if any(q_clean.startswith(prefix) for prefix in ("voice:", "voice ", "lang:", "switch to ", "language:")) or q_clean in ("en", "ta", "hi", "te", "ka", "kn"):
                target_str = q_clean.replace("switch to", "").replace("language:", "").strip()
                new_lang_info = resolve_language(target_str)
                selected_lang = new_lang_info["code"]
                lang_native = new_lang_info["native"]
                voice_tag = new_lang_info["voice_tag"]
                switch_msg = {
                    "ta": f"மொழி மாற்றப்பட்டது: தமிழ் ({voice_tag})",
                    "hi": f"भाषा बदल दी गई: हिन्दी ({voice_tag})",
                    "te": f"భాష మార్చబడింది: తెలుగు ({voice_tag})",
                    "kn": f"ಭಾಷೆ ಬದಲಾಯಿಸಲಾಗಿದೆ: ಕನ್ನಡ ({voice_tag})",
                    "en": f"Language switched to: English ({voice_tag})",
                }.get(selected_lang, f"Language switched to: {lang_native} ({voice_tag})")
                console.print(f"\n[bold green]✓ {switch_msg}[/bold green]\n")
                tts.speak(switch_msg, language=selected_lang)
                continue

            console.print("[bold yellow]⟳ Processing…[/bold yellow]")
            result = pipeline.query(question, keep_llm_loaded=True, language=selected_lang, query_en=query_en)

            console.print(Panel(result.answer, title=f"[bold green]Avera ({voice_tag} — {lang_native})[/bold green]"))
            console.print(f"[dim]Latency: {result.latency.get('total_ms','?')}ms[/dim]")

            console.print("[dim]🔊 Speaking… [italic](press Space, Enter, or Ctrl+C to stop)[/italic][/dim]")
            try:
                stopped = tts.speak_streaming(result.answer, language=selected_lang)
                if stopped:
                    console.print("\n[bold yellow]⏹ Speech stopped by user.[/bold yellow]\n")
            except KeyboardInterrupt:
                tts.stop()
                console.print("\n[bold yellow]⏹ Speech stopped by user.[/bold yellow]\n")

        except KeyboardInterrupt:
            console.print("\n[bold cyan]Interrupted. Goodbye.[/bold cyan]")
            break
        except Exception as e:
            logging.getLogger(__name__).error(f"Voice loop error: {e}", exc_info=True)
            err_msg = {
                "ta": "மன்னிக்கவும், ஒரு பிழை ஏற்பட்டது. மீண்டும் முயற்சிக்கவும்.",
                "hi": "क्षमा करें, कोई त्रुटि हुई। कृपया पुनः प्रयास करें।",
                "te": "క్షమించండి, లోపం సంభవించింది. దయచేసి మళ్లీ ప్రయత్నించండి.",
                "kn": "ಕ್ಷಮಿಸಿ, ದೋಷ ಸಂಭವಿಸಿದೆ. దಯವಿಟ್ಟು ಮತ್ತೆ ಪ್ರಯತ್ನಿಸಿ.",
                "en": "Sorry, I encountered an error. Please try again.",
            }.get(selected_lang, "Sorry, I encountered an error. Please try again.")
            try:
                tts.speak(err_msg, language=selected_lang)
            except Exception:
                pass

    pipeline.manager.unload_all()


# ── Argument parsing ──────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Avera Medical RAG — offline medical AI assistant for Jetson Orin Nano",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        choices=["ingest", "serve", "eval", "voice", "query"],
        required=True,
        help="Operating mode",
    )
    parser.add_argument(
        "--path",
        default=".",
        help="[ingest] Path to file or directory to ingest",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="[ingest] Re-ingest even if document is already indexed",
    )
    parser.add_argument(
        "--question", "-q",
        default="",
        help="[query] Question to ask the RAG pipeline",
    )
    parser.add_argument(
        "--language", "-l",
        default="",
        help="Preferred language: Voice:en, voice:ta, Voice:hi, Voice:te, Voice:ka (or en, ta, hi, te, kn/ka)",
    )
    parser.add_argument(
        "--qa-pairs",
        default=str(Path(__file__).parent / "eval" / "qa_pairs.jsonl"),
        help="[eval] Path to JSONL evaluation pairs file",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="[eval] Top-k for Recall@k evaluation (default: 5)",
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to config.yaml (default: ./config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)

    # Load config (use --config override if provided)
    config_path = Path(args.config)
    if not config_path.exists():
        console.print(f"[red]Config not found: {config_path}[/red]")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    console.print(
        f"\n[bold]Avera Medical RAG[/bold]  |  mode=[cyan]{args.mode}[/cyan]\n"
    )

    if args.mode == "ingest":
        mode_ingest(config, args.path, args.force)
    elif args.mode == "serve":
        mode_serve(config)
    elif args.mode == "eval":
        mode_eval(config, args.qa_pairs, args.k)
    elif args.mode == "query":
        if not args.question:
            console.print("[red]--question is required for query mode[/red]")
            sys.exit(1)
        mode_query(config, args.question, language=args.language or "en")
    elif args.mode == "voice":
        mode_voice(config, language=args.language)


if __name__ == "__main__":
    main()
