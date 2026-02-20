"""
Glix Pipeline Orchestrator
===========================
Wires all 9 agents together in sequence, manages logging, context snapshots,
and generates documentation after a successful run.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import config
from agents import (
    AudioExtractionAgent,
    TranscriptionAgent,
    SlangAnalysisAgent,
    ContextAnalysisAgent,
    TranslationAgent,
    TimingSyncAgent,
    RTLFormattingAgent,
    SRTExportAgent,
    CSVExportAgent,
)


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(log_file_path: str) -> logging.Logger:
    """
    Configure dual logging: console (stdout) + file (output/pipeline.log).

    Format: HH:MM:SS | glix.AgentName | LEVEL | message
    """
    root_logger = logging.getLogger("glix")
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    # Console handler — real-time progress
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler — persistent record
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    return root_logger


# ---------------------------------------------------------------------------
# Prerequisite validation
# ---------------------------------------------------------------------------

def validate_prerequisites() -> None:
    """
    Ensure the video file exists and the Anthropic API key is configured.
    Exits with code 1 on failure.
    """
    if not os.path.isfile(config.VIDEO_PATH):
        print(
            f"ERROR: Video file not found at:\n"
            f"  {config.VIDEO_PATH}\n"
            f"Please place the video in the project root and try again."
        )
        sys.exit(1)

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ERROR: ANTHROPIC_API_KEY is not set in the environment.\n"
            "Export it before running the pipeline:\n"
            '  export ANTHROPIC_API_KEY="sk-ant-..."'
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# Directory creation
# ---------------------------------------------------------------------------

def create_directories() -> None:
    """Create output/ and output/snapshots/ directories if they don't exist."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.OUTPUT_DIR, "snapshots"), exist_ok=True)


# ---------------------------------------------------------------------------
# Context snapshot
# ---------------------------------------------------------------------------

def save_snapshot(context: Dict[str, Any], stage_name: str) -> None:
    """
    Persist a JSON snapshot of the pipeline context after each agent.

    Saved to output/snapshots/{stage_name}.json with readable Unicode and
    indentation.  The ``default=str`` fallback serialises non-standard types
    (e.g. datetime, Path) as strings rather than crashing.
    """
    snapshot_path = os.path.join(
        config.OUTPUT_DIR, "snapshots", f"{stage_name}.json"
    )
    with open(snapshot_path, "w", encoding="utf-8") as fh:
        json.dump(context, fh, ensure_ascii=False, indent=2, default=str)


# ---------------------------------------------------------------------------
# Documentation generation
# ---------------------------------------------------------------------------

def generate_documentation() -> str:
    """
    Write output/models_and_tools.md summarising the models, tools, and
    architecture used by the pipeline.  Returns the path to the file.
    """
    doc_path = config.TOOLS_DOC_PATH
    content = """\
# Glix — Models and Tools Documentation

## Models Used

| Model | Parameters | Role |
|-------|-----------|------|
| OpenAI Whisper `medium` | 769 M | Speech-to-text transcription of English audio |
| Anthropic Claude Sonnet 4 | — | Slang analysis, context analysis, translation to Hebrew, quality flagging |

## Tools

| Tool | Purpose |
|------|---------|
| **ffmpeg** | Extract audio track from video as 16 kHz WAV |
| **Python `srt` library** | Parse and write SRT subtitle files |
| **Python `csv` module** | Export structured subtitle data to CSV |

## Architecture

The pipeline is composed of **9 sequential agents** that share a single
context dictionary.  Each agent reads the keys it needs, performs its work,
and writes its results back into the context before handing off to the next
agent.

```
AudioExtraction -> Transcription -> SlangAnalysis -> ContextAnalysis
    -> Translation -> TimingSync -> RTLFormatting -> SRTExport -> CSVExport
```

## Accuracy Approach

- **Word-level confidence** — Whisper provides per-word confidence scores;
  segments with low average confidence are flagged for human review.
- **Slang detection** — Claude identifies Regency-era slang, idioms, and
  culturally loaded phrases before translation so they receive special
  handling.
- **Context analysis** — Each segment receives a context score (0.0–1.0)
  based on pronoun usage, gendered words, and conversation continuity.
  High-scoring segments get extra attention during translation (gender cues,
  surrounding context). Global show metadata (genre, tone, characters) is
  injected into the translation prompt for coherent cross-segment choices.
- **Auto-flagging** — Any subtitle where confidence is below the threshold
  or the translation may be ambiguous is marked with a `REVIEW` flag in the
  CSV output for a human translator to verify.
"""
    with open(doc_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    return doc_path


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(
    total_time: float,
    context: Dict[str, Any],
) -> None:
    """Print a human-readable summary after the pipeline finishes."""
    segments = context.get("transcription_segments", [])
    subtitles = context.get("formatted_segments", segments)
    review_count = sum(
        1 for s in subtitles
        if "REVIEW" in s.get("accuracy_flag", s.get("flag", "")).upper()
    )

    print("\n" + "=" * 60)
    print("  GLIX PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time       : {total_time:.1f}s")
    print(f"  Segments         : {len(segments)}")
    print(f"  Subtitles        : {len(subtitles)}")
    print(f"  REVIEW-flagged   : {review_count}")
    print()
    print("  Output files:")
    print(f"    SRT            : {config.SRT_OUTPUT_PATH}")
    debug_srt = context.get(
        "srt_debug_path",
        config.SRT_OUTPUT_PATH.replace(".srt", "_debug.srt"),
    )
    print(f"    Debug SRT      : {debug_srt}")
    print(f"    CSV            : {config.CSV_OUTPUT_PATH}")
    print(f"    Docs           : {config.TOOLS_DOC_PATH}")
    print(f"    Pipeline log   : {os.path.join(config.OUTPUT_DIR, 'pipeline.log')}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_snapshot(stage_name: str) -> Dict[str, Any]:
    """Load a previously saved context snapshot from output/snapshots/."""
    snapshot_path = os.path.join(
        config.OUTPUT_DIR, "snapshots", f"{stage_name}.json"
    )
    if not os.path.isfile(snapshot_path):
        print(f"ERROR: Snapshot not found: {snapshot_path}")
        sys.exit(1)
    with open(snapshot_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Glix — Agentic Hebrew Subtitle Pipeline",
    )
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to video file (default: Bridgerton S1 Episode 1.mp4 in project root)",
    )
    parser.add_argument(
        "--skip-to",
        dest="skip_to",
        default=None,
        help=(
            "Skip to a specific pipeline stage by loading the previous snapshot. "
            "Example: --skip-to 04_context_analysis (loads 03_slang_analysis snapshot)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Override video path if provided via CLI
    if args.video:
        video_path = os.path.abspath(args.video)
        config.VIDEO_PATH = video_path
        # Derive output filenames from video name
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        config.SRT_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, f"{video_name}_Hebrew.srt")
        config.CSV_OUTPUT_PATH = os.path.join(config.OUTPUT_DIR, f"{video_name}_data.csv")

    # 1. Create output directories
    create_directories()

    # 2. Setup dual logging
    log_file_path = os.path.join(config.OUTPUT_DIR, "pipeline.log")
    logger = setup_logging(log_file_path)

    # 3. Define the agent pipeline
    pipeline: List[Tuple[str, Any]] = [
        ("01_audio_extraction",  AudioExtractionAgent()),
        ("02_transcription",     TranscriptionAgent()),
        ("03_slang_analysis",    SlangAnalysisAgent()),
        ("04_context_analysis",  ContextAnalysisAgent()),
        ("05_translation",       TranslationAgent()),
        ("06_timing_sync",       TimingSyncAgent()),
        ("07_rtl_formatting",    RTLFormattingAgent()),
        ("08_srt_export",        SRTExportAgent()),
        ("09_csv_export",        CSVExportAgent()),
    ]

    # 4. Handle --skip-to: load snapshot and skip completed stages
    start_index = 0
    context: Dict[str, Any] = {"video_path": config.VIDEO_PATH}

    if args.skip_to:
        # Find the target stage index
        stage_names = [name for name, _ in pipeline]
        if args.skip_to not in stage_names:
            print(
                f"ERROR: Unknown stage '{args.skip_to}'\n"
                f"Available stages: {', '.join(stage_names)}"
            )
            sys.exit(1)

        start_index = stage_names.index(args.skip_to)

        # Load the snapshot from the stage just before
        if start_index == 0:
            print("ERROR: Cannot skip to the first stage")
            sys.exit(1)

        prev_stage = stage_names[start_index - 1]
        logger.info(f"Loading snapshot from {prev_stage}...")
        context = load_snapshot(prev_stage)
        logger.info(
            f"Resuming pipeline from {args.skip_to} "
            f"(skipped stages 1-{start_index})"
        )
    else:
        # Full run — validate prerequisites
        validate_prerequisites()

    logger.info("Glix pipeline starting")

    # 5. Execute each agent and snapshot context
    pipeline_start = time.time()

    try:
        for stage_name, agent in pipeline[start_index:]:
            context = agent.execute(context)
            save_snapshot(context, stage_name)

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user (Ctrl+C)")
        print("\nPipeline interrupted by user. Partial results may be in output/.")
        sys.exit(130)
    except Exception:
        logger.exception("Pipeline failed")
        raise

    pipeline_elapsed = time.time() - pipeline_start
    logger.info(f"Pipeline finished in {pipeline_elapsed:.1f}s")

    # 7. Generate documentation
    doc_path = generate_documentation()
    logger.info(f"Documentation written to {doc_path}")

    # 8. Print summary
    print_summary(pipeline_elapsed, context)


if __name__ == "__main__":
    main()
