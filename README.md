# Glix — Agentic Hebrew Subtitle Pipeline

An automated, multi-agent pipeline that creates Hebrew subtitles from scratch for any English video file. Features intelligent context analysis for cross-segment translation coherence, special handling for period-specific language, and correct right-to-left (RTL) text rendering.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline Agents](#pipeline-agents)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Files](#output-files)
- [Glossary](#glossary)
- [Accuracy & Quality Flags](#accuracy--quality-flags)
- [RTL Rendering](#rtl-rendering)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Models & Tools](#models--tools)
- [Troubleshooting](#troubleshooting)

---

## Overview

Glix is a Python-based pipeline that takes an English video file and produces:

1. **Hebrew SRT subtitles** — ready to load in any video player (VLC, mpv, MPC-HC)
2. **Debug SRT** — bilingual English + Hebrew side-by-side for QA review
3. **CSV data export** — all extracted data with accuracy flags for Google Sheets / Excel
4. **Documentation** — auto-generated report of all models and tools used

The pipeline is fully agentic — 9 specialized agents run sequentially, each handling one step of the subtitle creation process. No pre-existing subtitles are used; everything is generated from the raw audio.

---

## Architecture

```
┌──────────────────┐
│  MP4 Video File  │
└────────┬─────────┘
         │
    Agent 1: ffmpeg
         │
         ▼
┌──────────────────┐
│  16kHz Mono WAV  │
└────────┬─────────┘
         │
    Agent 2: Whisper medium (769M params)
         │
         ▼
┌──────────────────────────────────────┐
│  Transcription Segments              │
│  [{id, start, end, text, words[      │
│     {word, start, end, probability}  │
│  ]}]                                 │
└────────┬─────────────────────────────┘
         │
    Agent 3: Claude Sonnet 4 (Slang Analysis)
         │
         ▼
┌────────────────────┐
│ Slang Annotations  │
│ [{phrase, segment,  │
│   meaning, approach}│
│ ]                   │
└────────┬───────────┘
         │
    Agent 4: Context Analysis (local + Claude)
         │
         ▼
┌─────────────────────────────────────┐
│  Segment Contexts + Show Context    │
│  [{id, context_score, before_text,  │
│    after_text, gender_cues}]        │
│  + {genre, tone, characters, ...}   │
└────────┬────────────────────────────┘
         │
    Agent 5: Claude Sonnet 4 (Hebrew Translation)
         │
         ▼
┌──────────────────────────────┐
│  Translated Segments         │
│  [{id, english, hebrew,      │
│    translation_notes}]       │
└────────┬─────────────────────┘
         │
    Agent 6: Timing Sync
         │
         ▼
┌──────────────────────────────┐
│  Timed Segments              │
│  [{id, start, end(adjusted), │
│    hebrew, duration}]        │
└────────┬─────────────────────┘
         │
    Agent 7: RTL Formatting
         │
         ▼
┌──────────────────────────────┐
│  Formatted Segments          │
│  [{..., hebrew_formatted     │
│    (with RLE/RLM/PDF)}]      │
└────────┬─────────────────────┘
         │
         ├──────────────────────┐
         │                      │
    Agent 8: SRT Export    Agent 9: CSV Export
         │                      │
         ▼                      ▼
┌────────────────┐    ┌──────────────────────┐
│ .srt file      │    │ .csv file            │
│ (UTF-8 BOM)    │    │ (UTF-8 BOM)          │
│ + debug .srt   │    │ with accuracy flags  │
└────────────────┘    └──────────────────────┘
```

### Data Flow

All agents share a single `context` dictionary (Python dict). Each agent reads the keys it needs and writes its results back. After each agent completes, a JSON snapshot of the entire context is saved to `output/snapshots/` for debugging and auditability.

---

## Pipeline Agents

### Agent 1 — Audio Extraction (`AudioExtractionAgent`)

- **Input:** MP4 video file
- **Output:** 16kHz mono WAV audio file
- **Tool:** ffmpeg
- **What it does:** Extracts the audio track from the video and converts it to the exact format Whisper expects — 16-bit PCM, 16kHz sample rate, mono channel. Validates the output file exists and has a reasonable size.

### Agent 2 — Transcription (`TranscriptionAgent`)

- **Input:** WAV audio file
- **Output:** Timestamped text segments with word-level confidence scores
- **Tool:** OpenAI Whisper `medium` model (769M parameters)
- **What it does:** Transcribes the English audio with `word_timestamps=True`, producing per-word timing and confidence (probability) data. This word-level data is critical for timing adjustment (Agent 5) and accuracy flagging (Agent 8).
- **Performance:** ~12-20 minutes on CPU for a 6-minute clip. The `medium` model was chosen as the best accuracy/speed tradeoff for CPU-only environments.

### Agent 3 — Slang & Context Analysis (`SlangAnalysisAgent`)

- **Input:** Full transcript text + Regency-era glossary
- **Output:** Annotated list of period-specific terms and phrases
- **Tool:** Claude Sonnet 4
- **What it does:** Sends the full transcript to Claude along with the pre-built glossary. Claude identifies:
  - Period vocabulary ("the ton", "rake", "modiste")
  - Formal speech patterns ("I dare say", "pray tell")
  - Double meanings and innuendo common to Bridgerton
  - Titles and social hierarchy terms ("Your Grace", "Viscount")
  - Phrases where literal translation would lose meaning
- **Why separate from translation:** Running analysis once over the full text produces better results than analyzing per-segment during translation. The annotations can also be manually reviewed before translation begins.

### Agent 4 — Context Analysis (`ContextAnalysisAgent`)

- **Input:** Transcription segments
- **Output:** Per-segment context windows with relevance scores + global show metadata
- **Tools:** Local analysis (no API) + one Claude Sonnet 4 call for global metadata
- **What it does:** Prepares contextual information so the translation agent can make coherent cross-segment choices (e.g., correct gender forms for "cousin" based on surrounding dialogue). Two phases:

  **Phase A — Local analysis (no API call):**
  - Builds a sliding context window of ~10 segments before and after each segment
  - Detects conversation boundaries using timing gaps (>2 seconds = new conversation)
  - Computes a **context_score** (0.0–1.0) for each segment indicating how dependent it is on surrounding context
  - Infers gender cues from pronouns near ambiguous words (e.g., "she... cousin" -> cousin is female)
  - Extracts character/pronoun mentions

  **Phase B — Global context (one Claude call):**
  - Sends a condensed ~500-word summary of the transcript to Claude
  - Extracts: genre, tone, setting, characters (with genders), translation directives
  - This metadata is injected into the translation agent's system prompt

- **Context score factors:** Segments score higher if they contain pronouns (needs antecedent), gendered words (needs gender consistency), are short (likely dialogue exchange), or share vocabulary with adjacent segments.

### Agent 5 — Hebrew Translation (`TranslationAgent`)

- **Input:** Transcription segments + slang annotations + glossary + context analysis (segment contexts + show context)
- **Output:** Each segment enriched with `hebrew` and `translation_notes` fields
- **Tool:** Claude Sonnet 4
- **What it does:** Translates each subtitle segment to literary Hebrew. Uses:
  - Batch processing (10 segments per API call) for efficiency
  - Detailed system prompt with glossary, annotations, show context (genre, tone, characters), and translation guidelines
  - Surrounding context from the context analysis agent — segments before and after each batch are included as reference (not translated)
  - For segments with high context scores (>=0.6), adds explicit notes about gender cues and pronoun dependencies
  - Fallback to single-segment translation if a batch response fails to parse
- **Translation guidelines enforced:**
  - Use literary Hebrew, not colloquial modern
  - Preserve social hierarchy through register shifts
  - Keep translations concise for subtitle readability
  - Transliterate names to Hebrew script ("Daphne" → "דפנה")
  - Translate titles ("Duke" → "הדוכס")
  - Preserve emotional tone (sarcasm, tenderness, authority)
  - Maintain cross-segment coherence — consistent Hebrew terms and correct gender forms throughout

### Agent 6 — Timing & Sync (`TimingSyncAgent`)

- **Input:** Translated segments with original timestamps
- **Output:** Timing-adjusted segments with line splitting
- **What it does:**
  1. **Reading speed calculation:** Hebrew characters (excluding spaces) / 14 CPS
  2. **Duration clamping:** Ensures each subtitle displays for 1–7 seconds
  3. **Overlap prevention:** Maintains 80ms minimum gap between consecutive subtitles
  4. **Line splitting:** If Hebrew text exceeds 42 characters, splits at the nearest space to the midpoint (max 2 lines per subtitle)
- **Why 14 CPS:** This is the established standard for Hebrew subtitle reading speed, accounting for right-to-left reading patterns and morphological density.

### Agent 7 — RTL Formatting (`RTLFormattingAgent`)

- **Input:** Timed segments with Hebrew text
- **Output:** Segments with Unicode-wrapped Hebrew for correct RTL rendering
- **What it does:** Applies triple-layer Unicode directional wrapping to each line:
  ```
  RLE (U+202B) + RLM (U+200F) + hebrew_text + RLM (U+200F) + PDF (U+202C)
  ```
  - **RLE:** Right-to-Left Embedding — tells the bidi algorithm the whole line is RTL
  - **RLM:** Right-to-Left Mark — anchors punctuation and spaces to the correct side
  - **PDF:** Pop Directional Formatting — prevents leaking into the video player UI

### Agent 8 — SRT Export (`SRTExportAgent`)

- **Input:** Formatted segments
- **Output:** Two SRT files (production + debug)
- **Tool:** Python `srt` library
- **What it does:**
  1. Generates the production Hebrew SRT file with RTL-formatted text
  2. Generates a debug SRT with English + Hebrew side-by-side (separated by `---`) for QA
  3. Both files use `utf-8-sig` encoding (UTF-8 with BOM) so Windows applications and VLC detect the encoding correctly

### Agent 9 — CSV Export (`CSVExportAgent`)

- **Input:** All pipeline data (formatted segments, transcription segments, slang annotations)
- **Output:** 12-column CSV file with accuracy flags
- **Tool:** Python `csv` module
- **What it does:** Exports a comprehensive data file with every piece of extracted data plus computed accuracy flags. See [Accuracy & Quality Flags](#accuracy--quality-flags) for the full column specification.

---

## Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Tested with 3.10.2 |
| ffmpeg | 7.0+ | Must be in PATH |
| Anthropic API key | — | For Claude Sonnet 4 (slang analysis + translation) |
| Disk space | ~2 GB | For Whisper model cache + audio extraction |

### Verify prerequisites

```bash
python --version    # Should be 3.10+
ffmpeg -version     # Should show version info
```

---

## Installation

```bash
# Clone the project
git clone <repo-url>
cd glix

# Install CPU-only PyTorch (avoids downloading ~2GB CUDA toolkit)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt

# Setup your API key
cp .env.example .env
# Edit .env and paste your ANTHROPIC_API_KEY
```

### Dependencies

| Package | Purpose |
|---------|---------|
| `openai-whisper` | Speech-to-text transcription |
| `torch` | PyTorch backend for Whisper (CPU-only) |
| `srt` | SRT subtitle file parsing and generation |
| `anthropic` | Claude API client for slang analysis + translation |
| `tqdm` | Progress bars |
| `python-dotenv` | Load `.env` file for API keys |

---

## Configuration

All configuration lives in `config.py`. No magic numbers anywhere else in the codebase.

### Key Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `WHISPER_MODEL` | `"medium"` | Whisper model size. Options: `tiny`, `base`, `small`, `medium`, `large` |
| `WHISPER_LANGUAGE` | `"en"` | Source audio language |
| `CLAUDE_MODEL` | `"claude-sonnet-4-20250514"` | Claude model for analysis + translation |
| `TRANSLATION_BATCH_SIZE` | `10` | Segments per Claude API call |
| `MAX_CHARS_PER_LINE` | `42` | Industry standard subtitle line length |
| `MAX_LINES_PER_SUBTITLE` | `2` | Maximum lines per subtitle block |
| `MIN_DISPLAY_DURATION_MS` | `1000` | Minimum 1 second on screen |
| `MAX_DISPLAY_DURATION_MS` | `7000` | Maximum 7 seconds on screen |
| `READING_SPEED_CPS` | `14` | Hebrew characters per second (standard) |
| `GAP_BETWEEN_SUBS_MS` | `80` | Minimum gap between consecutive subtitles |

### Whisper Model Comparison

| Model | Parameters | CPU Speed | Accuracy | Recommended For |
|-------|-----------|-----------|----------|-----------------|
| tiny | 39M | ~10x realtime | Low | Quick testing only |
| base | 74M | ~7x realtime | Low | Quick testing only |
| small | 244M | ~2x realtime | Good | Fast runs, simple dialogue |
| **medium** | **769M** | **~0.3-0.5x** | **Very good** | **Production use on CPU** |
| large | 1.5B | ~0.1x realtime | Best | GPU only (~60 min on CPU) |

---

## Usage

### Basic usage (default Bridgerton video)

```bash
python main.py
```

This uses the video file configured in `config.py` (`Bridgerton S1 Episode 1.mp4` in the project root).

### Custom video file

```bash
python main.py "path/to/your/video.mp4"
python main.py "C:\Videos\My Show S2E03.mp4"
```

Output filenames are automatically derived from the video name.

### Resume from a specific stage (`--skip-to`)

The pipeline saves context snapshots after each agent. If you need to re-run only part of the pipeline (e.g., after changing translation settings), use `--skip-to` to skip the expensive early stages:

```bash
# Re-run from context analysis onward (skips audio extraction, transcription, slang analysis)
python main.py --skip-to 04_context_analysis

# Re-run only translation and everything after
python main.py --skip-to 05_translation

# Re-run from timing sync onward
python main.py --skip-to 06_timing_sync
```

Available stage names:
| Stage | Name |
|-------|------|
| Audio Extraction | `01_audio_extraction` |
| Transcription | `02_transcription` |
| Slang Analysis | `03_slang_analysis` |
| Context Analysis | `04_context_analysis` |
| Translation | `05_translation` |
| Timing Sync | `06_timing_sync` |
| RTL Formatting | `07_rtl_formatting` |
| SRT Export | `08_srt_export` |
| CSV Export | `09_csv_export` |

This is especially useful since transcription (Agent 2) takes 12-20 minutes on CPU — skipping it saves significant time when iterating on downstream agents.

### Environment variable (alternative to .env)

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
python main.py
```

### Expected runtime

| Stage | Duration | Notes |
|-------|----------|-------|
| Audio extraction | ~30 seconds | ffmpeg, very fast |
| Transcription | ~12-20 minutes | Whisper `medium` on CPU, dominates total time |
| Slang analysis | ~10 seconds | Single Claude API call |
| Context analysis | ~8 seconds | Local analysis + one small Claude API call |
| Translation | ~2-3 minutes | ~12 Claude API calls (batched, 10 segments each) |
| Timing + RTL + Export | ~2 seconds | Local computation |
| **Total** | **~15-25 minutes** | Dominated by Whisper CPU transcription |

### Estimated API cost

~$0.12 per video (for a 6-minute clip). Hebrew tokenization uses ~4x more tokens than English, but the total is still negligible.

---

## Output Files

All output goes to the `output/` directory.

| File | Description |
|------|-------------|
| `Bridgerton_S1E1_Hebrew.srt` | Production Hebrew subtitles (RTL-formatted, UTF-8 BOM) |
| `Bridgerton_S1E1_Hebrew_debug.srt` | QA subtitles with English + Hebrew side-by-side |
| `Bridgerton_S1E1_data.csv` | Full data export with 12 columns and accuracy flags |
| `models_and_tools.md` | Auto-generated documentation of models and tools used |
| `audio.wav` | Extracted audio (16kHz mono PCM, ~11 MB per minute) |
| `pipeline.log` | Full execution log with timestamps |
| `snapshots/` | JSON context snapshots after each agent (for debugging) |

### SRT Format

```
1
00:00:01,200 --> 00:00:04,800
‫‏כמדומני שהעונה הזו תהיה מרתקת במיוחד‏‬

2
00:00:05,100 --> 00:00:08,300
‫‏הדוכס הצעיר חזר לעיר‏‬
‫‏כולם מדברים על כך‏‬
```

### Debug SRT Format

```
1
00:00:01,200 --> 00:00:04,800
I dare say this Season shall be most exciting
---
‫‏כמדומני שהעונה הזו תהיה מרתקת במיוחד‏‬
```

---

## Glossary

The pre-built glossary at `data/bridgerton_glossary.json` maps 12 key Regency-era terms to their Hebrew translations:

| English Term | Hebrew | Meaning |
|-------------|--------|---------|
| the ton | החברה הגבוהה | High society (from French "le bon ton") |
| rake | נואף מקסים | Charming womanizer |
| diamond of the first water | יהלום ללא דופי | Queen Charlotte's ultimate compliment |
| the Season | העונה | Annual London social season |
| promenade | טיול ראווה | Public social walk |
| modiste | תופרת אופנה | Fashionable dressmaker |
| calling card | כרטיס ביקור | Formal visiting card |
| on-dit | רכילות | Gossip (French: "one says") |
| Your Grace | הוד מעלתך | Address for Duke/Duchess |
| I dare say | אני מעז לומר | Common Regency filler |
| pray tell | אנא ספר לי | Archaic polite request |
| it would not do | זה לא יתאים | Understatement about propriety |

The glossary is loaded by Agent 3 and fed into both the slang analysis and translation prompts. You can add entries to cover vocabulary specific to other episodes or shows.

---

## Accuracy & Quality Flags

The CSV export contains 12 columns:

| # | Column | Type | Description |
|---|--------|------|-------------|
| 1 | `subtitle_id` | int | Sequential number (1, 2, 3, ...) |
| 2 | `start_time` | str | Start timestamp `HH:MM:SS,mmm` |
| 3 | `end_time` | str | End timestamp `HH:MM:SS,mmm` |
| 4 | `duration_sec` | float | Display duration in seconds |
| 5 | `english_text` | str | Original Whisper transcription |
| 6 | `hebrew_text` | str | Hebrew translation (plain text, no RTL markers) |
| 7 | `confidence_score` | float | Average Whisper word probability (0.0–1.0) |
| 8 | `low_confidence_words` | str | Words with probability < 0.7 (pipe-separated) |
| 9 | `slang_flags` | str | Regency terms detected (pipe-separated) |
| 10 | `translation_notes` | str | Claude's notes on translation choices |
| 11 | `chars_per_sec` | float | Hebrew reading speed for this subtitle |
| 12 | `accuracy_flag` | str | `REVIEW (reasons)` or empty |

### Flag logic

A subtitle gets flagged with `REVIEW` if any of these conditions apply:

| Condition | Trigger | Why |
|-----------|---------|-----|
| Low confidence | Average word probability < 0.7 | Whisper may have misheard the dialogue |
| Contains slang | Regency terms detected in segment | Translation may need human verification |
| Fast reading speed | > 16 Hebrew characters per second | Subtitle may be too fast to read |

Rows with a `REVIEW` flag should be manually inspected by a human translator to ensure 100% accuracy.

---

## RTL Rendering

Hebrew is a right-to-left (RTL) language. SRT files have no built-in text direction mechanism, so different video players handle bare Hebrew text inconsistently — reversed line order, punctuation on the wrong side, broken layout when mixed with Latin characters.

### The solution

Every line of Hebrew text is wrapped with a triple-layer Unicode pattern:

```
RLE (U+202B) + RLM (U+200F) + text + RLM (U+200F) + PDF (U+202C)
```

| Character | Name | Purpose |
|-----------|------|---------|
| `U+202B` | Right-to-Left Embedding | Tells the bidi algorithm the entire line is RTL |
| `U+200F` | Right-to-Left Mark | Anchors punctuation and spaces to the correct side |
| `U+202C` | Pop Directional Formatting | Prevents RTL context from leaking into the player UI |

### Compatibility

This approach has been tested with:
- VLC (Windows, macOS, Linux)
- mpv
- MPC-HC
- Most embedded video players

### UTF-8 BOM

Both SRT and CSV files are written with `utf-8-sig` encoding (UTF-8 with Byte Order Mark). The BOM (`EF BB BF`) tells Windows applications and VLC how to decode the file. Without it, Hebrew may display as garbled characters.

---

## Testing

The test suite covers all 9 agents with 91 unit tests across 13 test classes. All external dependencies (Whisper, Claude API, ffmpeg) are fully mocked.

### Run tests

```bash
python -m pytest tests/ -v
```

### Test coverage

| Test Class | Tests | What it covers |
|-----------|-------|----------------|
| `TestConfig` | 9 | All paths, constants, timing values |
| `TestBaseAgent` | 7 | Abstract class, execute wrapper, error handling |
| `TestAudioExtractionAgent` | 6 | ffmpeg flags, validation, error cases |
| `TestTranscriptionAgent` | 8 | Segment structure, word data, model loading |
| `TestSlangAnalysisAgent` | 6 | API calls, JSON parsing, fallbacks |
| `TestContextAnalysisAgent` | 10 | Context windows, conversation boundaries, scoring, gender cues, Claude integration |
| `TestTranslationAgent` | 11 | Batch processing, fallback, field preservation, show context, segment contexts |
| `TestTimingSyncAgent` | 10 | Reading speed, clamping, overlap, line splitting |
| `TestRTLFormattingAgent` | 6 | Unicode wrapping, multi-line handling |
| `TestSRTExportAgent` | 6 | SRT format, debug SRT, encoding |
| `TestCSVExportAgent` | 10 | All 12 columns, accuracy flags, encoding |
| `TestPipelineContextFlow` | 2 | Multi-agent integration flow |

---

## Project Structure

```
glix/
├── .env.example                       # Template for API key
├── .gitignore                         # Git ignore rules
├── config.py                          # Central configuration (all constants)
├── main.py                            # Pipeline orchestrator
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── PLAN.md                            # Detailed implementation plan
│
├── agents/
│   ├── __init__.py                    # Exports all 9 agent classes
│   ├── base_agent.py                  # Abstract base class (logging, timing)
│   ├── audio_extraction_agent.py      # Agent 1: ffmpeg audio extraction
│   ├── transcription_agent.py         # Agent 2: Whisper transcription
│   ├── slang_analysis_agent.py        # Agent 3: Claude slang detection
│   ├── context_analysis_agent.py      # Agent 4: Context windowing & scoring
│   ├── translation_agent.py           # Agent 5: Claude Hebrew translation
│   ├── timing_sync_agent.py           # Agent 6: Timing adjustment
│   ├── rtl_formatting_agent.py        # Agent 7: RTL Unicode formatting
│   ├── srt_export_agent.py            # Agent 8: SRT file generation
│   └── csv_export_agent.py            # Agent 9: CSV data export
│
├── data/
│   └── bridgerton_glossary.json       # Regency-era vocabulary → Hebrew
│
├── tests/
│   ├── __init__.py
│   └── test_pipeline.py               # 91 unit tests
│
└── output/                            # Created at runtime
    ├── audio.wav                      # Extracted audio
    ├── <video_name>_Hebrew.srt        # Production Hebrew subtitles
    ├── <video_name>_Hebrew_debug.srt  # QA bilingual subtitles
    ├── <video_name>_data.csv          # Full data with accuracy flags
    ├── models_and_tools.md            # Auto-generated documentation
    ├── pipeline.log                   # Execution log
    └── snapshots/                     # JSON state after each agent
        ├── 01_audio_extraction.json
        ├── 02_transcription.json
        ├── 03_slang_analysis.json
        ├── 04_context_analysis.json
        ├── 05_translation.json
        ├── 06_timing_sync.json
        ├── 07_rtl_formatting.json
        ├── 08_srt_export.json
        └── 09_csv_export.json
```

---

## Models & Tools

### Transcription — OpenAI Whisper `medium`

| Property | Value |
|----------|-------|
| Parameters | 769 million |
| Input | 16kHz mono WAV audio |
| Output | Timestamped segments with word-level confidence |
| Language | English (forced via `language="en"`) |
| Features | `word_timestamps=True` for per-word timing and confidence |
| Performance | ~0.3-0.5x realtime on CPU |

**Why Whisper medium:** Best accuracy/speed tradeoff for CPU-only environments. The `large` model provides marginally better accuracy but takes ~60 minutes on CPU for a 6-minute clip. The `small` model is ~4x faster but struggles with archaic vocabulary and overlapping speech.

### Translation — Claude Sonnet 4

| Property | Value |
|----------|-------|
| Model ID | `claude-sonnet-4-20250514` |
| Used for | Slang analysis (Agent 3) + Context analysis (Agent 4) + Hebrew translation (Agent 5) |
| Cost | ~$0.15 per 6-minute video |

**Why Claude Sonnet 4:** Ranked #1 in 9 of 11 WMT24 language pairs. 78% of translations rated "good" by professional translators (Lokalise 2025). Excels at literary and non-standard Hebrew (rabbinic, formal, literary register). Gemini 2.5 Flash was evaluated but Hebrew is not in its primary supported languages.

### Audio Extraction — ffmpeg

Extracts audio as 16-bit PCM WAV at 16kHz mono — the exact input format Whisper expects.

---

## Troubleshooting

### `ERROR: Video file not found`

The video file must exist at the path specified. Either:
- Place it in the project root as `Bridgerton S1 Episode 1.mp4`
- Pass a custom path: `python main.py "path/to/video.mp4"`

### `ERROR: ANTHROPIC_API_KEY is not set`

Set your API key in `.env`:
```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```
Or export it directly:
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Hebrew displays as garbled characters in VLC

Make sure you're using the SRT file from `output/` (not a copy). The file is encoded as UTF-8 with BOM. If the issue persists:
1. In VLC: Tools → Preferences → Subtitles/OSD → Default encoding → set to `UTF-8`
2. Ensure your system has Hebrew fonts installed

### Whisper transcription is very slow

The `medium` model takes ~12-20 minutes on CPU for a 6-minute clip. To speed it up:
- Switch to `small` model in `config.py`: `WHISPER_MODEL = "small"` (faster but less accurate)
- Use a GPU with CUDA-enabled PyTorch (reduces to ~1-2 minutes)

### Claude API returns errors

- Verify your API key is valid and has available credits
- Check `output/pipeline.log` for detailed error messages
- The translation agent has a fallback: if batch translation fails, it automatically retries each segment individually

### Pipeline was interrupted

The pipeline saves context snapshots after each agent to `output/snapshots/`. You can inspect these JSON files to see exactly where it stopped and what data was collected up to that point.

To resume from where it left off, use `--skip-to` with the stage that needs to run next:

```bash
# If it stopped during translation, resume from there
python main.py --skip-to 05_translation
```
