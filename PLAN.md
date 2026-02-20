# Glix: Agentic Hebrew Subtitle Pipeline — Detailed Plan

---

## 1. Context & Problem Statement

**Client brief (received Feb 18, 2026):**
> Build English subtitles from scratch for a Bridgerton S1 Episode 1 clip (~6 min),
> translate them to Hebrew, and deliver:
> 1. An SRT file with Hebrew subtitles
> 2. A CSV (or Google Sheet) with all extracted data + accuracy gap flags
> 3. Documentation of all models and tools used
> 4. Flagged points needing manual review for 100% accuracy

**Key constraints:**
- Subtitles must be created from scratch (no pre-existing subtitles)
- Special attention to Regency-era slang and period drama language
- Hebrew RTL must render correctly in video players
- The pipeline must be "agentic" — automated multi-step workflow
- Deadline: end of Tuesday, Feb 19, 2026

**What we're building:**
A Python project with 8 sequential agents orchestrated by a main script. Each agent handles one step of the subtitle creation pipeline: audio extraction → transcription → slang analysis → translation → timing → RTL formatting → SRT export → CSV export.

---

## 2. Environment & Prerequisites

**Confirmed system state (verified via exploration):**

| Component | Status | Details |
|-----------|--------|---------|
| Working directory | `c:/Users/user/OneDrive/Desktop/glix/` | Fresh git repo, no code |
| Video file | `Bridgerton S1 Episode 1.mp4` | 1.33 GB, ~358 seconds, 1920x1080 H.264, AAC stereo 48kHz |
| Python | 3.10.2 | Installed and working |
| ffmpeg | 7.1 | Installed and working |
| pip | 21.2.4 | Working |
| anthropic | 0.66.0 | Already installed |
| openai | 1.98.0 | Already installed |
| torch | NOT installed | Needed for Whisper |
| openai-whisper | NOT installed | Needed for transcription |
| GPU/CUDA | NOT available | Pipeline will run on CPU |

---

## 3. Model Selection Rationale

### Translation Model: Claude Sonnet 4

| Criteria | Claude Sonnet 4 | Gemini 2.5 Flash | GPT-4o |
|----------|------------------|-------------------|--------|
| Hebrew translation quality | Strong — #1 in 9/11 WMT24 pairs | **Weak — Hebrew not in primary languages** | Strong — good RTL handling |
| Literary/period text | Excellent — best at tone, nuance, literary register | Untested for Hebrew | Good |
| Professional rating | 78% rated "good" by pro translators (Lokalise 2025) | No Hebrew data | Comparable |
| Creative Hebrew | Excels at non-standard Hebrew (rabbinic, literary) | Unknown | Good |
| Cost (this video) | ~$0.12 | ~$0.02 | ~$0.10 |

**Decision:** Claude Sonnet 4 (`claude-sonnet-4-20250514`). Cost is negligible for a single video. Quality matters most.

### Transcription Model: OpenAI Whisper `medium`

| Model | Parameters | CPU Speed (vs realtime) | English Accuracy | Notes |
|-------|-----------|------------------------|------------------|-------|
| tiny | 39M | ~10x | Lowest | Too inaccurate for period drama |
| base | 74M | ~7x | Low | Misses accents, overlapping speech |
| small | 244M | ~2x | Good | Acceptable but struggles with archaic vocab |
| **medium** | **769M** | **~0.3-0.5x** | **Very good** | **Best accuracy/speed for CPU** |
| large | 1.5B | ~0.1x | Best | ~60 min for 6 min clip — too slow on CPU |

**Decision:** Whisper `medium`. Estimated transcription time: 12-20 minutes on CPU.

> Note: The video audio is English (not Hebrew), so standard Whisper is correct.
> Whisper Ivrit would only be needed if we were transcribing Hebrew audio.

### Total estimated API cost: ~$0.12

---

## 4. Project File Structure

```
c:/Users/user/OneDrive/Desktop/glix/
│
├── Bridgerton S1 Episode 1.mp4      # Input video (exists, 1.33 GB)
│
├── requirements.txt                   # All pip dependencies
├── config.py                          # Central configuration file
├── main.py                            # Pipeline orchestrator
│
├── agents/
│   ├── __init__.py                    # Package init (exports all agents)
│   ├── base_agent.py                  # Abstract base class
│   ├── audio_extraction_agent.py      # Agent 1: ffmpeg audio extraction
│   ├── transcription_agent.py         # Agent 2: Whisper transcription
│   ├── slang_analysis_agent.py        # Agent 3: Claude slang detection
│   ├── translation_agent.py           # Agent 4: Claude Hebrew translation
│   ├── timing_sync_agent.py           # Agent 5: Timing adjustment
│   ├── rtl_formatting_agent.py        # Agent 6: RTL Unicode formatting
│   ├── srt_export_agent.py            # Agent 7: SRT file generation
│   └── csv_export_agent.py            # Agent 8: CSV data export
│
├── data/
│   └── bridgerton_glossary.json       # Regency-era vocabulary → Hebrew
│
└── output/                            # Created at runtime
    ├── audio.wav                      # Extracted audio (16kHz mono)
    ├── Bridgerton_S1E1_Hebrew.srt     # Final Hebrew subtitles
    ├── Bridgerton_S1E1_Hebrew_debug.srt  # QA: English + Hebrew side-by-side
    ├── Bridgerton_S1E1_data.csv       # Full data with accuracy flags
    ├── models_and_tools.md            # Documentation of tools/models
    ├── pipeline.log                   # Execution log
    └── snapshots/                     # JSON state after each agent
        ├── 01_audio_extraction.json
        ├── 02_transcription.json
        ├── 03_slang_analysis.json
        ├── 04_translation.json
        ├── 05_timing_sync.json
        ├── 06_rtl_formatting.json
        ├── 07_srt_export.json
        └── 08_csv_export.json
```

---

## 5. Dependencies

### `requirements.txt`
```
openai-whisper>=20231117
torch>=2.0.0
srt>=3.5.3
anthropic>=0.66.0
tqdm>=4.65.0
```

### Installation commands
```bash
# CPU-only PyTorch (avoids downloading ~2GB CUDA toolkit)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies
pip install openai-whisper srt tqdm
```

---

## 6. Detailed Configuration — `config.py`

Central file for ALL tunable parameters. No magic numbers anywhere else.

```python
import os

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = os.path.join(BASE_DIR, "Bridgerton S1 Episode 1.mp4")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
GLOSSARY_PATH = os.path.join(BASE_DIR, "data", "bridgerton_glossary.json")

# === Audio Extraction ===
AUDIO_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "audio.wav")
AUDIO_SAMPLE_RATE = 16000   # Whisper expects 16kHz

# === Transcription ===
WHISPER_MODEL = "medium"     # 769M params, best CPU accuracy/speed
WHISPER_LANGUAGE = "en"      # Source audio is English

# === Claude API ===
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = "claude-sonnet-4-20250514"
TRANSLATION_BATCH_SIZE = 10  # Segments per API call

# === Subtitle Timing ===
MAX_CHARS_PER_LINE = 42        # Industry standard line length
MAX_LINES_PER_SUBTITLE = 2    # Max 2 lines per subtitle
MIN_DISPLAY_DURATION_MS = 1000 # Minimum 1 second on screen
MAX_DISPLAY_DURATION_MS = 7000 # Maximum 7 seconds on screen
READING_SPEED_CPS = 14         # Hebrew characters per second (standard)
GAP_BETWEEN_SUBS_MS = 80       # Minimum gap between subtitles

# === RTL Unicode Characters ===
RLM = "\u200F"  # Right-to-Left Mark
RLE = "\u202B"  # Right-to-Left Embedding
PDF = "\u202C"  # Pop Directional Formatting

# === Output Paths ===
SRT_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "Bridgerton_S1E1_Hebrew.srt")
CSV_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "Bridgerton_S1E1_data.csv")
TOOLS_DOC_PATH = os.path.join(OUTPUT_DIR, "models_and_tools.md")
```

---

## 7. Agent Architecture

### 7.1 Base Agent — `agents/base_agent.py`

Every agent inherits from `BaseAgent`. It provides:
- Consistent `run(context) -> context` interface
- Automatic logging with agent name
- Execution timing
- Error handling with full traceback

**Data flow:** A shared `context` dictionary (Python dict) passes between agents. Each agent reads its inputs and writes its outputs to this dict. Benefits:
- Simple to debug (serialize to JSON at any stage)
- Easy to resume from any stage (load a snapshot)
- Full audit trail of the pipeline

```
context = {
    "video_path": str,              # Set by orchestrator
    "audio_path": str,              # Set by Agent 1
    "transcription_segments": [...],# Set by Agent 2
    "full_text": str,               # Set by Agent 2
    "slang_annotations": [...],     # Set by Agent 3
    "glossary": {...},              # Set by Agent 3
    "translated_segments": [...],   # Set by Agent 4
    "timed_segments": [...],        # Set by Agent 5
    "formatted_segments": [...],    # Set by Agent 6
    "srt_output_path": str,         # Set by Agent 7
    "csv_output_path": str,         # Set by Agent 8
}
```

### 7.2 Agent 1 — Audio Extraction (`audio_extraction_agent.py`)

**Purpose:** Extract audio from MP4 in the exact format Whisper expects.

**Logic:**
1. Create `output/` directory if it doesn't exist
2. Run ffmpeg subprocess:
   ```
   ffmpeg -y -i "Bridgerton S1 Episode 1.mp4" -vn -acodec pcm_s16le -ar 16000 -ac 1 output/audio.wav
   ```
   - `-y`: overwrite without asking
   - `-vn`: no video
   - `-acodec pcm_s16le`: 16-bit PCM (uncompressed)
   - `-ar 16000`: 16kHz sample rate
   - `-ac 1`: mono channel
3. Validate output file exists and has reasonable size
4. Add `audio_path` to context

**Expected output:** `output/audio.wav`, ~11.4 MB (358s x 16000 samples/s x 2 bytes)

**Error handling:** If ffmpeg fails (non-zero return code), raise with stderr output.

### 7.3 Agent 2 — Transcription (`transcription_agent.py`)

**Purpose:** Transcribe English audio using Whisper with word-level timestamps.

**Logic:**
1. Load Whisper `medium` model (downloads ~1.5GB on first run, cached afterward)
2. Transcribe with parameters:
   - `language="en"` — force English (avoids misdetection)
   - `word_timestamps=True` — enables per-word timing (critical for Agent 5)
   - `verbose=False` — suppress console output
3. Extract segments, each containing:
   ```python
   {
       "id": int,                    # Segment index (0, 1, 2, ...)
       "start": float,               # Start time in seconds (e.g., 1.24)
       "end": float,                 # End time in seconds (e.g., 4.80)
       "text": str,                  # Full segment text, stripped
       "words": [                    # Per-word data
           {
               "word": str,          # Individual word
               "start": float,       # Word start time
               "end": float,         # Word end time
               "probability": float  # Whisper confidence (0.0-1.0)
           }
       ]
   }
   ```
4. Add `transcription_segments` and `full_text` to context

**Why `word_timestamps=True` matters:**
- Enables Agent 5 to split/merge segments based on Hebrew translation length
- Word-level `probability` scores feed into CSV accuracy flags
- Allows precise frame-level sync

**Estimated time:** 12-20 minutes on CPU for 6-min clip.

**Expected output:** ~30-50 segments for a 6-minute dialogue-heavy clip.

### 7.4 Agent 3 — Slang & Context Analysis (`slang_analysis_agent.py`)

**Purpose:** Identify Regency-era vocabulary and period-specific language patterns that require special translation attention.

**Logic:**
1. Load `bridgerton_glossary.json` (pre-built vocabulary map)
2. Send full transcript + glossary to Claude with a specialized system prompt
3. System prompt instructs Claude to identify:
   - **Period vocabulary:** "the ton", "rake", "modiste", "promenade", "calling card", "diamond of the first water", "on-dit", "Season"
   - **Formal speech patterns:** "I dare say", "I am most obliged", "pray tell", "it would not do"
   - **Double meanings/innuendo** common to Bridgerton
   - **Titles & social hierarchy:** "Your Grace", "Lady Whistledown", "Viscount"
   - **Phrases where literal translation loses meaning**
4. For each identified item, Claude returns:
   ```python
   {
       "phrase": str,              # Exact text from transcript
       "segment_id": int,          # Which segment it appears in
       "meaning": str,             # Explanation of meaning/connotation
       "translation_approach": str  # Suggested Hebrew approach
   }
   ```
5. Parse JSON response (with fallback for markdown-wrapped JSON)
6. Add `slang_annotations` and `glossary` to context

**Why separate from translation:** The analysis runs once over the full text. The translation agent then receives structured annotations alongside each batch. This means:
- Annotations can be manually reviewed/edited before translation
- The translation prompt stays focused (not bloated with analysis instructions)
- Reusable: if you re-translate, you don't re-analyze

**Error handling:** If Claude's JSON response fails to parse, fall back to raw text annotation.

### 7.5 Glossary — `data/bridgerton_glossary.json`

Pre-built mapping of ~12 key Regency-era terms. Ships with the project.

**Structure per entry:**
```json
{
  "the ton": {
    "meaning": "High society, the fashionable elite of London",
    "hebrew": "החברה הגבוהה",
    "notes": "From French 'le bon ton'. Central concept in Bridgerton."
  }
}
```

**Full entries:**
| English Term | Hebrew | Notes |
|-------------|--------|-------|
| the ton | החברה הגבוהה | French origin, high society |
| rake | נואף מקסים | Charming womanizer |
| diamond of the first water | יהלום ללא דופי | Queen Charlotte's ultimate compliment |
| the Season | העונה | Annual London social season, capitalize |
| promenade | טיול ראווה | Public social walk |
| modiste | תופרת אופנה | Fashionable dressmaker |
| calling card | כרטיס ביקור | Formal visiting card |
| on-dit | רכילות | Gossip (French: "one says") |
| Your Grace | הוד מעלתך | Address for Duke/Duchess |
| I dare say | אני מעז לומר | Common Regency filler |
| pray tell | אנא ספר לי | Archaic polite request |
| it would not do | זה לא יתאים | Understatement about propriety |

### 7.6 Agent 4 — Hebrew Translation (`translation_agent.py`)

**Purpose:** Translate each subtitle segment to literary Hebrew using Claude, incorporating slang annotations and glossary context.

**Logic:**
1. Build a detailed system prompt containing:
   - Role: "Expert Hebrew translator specializing in period drama dialogue"
   - Full glossary with established Hebrew translations
   - All slang annotations from Agent 3
   - Translation guidelines:
     - Use literary Hebrew (not colloquial modern)
     - Preserve social hierarchy through register shifts
     - Keep translations concise (subtitle readability)
     - Transliterate common names to Hebrew script ("Daphne" → "דפנה")
     - Keep titles in Hebrew ("Duke" → "הדוכס")
     - Preserve emotional tone (sarcasm, tenderness, authority)
     - Find Hebrew equivalents for double entendres where possible
   - Output format: JSON array with `{id, hebrew, notes}`

2. Process segments in batches of 10:
   - Send: `[{"id": 0, "english": "I dare say..."}, ...]`
   - Receive: `[{"id": 0, "hebrew": "אני מעז לומר...", "notes": "..."}, ...]`
   - ~3-5 API calls for ~30-50 segments

3. Parse JSON response for each batch

4. **Fallback mechanism:** If batch JSON parsing fails:
   - Fall back to single-segment translation
   - Send each segment individually with "Return ONLY the Hebrew text"
   - Ensures pipeline never fails completely due to formatting

5. Merge translations back: each segment gets a `hebrew` field
6. Add `translated_segments` to context

**Why batch processing:**
- Reduces API calls (5 instead of 40+)
- Cheaper and faster
- Context window easily handles 10 segments

**Translation notes:** The `notes` field from Claude captures non-obvious translation choices (e.g., "Used formal register because speaker is addressing a Duke"). These flow into the CSV for human review.

### 7.7 Agent 5 — Timing & Sync (`timing_sync_agent.py`)

**Purpose:** Adjust subtitle display timing so Hebrew text is readable. Hebrew translations may differ in length from English.

**Logic for each segment:**

1. **Calculate minimum reading time:**
   ```
   hebrew_char_count = len(hebrew_text.replace(" ", ""))
   min_reading_duration = hebrew_char_count / 14  # 14 CPS for Hebrew
   ```

2. **Choose display duration:**
   ```
   duration = max(original_duration, min_reading_duration)
   duration = clamp(duration, min=1.0s, max=7.0s)
   ```

3. **Prevent overlap with next subtitle:**
   ```
   if new_end > next_segment_start - 0.08:  # 80ms gap
       new_end = next_segment_start - 0.08
   ```

4. **Line splitting** (if Hebrew text > 42 chars):
   - Find space nearest to the middle of the text
   - Split into 2 lines
   - If either line still > 42 chars, truncate with "..."

5. Output per segment:
   ```python
   {
       "id": int,
       "start": float,               # Original start (anchored to speech)
       "end": float,                  # Adjusted end
       "english": str,               # Original English
       "hebrew": str,                # Hebrew (possibly line-split)
       "duration_original": float,    # Original duration
       "duration_adjusted": float,    # New duration
   }
   ```

6. Add `timed_segments` to context. Log average duration.

**Why 14 CPS:** This is the established standard for Hebrew subtitle reading speed (characters per second). It accounts for Hebrew's right-to-left reading pattern and morphological density.

### 7.8 Agent 6 — RTL Formatting (`rtl_formatting_agent.py`)

**Purpose:** Insert Unicode directional control characters so Hebrew renders correctly across all major video players.

**The RTL problem:**
- SRT has no built-in text direction mechanism
- Different players handle bare Hebrew inconsistently:
  - Reversed line order on multi-line subtitles
  - Punctuation on wrong side (`,well` instead of `well,`)
  - Mixing with Latin characters (names, numbers) breaks layout

**Solution — triple-layer Unicode wrapping per line:**
```
RLE + RLM + hebrew_text + RLM + PDF
```
- **RLE (U+202B):** Right-to-Left Embedding — tells the bidi algorithm the whole line is RTL
- **RLM (U+200F):** Right-to-Left Mark — anchors weak/neutral chars (punctuation, spaces) to correct side
- **PDF (U+202C):** Pop Directional Formatting — prevents leaking into player UI

**Logic:**
1. For each segment, split text by `\n` (may be 2 lines from Agent 5)
2. Wrap each line with the triple-layer pattern
3. Rejoin with `\n`
4. Store as `hebrew_formatted`
5. Add `formatted_segments` to context

**Compatibility:** Tested to work with VLC, mpv, MPC-HC, and most embedded players.

### 7.9 Agent 7 — SRT Export (`srt_export_agent.py`)

**Purpose:** Generate a standards-compliant SRT subtitle file.

**SRT format example:**
```
1
00:00:01,200 --> 00:00:04,800
כמדומני שהעונה הזו תהיה מרתקת במיוחד

2
00:00:05,100 --> 00:00:08,300
הדוכס הצעיר חזר לעיר
כולם מדברים על כך
```

**Logic:**
1. Convert each segment's `start`/`end` (float seconds) to `timedelta`
2. Create `srt.Subtitle` objects with index, start, end, content
3. Compose using `srt.compose()` (handles formatting)
4. Write to file with `utf-8-sig` encoding (UTF-8 BOM)

**Why `utf-8-sig`:** Prepends the UTF-8 BOM (`EF BB BF`). Many Windows apps (including VLC on Windows) use BOM to detect encoding. Without it, Hebrew may display as garbled characters.

5. **Also generate a debug SRT:**
   - Same timestamps
   - Content: `english_text\n---\nhebrew_formatted`
   - For QA: load in VLC to see both languages at once

6. Add `srt_output_path` and `srt_debug_path` to context

### 7.10 Agent 8 — CSV Export (`csv_export_agent.py`)

**Purpose:** Generate a comprehensive data CSV for Google Sheets/Excel with all extracted data and accuracy flags.

**CSV columns:**

| # | Column | Type | Source | Description |
|---|--------|------|--------|-------------|
| 1 | `subtitle_id` | int | Agent 2 | Sequential number (1, 2, 3, ...) |
| 2 | `start_time` | str | Agent 2 | Timestamp `HH:MM:SS,mmm` |
| 3 | `end_time` | str | Agent 5 | Adjusted timestamp `HH:MM:SS,mmm` |
| 4 | `duration_sec` | float | Agent 5 | Display duration in seconds |
| 5 | `english_text` | str | Agent 2 | Original Whisper transcription |
| 6 | `hebrew_text` | str | Agent 4 | Hebrew translation (plain, no RTL chars) |
| 7 | `confidence_score` | float | Agent 2 | Average Whisper word probability (0.0-1.0) |
| 8 | `low_confidence_words` | str | Agent 2 | Words with probability < 0.7, pipe-separated |
| 9 | `slang_flags` | str | Agent 3 | Regency terms detected in this segment, pipe-separated |
| 10 | `translation_notes` | str | Agent 4 | Non-obvious translation choices |
| 11 | `chars_per_sec` | float | Agent 5 | Hebrew reading speed for this subtitle |
| 12 | `accuracy_flag` | str | Computed | `"REVIEW"` if any accuracy concern |

**Accuracy flag logic:**
```python
flag = ""
reasons = []
if avg_confidence < 0.7:
    reasons.append("low_confidence")
if slang_flags:
    reasons.append("contains_slang")
if chars_per_sec > 16:
    reasons.append("fast_reading_speed")
if reasons:
    flag = f"REVIEW ({', '.join(reasons)})"
```

Any row with `accuracy_flag` starting with "REVIEW" needs manual checking for 100% accuracy.

**Encoding:** `utf-8-sig` (BOM) so Excel/Google Sheets display Hebrew correctly.

### 7.11 Orchestrator — `main.py`

**Purpose:** Wire all 8 agents together, handle logging, snapshots, and generate documentation.

**Execution flow:**
```
1. Validate prerequisites (video file exists, ANTHROPIC_API_KEY set)
2. Create output/ directory
3. Initialize context dict
4. For each agent in pipeline:
   a. agent.execute(context) -> updated context
   b. Save context snapshot to output/snapshots/{stage}.json
5. Generate models_and_tools.md
6. Print summary (total time, output paths)
```

**Pipeline definition:**
```python
pipeline = [
    ("01_audio_extraction",  AudioExtractionAgent()),
    ("02_transcription",     TranscriptionAgent()),
    ("03_slang_analysis",    SlangAnalysisAgent()),
    ("04_translation",       TranslationAgent()),
    ("05_timing_sync",       TimingSyncAgent()),
    ("06_rtl_formatting",    RTLFormattingAgent()),
    ("07_srt_export",        SRTExportAgent()),
    ("08_csv_export",        CSVExportAgent()),
]
```

**Context snapshots:** After each agent, the full context dict is serialized to JSON. This enables:
- Debugging any stage by examining input/output
- Resuming the pipeline mid-way (load snapshot, skip completed agents)
- Full audit trail

**Logging:** Dual output via Python `logging`:
- Console (`stdout`) — real-time progress
- File (`output/pipeline.log`) — persistent record

Format: `HH:MM:SS | glix.AgentName | INFO | message`

**Documentation generation (`models_and_tools.md`):**
Generated automatically at pipeline completion. Contains:
- Models used: Whisper medium (769M params), Claude Sonnet 4
- Tools: ffmpeg 7.1, Python srt library, Python csv module
- Architecture: 8 sequential agents with shared context
- Accuracy approach: word-level confidence, slang detection, auto-flagging

---

## 8. Data Flow Diagram

```
+------------------+
|  MP4 Video File  |
+--------+---------+
         |
    Agent 1: ffmpeg
         |
         v
+------------------+
|  16kHz Mono WAV  |
+--------+---------+
         |
    Agent 2: Whisper medium
         |
         v
+--------------------------------------+
|  Transcription Segments              |
|  [{id, start, end, text, words[      |
|     {word, start, end, probability}  |
|  ]}]                                 |
+--------+-----------------------------+
         |
         +----------------------+
         |                      |
    Agent 3: Claude         (passes through)
    Slang Analysis              |
         |                      |
         v                      |
+--------------------+          |
| Slang Annotations  |          |
| [{phrase, segment,  |          |
|   meaning, approach}|          |
| ]                   |          |
+--------+-----------+          |
         |                      |
         +------+---------------+
                |
           Agent 4: Claude
           Hebrew Translation
                |
                v
+------------------------------+
|  Translated Segments         |
|  [{id, start, end, text,     |
|    hebrew, translation_notes}|
|  ]                           |
+--------+---------------------+
         |
    Agent 5: Timing Sync
         |
         v
+------------------------------+
|  Timed Segments              |
|  [{id, start, end(adjusted), |
|    english, hebrew,          |
|    duration_adjusted}]       |
+--------+---------------------+
         |
    Agent 6: RTL Formatting
         |
         v
+------------------------------+
|  Formatted Segments          |
|  [{..., hebrew_formatted     |
|    (with RLE/RLM/PDF)}]      |
+--------+---------------------+
         |
         +----------------------+
         |                      |
    Agent 7: SRT Export    Agent 8: CSV Export
         |                      |
         v                      v
+----------------+    +----------------------+
| .srt file      |    | .csv file            |
| (UTF-8 BOM)    |    | (UTF-8 BOM)          |
| + debug .srt   |    | with accuracy flags  |
+----------------+    +----------------------+
```

---

## 9. Implementation Order

| Step | File(s) | Description |
|------|---------|-------------|
| 1 | `requirements.txt` | Write dependencies, install via pip |
| 2 | `config.py` | All paths, constants, model names |
| 3 | `agents/__init__.py` | Package init |
| 4 | `agents/base_agent.py` | Abstract base class |
| 5 | `data/bridgerton_glossary.json` | Pre-built Regency vocabulary |
| 6 | `agents/audio_extraction_agent.py` | Agent 1 |
| 7 | `agents/transcription_agent.py` | Agent 2 |
| 8 | `agents/slang_analysis_agent.py` | Agent 3 |
| 9 | `agents/translation_agent.py` | Agent 4 |
| 10 | `agents/timing_sync_agent.py` | Agent 5 |
| 11 | `agents/rtl_formatting_agent.py` | Agent 6 |
| 12 | `agents/srt_export_agent.py` | Agent 7 |
| 13 | `agents/csv_export_agent.py` | Agent 8 |
| 14 | `main.py` | Orchestrator + docs generation |
| 15 | Run pipeline | `python main.py` |

---

## 10. Deliverables

| # | File | Purpose | Format |
|---|------|---------|--------|
| 1 | `Bridgerton_S1E1_Hebrew.srt` | Production Hebrew subtitles | SRT, UTF-8 BOM, RTL-formatted |
| 2 | `Bridgerton_S1E1_data.csv` | Full data export for Google Sheets | CSV, UTF-8 BOM, 12 columns |
| 3 | `models_and_tools.md` | Documentation of all models/tools | Markdown |
| 4 | `Bridgerton_S1E1_Hebrew_debug.srt` | QA review (English + Hebrew) | SRT, UTF-8 BOM |

---

## 11. Verification Plan

### Automated checks (in pipeline):
- [ ] Agent 1: audio.wav exists and size > 0
- [ ] Agent 2: segments list is non-empty
- [ ] Agent 3: annotations parsed as JSON
- [ ] Agent 4: every segment has `hebrew` field
- [ ] Agent 5: no overlapping subtitles, all durations within 1-7s
- [ ] Agent 7: .srt file is valid (parseable by srt library)
- [ ] Agent 8: .csv has correct number of rows = number of segments

### Manual checks (after pipeline):
1. **Run:** `python main.py` with `ANTHROPIC_API_KEY` set
2. **Log:** Check `output/pipeline.log` — all 8 agents complete without errors
3. **Debug SRT:** Open `_debug.srt` — verify English/Hebrew pairs are sensible
4. **VLC test:** Load `Bridgerton_S1E1_Hebrew.srt` in VLC with the video:
   - Hebrew displays right-to-left
   - Timing matches when characters speak
   - Subtitles stay on screen long enough to read
   - Regency terms translated appropriately (check glossary terms)
5. **CSV in Sheets:** Open `.csv` in Google Sheets:
   - Hebrew column displays correctly
   - Filter by `accuracy_flag = "REVIEW"` — inspect flagged rows
   - Verify confidence scores look reasonable
6. **Documentation:** Review `models_and_tools.md` for completeness

---

## 12. Estimated Timeline

| Phase | Duration |
|-------|----------|
| Write all code (steps 1-14) | ~30-45 min |
| Install dependencies | ~5-10 min |
| Pipeline execution (step 15) | ~15-25 min (dominated by Whisper CPU) |
| Manual verification | ~10 min |
| **Total** | **~60-90 min** |

---

## 13. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Whisper misrecognizes period vocabulary | Wrong English -> wrong Hebrew | Glossary cross-reference in Agent 3; Claude can infer from context |
| Claude returns malformed JSON | Translation batch fails | Fallback to single-segment translation |
| Hebrew translation too long for timing | Unreadable subtitles | Agent 5 extends duration based on char count |
| RTL renders differently per player | Hebrew looks broken | Triple-layer Unicode approach (most compatible) |
| Whisper CPU is slow (~20 min) | Pipeline takes long | Configurable: switch to `small` model in config.py if needed |
| ANTHROPIC_API_KEY not set | Pipeline fails immediately | Validated at startup with clear error message |
| Hebrew tokenization 4x cost | Higher API cost | Still only ~$0.12 total, negligible |
