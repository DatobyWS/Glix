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
