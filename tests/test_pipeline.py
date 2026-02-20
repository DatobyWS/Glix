"""
Comprehensive unit tests for the Glix Hebrew subtitle pipeline.

Covers all 9 agents, config validation, and the base agent abstraction.
All external dependencies (whisper, anthropic, ffmpeg subprocess, file I/O)
are mocked so tests run without network access or heavy model weights.

Run with:
    pytest tests/test_pipeline.py -v
"""

import abc
import csv
import io
import json
import os
import sys
import time
from typing import Any, Dict
from unittest import mock
from unittest.mock import MagicMock, Mock, mock_open, patch, call

import pytest

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so agents/config can be imported.
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config
from agents.base_agent import BaseAgent
from agents.audio_extraction_agent import AudioExtractionAgent
from agents.transcription_agent import TranscriptionAgent
from agents.slang_analysis_agent import SlangAnalysisAgent
from agents.context_analysis_agent import ContextAnalysisAgent
from agents.translation_agent import TranslationAgent
from agents.timing_sync_agent import TimingSyncAgent
from agents.rtl_formatting_agent import RTLFormattingAgent
from agents.srt_export_agent import SRTExportAgent
from agents.csv_export_agent import CSVExportAgent


# ===========================================================================
# Shared mock data fixtures
# ===========================================================================

@pytest.fixture
def mock_transcription_segments():
    """Segments as produced by TranscriptionAgent (Agent 2)."""
    return [
        {
            "id": 0,
            "start": 0.0,
            "end": 3.5,
            "text": "I dare say, the Season has begun.",
            "words": [
                {"word": "I", "start": 0.0, "end": 0.2, "probability": 0.95},
                {"word": "dare", "start": 0.2, "end": 0.5, "probability": 0.92},
                {"word": "say,", "start": 0.5, "end": 0.8, "probability": 0.88},
                {"word": "the", "start": 0.9, "end": 1.0, "probability": 0.97},
                {"word": "Season", "start": 1.0, "end": 1.5, "probability": 0.90},
                {"word": "has", "start": 1.5, "end": 1.7, "probability": 0.93},
                {"word": "begun.", "start": 1.7, "end": 2.1, "probability": 0.91},
            ],
        },
        {
            "id": 1,
            "start": 4.0,
            "end": 7.2,
            "text": "Your Grace, pray tell, who is the diamond of the first water?",
            "words": [
                {"word": "Your", "start": 4.0, "end": 4.2, "probability": 0.96},
                {"word": "Grace,", "start": 4.2, "end": 4.6, "probability": 0.94},
                {"word": "pray", "start": 4.7, "end": 5.0, "probability": 0.85},
                {"word": "tell,", "start": 5.0, "end": 5.3, "probability": 0.83},
                {"word": "who", "start": 5.4, "end": 5.6, "probability": 0.97},
                {"word": "is", "start": 5.6, "end": 5.7, "probability": 0.98},
                {"word": "the", "start": 5.8, "end": 5.9, "probability": 0.97},
                {"word": "diamond", "start": 5.9, "end": 6.3, "probability": 0.91},
                {"word": "of", "start": 6.3, "end": 6.4, "probability": 0.96},
                {"word": "the", "start": 6.4, "end": 6.5, "probability": 0.97},
                {"word": "first", "start": 6.5, "end": 6.8, "probability": 0.93},
                {"word": "water?", "start": 6.8, "end": 7.2, "probability": 0.90},
            ],
        },
        {
            "id": 2,
            "start": 8.0,
            "end": 10.5,
            "text": "The rake has been spotted at the promenade.",
            "words": [
                {"word": "The", "start": 8.0, "end": 8.2, "probability": 0.60},
                {"word": "rake", "start": 8.2, "end": 8.5, "probability": 0.55},
                {"word": "has", "start": 8.5, "end": 8.7, "probability": 0.62},
                {"word": "been", "start": 8.7, "end": 8.9, "probability": 0.58},
                {"word": "spotted", "start": 8.9, "end": 9.3, "probability": 0.65},
                {"word": "at", "start": 9.3, "end": 9.4, "probability": 0.70},
                {"word": "the", "start": 9.4, "end": 9.5, "probability": 0.72},
                {"word": "promenade.", "start": 9.5, "end": 10.0, "probability": 0.68},
            ],
        },
    ]


@pytest.fixture
def mock_slang_annotations():
    """Annotations as produced by SlangAnalysisAgent (Agent 3)."""
    return [
        {
            "phrase": "I dare say",
            "term": "I dare say",
            "segment_id": 0,
            "meaning": "I suppose, I venture to say",
            "translation_approach": "Use archaic Hebrew equivalent",
        },
        {
            "phrase": "the Season",
            "term": "the Season",
            "segment_id": 0,
            "meaning": "The annual London social season",
            "translation_approach": "Capitalize and preserve formality",
        },
        {
            "phrase": "diamond of the first water",
            "term": "diamond of the first water",
            "segment_id": 1,
            "meaning": "The most beautiful debutante",
            "translation_approach": "Use established glossary term",
        },
        {
            "phrase": "rake",
            "term": "rake",
            "segment_id": 2,
            "meaning": "A charming but dissolute man",
            "translation_approach": "Literary Hebrew equivalent",
        },
    ]


@pytest.fixture
def mock_glossary():
    """Subset of the bridgerton_glossary.json."""
    return {
        "the ton": {
            "meaning": "High society",
            "hebrew": "\u05d4\u05d7\u05d1\u05e8\u05d4 \u05d4\u05d2\u05d1\u05d5\u05d4\u05d4",
            "notes": "From French 'le bon ton'.",
        },
        "rake": {
            "meaning": "A charming but dissolute man",
            "hebrew": "\u05e0\u05d5\u05d0\u05e3 \u05de\u05e7\u05e1\u05d9\u05dd",
            "notes": "Short for 'rakehell'.",
        },
        "Your Grace": {
            "meaning": "Form of address for a Duke or Duchess",
            "hebrew": "\u05d4\u05d5\u05d3 \u05de\u05e2\u05dc\u05ea\u05da",
            "notes": "Formal title.",
        },
    }


@pytest.fixture
def mock_translated_segments(mock_transcription_segments):
    """Segments enriched with Hebrew translations (Agent 4 output)."""
    hebrew_texts = [
        "\u05d0\u05e0\u05d9 \u05de\u05e2\u05d6 \u05dc\u05d5\u05de\u05e8, \u05d4\u05e2\u05d5\u05e0\u05d4 \u05d4\u05d7\u05dc\u05d4.",
        "\u05d4\u05d5\u05d3 \u05de\u05e2\u05dc\u05ea\u05da, \u05d0\u05e0\u05d0 \u05e1\u05e4\u05e8 \u05dc\u05d9, \u05de\u05d9 \u05d4\u05d5\u05d0 \u05d4\u05d9\u05d4\u05dc\u05d5\u05dd \u05dc\u05dc\u05d0 \u05d3\u05d5\u05e4\u05d9?",
        "\u05d4\u05e0\u05d5\u05d0\u05e3 \u05d4\u05de\u05e7\u05e1\u05d9\u05dd \u05e0\u05e6\u05e4\u05d4 \u05d1\u05d8\u05d9\u05d5\u05dc \u05d4\u05e8\u05d0\u05d5\u05d5\u05d4.",
    ]
    segments = []
    for i, seg in enumerate(mock_transcription_segments):
        merged = dict(seg)
        merged["hebrew"] = hebrew_texts[i]
        merged["translation_notes"] = ""
        segments.append(merged)
    return segments


@pytest.fixture
def mock_timed_segments(mock_translated_segments):
    """Segments after timing adjustment (Agent 5 output)."""
    timed = []
    for seg in mock_translated_segments:
        timed.append({
            "id": seg["id"],
            "start": seg["start"],
            "end": seg["end"],
            "english": seg["text"],
            "hebrew": seg["hebrew"],
            "duration_original": round(seg["end"] - seg["start"], 3),
            "duration_adjusted": round(seg["end"] - seg["start"], 3),
        })
    return timed


@pytest.fixture
def mock_formatted_segments(mock_timed_segments):
    """Segments after RTL formatting (Agent 6 output)."""
    formatted = []
    for seg in mock_timed_segments:
        lines = seg["hebrew"].split("\n")
        wrapped = [
            config.RLE + config.RLM + line + config.RLM + config.PDF
            for line in lines
        ]
        formatted.append({
            **seg,
            "hebrew_formatted": "\n".join(wrapped),
        })
    return formatted


# ===========================================================================
# 1. Config tests
# ===========================================================================

class TestConfig:
    """Validate config.py constants and paths."""

    def test_paths_are_strings(self):
        """All path config values must be strings."""
        assert isinstance(config.BASE_DIR, str)
        assert isinstance(config.VIDEO_PATH, str)
        assert isinstance(config.OUTPUT_DIR, str)
        assert isinstance(config.GLOSSARY_PATH, str)
        assert isinstance(config.AUDIO_OUTPUT_PATH, str)
        assert isinstance(config.SRT_OUTPUT_PATH, str)
        assert isinstance(config.CSV_OUTPUT_PATH, str)
        assert isinstance(config.TOOLS_DOC_PATH, str)

    def test_whisper_model_is_valid(self):
        """WHISPER_MODEL must be one of the standard Whisper model sizes."""
        valid_models = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
        assert config.WHISPER_MODEL in valid_models, (
            f"WHISPER_MODEL '{config.WHISPER_MODEL}' is not a known Whisper model size"
        )

    def test_whisper_language_is_english(self):
        """Source language should be English."""
        assert config.WHISPER_LANGUAGE == "en"

    def test_timing_constants_positive(self):
        """All timing constants must be positive."""
        assert config.MIN_DISPLAY_DURATION_MS > 0
        assert config.MAX_DISPLAY_DURATION_MS > 0
        assert config.READING_SPEED_CPS > 0
        assert config.GAP_BETWEEN_SUBS_MS > 0

    def test_timing_constants_reasonable_ranges(self):
        """Timing constants should be in reasonable ranges."""
        assert config.MIN_DISPLAY_DURATION_MS < config.MAX_DISPLAY_DURATION_MS
        assert 500 <= config.MIN_DISPLAY_DURATION_MS <= 2000
        assert 5000 <= config.MAX_DISPLAY_DURATION_MS <= 10000
        assert 8 <= config.READING_SPEED_CPS <= 25
        assert 20 <= config.GAP_BETWEEN_SUBS_MS <= 200

    def test_max_chars_per_line_positive(self):
        """MAX_CHARS_PER_LINE must be positive and reasonable."""
        assert config.MAX_CHARS_PER_LINE > 0
        assert 30 <= config.MAX_CHARS_PER_LINE <= 60

    def test_translation_batch_size_positive(self):
        """TRANSLATION_BATCH_SIZE must be a positive integer."""
        assert isinstance(config.TRANSLATION_BATCH_SIZE, int)
        assert config.TRANSLATION_BATCH_SIZE > 0

    def test_rtl_unicode_chars_defined(self):
        """RTL Unicode control characters must be correctly defined."""
        assert config.RLM == "\u200F"
        assert config.RLE == "\u202B"
        assert config.PDF == "\u202C"

    def test_audio_sample_rate(self):
        """Whisper expects 16 kHz audio."""
        assert config.AUDIO_SAMPLE_RATE == 16000


# ===========================================================================
# 2. Base agent tests
# ===========================================================================

class TestBaseAgent:
    """Test the BaseAgent abstract class."""

    def test_cannot_instantiate_directly(self):
        """BaseAgent is abstract; direct instantiation must raise TypeError."""
        with pytest.raises(TypeError):
            BaseAgent("test")

    def test_subclass_must_implement_run(self):
        """A subclass without run() must also raise TypeError."""
        class IncompleteAgent(BaseAgent):
            pass

        with pytest.raises(TypeError):
            IncompleteAgent("incomplete")

    def test_concrete_subclass_instantiation(self):
        """A concrete subclass that implements run() should instantiate fine."""
        class ConcreteAgent(BaseAgent):
            def run(self, context):
                context["done"] = True
                return context

        agent = ConcreteAgent("concrete")
        assert agent.name == "concrete"

    def test_execute_calls_run_and_returns_context(self):
        """execute() should call run() and return its result."""
        class ConcreteAgent(BaseAgent):
            def run(self, context):
                context["processed"] = True
                return context

        agent = ConcreteAgent("test_agent")
        result = agent.execute({"input": "data"})
        assert result["processed"] is True
        assert result["input"] == "data"

    def test_execute_wraps_with_timing(self):
        """execute() should log timing information."""
        class SlowAgent(BaseAgent):
            def run(self, context):
                return context

        agent = SlowAgent("slow_agent")
        with patch.object(agent.logger, "info") as mock_log:
            agent.execute({})
            # Should have at least 2 info calls: starting + completed with timing
            assert mock_log.call_count >= 2
            # The completion message should contain timing info
            completion_msg = mock_log.call_args_list[-1][0][0]
            assert "Completed" in completion_msg

    def test_execute_propagates_exception(self):
        """execute() should re-raise exceptions from run()."""
        class FailingAgent(BaseAgent):
            def run(self, context):
                raise ValueError("intentional failure")

        agent = FailingAgent("failing")
        with pytest.raises(ValueError, match="intentional failure"):
            agent.execute({})

    def test_execute_logs_error_on_exception(self):
        """execute() should log error when run() raises."""
        class FailingAgent(BaseAgent):
            def run(self, context):
                raise RuntimeError("boom")

        agent = FailingAgent("failing")
        with patch.object(agent.logger, "error") as mock_error:
            with pytest.raises(RuntimeError):
                agent.execute({})
            mock_error.assert_called_once()


# ===========================================================================
# 3. AudioExtractionAgent tests
# ===========================================================================

class TestAudioExtractionAgent:
    """Test AudioExtractionAgent (Agent 1)."""

    @patch("agents.audio_extraction_agent.os.path.getsize", return_value=1024 * 1024)
    @patch("agents.audio_extraction_agent.os.path.isfile", return_value=True)
    @patch("agents.audio_extraction_agent.os.makedirs")
    @patch("agents.audio_extraction_agent.subprocess.run")
    def test_ffmpeg_command_flags(self, mock_run, mock_makedirs, mock_isfile, mock_getsize):
        """Verify ffmpeg is called with correct flags for Whisper-compatible audio."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        agent = AudioExtractionAgent()
        context = agent.run({})

        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]

        # Verify essential ffmpeg flags
        assert cmd[0] == "ffmpeg"
        assert "-y" in cmd
        assert "-i" in cmd
        assert "-vn" in cmd
        assert "-acodec" in cmd
        assert "pcm_s16le" in cmd
        assert "-ar" in cmd
        assert str(config.AUDIO_SAMPLE_RATE) in cmd
        assert "-ac" in cmd
        assert "1" in cmd

    @patch("agents.audio_extraction_agent.os.path.getsize", return_value=5 * 1024 * 1024)
    @patch("agents.audio_extraction_agent.os.path.isfile", return_value=True)
    @patch("agents.audio_extraction_agent.os.makedirs")
    @patch("agents.audio_extraction_agent.subprocess.run")
    def test_context_gets_audio_path(self, mock_run, mock_makedirs, mock_isfile, mock_getsize):
        """Verify context receives 'audio_path' key after successful extraction."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        agent = AudioExtractionAgent()
        context = agent.run({})

        assert "audio_path" in context
        assert context["audio_path"] == config.AUDIO_OUTPUT_PATH

    @patch("agents.audio_extraction_agent.os.makedirs")
    @patch("agents.audio_extraction_agent.subprocess.run")
    def test_raises_on_ffmpeg_failure(self, mock_run, mock_makedirs):
        """Verify RuntimeError raised when ffmpeg exits with non-zero code."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout="",
            stderr="No such file or directory",
        )

        agent = AudioExtractionAgent()
        with pytest.raises(RuntimeError, match="ffmpeg exited with code 1"):
            agent.run({})

    @patch("agents.audio_extraction_agent.os.path.isfile", return_value=False)
    @patch("agents.audio_extraction_agent.os.makedirs")
    @patch("agents.audio_extraction_agent.subprocess.run")
    def test_raises_on_missing_output_file(self, mock_run, mock_makedirs, mock_isfile):
        """Verify FileNotFoundError when output WAV is not created."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        agent = AudioExtractionAgent()
        with pytest.raises(FileNotFoundError, match="Audio output file was not created"):
            agent.run({})

    @patch("agents.audio_extraction_agent.os.path.getsize", return_value=0)
    @patch("agents.audio_extraction_agent.os.path.isfile", return_value=True)
    @patch("agents.audio_extraction_agent.os.makedirs")
    @patch("agents.audio_extraction_agent.subprocess.run")
    def test_raises_on_empty_output_file(self, mock_run, mock_makedirs, mock_isfile, mock_getsize):
        """Verify RuntimeError when output WAV is empty (0 bytes)."""
        mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")

        agent = AudioExtractionAgent()
        with pytest.raises(RuntimeError, match="Audio output file is empty"):
            agent.run({})

    @patch("agents.audio_extraction_agent.os.makedirs")
    @patch("agents.audio_extraction_agent.subprocess.run")
    def test_output_directory_created(self, mock_run, mock_makedirs):
        """Verify output directory is created with exist_ok=True."""
        mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="fail")

        agent = AudioExtractionAgent()
        with pytest.raises(RuntimeError):
            agent.run({})

        mock_makedirs.assert_called_once_with(config.OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# 4. TranscriptionAgent tests
# ===========================================================================

class TestTranscriptionAgent:
    """Test TranscriptionAgent (Agent 2)."""

    def _make_whisper_result(self):
        """Build a mock Whisper transcription result."""
        return {
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.5,
                    "text": " I dare say, the Season has begun.",
                    "words": [
                        {"word": " I", "start": 0.0, "end": 0.2, "probability": 0.95},
                        {"word": " dare", "start": 0.2, "end": 0.5, "probability": 0.92},
                        {"word": " say,", "start": 0.5, "end": 0.8, "probability": 0.88},
                        {"word": " the", "start": 0.9, "end": 1.0, "probability": 0.97},
                        {"word": " Season", "start": 1.0, "end": 1.5, "probability": 0.90},
                        {"word": " has", "start": 1.5, "end": 1.7, "probability": 0.93},
                        {"word": " begun.", "start": 1.7, "end": 2.1, "probability": 0.91},
                    ],
                },
                {
                    "start": 4.0,
                    "end": 7.2,
                    "text": " Your Grace, who is the diamond of the first water?",
                    "words": [
                        {"word": " Your", "start": 4.0, "end": 4.2, "probability": 0.96},
                        {"word": " Grace,", "start": 4.2, "end": 4.6, "probability": 0.94},
                        {"word": " who", "start": 5.4, "end": 5.6, "probability": 0.97},
                        {"word": " is", "start": 5.6, "end": 5.7, "probability": 0.98},
                        {"word": " the", "start": 5.8, "end": 5.9, "probability": 0.97},
                        {"word": " diamond", "start": 5.9, "end": 6.3, "probability": 0.91},
                    ],
                },
            ],
            "text": " I dare say, the Season has begun. Your Grace, who is the diamond of the first water?",
        }

    @patch("agents.transcription_agent.whisper")
    def test_context_gets_transcription_segments(self, mock_whisper):
        """Verify context receives 'transcription_segments' key."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = self._make_whisper_result()
        mock_whisper.load_model.return_value = mock_model

        agent = TranscriptionAgent()
        context = agent.run({"audio_path": "/tmp/audio.wav"})

        assert "transcription_segments" in context
        assert len(context["transcription_segments"]) == 2

    @patch("agents.transcription_agent.whisper")
    def test_context_gets_full_text(self, mock_whisper):
        """Verify context receives 'full_text' key."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = self._make_whisper_result()
        mock_whisper.load_model.return_value = mock_model

        agent = TranscriptionAgent()
        context = agent.run({"audio_path": "/tmp/audio.wav"})

        assert "full_text" in context
        assert isinstance(context["full_text"], str)
        assert len(context["full_text"]) > 0

    @patch("agents.transcription_agent.whisper")
    def test_segment_structure(self, mock_whisper):
        """Verify each segment has required fields: id, start, end, text, words."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = self._make_whisper_result()
        mock_whisper.load_model.return_value = mock_model

        agent = TranscriptionAgent()
        context = agent.run({"audio_path": "/tmp/audio.wav"})

        for seg in context["transcription_segments"]:
            assert "id" in seg
            assert "start" in seg
            assert "end" in seg
            assert "text" in seg
            assert "words" in seg
            assert isinstance(seg["id"], int)
            assert isinstance(seg["start"], float)
            assert isinstance(seg["end"], float)
            assert isinstance(seg["text"], str)
            assert isinstance(seg["words"], list)

    @patch("agents.transcription_agent.whisper")
    def test_word_data_structure(self, mock_whisper):
        """Verify word-level data has word, start, end, probability."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = self._make_whisper_result()
        mock_whisper.load_model.return_value = mock_model

        agent = TranscriptionAgent()
        context = agent.run({"audio_path": "/tmp/audio.wav"})

        for seg in context["transcription_segments"]:
            for word in seg["words"]:
                assert "word" in word
                assert "start" in word
                assert "end" in word
                assert "probability" in word
                assert isinstance(word["probability"], float)

    @patch("agents.transcription_agent.whisper")
    def test_text_is_stripped(self, mock_whisper):
        """Verify segment text is stripped of leading/trailing whitespace."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = self._make_whisper_result()
        mock_whisper.load_model.return_value = mock_model

        agent = TranscriptionAgent()
        context = agent.run({"audio_path": "/tmp/audio.wav"})

        for seg in context["transcription_segments"]:
            assert seg["text"] == seg["text"].strip()

    @patch("agents.transcription_agent.whisper")
    def test_whisper_model_loaded_with_correct_size(self, mock_whisper):
        """Verify whisper.load_model is called with the configured model size."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}
        mock_whisper.load_model.return_value = mock_model

        agent = TranscriptionAgent()
        agent.run({"audio_path": "/tmp/audio.wav"})

        mock_whisper.load_model.assert_called_once_with(config.WHISPER_MODEL)

    @patch("agents.transcription_agent.whisper")
    def test_whisper_transcribe_with_word_timestamps(self, mock_whisper):
        """Verify transcribe is called with word_timestamps=True."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}
        mock_whisper.load_model.return_value = mock_model

        agent = TranscriptionAgent()
        agent.run({"audio_path": "/tmp/audio.wav"})

        call_kwargs = mock_model.transcribe.call_args
        assert call_kwargs[1]["word_timestamps"] is True
        assert call_kwargs[1]["language"] == config.WHISPER_LANGUAGE

    @patch("agents.transcription_agent.whisper")
    def test_segment_ids_are_sequential(self, mock_whisper):
        """Verify segment IDs start at 0 and are sequential."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = self._make_whisper_result()
        mock_whisper.load_model.return_value = mock_model

        agent = TranscriptionAgent()
        context = agent.run({"audio_path": "/tmp/audio.wav"})

        ids = [seg["id"] for seg in context["transcription_segments"]]
        assert ids == list(range(len(ids)))


# ===========================================================================
# 5. SlangAnalysisAgent tests
# ===========================================================================

class TestSlangAnalysisAgent:
    """Test SlangAnalysisAgent (Agent 3)."""

    def _mock_claude_response(self, text):
        """Build a mock anthropic message response."""
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = text
        mock_response.content = [mock_content_block]
        return mock_response

    @patch("agents.slang_analysis_agent.anthropic.Anthropic")
    @patch("builtins.open", new_callable=mock_open,
           read_data='{"the ton": {"meaning": "High society", "hebrew": "test"}}')
    def test_context_gets_slang_annotations(self, mock_file, mock_anthropic_cls):
        """Verify context receives 'slang_annotations' list."""
        annotations_json = json.dumps([
            {"phrase": "the ton", "segment_id": 0, "meaning": "High society",
             "translation_approach": "Use glossary term"}
        ])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response(annotations_json)
        mock_anthropic_cls.return_value = mock_client

        agent = SlangAnalysisAgent()
        context = agent.run({
            "full_text": "The ton gathered for the Season.",
            "transcription_segments": [{"id": 0, "text": "The ton gathered."}],
        })

        assert "slang_annotations" in context
        assert isinstance(context["slang_annotations"], list)
        assert len(context["slang_annotations"]) == 1
        assert context["slang_annotations"][0]["phrase"] == "the ton"

    @patch("agents.slang_analysis_agent.anthropic.Anthropic")
    @patch("builtins.open", new_callable=mock_open,
           read_data='{"rake": {"meaning": "A dissolute man", "hebrew": "test"}}')
    def test_context_gets_glossary(self, mock_file, mock_anthropic_cls):
        """Verify context receives 'glossary' dict loaded from file."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response("[]")
        mock_anthropic_cls.return_value = mock_client

        agent = SlangAnalysisAgent()
        context = agent.run({
            "full_text": "Some text.",
            "transcription_segments": [],
        })

        assert "glossary" in context
        assert isinstance(context["glossary"], dict)
        assert "rake" in context["glossary"]

    @patch("agents.slang_analysis_agent.anthropic.Anthropic")
    @patch("builtins.open", side_effect=FileNotFoundError("glossary missing"))
    def test_handles_missing_glossary_file(self, mock_file, mock_anthropic_cls):
        """Verify agent continues with empty glossary when file is missing."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response("[]")
        mock_anthropic_cls.return_value = mock_client

        agent = SlangAnalysisAgent()
        context = agent.run({
            "full_text": "Some dialogue.",
            "transcription_segments": [],
        })

        assert context["glossary"] == {}
        assert context["slang_annotations"] == []

    @patch("agents.slang_analysis_agent.anthropic.Anthropic")
    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    def test_handles_markdown_fenced_response(self, mock_file, mock_anthropic_cls):
        """Verify agent strips ```json ... ``` fences from Claude response."""
        annotations_json = '```json\n[{"phrase": "pray tell", "segment_id": 0, "meaning": "please tell me", "translation_approach": "use archaic form"}]\n```'
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response(annotations_json)
        mock_anthropic_cls.return_value = mock_client

        agent = SlangAnalysisAgent()
        context = agent.run({
            "full_text": "Pray tell, what happened?",
            "transcription_segments": [],
        })

        assert len(context["slang_annotations"]) == 1
        assert context["slang_annotations"][0]["phrase"] == "pray tell"

    @patch("agents.slang_analysis_agent.anthropic.Anthropic")
    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    def test_handles_invalid_json_response(self, mock_file, mock_anthropic_cls):
        """Verify agent returns empty annotations when Claude returns invalid JSON."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response(
            "This is not valid JSON at all"
        )
        mock_anthropic_cls.return_value = mock_client

        agent = SlangAnalysisAgent()
        context = agent.run({
            "full_text": "Some text.",
            "transcription_segments": [],
        })

        assert context["slang_annotations"] == []

    def test_skips_analysis_when_no_full_text(self):
        """Verify agent returns empty annotations when full_text is empty."""
        agent = SlangAnalysisAgent()
        context = agent.run({"full_text": "", "transcription_segments": []})

        assert context["slang_annotations"] == []
        assert "glossary" in context


# ===========================================================================
# 5.5 ContextAnalysisAgent tests
# ===========================================================================

class TestContextAnalysisAgent:
    """Test ContextAnalysisAgent (Agent 3.5)."""

    def _mock_claude_response(self, text):
        """Build a mock anthropic message response."""
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = text
        mock_response.content = [mock_content_block]
        return mock_response

    def test_empty_segments_returns_defaults(self):
        """Verify agent returns default context when no segments provided."""
        agent = ContextAnalysisAgent()
        context = agent.run({"transcription_segments": []})

        assert context["segment_contexts"] == []
        assert context["show_context"]["genre"] == "unknown"
        assert context["show_context"]["characters"] == []

    @patch("agents.context_analysis_agent.anthropic.Anthropic")
    def test_segment_contexts_structure(self, mock_anthropic_cls, mock_transcription_segments):
        """Verify each segment context has required fields."""
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response(
            json.dumps({
                "genre": "period drama",
                "tone": "formal",
                "setting": "Regency-era London",
                "characters": [],
                "translation_directives": [],
            })
        )
        mock_anthropic_cls.return_value = mock_client

        agent = ContextAnalysisAgent()
        context = agent.run({"transcription_segments": mock_transcription_segments})

        assert "segment_contexts" in context
        assert len(context["segment_contexts"]) == len(mock_transcription_segments)

        for sc in context["segment_contexts"]:
            assert "id" in sc
            assert "context_score" in sc
            assert "before_text" in sc
            assert "after_text" in sc
            assert "character_mentions" in sc
            assert "gender_cues" in sc
            assert "conversation_id" in sc
            assert 0.0 <= sc["context_score"] <= 1.0

    def test_context_window_size(self):
        """Verify context window includes up to 10 segments before and after."""
        segments = [
            {"id": i, "start": i * 1.0, "end": i * 1.0 + 0.9, "text": f"Seg {i}"}
            for i in range(25)
        ]
        agent = ContextAnalysisAgent()
        results = agent._build_segment_contexts(segments)

        # Middle segment should have 10 before and 10 after
        mid = results[12]
        before_parts = mid["before_text"].split()
        after_parts = mid["after_text"].split()
        # Before text has segments 2-11 text (10 segs)
        assert len([p for p in before_parts if p.startswith("Seg")]) == 10
        # After text has segments 13-22 text (10 segs)
        assert len([p for p in after_parts if p.startswith("Seg")]) == 10

    def test_conversation_boundary_detection(self):
        """Verify conversation IDs change when timing gap > 2s."""
        segments = [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "First line."},
            {"id": 1, "start": 1.2, "end": 2.0, "text": "Still same convo."},
            {"id": 2, "start": 5.0, "end": 6.0, "text": "New conversation."},
            {"id": 3, "start": 6.5, "end": 7.0, "text": "Same new convo."},
        ]
        agent = ContextAnalysisAgent()
        conv_ids = agent._detect_conversations(segments)

        assert conv_ids[0] == conv_ids[1]  # Same conversation
        assert conv_ids[2] == conv_ids[3]  # Same conversation
        assert conv_ids[0] != conv_ids[2]  # Different conversations

    def test_context_score_increases_with_pronouns(self):
        """Verify segments with pronouns get higher context scores."""
        agent = ContextAnalysisAgent()

        seg_with_pronouns = {"id": 0, "text": "She told him about her cousin."}
        seg_without = {"id": 1, "text": "The weather is pleasant today."}

        words_with = agent._extract_words(seg_with_pronouns["text"])
        words_without = agent._extract_words(seg_without["text"])

        score_with = agent._compute_context_score(
            seg_with_pronouns, words_with, None, None
        )
        score_without = agent._compute_context_score(
            seg_without, words_without, None, None
        )

        assert score_with > score_without

    def test_gender_cues_inferred_from_context(self):
        """Verify gender cues are inferred from surrounding pronouns."""
        text = "She is my cousin. Her cousin is arriving today."
        cues = ContextAnalysisAgent._infer_gender_cues(text)

        assert "cousin" in cues
        assert cues["cousin"] == "female"

    def test_gender_cues_male(self):
        """Verify male gender cues detected correctly."""
        text = "He introduced his cousin to the Duke."
        cues = ContextAnalysisAgent._infer_gender_cues(text)

        assert "cousin" in cues
        assert cues["cousin"] == "male"

    @patch("agents.context_analysis_agent.anthropic.Anthropic")
    def test_show_context_from_claude(self, mock_anthropic_cls, mock_transcription_segments):
        """Verify show_context is populated from Claude response."""
        show_context = {
            "genre": "period drama",
            "tone": "formal",
            "setting": "Regency-era London, 1813",
            "characters": [
                {"name": "Daphne", "gender": "female", "role": "protagonist"}
            ],
            "translation_directives": [
                "Use feminine Hebrew forms for Daphne"
            ],
        }
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response(
            json.dumps(show_context)
        )
        mock_anthropic_cls.return_value = mock_client

        agent = ContextAnalysisAgent()
        context = agent.run({"transcription_segments": mock_transcription_segments})

        assert context["show_context"]["genre"] == "period drama"
        assert len(context["show_context"]["characters"]) == 1
        assert context["show_context"]["characters"][0]["name"] == "Daphne"

    @patch("agents.context_analysis_agent.anthropic.Anthropic")
    def test_claude_failure_returns_defaults(self, mock_anthropic_cls, mock_transcription_segments):
        """Verify agent returns defaults when Claude API fails."""
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("API error")
        mock_anthropic_cls.return_value = mock_client

        agent = ContextAnalysisAgent()
        context = agent.run({"transcription_segments": mock_transcription_segments})

        # Should still have segment_contexts (local analysis works)
        assert len(context["segment_contexts"]) == len(mock_transcription_segments)
        # show_context should be defaults
        assert context["show_context"]["genre"] == "unknown"

    def test_extract_character_mentions(self):
        """Verify pronouns and gendered words are extracted."""
        text = "She told the Duke about her cousin."
        mentions = ContextAnalysisAgent._extract_character_mentions(text)

        assert "She" in mentions
        assert "her" in mentions
        assert "Duke" in mentions
        assert "cousin" in mentions


# ===========================================================================
# 6. TranslationAgent tests
# ===========================================================================

class TestTranslationAgent:
    """Test TranslationAgent (Agent 4)."""

    def _mock_claude_response(self, text):
        """Build a mock anthropic message response."""
        mock_response = MagicMock()
        mock_content_block = MagicMock()
        mock_content_block.text = text
        mock_response.content = [mock_content_block]
        return mock_response

    @patch("agents.translation_agent.anthropic.Anthropic")
    def test_context_gets_translated_segments(
        self, mock_anthropic_cls, mock_transcription_segments, mock_slang_annotations, mock_glossary
    ):
        """Verify context receives 'translated_segments' with 'hebrew' field."""
        response_json = json.dumps([
            {"id": 0, "hebrew": "\u05d0\u05e0\u05d9 \u05de\u05e2\u05d6 \u05dc\u05d5\u05de\u05e8", "notes": ""},
            {"id": 1, "hebrew": "\u05d4\u05d5\u05d3 \u05de\u05e2\u05dc\u05ea\u05da", "notes": ""},
            {"id": 2, "hebrew": "\u05d4\u05e0\u05d5\u05d0\u05e3 \u05d4\u05de\u05e7\u05e1\u05d9\u05dd", "notes": ""},
        ])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response(response_json)
        mock_anthropic_cls.return_value = mock_client

        agent = TranslationAgent()
        context = agent.run({
            "transcription_segments": mock_transcription_segments,
            "slang_annotations": mock_slang_annotations,
            "glossary": mock_glossary,
        })

        assert "translated_segments" in context
        assert len(context["translated_segments"]) == 3
        for seg in context["translated_segments"]:
            assert "hebrew" in seg
            assert isinstance(seg["hebrew"], str)

    @patch("agents.translation_agent.anthropic.Anthropic")
    def test_batch_processing(self, mock_anthropic_cls, mock_transcription_segments):
        """Verify segments are sent in batches of TRANSLATION_BATCH_SIZE."""
        # Create 25 segments to force multiple batches (batch size = 10)
        segments = []
        for i in range(25):
            segments.append({
                "id": i,
                "start": i * 3.0,
                "end": i * 3.0 + 2.5,
                "text": f"Segment {i} text.",
                "words": [],
            })

        response_data = [{"id": i, "hebrew": f"Hebrew {i}", "notes": ""} for i in range(10)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response(
            json.dumps(response_data)
        )
        mock_anthropic_cls.return_value = mock_client

        agent = TranslationAgent()

        # Dynamically adjust response per batch call
        call_count = [0]
        def make_response(*args, **kwargs):
            batch_start = call_count[0] * config.TRANSLATION_BATCH_SIZE
            batch_end = min(batch_start + config.TRANSLATION_BATCH_SIZE, 25)
            data = [{"id": i, "hebrew": f"Hebrew {i}", "notes": ""} for i in range(batch_start, batch_end)]
            call_count[0] += 1
            return self._mock_claude_response(json.dumps(data))

        mock_client.messages.create.side_effect = make_response

        context = agent.run({
            "transcription_segments": segments,
            "slang_annotations": [],
            "glossary": {},
        })

        expected_batches = (25 + config.TRANSLATION_BATCH_SIZE - 1) // config.TRANSLATION_BATCH_SIZE
        assert mock_client.messages.create.call_count == expected_batches
        assert len(context["translated_segments"]) == 25

    @patch("agents.translation_agent.anthropic.Anthropic")
    def test_fallback_to_single_segment_on_parse_error(self, mock_anthropic_cls):
        """Verify fallback to single-segment translation when batch parse fails."""
        segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "Hello world.", "words": []},
            {"id": 1, "start": 3.0, "end": 5.0, "text": "Good day.", "words": []},
        ]

        mock_client = MagicMock()

        # First call returns invalid JSON (batch fails), subsequent calls return
        # single Hebrew lines (fallback).
        call_count = [0]
        def side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # Batch attempt returns invalid JSON
                return self._mock_claude_response("NOT VALID JSON {{{")
            else:
                # Single-segment fallback returns plain Hebrew
                return self._mock_claude_response("\u05e9\u05dc\u05d5\u05dd")

        mock_client.messages.create.side_effect = side_effect
        mock_anthropic_cls.return_value = mock_client

        agent = TranslationAgent()
        context = agent.run({
            "transcription_segments": segments,
            "slang_annotations": [],
            "glossary": {},
        })

        assert "translated_segments" in context
        assert len(context["translated_segments"]) == 2
        # The fallback should have been triggered:
        # 1 batch call + 2 single-segment calls = 3 total
        assert mock_client.messages.create.call_count == 3

    @patch("agents.translation_agent.anthropic.Anthropic")
    def test_empty_segments_skips_translation(self, mock_anthropic_cls):
        """Verify empty segments list returns empty translated_segments."""
        agent = TranslationAgent()
        context = agent.run({
            "transcription_segments": [],
            "slang_annotations": [],
            "glossary": {},
        })

        assert context["translated_segments"] == []
        mock_anthropic_cls.assert_not_called()

    @patch("agents.translation_agent.anthropic.Anthropic")
    def test_translated_segments_preserve_original_fields(
        self, mock_anthropic_cls, mock_transcription_segments
    ):
        """Verify translated segments contain original fields plus hebrew and notes."""
        response_json = json.dumps([
            {"id": seg["id"], "hebrew": f"Hebrew {seg['id']}", "notes": ""}
            for seg in mock_transcription_segments
        ])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response(response_json)
        mock_anthropic_cls.return_value = mock_client

        agent = TranslationAgent()
        context = agent.run({
            "transcription_segments": mock_transcription_segments,
            "slang_annotations": [],
            "glossary": {},
        })

        for seg in context["translated_segments"]:
            assert "id" in seg
            assert "start" in seg
            assert "end" in seg
            assert "text" in seg
            assert "words" in seg
            assert "hebrew" in seg
            assert "translation_notes" in seg

    def test_strip_code_fences_json(self):
        """Verify _strip_code_fences removes ```json ... ``` wrapper."""
        agent = TranslationAgent()
        result = agent._strip_code_fences('```json\n[{"id": 1}]\n```')
        assert result == '[{"id": 1}]'

    def test_strip_code_fences_plain(self):
        """Verify _strip_code_fences removes bare ``` wrapper."""
        agent = TranslationAgent()
        result = agent._strip_code_fences('```\n[1, 2, 3]\n```')
        assert result == '[1, 2, 3]'

    def test_strip_code_fences_no_fences(self):
        """Verify _strip_code_fences is a no-op when there are no fences."""
        agent = TranslationAgent()
        result = agent._strip_code_fences('[1, 2, 3]')
        assert result == '[1, 2, 3]'

    def test_system_prompt_includes_show_context(self):
        """Verify system prompt includes show context when provided."""
        agent = TranslationAgent()
        show_context = {
            "genre": "period drama",
            "tone": "formal",
            "setting": "Regency-era London",
            "characters": [
                {"name": "Daphne", "gender": "female", "role": "protagonist"}
            ],
            "translation_directives": [
                "Use feminine Hebrew forms for Daphne"
            ],
        }
        prompt = agent._build_system_prompt({}, [], show_context)

        assert "period drama" in prompt
        assert "Daphne" in prompt
        assert "female" in prompt
        assert "feminine Hebrew forms" in prompt
        assert "cross-segment coherence" in prompt.lower()

    def test_system_prompt_works_without_show_context(self):
        """Verify system prompt works fine without show context."""
        agent = TranslationAgent()
        prompt = agent._build_system_prompt({}, [], None)

        assert "Show Context" not in prompt
        assert "Hebrew translator" in prompt

    @patch("agents.translation_agent.anthropic.Anthropic")
    def test_translation_with_segment_contexts(self, mock_anthropic_cls):
        """Verify segment contexts are included in translation batch."""
        segments = [
            {"id": 0, "start": 0.0, "end": 2.0, "text": "She told him.", "words": []},
            {"id": 1, "start": 2.5, "end": 4.0, "text": "Her cousin arrived.", "words": []},
        ]
        segment_contexts = [
            {
                "id": 0, "context_score": 0.7, "before_text": "",
                "after_text": "Her cousin arrived.",
                "character_mentions": ["She", "him"],
                "gender_cues": {}, "conversation_id": 0,
            },
            {
                "id": 1, "context_score": 0.8, "before_text": "She told him.",
                "after_text": "",
                "character_mentions": ["Her", "cousin"],
                "gender_cues": {"cousin": "female"}, "conversation_id": 0,
            },
        ]

        response_json = json.dumps([
            {"id": 0, "hebrew": "היא סיפרה לו", "notes": ""},
            {"id": 1, "hebrew": "בת דודתה הגיעה", "notes": "female cousin"},
        ])
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_claude_response(response_json)
        mock_anthropic_cls.return_value = mock_client

        agent = TranslationAgent()
        context = agent.run({
            "transcription_segments": segments,
            "slang_annotations": [],
            "glossary": {},
            "show_context": {},
            "segment_contexts": segment_contexts,
        })

        assert len(context["translated_segments"]) == 2
        # Check that the API call included context info
        call_args = mock_client.messages.create.call_args
        user_content = call_args[1]["messages"][0]["content"]
        assert "Surrounding Context" in user_content or "Context Notes" in user_content


# ===========================================================================
# 7. TimingSyncAgent tests
# ===========================================================================

class TestTimingSyncAgent:
    """Test TimingSyncAgent (Agent 5)."""

    def test_reading_speed_calculation(self):
        """Test that reading duration = chars / 14 CPS."""
        agent = TimingSyncAgent()
        # Hebrew text with 28 non-space chars => 28 / 14 = 2.0s reading time
        hebrew_text = "a" * 14 + " " + "b" * 14  # 28 non-space chars
        segments = [{
            "id": 0,
            "start": 0.0,
            "end": 1.0,  # original 1.0s, but reading needs 2.0s
            "text": "English text here",
            "hebrew": hebrew_text,
        }]

        context = agent.run({"translated_segments": segments})
        timed = context["timed_segments"]

        # Duration should be extended to accommodate reading speed
        # 28 chars / 14 CPS = 2.0s, which is > original 1.0s
        duration = timed[0]["end"] - timed[0]["start"]
        assert duration >= 2.0

    def test_duration_clamping_minimum(self):
        """Test that very short durations are clamped to MIN_DISPLAY_DURATION_MS."""
        agent = TimingSyncAgent()
        # Very short text, very short original duration
        segments = [{
            "id": 0,
            "start": 0.0,
            "end": 0.1,  # 100ms original
            "text": "Hi",
            "hebrew": "X",  # 1 char / 14 CPS = 0.071s
        }]

        context = agent.run({"translated_segments": segments})
        timed = context["timed_segments"]

        duration = timed[0]["end"] - timed[0]["start"]
        min_dur = config.MIN_DISPLAY_DURATION_MS / 1000.0
        assert duration >= min_dur

    def test_duration_clamping_maximum(self):
        """Test that very long durations are clamped to MAX_DISPLAY_DURATION_MS."""
        agent = TimingSyncAgent()
        # Very long text with very long original duration
        segments = [{
            "id": 0,
            "start": 0.0,
            "end": 30.0,  # 30 seconds original
            "text": "A very long English segment that goes on and on.",
            "hebrew": "X" * 500,  # 500 chars / 14 CPS = 35.7s needed
        }]

        context = agent.run({"translated_segments": segments})
        timed = context["timed_segments"]

        duration = timed[0]["end"] - timed[0]["start"]
        max_dur = config.MAX_DISPLAY_DURATION_MS / 1000.0
        assert duration <= max_dur

    def test_overlap_prevention(self):
        """Test that consecutive subtitles maintain GAP_BETWEEN_SUBS_MS gap."""
        agent = TimingSyncAgent()
        gap = config.GAP_BETWEEN_SUBS_MS / 1000.0

        segments = [
            {
                "id": 0,
                "start": 0.0,
                "end": 5.0,
                "text": "First segment.",
                "hebrew": "X" * 100,  # Needs long display time
            },
            {
                "id": 1,
                "start": 3.0,  # Starts at 3s, but first segment wants to end at 5+
                "end": 6.0,
                "text": "Second segment.",
                "hebrew": "Y" * 10,
            },
        ]

        context = agent.run({"translated_segments": segments})
        timed = context["timed_segments"]

        # End of first subtitle should be at most (start of second - gap)
        assert timed[0]["end"] <= timed[1]["start"] - gap + 0.001  # small float tolerance

    def test_overlap_prevention_80ms_gap(self):
        """Specifically test the 80ms gap between subtitles."""
        agent = TimingSyncAgent()
        gap_s = config.GAP_BETWEEN_SUBS_MS / 1000.0
        assert gap_s == 0.08  # Confirm 80ms

        segments = [
            {"id": 0, "start": 0.0, "end": 2.95, "text": "A", "hebrew": "A" * 40},
            {"id": 1, "start": 3.0, "end": 5.0, "text": "B", "hebrew": "B" * 10},
        ]

        context = agent.run({"translated_segments": segments})
        timed = context["timed_segments"]

        actual_gap = timed[1]["start"] - timed[0]["end"]
        assert actual_gap >= gap_s - 0.001  # Allow small float rounding

    def test_line_splitting_at_42_chars(self):
        """Test that lines exceeding 42 chars are split into two lines."""
        agent = TimingSyncAgent()

        # Create text > 42 chars
        long_hebrew = "A" * 20 + " " + "B" * 25  # 46 chars total
        assert len(long_hebrew) > config.MAX_CHARS_PER_LINE

        segments = [{
            "id": 0,
            "start": 0.0,
            "end": 5.0,
            "text": "Long English sentence.",
            "hebrew": long_hebrew,
        }]

        context = agent.run({"translated_segments": segments})
        timed = context["timed_segments"]

        # Should contain a newline (split into two lines)
        assert "\n" in timed[0]["hebrew"]
        lines = timed[0]["hebrew"].split("\n")
        assert len(lines) == 2

    def test_short_line_not_split(self):
        """Test that lines <= 42 chars are not split."""
        agent = TimingSyncAgent()

        short_hebrew = "A" * 20  # 20 chars, well under limit
        segments = [{
            "id": 0, "start": 0.0, "end": 3.0,
            "text": "Short text.", "hebrew": short_hebrew,
        }]

        context = agent.run({"translated_segments": segments})
        timed = context["timed_segments"]

        assert "\n" not in timed[0]["hebrew"]

    def test_context_gets_timed_segments(self, mock_translated_segments):
        """Verify context receives 'timed_segments' key."""
        agent = TimingSyncAgent()
        context = agent.run({"translated_segments": mock_translated_segments})

        assert "timed_segments" in context
        assert isinstance(context["timed_segments"], list)
        assert len(context["timed_segments"]) == len(mock_translated_segments)

    def test_timed_segment_structure(self, mock_translated_segments):
        """Verify each timed segment has all required fields."""
        agent = TimingSyncAgent()
        context = agent.run({"translated_segments": mock_translated_segments})

        for seg in context["timed_segments"]:
            assert "id" in seg
            assert "start" in seg
            assert "end" in seg
            assert "english" in seg
            assert "hebrew" in seg
            assert "duration_original" in seg
            assert "duration_adjusted" in seg

    def test_split_line_no_space_fallback(self):
        """Test _split_line truncation when no space is found."""
        long_text = "A" * 60  # 60 chars, no spaces
        result = TimingSyncAgent._split_line(long_text)
        # Should be truncated with "..."
        assert result.endswith("...")
        assert len(result) == config.MAX_CHARS_PER_LINE


# ===========================================================================
# 8. RTLFormattingAgent tests
# ===========================================================================

class TestRTLFormattingAgent:
    """Test RTLFormattingAgent (Agent 6)."""

    def test_rtl_wrapping_pattern(self, mock_timed_segments):
        """Verify RLE+RLM+text+RLM+PDF wrapping pattern on each line."""
        agent = RTLFormattingAgent()
        context = agent.run({"timed_segments": mock_timed_segments})

        for seg in context["formatted_segments"]:
            formatted = seg["hebrew_formatted"]
            for line in formatted.split("\n"):
                assert line.startswith(config.RLE + config.RLM), (
                    f"Line must start with RLE+RLM: {repr(line)}"
                )
                assert line.endswith(config.RLM + config.PDF), (
                    f"Line must end with RLM+PDF: {repr(line)}"
                )

    def test_multi_line_handling(self):
        """Verify multi-line Hebrew text has each line independently wrapped."""
        agent = RTLFormattingAgent()
        timed_segments = [{
            "id": 0,
            "start": 0.0,
            "end": 3.0,
            "english": "Two line subtitle.",
            "hebrew": "Line one\nLine two",
            "duration_original": 3.0,
            "duration_adjusted": 3.0,
        }]

        context = agent.run({"timed_segments": timed_segments})
        formatted = context["formatted_segments"][0]["hebrew_formatted"]

        lines = formatted.split("\n")
        assert len(lines) == 2

        # Each line should be independently wrapped
        for line in lines:
            assert line.startswith(config.RLE + config.RLM)
            assert line.endswith(config.RLM + config.PDF)

        # Verify the actual text content is preserved inside the wrapping
        inner_1 = lines[0][len(config.RLE + config.RLM):-len(config.RLM + config.PDF)]
        inner_2 = lines[1][len(config.RLE + config.RLM):-len(config.RLM + config.PDF)]
        assert inner_1 == "Line one"
        assert inner_2 == "Line two"

    def test_context_gets_formatted_segments(self, mock_timed_segments):
        """Verify context receives 'formatted_segments' with 'hebrew_formatted'."""
        agent = RTLFormattingAgent()
        context = agent.run({"timed_segments": mock_timed_segments})

        assert "formatted_segments" in context
        assert isinstance(context["formatted_segments"], list)
        assert len(context["formatted_segments"]) == len(mock_timed_segments)

        for seg in context["formatted_segments"]:
            assert "hebrew_formatted" in seg

    def test_preserves_all_original_fields(self, mock_timed_segments):
        """Verify formatted segments contain all fields from timed segments."""
        agent = RTLFormattingAgent()
        context = agent.run({"timed_segments": mock_timed_segments})

        for original, formatted in zip(mock_timed_segments, context["formatted_segments"]):
            assert formatted["id"] == original["id"]
            assert formatted["start"] == original["start"]
            assert formatted["end"] == original["end"]
            assert formatted["english"] == original["english"]
            assert formatted["hebrew"] == original["hebrew"]
            assert formatted["duration_original"] == original["duration_original"]
            assert formatted["duration_adjusted"] == original["duration_adjusted"]

    def test_single_line_wrapping(self):
        """Verify single-line text gets proper wrapping without newlines."""
        agent = RTLFormattingAgent()
        timed_segments = [{
            "id": 0,
            "start": 0.0, "end": 2.0,
            "english": "Hello",
            "hebrew": "\u05e9\u05dc\u05d5\u05dd",
            "duration_original": 2.0,
            "duration_adjusted": 2.0,
        }]

        context = agent.run({"timed_segments": timed_segments})
        formatted = context["formatted_segments"][0]["hebrew_formatted"]

        assert "\n" not in formatted
        expected = config.RLE + config.RLM + "\u05e9\u05dc\u05d5\u05dd" + config.RLM + config.PDF
        assert formatted == expected

    def test_empty_hebrew_text(self):
        """Verify agent handles empty Hebrew text gracefully."""
        agent = RTLFormattingAgent()
        timed_segments = [{
            "id": 0,
            "start": 0.0, "end": 1.0,
            "english": "Test",
            "hebrew": "",
            "duration_original": 1.0,
            "duration_adjusted": 1.0,
        }]

        context = agent.run({"timed_segments": timed_segments})
        formatted = context["formatted_segments"][0]["hebrew_formatted"]

        # Even empty text should be wrapped
        expected = config.RLE + config.RLM + "" + config.RLM + config.PDF
        assert formatted == expected


# ===========================================================================
# 9. SRTExportAgent tests
# ===========================================================================

class TestSRTExportAgent:
    """Test SRTExportAgent (Agent 7)."""

    @patch("agents.srt_export_agent.srt.compose")
    @patch("builtins.open", new_callable=mock_open)
    def test_srt_format_correct(self, mock_file, mock_compose, mock_formatted_segments):
        """Verify SRT file is written with correct structure."""
        mock_compose.return_value = "1\n00:00:00,000 --> 00:00:03,500\nHebrew text\n\n"

        agent = SRTExportAgent()
        context = agent.run({"formatted_segments": mock_formatted_segments})

        # srt.compose should have been called (for main + debug = 2 calls)
        assert mock_compose.call_count == 2

    @patch("agents.srt_export_agent.srt.compose")
    @patch("builtins.open", new_callable=mock_open)
    def test_debug_srt_includes_english_and_hebrew(
        self, mock_file, mock_compose, mock_formatted_segments
    ):
        """Verify the debug SRT contains English+Hebrew separated by ---."""
        # Capture the subtitles passed to srt.compose
        compose_calls = []
        def capture_compose(subtitles):
            compose_calls.append(list(subtitles))
            return "mocked srt content"
        mock_compose.side_effect = capture_compose

        agent = SRTExportAgent()
        context = agent.run({"formatted_segments": mock_formatted_segments})

        # Second call is the debug SRT
        assert len(compose_calls) == 2
        debug_subtitles = compose_calls[1]

        for sub in debug_subtitles:
            content = sub.content
            assert "---" in content, "Debug SRT should contain '---' separator"
            parts = content.split("\n---\n")
            assert len(parts) == 2, "Debug SRT should have English + Hebrew parts"

    @patch("agents.srt_export_agent.srt.compose")
    @patch("builtins.open", new_callable=mock_open)
    def test_utf8_sig_encoding(self, mock_file, mock_compose, mock_formatted_segments):
        """Verify SRT files are written with utf-8-sig encoding (BOM)."""
        mock_compose.return_value = "srt content"

        agent = SRTExportAgent()
        agent.run({"formatted_segments": mock_formatted_segments})

        # Check all open calls use utf-8-sig encoding
        open_calls = mock_file.call_args_list
        for call_obj in open_calls:
            kwargs = call_obj[1] if call_obj[1] else {}
            args = call_obj[0] if call_obj[0] else ()
            # encoding can be positional or keyword
            if "encoding" in kwargs:
                assert kwargs["encoding"] == "utf-8-sig"

    @patch("agents.srt_export_agent.srt.compose")
    @patch("builtins.open", new_callable=mock_open)
    def test_context_gets_srt_paths(self, mock_file, mock_compose, mock_formatted_segments):
        """Verify context receives srt_output_path and srt_debug_path."""
        mock_compose.return_value = "srt content"

        agent = SRTExportAgent()
        context = agent.run({"formatted_segments": mock_formatted_segments})

        assert "srt_output_path" in context
        assert "srt_debug_path" in context
        assert context["srt_output_path"] == config.SRT_OUTPUT_PATH
        assert context["srt_debug_path"].endswith("_debug.srt")

    @patch("agents.srt_export_agent.srt.compose")
    @patch("builtins.open", new_callable=mock_open)
    def test_srt_index_is_1_based(self, mock_file, mock_compose, mock_formatted_segments):
        """Verify SRT subtitle indices are 1-based (id + 1)."""
        compose_calls = []
        def capture_compose(subtitles):
            compose_calls.append(list(subtitles))
            return "content"
        mock_compose.side_effect = capture_compose

        agent = SRTExportAgent()
        agent.run({"formatted_segments": mock_formatted_segments})

        # Main SRT subtitles (first compose call)
        main_subs = compose_calls[0]
        for i, sub in enumerate(main_subs):
            expected_index = mock_formatted_segments[i]["id"] + 1
            assert sub.index == expected_index

    @patch("agents.srt_export_agent.srt.compose")
    @patch("builtins.open", new_callable=mock_open)
    def test_main_srt_uses_hebrew_formatted(
        self, mock_file, mock_compose, mock_formatted_segments
    ):
        """Verify main SRT uses hebrew_formatted (with RTL chars), not plain hebrew."""
        compose_calls = []
        def capture_compose(subtitles):
            compose_calls.append(list(subtitles))
            return "content"
        mock_compose.side_effect = capture_compose

        agent = SRTExportAgent()
        agent.run({"formatted_segments": mock_formatted_segments})

        main_subs = compose_calls[0]
        for i, sub in enumerate(main_subs):
            assert sub.content == mock_formatted_segments[i]["hebrew_formatted"]


# ===========================================================================
# 10. CSVExportAgent tests
# ===========================================================================

class TestCSVExportAgent:
    """Test CSVExportAgent (Agent 8)."""

    def _build_full_context(
        self, mock_formatted_segments, mock_transcription_segments, mock_slang_annotations
    ):
        """Build a complete pipeline context for CSV export."""
        return {
            "formatted_segments": mock_formatted_segments,
            "transcription_segments": mock_transcription_segments,
            "slang_annotations": mock_slang_annotations,
        }

    @patch("builtins.open", new_callable=mock_open)
    def test_all_12_column_headers(
        self, mock_file, mock_formatted_segments,
        mock_transcription_segments, mock_slang_annotations
    ):
        """Verify the CSV has all 12 required column headers."""
        expected_headers = [
            "subtitle_id", "start_time", "end_time", "duration_sec",
            "english_text", "hebrew_text", "confidence_score",
            "low_confidence_words", "slang_flags", "translation_notes",
            "chars_per_sec", "accuracy_flag",
        ]

        # Capture what gets written to the CSV
        written_data = io.StringIO()
        mock_file.return_value.write = written_data.write

        agent = CSVExportAgent()
        context = self._build_full_context(
            mock_formatted_segments, mock_transcription_segments, mock_slang_annotations
        )
        agent.run(context)

        written_data.seek(0)
        content = written_data.getvalue()

        # The header line should contain all expected fields
        for header in expected_headers:
            assert header in content, f"Missing header: {header}"

    @patch("builtins.open", new_callable=mock_open)
    def test_low_confidence_triggers_review(
        self, mock_file, mock_formatted_segments,
        mock_transcription_segments, mock_slang_annotations
    ):
        """Verify low_confidence words trigger REVIEW flag."""
        # Segment id=2 has low probability words (all < 0.7)
        written_data = io.StringIO()
        mock_file.return_value.write = written_data.write

        agent = CSVExportAgent()
        context = self._build_full_context(
            mock_formatted_segments, mock_transcription_segments, mock_slang_annotations
        )
        agent.run(context)

        written_data.seek(0)
        content = written_data.getvalue()
        assert "low_confidence" in content, "Low-confidence segment should trigger REVIEW"

    @patch("builtins.open", new_callable=mock_open)
    def test_slang_triggers_review(
        self, mock_file, mock_formatted_segments,
        mock_transcription_segments, mock_slang_annotations
    ):
        """Verify segments with slang annotations trigger REVIEW flag."""
        written_data = io.StringIO()
        mock_file.return_value.write = written_data.write

        agent = CSVExportAgent()
        context = self._build_full_context(
            mock_formatted_segments, mock_transcription_segments, mock_slang_annotations
        )
        agent.run(context)

        written_data.seek(0)
        content = written_data.getvalue()
        assert "contains_slang" in content, "Slang-annotated segment should trigger REVIEW"

    @patch("builtins.open", new_callable=mock_open)
    def test_fast_reading_speed_triggers_review(self, mock_file):
        """Verify fast reading speed (>16 CPS) triggers REVIEW flag."""
        # Create a segment with very long Hebrew text and very short duration
        formatted_segments = [{
            "id": 0,
            "start": 0.0,
            "end": 1.001,  # ~1 second
            "english_text": "Test",
            "hebrew_text": "A" * 50,  # 50 chars / 1s = 50 CPS > 16
            "hebrew": "A" * 50,
            "hebrew_formatted": "A" * 50,
            "duration_original": 1.0,
            "duration_adjusted": 1.0,
            "translation_notes": "",
        }]

        written_data = io.StringIO()
        mock_file.return_value.write = written_data.write

        agent = CSVExportAgent()
        context = {
            "formatted_segments": formatted_segments,
            "transcription_segments": [
                {"id": 0, "text": "Test", "words": [
                    {"word": "Test", "start": 0.0, "end": 1.0, "probability": 0.99}
                ]},
            ],
            "slang_annotations": [],
        }
        agent.run(context)

        written_data.seek(0)
        content = written_data.getvalue()
        assert "fast_reading_speed" in content, "Fast reading speed should trigger REVIEW"

    @patch("builtins.open", new_callable=mock_open)
    def test_utf8_sig_encoding(
        self, mock_file, mock_formatted_segments,
        mock_transcription_segments, mock_slang_annotations
    ):
        """Verify CSV is written with utf-8-sig encoding."""
        agent = CSVExportAgent()
        context = self._build_full_context(
            mock_formatted_segments, mock_transcription_segments, mock_slang_annotations
        )
        agent.run(context)

        # Verify open was called with utf-8-sig
        open_calls = mock_file.call_args_list
        assert len(open_calls) >= 1
        first_call = open_calls[0]
        kwargs = first_call[1] if first_call[1] else {}
        assert kwargs.get("encoding") == "utf-8-sig"

    @patch("builtins.open", new_callable=mock_open)
    def test_context_gets_csv_output_path(
        self, mock_file, mock_formatted_segments,
        mock_transcription_segments, mock_slang_annotations
    ):
        """Verify context receives csv_output_path."""
        agent = CSVExportAgent()
        context = self._build_full_context(
            mock_formatted_segments, mock_transcription_segments, mock_slang_annotations
        )
        result = agent.run(context)

        assert "csv_output_path" in result
        assert result["csv_output_path"] == config.CSV_OUTPUT_PATH

    @patch("builtins.open", new_callable=mock_open)
    def test_no_review_flag_for_clean_segment(self, mock_file):
        """Verify clean segments (high confidence, no slang, normal speed) have no flag."""
        formatted_segments = [{
            "id": 0,
            "start": 0.0,
            "end": 5.0,
            "english_text": "Good morning.",
            "hebrew_text": "boker tov",
            "hebrew": "boker tov",
            "hebrew_formatted": "boker tov",
            "duration_original": 5.0,
            "duration_adjusted": 5.0,
            "translation_notes": "",
        }]

        written_data = io.StringIO()
        mock_file.return_value.write = written_data.write

        agent = CSVExportAgent()
        context = {
            "formatted_segments": formatted_segments,
            "transcription_segments": [{
                "id": 0, "text": "Good morning.",
                "words": [
                    {"word": "Good", "start": 0.0, "end": 0.5, "probability": 0.99},
                    {"word": "morning.", "start": 0.5, "end": 1.0, "probability": 0.98},
                ],
            }],
            "slang_annotations": [],
        }
        agent.run(context)

        written_data.seek(0)
        content = written_data.getvalue()
        assert "REVIEW" not in content, "Clean segment should NOT be flagged for REVIEW"

    @patch("builtins.open", new_callable=mock_open)
    def test_multiple_review_reasons(self, mock_file):
        """Verify a segment can have multiple REVIEW reasons."""
        formatted_segments = [{
            "id": 0,
            "start": 0.0,
            "end": 1.001,
            "english_text": "The rake is here.",
            "hebrew_text": "A" * 50,
            "hebrew": "A" * 50,
            "hebrew_formatted": "A" * 50,
            "duration_original": 1.0,
            "duration_adjusted": 1.0,
            "translation_notes": "",
        }]

        written_data = io.StringIO()
        mock_file.return_value.write = written_data.write

        agent = CSVExportAgent()
        context = {
            "formatted_segments": formatted_segments,
            "transcription_segments": [{
                "id": 0, "text": "The rake is here.",
                "words": [
                    {"word": "The", "start": 0.0, "end": 0.2, "probability": 0.3},
                    {"word": "rake", "start": 0.2, "end": 0.5, "probability": 0.4},
                    {"word": "is", "start": 0.5, "end": 0.6, "probability": 0.5},
                    {"word": "here.", "start": 0.6, "end": 1.0, "probability": 0.3},
                ],
            }],
            "slang_annotations": [
                {"segment_id": 0, "term": "rake", "meaning": "dissolute man"},
            ],
        }
        agent.run(context)

        written_data.seek(0)
        content = written_data.getvalue()
        assert "low_confidence" in content
        assert "contains_slang" in content
        assert "fast_reading_speed" in content

    @patch("builtins.open", new_callable=mock_open)
    def test_timestamp_format(self, mock_file):
        """Verify timestamps are formatted as HH:MM:SS,mmm."""
        from agents.csv_export_agent import _format_timestamp

        assert _format_timestamp(0.0) == "00:00:00,000"
        assert _format_timestamp(61.5) == "00:01:01,500"
        assert _format_timestamp(3661.123) == "01:01:01,123"
        assert _format_timestamp(7200.0) == "02:00:00,000"

    @patch("builtins.open", new_callable=mock_open)
    def test_csv_row_count(
        self, mock_file, mock_formatted_segments,
        mock_transcription_segments, mock_slang_annotations
    ):
        """Verify the CSV has one row per formatted segment."""
        written_data = io.StringIO()
        mock_file.return_value.write = written_data.write

        agent = CSVExportAgent()
        context = self._build_full_context(
            mock_formatted_segments, mock_transcription_segments, mock_slang_annotations
        )
        agent.run(context)

        written_data.seek(0)
        reader = csv.reader(written_data)
        rows = list(reader)
        # Header + data rows
        assert len(rows) == 1 + len(mock_formatted_segments)


# ===========================================================================
# Integration-style test: Verify context keys flow through agents
# ===========================================================================

class TestPipelineContextFlow:
    """Verify that context keys produced by each agent are consumed correctly
    by downstream agents (using fully mocked data)."""

    def test_timing_to_rtl_to_srt_flow(self, mock_translated_segments):
        """Test that TimingSync -> RTLFormatting -> SRTExport flows correctly."""
        # Agent 5: Timing
        timing_agent = TimingSyncAgent()
        context = {"translated_segments": mock_translated_segments}
        context = timing_agent.run(context)
        assert "timed_segments" in context

        # Agent 6: RTL Formatting
        rtl_agent = RTLFormattingAgent()
        context = rtl_agent.run(context)
        assert "formatted_segments" in context

        # Agent 7: SRT Export
        with patch("agents.srt_export_agent.srt.compose", return_value="srt content"):
            with patch("builtins.open", mock_open()):
                srt_agent = SRTExportAgent()
                context = srt_agent.run(context)
                assert "srt_output_path" in context

    def test_timing_to_rtl_to_csv_flow(
        self, mock_translated_segments, mock_transcription_segments, mock_slang_annotations
    ):
        """Test that TimingSync -> RTLFormatting -> CSVExport flows correctly."""
        context = {
            "translated_segments": mock_translated_segments,
            "transcription_segments": mock_transcription_segments,
            "slang_annotations": mock_slang_annotations,
        }

        # Agent 5
        timing_agent = TimingSyncAgent()
        context = timing_agent.run(context)

        # Agent 6
        rtl_agent = RTLFormattingAgent()
        context = rtl_agent.run(context)

        # Agent 8
        with patch("builtins.open", mock_open()):
            csv_agent = CSVExportAgent()
            context = csv_agent.run(context)
            assert "csv_output_path" in context
