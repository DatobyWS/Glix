import whisper
from typing import Any, Dict

from agents.base_agent import BaseAgent
import config


class TranscriptionAgent(BaseAgent):
    """Agent 2: Transcribe English audio using Whisper with word-level timestamps."""

    def __init__(self):
        super().__init__("TranscriptionAgent")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        audio_path = context["audio_path"]
        self.logger.info(f"Loading Whisper model '{config.WHISPER_MODEL}'...")
        model = whisper.load_model(config.WHISPER_MODEL)

        self.logger.info(f"Transcribing '{audio_path}' (language={config.WHISPER_LANGUAGE}, word_timestamps=True)...")
        result = model.transcribe(
            audio_path,
            language=config.WHISPER_LANGUAGE,
            word_timestamps=True,
            verbose=False,
        )

        segments = []
        for idx, seg in enumerate(result.get("segments", [])):
            words = []
            for w in seg.get("words", []):
                words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", 0.0),
                    "end": w.get("end", 0.0),
                    "probability": w.get("probability", 0.0),
                })

            segments.append({
                "id": idx,
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip(),
                "words": words,
            })

        full_text = " ".join(seg["text"] for seg in segments)

        context["transcription_segments"] = segments
        context["full_text"] = full_text

        self.logger.info(f"Transcription complete: {len(segments)} segments found")
        return context
