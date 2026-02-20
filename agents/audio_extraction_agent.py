import os
import subprocess
from typing import Any, Dict

from agents.base_agent import BaseAgent
import config


class AudioExtractionAgent(BaseAgent):
    """
    Agent 1 -- Extract audio from the source MP4 video file.

    Runs ffmpeg to produce a 16-bit PCM WAV file at 16 kHz mono,
    which is the exact format OpenAI Whisper expects for transcription.
    """

    def __init__(self):
        super().__init__(name="AudioExtractionAgent")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # 1. Ensure the output directory exists
        # ------------------------------------------------------------------
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)
        self.logger.info(f"Output directory ready: {config.OUTPUT_DIR}")

        # ------------------------------------------------------------------
        # 2. Build and run the ffmpeg command
        # ------------------------------------------------------------------
        cmd = [
            "ffmpeg",
            "-y",                       # overwrite without asking
            "-i", config.VIDEO_PATH,    # input video
            "-vn",                      # no video stream
            "-acodec", "pcm_s16le",     # 16-bit PCM (uncompressed)
            "-ar", str(config.AUDIO_SAMPLE_RATE),  # 16 kHz sample rate
            "-ac", "1",                 # mono channel
            config.AUDIO_OUTPUT_PATH,
        ]

        self.logger.info(f"Running ffmpeg: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg exited with code {result.returncode}.\n"
                f"stderr:\n{result.stderr}"
            )

        self.logger.info("ffmpeg completed successfully")

        # ------------------------------------------------------------------
        # 3. Validate the output file exists and has a reasonable size
        # ------------------------------------------------------------------
        if not os.path.isfile(config.AUDIO_OUTPUT_PATH):
            raise FileNotFoundError(
                f"Audio output file was not created: {config.AUDIO_OUTPUT_PATH}"
            )

        file_size = os.path.getsize(config.AUDIO_OUTPUT_PATH)
        if file_size == 0:
            raise RuntimeError(
                f"Audio output file is empty (0 bytes): {config.AUDIO_OUTPUT_PATH}"
            )

        size_mb = file_size / (1024 * 1024)
        self.logger.info(
            f"Audio extracted: {config.AUDIO_OUTPUT_PATH} ({size_mb:.1f} MB)"
        )

        # ------------------------------------------------------------------
        # 4. Store the audio path in the pipeline context
        # ------------------------------------------------------------------
        context["audio_path"] = config.AUDIO_OUTPUT_PATH

        return context
