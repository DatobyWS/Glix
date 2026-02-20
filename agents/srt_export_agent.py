"""Agent 7 -- SRT Export: Generate standards-compliant SRT subtitle files."""

from datetime import timedelta
from typing import Any, Dict

import srt

import config
from agents.base_agent import BaseAgent


class SRTExportAgent(BaseAgent):
    """Generate a main Hebrew SRT file and a debug bilingual SRT for QA."""

    def __init__(self):
        super().__init__("SRTExport")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        formatted_segments = context["formatted_segments"]

        # ---- Build main subtitle list (Hebrew only) ----
        subtitles = []
        for seg in formatted_segments:
            subtitles.append(
                srt.Subtitle(
                    index=seg["id"] + 1,  # SRT is 1-indexed
                    start=timedelta(seconds=seg["start"]),
                    end=timedelta(seconds=seg["end"]),
                    content=seg["hebrew_formatted"],
                )
            )

        # ---- Write main SRT file (UTF-8 BOM) ----
        srt_text = srt.compose(subtitles)
        with open(config.SRT_OUTPUT_PATH, "w", encoding="utf-8-sig") as f:
            f.write(srt_text)

        self.logger.info(
            "Wrote %d subtitles to %s", len(subtitles), config.SRT_OUTPUT_PATH
        )

        # ---- Build debug subtitle list (English + Hebrew) ----
        debug_subtitles = []
        for seg in formatted_segments:
            debug_content = seg["english"] + "\n---\n" + seg["hebrew_formatted"]
            debug_subtitles.append(
                srt.Subtitle(
                    index=seg["id"] + 1,
                    start=timedelta(seconds=seg["start"]),
                    end=timedelta(seconds=seg["end"]),
                    content=debug_content,
                )
            )

        # ---- Write debug SRT file (UTF-8 BOM) ----
        debug_path = config.SRT_OUTPUT_PATH.replace(".srt", "_debug.srt")
        debug_text = srt.compose(debug_subtitles)
        with open(debug_path, "w", encoding="utf-8-sig") as f:
            f.write(debug_text)

        self.logger.info("Wrote %d debug subtitles to %s", len(debug_subtitles), debug_path)

        # ---- Update context ----
        context["srt_output_path"] = config.SRT_OUTPUT_PATH
        context["srt_debug_path"] = debug_path

        return context
