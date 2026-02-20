"""Agent 6 — RTL Formatting: Insert Unicode directional control characters
so Hebrew renders correctly across all major video players.

SRT has no built-in text direction mechanism, and different players handle
bare Hebrew text inconsistently. The solution is triple-layer Unicode
wrapping per line:

    RLE + RLM + hebrew_text + RLM + PDF

- RLE (U+202B): Right-to-Left Embedding — opens an RTL embedding scope
- RLM (U+200F): Right-to-Left Mark — reinforces direction at boundaries
- PDF (U+202C): Pop Directional Formatting — closes the embedding scope
"""

from typing import Any, Dict

from agents.base_agent import BaseAgent
import config


class RTLFormattingAgent(BaseAgent):
    """Wraps each Hebrew subtitle line with Unicode directional characters
    to guarantee correct right-to-left rendering in all major video players."""

    def __init__(self):
        super().__init__(name="RTLFormatting")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply triple-layer RTL wrapping to every Hebrew subtitle line.

        Reads:
            context["timed_segments"] — list of dicts with at least:
                id, start, end, english, hebrew, duration_original,
                duration_adjusted

        Writes:
            context["formatted_segments"] — same list with an added
                ``hebrew_formatted`` field on each segment.
        """
        timed_segments = context["timed_segments"]
        formatted_segments = []

        for segment in timed_segments:
            hebrew_text = segment["hebrew"]

            # Split into individual lines (Agent 5 may have produced 2 lines)
            lines = hebrew_text.split("\n")

            # Wrap each line: RLE + RLM + line + RLM + PDF
            wrapped_lines = [
                config.RLE + config.RLM + line + config.RLM + config.PDF
                for line in lines
            ]

            # Rejoin with newline
            hebrew_formatted = "\n".join(wrapped_lines)

            # Build the formatted segment, preserving all existing fields
            formatted_segment = {
                "id": segment["id"],
                "start": segment["start"],
                "end": segment["end"],
                "english": segment["english"],
                "hebrew": segment["hebrew"],
                "duration_original": segment["duration_original"],
                "duration_adjusted": segment["duration_adjusted"],
                "hebrew_formatted": hebrew_formatted,
            }
            formatted_segments.append(formatted_segment)

        context["formatted_segments"] = formatted_segments

        self.logger.info(
            f"[{self.name}] Formatted {len(formatted_segments)} segments "
            f"with triple-layer RTL wrapping (RLE+RLM+text+RLM+PDF)"
        )

        return context
