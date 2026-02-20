"""
Agent 5 — Timing & Sync

Adjusts subtitle display timing so Hebrew text is readable.
Hebrew translations may differ in length from English originals,
so display durations are recalculated based on Hebrew character count
and clamped to industry-standard limits.  Overlap between consecutive
subtitles is prevented and long lines are split for readability.
"""

from typing import Any, Dict, List

from agents.base_agent import BaseAgent
from config import (
    GAP_BETWEEN_SUBS_MS,
    MAX_CHARS_PER_LINE,
    MAX_DISPLAY_DURATION_MS,
    MAX_LINES_PER_SUBTITLE,
    MIN_DISPLAY_DURATION_MS,
    READING_SPEED_CPS,
)


class TimingSyncAgent(BaseAgent):
    """Adjust subtitle timing for Hebrew readability and split long lines."""

    def __init__(self) -> None:
        super().__init__(name="TimingSync")

    # ------------------------------------------------------------------
    # Line splitting
    # ------------------------------------------------------------------
    @staticmethod
    def _split_line(text: str) -> str:
        """Split *text* into at most two lines for subtitle display.

        If the text is within MAX_CHARS_PER_LINE it is returned as-is.
        Otherwise, find the space nearest to the middle and split there.
        If either resulting line still exceeds MAX_CHARS_PER_LINE,
        truncate it with '...'.
        """
        if len(text) <= MAX_CHARS_PER_LINE:
            return text

        mid = len(text) // 2

        # Search outward from the midpoint for the nearest space.
        best_pos: int | None = None
        for offset in range(mid + 1):
            left = mid - offset
            right = mid + offset
            if left >= 0 and text[left] == " ":
                best_pos = left
                break
            if right < len(text) and text[right] == " ":
                best_pos = right
                break

        if best_pos is None:
            # No space found — hard-truncate the single line.
            return text[:MAX_CHARS_PER_LINE - 3] + "..."

        line1 = text[:best_pos]
        line2 = text[best_pos + 1:]  # skip the space itself

        # Truncate individual lines that are still too long.
        if len(line1) > MAX_CHARS_PER_LINE:
            line1 = line1[:MAX_CHARS_PER_LINE - 3] + "..."
        if len(line2) > MAX_CHARS_PER_LINE:
            line2 = line2[:MAX_CHARS_PER_LINE - 3] + "..."

        return f"{line1}\n{line2}"

    # ------------------------------------------------------------------
    # Core timing logic
    # ------------------------------------------------------------------
    def _adjust_segments(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Return a new list of timed segment dicts."""
        min_dur = MIN_DISPLAY_DURATION_MS / 1000.0
        max_dur = MAX_DISPLAY_DURATION_MS / 1000.0
        gap = GAP_BETWEEN_SUBS_MS / 1000.0

        timed: List[Dict[str, Any]] = []

        for idx, seg in enumerate(segments):
            start: float = seg["start"]
            end: float = seg["end"]
            english: str = seg.get("english", seg.get("text", ""))
            hebrew: str = seg.get("hebrew", "")

            original_duration = end - start

            # 1. Minimum reading time based on Hebrew character count.
            hebrew_char_count = len(hebrew.replace(" ", ""))
            min_reading_duration = (
                hebrew_char_count / READING_SPEED_CPS if READING_SPEED_CPS else 0
            )

            # 2. Choose display duration and clamp.
            duration = max(original_duration, min_reading_duration)
            duration = max(min_dur, min(duration, max_dur))

            new_end = start + duration

            # 3. Prevent overlap with the next subtitle.
            if idx < len(segments) - 1:
                next_start: float = segments[idx + 1]["start"]
                if new_end > next_start - gap:
                    new_end = next_start - gap
                    # Ensure end never precedes start.
                    if new_end < start:
                        new_end = start + min_dur

            duration_adjusted = new_end - start

            # 4. Line splitting for long Hebrew text.
            hebrew_display = self._split_line(hebrew)

            timed.append(
                {
                    "id": seg.get("id", idx + 1),
                    "start": start,
                    "end": round(new_end, 3),
                    "english": english,
                    "hebrew": hebrew_display,
                    "duration_original": round(original_duration, 3),
                    "duration_adjusted": round(duration_adjusted, 3),
                }
            )

        return timed

    # ------------------------------------------------------------------
    # Agent entry point
    # ------------------------------------------------------------------
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        translated_segments: List[Dict[str, Any]] = context["translated_segments"]
        self.logger.info(
            "Adjusting timing for %d segments", len(translated_segments)
        )

        timed_segments = self._adjust_segments(translated_segments)

        # Log average adjusted duration.
        if timed_segments:
            avg_duration = sum(
                s["duration_adjusted"] for s in timed_segments
            ) / len(timed_segments)
            self.logger.info(
                "Average adjusted duration: %.2fs (across %d segments)",
                avg_duration,
                len(timed_segments),
            )

        context["timed_segments"] = timed_segments
        return context
