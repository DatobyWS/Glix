import csv
import datetime
from typing import Any, Dict

from agents.base_agent import BaseAgent
import config


def _format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class CSVExportAgent(BaseAgent):
    """Agent 8: Generate a comprehensive data CSV for Google Sheets/Excel
    with all extracted data and accuracy flags."""

    def __init__(self):
        super().__init__("CSVExport")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # 1. Read inputs from context
        # ------------------------------------------------------------------
        formatted_segments = context.get("formatted_segments", [])
        transcription_segments = context.get("transcription_segments", [])
        slang_annotations = context.get("slang_annotations", [])

        # ------------------------------------------------------------------
        # 2. Build lookup of transcription segments by id for word-level data
        # ------------------------------------------------------------------
        transcription_by_id: Dict[int, dict] = {}
        for seg in transcription_segments:
            seg_id = seg.get("id")
            if seg_id is not None:
                transcription_by_id[seg_id] = seg

        # ------------------------------------------------------------------
        # 3. Build lookup of slang annotations by segment_id
        # ------------------------------------------------------------------
        slang_by_segment: Dict[int, list] = {}
        for ann in slang_annotations:
            seg_id = ann.get("segment_id")
            if seg_id is not None:
                slang_by_segment.setdefault(seg_id, []).append(ann)

        # ------------------------------------------------------------------
        # 4. Build CSV rows
        # ------------------------------------------------------------------
        fieldnames = [
            "subtitle_id",
            "start_time",
            "end_time",
            "duration_sec",
            "english_text",
            "hebrew_text",
            "confidence_score",
            "low_confidence_words",
            "slang_flags",
            "translation_notes",
            "chars_per_sec",
            "accuracy_flag",
        ]

        rows = []
        review_count = 0

        for idx, seg in enumerate(formatted_segments, start=1):
            seg_id = seg.get("id", idx)

            # --- Timestamps & duration ---
            start = seg.get("start", 0.0)
            end = seg.get("end", 0.0)
            duration = round(end - start, 3)
            if duration <= 0:
                duration = 0.001  # avoid division by zero

            # --- English text ---
            english_text = seg.get("english", seg.get("english_text", seg.get("text", "")))

            # --- Hebrew text (plain, no RTL chars) ---
            hebrew_text = seg.get("hebrew", seg.get("hebrew_text", ""))
            # Strip any RTL unicode marks that may have been embedded
            for ch in (config.RLM, config.RLE, config.PDF):
                hebrew_text = hebrew_text.replace(ch, "")

            # --- Word-level data from transcription segments ---
            trans_seg = transcription_by_id.get(seg_id, {})
            words = trans_seg.get("words", [])

            # Average confidence from word probabilities
            if words:
                probabilities = [
                    w.get("probability", w.get("confidence", 0.0))
                    for w in words
                ]
                avg_confidence = round(sum(probabilities) / len(probabilities), 4)
            else:
                avg_confidence = 0.0

            # Low confidence words (probability < 0.7)
            low_conf_words = []
            for w in words:
                prob = w.get("probability", w.get("confidence", 0.0))
                if prob < 0.7:
                    word_text = w.get("word", w.get("text", "")).strip()
                    if word_text:
                        low_conf_words.append(word_text)
            low_confidence_str = "|".join(low_conf_words)

            # --- Slang flags for this segment ---
            segment_slang = slang_by_segment.get(seg_id, [])
            slang_terms = []
            for ann in segment_slang:
                term = ann.get("phrase", ann.get("term", ann.get("word", "")))
                if term:
                    slang_terms.append(term)
            slang_flags_str = "|".join(slang_terms)

            # --- Translation notes ---
            translation_notes = seg.get("translation_notes", "")

            # --- Hebrew reading speed (chars per second) ---
            hebrew_no_spaces = hebrew_text.replace(" ", "")
            chars_per_sec = round(len(hebrew_no_spaces) / duration, 2)

            # --- Accuracy flag ---
            reasons = []
            if avg_confidence < 0.7:
                reasons.append("low_confidence")
            if slang_terms:
                reasons.append("contains_slang")
            if chars_per_sec > 16:
                reasons.append("fast_reading_speed")

            if reasons:
                accuracy_flag = f"REVIEW ({', '.join(reasons)})"
                review_count += 1
            else:
                accuracy_flag = ""

            # --- Format timestamps ---
            start_time_str = _format_timestamp(start)
            end_time_str = _format_timestamp(end)

            rows.append({
                "subtitle_id": idx,
                "start_time": start_time_str,
                "end_time": end_time_str,
                "duration_sec": duration,
                "english_text": english_text,
                "hebrew_text": hebrew_text,
                "confidence_score": avg_confidence,
                "low_confidence_words": low_confidence_str,
                "slang_flags": slang_flags_str,
                "translation_notes": translation_notes,
                "chars_per_sec": chars_per_sec,
                "accuracy_flag": accuracy_flag,
            })

        # ------------------------------------------------------------------
        # 5. Write CSV with UTF-8 BOM for Excel compatibility
        # ------------------------------------------------------------------
        output_path = config.CSV_OUTPUT_PATH
        with open(output_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        # ------------------------------------------------------------------
        # 6. Update context and log summary
        # ------------------------------------------------------------------
        context["csv_output_path"] = output_path

        self.logger.info(
            f"CSV exported: {len(rows)} rows, {review_count} flagged for REVIEW "
            f"-> {output_path}"
        )

        return context
