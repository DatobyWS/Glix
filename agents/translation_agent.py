import json
from typing import Any, Dict, List

import anthropic

from agents.base_agent import BaseAgent
import config


class TranslationAgent(BaseAgent):
    """
    Agent 4 — Hebrew Translation

    Translates each subtitle segment to literary Hebrew using Claude,
    incorporating slang annotations, glossary context, and show context
    from upstream agents.

    Reads:
        context["transcription_segments"] — list of transcript segments
        context["slang_annotations"]      — annotated terms from SlangAnalysisAgent
        context["glossary"]               — glossary dict with established translations
        context["show_context"]           — global metadata from ContextAnalysisAgent
        context["segment_contexts"]       — per-segment context windows and scores

    Writes:
        context["translated_segments"] — segments enriched with 'hebrew' and
                                         'translation_notes' fields
    """

    def __init__(self):
        super().__init__("TranslationAgent")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(
        self,
        glossary: Dict[str, str],
        annotations: List[Dict[str, Any]],
        show_context: Dict[str, Any] | None = None,
    ) -> str:
        """Build the detailed system prompt for Hebrew translation."""

        glossary_block = json.dumps(glossary, indent=2, ensure_ascii=False)
        annotations_block = json.dumps(annotations, indent=2, ensure_ascii=False)

        # Build show context section if available
        context_section = ""
        if show_context:
            genre = show_context.get("genre", "unknown")
            tone = show_context.get("tone", "unknown")
            setting = show_context.get("setting", "unknown")
            characters = show_context.get("characters", [])
            directives = show_context.get("translation_directives", [])

            context_section = (
                "## Show Context\n"
                f"- **Genre:** {genre}\n"
                f"- **Tone:** {tone}\n"
                f"- **Setting:** {setting}\n"
            )

            if characters:
                context_section += "\n**Characters:**\n"
                for char in characters:
                    name = char.get("name", "Unknown")
                    gender = char.get("gender", "unknown")
                    role = char.get("role", "")
                    context_section += f"- {name} (gender: {gender}) — {role}\n"

            if directives:
                context_section += "\n**Translation Directives:**\n"
                for d in directives:
                    context_section += f"- {d}\n"

            context_section += "\n"

        return (
            "You are an expert Hebrew translator specializing in subtitle "
            "dialogue. Your task is to translate English subtitle segments into "
            "literary Hebrew suitable for on-screen subtitles.\n\n"

            f"{context_section}"

            "## Established Glossary\n"
            "Use the following pre-approved Hebrew translations whenever the "
            "corresponding English term appears:\n"
            f"```json\n{glossary_block}\n```\n\n"

            "## Slang & Context Annotations\n"
            "The following annotations describe vocabulary and "
            "language that requires special translation care:\n"
            f"```json\n{annotations_block}\n```\n\n"

            "## Translation Guidelines\n"
            "1. Use **literary Hebrew** (not colloquial modern Hebrew). The "
            "   register should feel elevated and period-appropriate.\n"
            "2. **Preserve social hierarchy** through register shifts — "
            "   servants speak differently from dukes.\n"
            "3. Keep translations **concise** for subtitle readability. Every "
            "   word must earn its place on screen.\n"
            "4. **Transliterate common names** to Hebrew script (e.g. "
            '   "Daphne" \u2192 "\u05d3\u05e4\u05e0\u05d4", "Simon" \u2192 "\u05e1\u05d9\u05de\u05d5\u05df").\n'
            "5. Keep **titles in Hebrew** (e.g. "
            '   "Duke" \u2192 "\u05d4\u05d3\u05d5\u05db\u05e1", "Viscountess" \u2192 "\u05d4\u05d5\u05d9\u05e7\u05d5\u05e0\u05d8\u05e1\u05d4").\n'
            "6. **Preserve emotional tone** — sarcasm, tenderness, authority, "
            "   and irony should come through in the Hebrew.\n"
            "7. Where possible, find **Hebrew equivalents for double "
            "   entendres** rather than flattening them.\n"
            "8. **Maintain cross-segment coherence** — if a character or "
            "   concept is mentioned across multiple segments, use consistent "
            "   Hebrew terms and correct gender forms throughout.\n\n"

            "## Output Format\n"
            "Return a **JSON array** (no extra commentary) where each element "
            "has exactly these fields:\n"
            "```\n"
            "{\n"
            '    "id": <integer \u2014 matching the input segment id>,\n'
            '    "hebrew": "<translated Hebrew text>",\n'
            '    "notes": "<brief note on translation choices, or empty string>"\n'
            "}\n"
            "```\n"
            "Return ONLY the JSON array."
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove markdown code fences from a string."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[len("```json"):]
        elif cleaned.startswith("```"):
            cleaned = cleaned[len("```"):]
        if cleaned.endswith("```"):
            cleaned = cleaned[: -len("```")]
        return cleaned.strip()

    def _translate_batch(
        self,
        client: anthropic.Anthropic,
        system_prompt: str,
        batch: List[Dict[str, Any]],
        segment_contexts_by_id: Dict[int, Dict[str, Any]] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Translate a batch of segments via Claude.

        Returns a list of dicts with keys: id, hebrew, notes.
        Falls back to single-segment translation if batch parsing fails.
        """

        # Build the user payload — only id and english text
        payload = [{"id": seg["id"], "english": seg["text"]} for seg in batch]
        user_message = json.dumps(payload, indent=2, ensure_ascii=False)

        # Build surrounding context section
        context_section = ""
        if segment_contexts_by_id:
            first_id = batch[0]["id"]
            last_id = batch[-1]["id"]
            first_ctx = segment_contexts_by_id.get(first_id, {})
            last_ctx = segment_contexts_by_id.get(last_id, {})

            before_text = first_ctx.get("before_text", "")
            after_text = last_ctx.get("after_text", "")

            if before_text or after_text:
                context_section = "\n\n## Surrounding Context (DO NOT translate — for reference only)\n"
                if before_text:
                    context_section += f"**Before:** {before_text}\n"
                if after_text:
                    context_section += f"**After:** {after_text}\n"

            # Add notes for high-scoring segments
            high_score_notes = []
            for seg in batch:
                ctx = segment_contexts_by_id.get(seg["id"], {})
                score = ctx.get("context_score", 0)
                if score >= 0.6:
                    gender_cues = ctx.get("gender_cues", {})
                    if gender_cues:
                        cue_str = ", ".join(
                            f"{k}={v}" for k, v in gender_cues.items()
                        )
                        high_score_notes.append(
                            f"Segment {seg['id']}: high context dependency "
                            f"(score={score}) — gender cues: {cue_str}"
                        )
                    else:
                        high_score_notes.append(
                            f"Segment {seg['id']}: high context dependency "
                            f"(score={score}) — pay attention to pronouns/gender"
                        )

            if high_score_notes:
                context_section += "\n**Context Notes:**\n"
                for note in high_score_notes:
                    context_section += f"- {note}\n"

        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Translate the following subtitle segments to Hebrew. "
                        "Return a JSON array:\n\n"
                        f"{user_message}"
                        f"{context_section}"
                    ),
                }
            ],
        )

        raw_text = response.content[0].text.strip()
        cleaned = self._strip_code_fences(raw_text)

        try:
            results = json.loads(cleaned)
            if isinstance(results, list) and len(results) > 0:
                return results
            raise ValueError("Parsed JSON is not a non-empty list")
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            self.logger.warning(
                f"Batch JSON parse failed ({exc}); falling back to "
                "single-segment translation"
            )
            return self._translate_singles(client, system_prompt, batch)

    def _translate_singles(
        self,
        client: anthropic.Anthropic,
        system_prompt: str,
        segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Fallback: translate each segment one at a time.

        Asks Claude to return ONLY the Hebrew text (no JSON), so parsing
        is trivial and the pipeline never fails completely.
        """
        results: List[Dict[str, Any]] = []

        for seg in segments:
            try:
                response = client.messages.create(
                    model=config.CLAUDE_MODEL,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=[
                        {
                            "role": "user",
                            "content": (
                                "Translate the following single subtitle line "
                                "to Hebrew. Return ONLY the Hebrew text, "
                                "nothing else.\n\n"
                                f"{seg['text']}"
                            ),
                        }
                    ],
                )
                hebrew = response.content[0].text.strip()
                results.append({
                    "id": seg["id"],
                    "hebrew": hebrew,
                    "notes": "single-segment fallback",
                })
            except Exception as exc:
                self.logger.error(
                    f"Single-segment translation failed for segment "
                    f"{seg['id']}: {exc}"
                )
                results.append({
                    "id": seg["id"],
                    "hebrew": seg["text"],
                    "notes": f"translation failed: {exc}",
                })

        return results

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        segments = context.get("transcription_segments", [])
        annotations = context.get("slang_annotations", [])
        glossary = context.get("glossary", {})
        show_context = context.get("show_context", {})
        segment_contexts = context.get("segment_contexts", [])

        if not segments:
            self.logger.warning(
                "No transcription segments found in context; "
                "skipping translation"
            )
            context["translated_segments"] = []
            return context

        # Build the system prompt once (shared by all batches)
        system_prompt = self._build_system_prompt(
            glossary, annotations, show_context or None
        )

        # Build lookup for segment contexts by id
        segment_contexts_by_id: Dict[int, Dict[str, Any]] = {}
        for sc in segment_contexts:
            segment_contexts_by_id[sc["id"]] = sc

        # Create Anthropic client
        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        # ------------------------------------------------------------------
        # Process segments in batches
        # ------------------------------------------------------------------
        batch_size = config.TRANSLATION_BATCH_SIZE
        total_batches = (len(segments) + batch_size - 1) // batch_size
        all_translations: Dict[int, Dict[str, Any]] = {}

        for i in range(0, len(segments), batch_size):
            batch = segments[i : i + batch_size]
            batch_num = (i // batch_size) + 1

            self.logger.info(
                f"Translating batch {batch_num} of {total_batches} "
                f"({len(batch)} segments)"
            )

            results = self._translate_batch(
                client, system_prompt, batch,
                segment_contexts_by_id if segment_contexts_by_id else None,
            )

            # Index results by segment id for easy lookup
            for item in results:
                all_translations[item["id"]] = item

        # ------------------------------------------------------------------
        # Merge translations back into segments
        # ------------------------------------------------------------------
        translated_segments: List[Dict[str, Any]] = []

        for seg in segments:
            merged = dict(seg)  # shallow copy of original segment
            tr = all_translations.get(seg["id"], {})
            merged["hebrew"] = tr.get("hebrew", "")
            merged["translation_notes"] = tr.get("notes", "")
            translated_segments.append(merged)

        context["translated_segments"] = translated_segments

        translated_count = sum(
            1 for s in translated_segments if s.get("hebrew")
        )
        self.logger.info(
            f"Translation complete: {translated_count}/{len(segments)} "
            "segments translated"
        )

        return context
