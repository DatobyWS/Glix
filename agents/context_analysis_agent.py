import json
import re
from typing import Any, Dict, List, Set

import anthropic

from agents.base_agent import BaseAgent
import config


# Pronouns and gendered words that signal context dependency
_PRONOUNS = {"he", "she", "her", "him", "his", "hers", "they", "them", "their"}
_GENDERED_WORDS = {
    "brother", "sister", "father", "mother", "son", "daughter",
    "husband", "wife", "uncle", "aunt", "nephew", "niece",
    "cousin", "gentleman", "lady", "lord", "duke", "duchess",
    "viscount", "viscountess", "baron", "baroness", "king", "queen",
    "prince", "princess", "mr", "mrs", "miss", "sir", "madam",
}
_WORD_RE = re.compile(r"\b\w+\b")

# Gap threshold for conversation boundary detection (seconds)
_CONVERSATION_GAP = 2.0


class ContextAnalysisAgent(BaseAgent):
    """
    Agent 3.5 — Context Analysis

    Analyzes the transcript to build per-segment context windows,
    relevance scores, and global show metadata for the translation agent.

    Reads:
        context["transcription_segments"] — list of transcript segments

    Writes:
        context["segment_contexts"] — per-segment context windows and scores
        context["show_context"]     — global metadata (genre, tone, characters)
    """

    def __init__(self):
        super().__init__("ContextAnalysisAgent")

    # ------------------------------------------------------------------
    # Phase A: Local analysis (no API call)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_words(text: str) -> Set[str]:
        """Extract lowercase words from text."""
        return {w.lower() for w in _WORD_RE.findall(text)}

    @staticmethod
    def _has_pronouns(words: Set[str]) -> bool:
        return bool(words & _PRONOUNS)

    @staticmethod
    def _has_gendered_words(words: Set[str]) -> bool:
        return bool(words & _GENDERED_WORDS)

    @staticmethod
    def _extract_character_mentions(text: str) -> List[str]:
        """Extract pronoun and gendered word mentions from text."""
        words = _WORD_RE.findall(text)
        mentions = []
        for w in words:
            wl = w.lower()
            if wl in _PRONOUNS or wl in _GENDERED_WORDS:
                mentions.append(w)
        return mentions

    def _detect_conversations(
        self, segments: List[Dict[str, Any]]
    ) -> List[int]:
        """
        Assign a conversation_id to each segment based on timing gaps.
        Returns a list of conversation IDs (one per segment).
        """
        if not segments:
            return []

        conversation_ids = [0]
        current_conv = 0

        for i in range(1, len(segments)):
            prev_end = segments[i - 1].get("end", 0.0)
            curr_start = segments[i].get("start", 0.0)
            if curr_start - prev_end > _CONVERSATION_GAP:
                current_conv += 1
            conversation_ids.append(current_conv)

        return conversation_ids

    def _compute_context_score(
        self,
        seg: Dict[str, Any],
        seg_words: Set[str],
        prev_seg: Dict[str, Any] | None,
        next_seg: Dict[str, Any] | None,
    ) -> float:
        """
        Compute how dependent this segment is on surrounding context.
        Returns a score between 0.0 and 1.0.
        """
        score = 0.2  # base score

        # Contains pronouns — needs antecedent from context
        if self._has_pronouns(seg_words):
            score += 0.25

        # Contains gendered words — needs gender consistency
        if self._has_gendered_words(seg_words):
            score += 0.15

        # Short segment — likely part of dialogue exchange
        text = seg.get("text", "")
        if len(text) < 30:
            score += 0.15

        # Adjacent segments share words (conversation continuity)
        if prev_seg or next_seg:
            shared = 0
            if prev_seg:
                prev_words = self._extract_words(prev_seg.get("text", ""))
                # Exclude common words
                meaningful = (seg_words & prev_words) - {"the", "a", "an", "is", "are", "was", "were", "to", "and", "of", "in", "it", "i"}
                shared += len(meaningful)
            if next_seg:
                next_words = self._extract_words(next_seg.get("text", ""))
                meaningful = (seg_words & next_words) - {"the", "a", "an", "is", "are", "was", "were", "to", "and", "of", "in", "it", "i"}
                shared += len(meaningful)
            if shared > 0:
                score += min(0.15, shared * 0.05)

        return min(1.0, round(score, 2))

    def _build_segment_contexts(
        self, segments: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build per-segment context windows with scores.
        """
        conversation_ids = self._detect_conversations(segments)
        window_size = 10
        results = []

        for i, seg in enumerate(segments):
            # Build context window
            before_start = max(0, i - window_size)
            after_end = min(len(segments), i + window_size + 1)

            before_texts = [s.get("text", "") for s in segments[before_start:i]]
            after_texts = [s.get("text", "") for s in segments[i + 1:after_end]]

            before_text = " ".join(before_texts)
            after_text = " ".join(after_texts)

            # Extract words and mentions
            seg_text = seg.get("text", "")
            seg_words = self._extract_words(seg_text)
            character_mentions = self._extract_character_mentions(
                before_text + " " + seg_text + " " + after_text
            )

            # Infer gender cues from surrounding context
            gender_cues = self._infer_gender_cues(
                before_text + " " + seg_text + " " + after_text
            )

            # Compute context score
            prev_seg = segments[i - 1] if i > 0 else None
            next_seg = segments[i + 1] if i < len(segments) - 1 else None
            context_score = self._compute_context_score(
                seg, seg_words, prev_seg, next_seg
            )

            results.append({
                "id": seg.get("id", i),
                "context_score": context_score,
                "before_text": before_text,
                "after_text": after_text,
                "character_mentions": list(set(character_mentions)),
                "gender_cues": gender_cues,
                "conversation_id": conversation_ids[i] if i < len(conversation_ids) else 0,
            })

        return results

    @staticmethod
    def _infer_gender_cues(text: str) -> Dict[str, str]:
        """
        Infer gender of referenced entities from pronouns in surrounding text.
        Simple heuristic: look for patterns like 'she ... cousin' or 'his ... brother'.
        """
        cues: Dict[str, str] = {}
        text_lower = text.lower()

        # Check for gendered pronoun proximity to gendered/ambiguous words
        sentences = re.split(r'[.!?]', text_lower)
        for sentence in sentences:
            words = _WORD_RE.findall(sentence)
            has_female = any(w in {"she", "her", "hers", "herself", "lady", "miss", "mrs", "madam", "duchess", "viscountess", "baroness", "queen", "princess", "mother", "sister", "daughter", "wife", "aunt", "niece"} for w in words)
            has_male = any(w in {"he", "him", "his", "himself", "lord", "sir", "mr", "duke", "viscount", "baron", "king", "prince", "father", "brother", "son", "husband", "uncle", "nephew"} for w in words)

            # Tag ambiguous words with inferred gender
            for w in words:
                if w == "cousin":
                    if has_female and not has_male:
                        cues["cousin"] = "female"
                    elif has_male and not has_female:
                        cues["cousin"] = "male"
                elif w == "friend":
                    if has_female and not has_male:
                        cues["friend"] = "female"
                    elif has_male and not has_female:
                        cues["friend"] = "male"

        return cues

    # ------------------------------------------------------------------
    # Phase B: Global context via Claude (one small call)
    # ------------------------------------------------------------------

    def _build_summary_for_claude(
        self,
        segments: List[Dict[str, Any]],
        conversation_ids: List[int],
    ) -> str:
        """
        Build a condensed summary of the transcript for Claude.
        Takes first and last segment of each conversation group.
        """
        if not segments:
            return ""

        conv_groups: Dict[int, List[Dict[str, Any]]] = {}
        for seg, conv_id in zip(segments, conversation_ids):
            conv_groups.setdefault(conv_id, []).append(seg)

        summary_parts = []
        for conv_id in sorted(conv_groups.keys()):
            group = conv_groups[conv_id]
            texts = [s.get("text", "") for s in group]
            if len(texts) <= 3:
                summary_parts.append(" ".join(texts))
            else:
                # First 2 and last 1 sentences
                summary_parts.append(
                    f"{texts[0]} {texts[1]} [...] {texts[-1]}"
                )

        return "\n\n".join(summary_parts)

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

    def _get_global_context(
        self,
        segments: List[Dict[str, Any]],
        conversation_ids: List[int],
    ) -> Dict[str, Any]:
        """
        Use one Claude call on a condensed summary to extract global metadata.
        """
        summary = self._build_summary_for_claude(segments, conversation_ids)
        if not summary:
            return self._default_show_context()

        system_prompt = (
            "You are an expert media analyst. Analyze the following transcript "
            "excerpt and identify key metadata for a subtitle translator.\n\n"
            "## Output Format\n"
            "Return a JSON object (no extra commentary) with these fields:\n"
            "```\n"
            "{\n"
            '    "genre": "<detected genre/style, e.g. period drama, comedy, thriller>",\n'
            '    "tone": "<overall tone, e.g. formal, casual, dramatic, mixed>",\n'
            '    "setting": "<time period and location if detectable>",\n'
            '    "characters": [\n'
            '        {"name": "<name>", "gender": "<male/female/unknown>", "role": "<brief role>"}\n'
            "    ],\n"
            '    "translation_directives": [\n'
            '        "<specific instruction for translator, e.g. Use feminine forms for X>"\n'
            "    ]\n"
            "}\n"
            "```\n"
            "Return ONLY the JSON object."
        )

        try:
            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            response = client.messages.create(
                model=config.CLAUDE_MODEL,
                max_tokens=2048,
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Analyze this transcript and return the JSON metadata:\n\n"
                            f"{summary}"
                        ),
                    }
                ],
            )

            raw_text = response.content[0].text.strip()
            cleaned = self._strip_code_fences(raw_text)
            result = json.loads(cleaned)

            if isinstance(result, dict):
                return result

        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            self.logger.warning(
                f"Failed to parse Claude context response ({exc}); "
                "using default context"
            )
        except Exception as exc:
            self.logger.warning(
                f"Claude API call for context analysis failed ({exc}); "
                "using default context"
            )

        return self._default_show_context()

    @staticmethod
    def _default_show_context() -> Dict[str, Any]:
        return {
            "genre": "unknown",
            "tone": "unknown",
            "setting": "unknown",
            "characters": [],
            "translation_directives": [],
        }

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        segments = context.get("transcription_segments", [])

        if not segments:
            self.logger.warning(
                "No transcription segments found; skipping context analysis"
            )
            context["segment_contexts"] = []
            context["show_context"] = self._default_show_context()
            return context

        # Phase A: Local analysis
        self.logger.info(
            f"Building context windows for {len(segments)} segments..."
        )
        segment_contexts = self._build_segment_contexts(segments)

        # Log score distribution
        scores = [sc["context_score"] for sc in segment_contexts]
        high_score_count = sum(1 for s in scores if s >= 0.6)
        avg_score = sum(scores) / len(scores) if scores else 0
        self.logger.info(
            f"Context scores: avg={avg_score:.2f}, "
            f"{high_score_count}/{len(scores)} segments with score >= 0.6"
        )

        # Phase B: Global context via Claude
        conversation_ids = [sc["conversation_id"] for sc in segment_contexts]
        self.logger.info("Requesting global context analysis from Claude...")
        show_context = self._get_global_context(segments, conversation_ids)

        char_count = len(show_context.get("characters", []))
        self.logger.info(
            f"Global context: genre={show_context.get('genre', 'unknown')}, "
            f"{char_count} characters identified"
        )

        context["segment_contexts"] = segment_contexts
        context["show_context"] = show_context

        return context
