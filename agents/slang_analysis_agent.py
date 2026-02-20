import json
import logging
from typing import Any, Dict

import anthropic

from agents.base_agent import BaseAgent
import config


class SlangAnalysisAgent(BaseAgent):
    """
    Agent 3 — Slang & Context Analysis

    Identifies Regency-era vocabulary and period-specific language patterns
    in the transcript that require special translation attention.

    Reads:
        context["full_text"]   — complete transcript text
        context["segments"]    — list of transcript segments (for segment IDs)

    Writes:
        context["slang_annotations"] — list of annotated terms/phrases
        context["glossary"]          — glossary dict loaded from disk
    """

    def __init__(self):
        super().__init__("SlangAnalysisAgent")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # ------------------------------------------------------------------
        # 1. Load glossary from disk
        # ------------------------------------------------------------------
        glossary: Dict[str, str] = {}
        try:
            with open(config.GLOSSARY_PATH, "r", encoding="utf-8") as f:
                glossary = json.load(f)
            self.logger.info(
                f"Loaded glossary with {len(glossary)} entries from {config.GLOSSARY_PATH}"
            )
        except FileNotFoundError:
            self.logger.warning(
                f"Glossary file not found at {config.GLOSSARY_PATH}; "
                "proceeding with empty glossary"
            )
        except json.JSONDecodeError as exc:
            self.logger.warning(
                f"Glossary file is not valid JSON ({exc}); "
                "proceeding with empty glossary"
            )

        # ------------------------------------------------------------------
        # 2. Build the system prompt
        # ------------------------------------------------------------------
        glossary_block = json.dumps(glossary, indent=2, ensure_ascii=False)

        system_prompt = (
            "You are an expert analyzer of Regency-era English dialogue, "
            "specializing in the language of the Netflix series *Bridgerton*.\n\n"
            "Your task is to examine the provided transcript and identify every "
            "instance of period-specific vocabulary, formal speech patterns, "
            "double meanings, and social-hierarchy references that would need "
            "special care when translating into Hebrew.\n\n"
            "## Reference Glossary\n"
            f"```json\n{glossary_block}\n```\n\n"
            "## What to Identify\n"
            "1. **Period vocabulary** — terms such as: \"the ton\", \"rake\", "
            "\"modiste\", \"promenade\", \"calling card\", "
            "\"diamond of the first water\", \"on-dit\", \"Season\".\n"
            "2. **Formal speech patterns** — phrases like: \"I dare say\", "
            "\"I am most obliged\", \"pray tell\", \"it would not do\".\n"
            "3. **Double meanings / innuendo** — language that carries a "
            "subtext common in Bridgerton dialogue.\n"
            "4. **Titles & social hierarchy** — forms of address such as "
            "\"Your Grace\", \"Lady Whistledown\", \"Viscount\", and any "
            "rank-related language.\n"
            "5. **Phrases where literal translation loses meaning** — idioms "
            "or culturally loaded expressions whose sense would be lost in "
            "a word-for-word Hebrew translation.\n\n"
            "## Output Format\n"
            "Return a **JSON array** (no markdown fences, no extra commentary) "
            "where each element has exactly these fields:\n"
            "```\n"
            "{\n"
            '    "phrase": "<exact text from transcript>",\n'
            '    "segment_id": <integer — which transcript segment it appears in>,\n'
            '    "meaning": "<explanation of meaning / connotation>",\n'
            '    "translation_approach": "<suggested Hebrew translation approach>"\n'
            "}\n"
            "```\n"
            "Return ONLY the JSON array."
        )

        # ------------------------------------------------------------------
        # 3. Call Claude API
        # ------------------------------------------------------------------
        full_text = context.get("full_text", "")
        if not full_text:
            self.logger.warning("No full_text found in context; skipping analysis")
            context["slang_annotations"] = []
            context["glossary"] = glossary
            return context

        client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)

        self.logger.info("Sending transcript to Claude for slang/context analysis...")
        response = client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Analyze the following transcript and return the JSON "
                        "array of Regency-era terms and special phrases:\n\n"
                        f"{full_text}"
                    ),
                }
            ],
        )

        raw_text = response.content[0].text.strip()

        # ------------------------------------------------------------------
        # 4. Parse JSON response (handle markdown code fences)
        # ------------------------------------------------------------------
        annotations = []
        try:
            # Strip markdown code fences if present
            cleaned = raw_text
            if cleaned.startswith("```json"):
                cleaned = cleaned[len("```json"):]
            elif cleaned.startswith("```"):
                cleaned = cleaned[len("```"):]
            if cleaned.endswith("```"):
                cleaned = cleaned[: -len("```")]
            cleaned = cleaned.strip()

            annotations = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError) as exc:
            self.logger.warning(
                f"Failed to parse Claude response as JSON ({exc}); "
                "storing empty annotations list"
            )
            annotations = []

        # ------------------------------------------------------------------
        # 5. Store results in context
        # ------------------------------------------------------------------
        context["slang_annotations"] = annotations
        context["glossary"] = glossary

        self.logger.info(
            f"Slang analysis complete: {len(annotations)} annotations found"
        )

        return context
