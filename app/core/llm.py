"""LLM-driven directive interpretation.

Wraps Google Gemini with:
  * structured prompt engineering
  * graceful fallback to the deterministic parser
  * JSON parsing with permissiveness around code fences
  * model fallback chain
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from app.core.config import settings
from app.models.schemas import BatteryData
from app.utils.parser import extract_directive_fallback


logger = logging.getLogger("gridwise.llm")


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
def build_llm_prompt(
    notes: List[str], battery_capacity: float, base_min_energy: float
) -> str:
    """Build the prompt sent to Gemini for directive extraction."""
    notes_text = "\n".join(f"{i}. {note}" for i, note in enumerate(notes))
    return (
        "You are an expert energy management system interpreter for a smart "
        "campus microgrid. Analyze the following operator notes for a 24-hour "
        "campus energy schedule (hours 0 through 23).\n\n"
        f"Campus Battery Information:\n"
        f"- Total Battery Capacity: {battery_capacity} kWh\n"
        f"- Base Minimum Energy Reserve: {base_min_energy} kWh\n\n"
        f"OPERATOR NOTES:\n{notes_text}\n\n"
        "For EACH note, extract exactly one directive. Allowed "
        "directive_type values:\n\n"
        "- solar_reduction: rooftop solar output is reduced during specific "
        'hours. Required structured_adjustment: {"hours": [list of integers '
        '0-23 in ascending order], "factor": float between 0.0 and 1.0}\n'
        '  NOTE: "factor" is the USABLE fraction remaining!\n'
        '    - "80% reduction" -> factor is 0.20\n'
        '    - "reduced by 60%" -> factor is 0.40\n'
        '    - "reduced to 25%" -> factor is 0.25\n'
        '    - "leave about half" -> factor is 0.50\n'
        '    - "leave roughly one-fifth" -> factor is 0.20\n'
        '    - "leave roughly one-fourth" or "quarter" -> factor is 0.25\n\n'
        "- minimum_battery_reserve: keep battery energy at or above a required "
        'level during specific hours. Required structured_adjustment: {"hours": '
        f'[...,], "minimum_energy_kwh": float}}.  NOTE: If note specifies a '
        'percentage of battery capacity (e.g. "50% of the battery capacity"), '
        f"calculate the kWh based on total capacity of {battery_capacity} kWh "
        f"(e.g. 50% = {0.5 * battery_capacity} kWh).\n\n"
        "- no_charge_window: battery charging is disabled/unavailable during "
        'specific hours. Required structured_adjustment: {"hours": [...]}\n\n'
        "- no_discharge_window: battery discharging is disabled/unavailable "
        'during specific hours. Required structured_adjustment: {"hours": '
        "[...]}\n\n"
        "- max_grid_window: campus grid import/intake must not exceed a stated "
        'limit during specific hours. Required structured_adjustment: {"hours": '
        '[...,], "max_grid_kwh": float}\n\n'
        "- no_op: the note does not impose any actionable constraint on the "
        "24-hour energy schedule (e.g. cafeteria menu, sports registration, "
        "library hours, seminar bookings). For no_op, applies MUST be false "
        "and structured_adjustment MUST be null.\n\n"
        "Time Window Rules:\n"
        "- Whole-hour intervals only. Start hour is INCLUDED, end hour is "
        "EXCLUDED.\n"
        '- "1 PM to 3 PM" -> hours [13, 14]\n'
        '- "noon until 2 PM" -> hours [12, 13]\n'
        '- "10 AM until noon" -> hours [10, 11]\n'
        '- "2 AM until 5 AM" -> hours [2, 3, 4]\n'
        '- "6 PM until 9 PM" -> hours [18, 19, 20]\n'
        '- "7 PM until 10 PM" -> hours [19, 20, 21]\n'
        '- "10 PM until midnight" -> hours [22, 23]\n'
        '- "10 PM until 2 AM" (overnight) -> hours [0, 1, 22, 23]\n'
        "- Hours must be unique integers from 0 through 23, sorted in "
        "ascending order.\n\n"
        "Return ONLY a valid JSON array of objects with keys:\n"
        "[\n"
        "  {\n"
        '    "note_index": 0,\n'
        '    "applies": true,\n'
        '    "directive_type": "...",\n'
        '    "structured_adjustment": { ... },\n'
        '    "explanation": "concise explanation"\n'
        "  }\n"
        "]\n"
    )


# ---------------------------------------------------------------------------
# LLM response parsing
# ---------------------------------------------------------------------------
def _parse_llm_json(text: str) -> List[Dict[str, Any]]:
    """Permissively parse a JSON array from an LLM response."""
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start == -1 or end <= start:
            raise
        data = json.loads(cleaned[start : end + 1])

    # Unwrap common shapes
    if isinstance(data, dict):
        for key in ("directives", "results", "interpretations", "notes"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
        else:
            data = [data] if "directive_type" in data else data
    if not isinstance(data, list):
        raise ValueError("LLM response is not a JSON array")
    return [x for x in data if isinstance(x, dict)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def interpret_notes_with_llm(
    notes: List[str], battery: BatteryData
) -> Tuple[List[Dict[str, Any]], str]:
    """Interpret operator notes via Gemini with deterministic fallback.

    Returns a ``(directives, interpreter_label)`` tuple.  The interpreter
    label is one of:
        ``"gemini:<model>"``        — Gemini returned valid output
        ``"rule-based (<reason>)"`` — fallback was used
    """

    def _fallback(reason: str) -> Tuple[List[Dict[str, Any]], str]:
        return (
            [extract_directive_fallback(n, battery.capacity_kwh) for n in notes],
            f"rule-based ({reason})",
        )

    api_key: Optional[str] = settings.gemini_api_key
    if not api_key:
        logger.info(
            "No GEMINI_API_KEY configured. Using deterministic parser for all notes."
        )
        return _fallback("no API key")

    try:
        import google.generativeai as genai  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - SDK is in requirements.txt
        logger.warning("Gemini SDK unavailable: %s. Using deterministic parser.", exc)
        return _fallback("SDK unavailable")

    try:
        genai.configure(api_key=api_key)
    except Exception as exc:  # pragma: no cover - SDK errors
        logger.warning("Failed to configure Gemini SDK: %s", exc)
        return _fallback("SDK configure error")

    prompt = build_llm_prompt(
        notes, battery.capacity_kwh, battery.minimum_energy_kwh
    )
    last_error: Optional[Exception] = None

    for model_name in settings.candidate_models():
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": settings.LLM_TEMPERATURE,
                    "response_mime_type": "application/json",
                },
                request_options={"timeout": settings.LLM_TIMEOUT_SECONDS},
            )
            parsed = _parse_llm_json(response.text)
            return parsed, f"gemini:{model_name}"
        except Exception as exc:
            last_error = exc
            logger.warning("LLM model '%s' failed: %s", model_name, exc)

    logger.warning(
        "All configured LLM models failed (%s). Using deterministic parser.",
        last_error,
    )
    return _fallback("LLM failed")
