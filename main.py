"""
GridWise LLM - BUP CSE Fest 2026 Hackathon Preliminary
Smart Campus Energy Optimization with LLM-assisted Operator Directive Interpretation

v1.1.0 changes (see README "What's new"):
  * Fixed: "reduced by 60%" style notes no longer fall back to a 50% default
  * Added: overnight time windows ("10 PM to 2 AM", "until midnight")
  * Added: battery consistency validation (422 instead of a solver crash)
  * Added: graceful handling of infeasible max_grid_window directives
  * Added: Gemini model fallback chain + request timeout (gemini-1.5-flash is retired)
  * Added: POST /api/analyze  (savings vs no-battery baseline, per-hour constraints, warnings)
  * Added: POST /api/export-csv, GET /api/status
  * /optimize-energy request/response contract is UNCHANGED.
"""

import csv
import io
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)
from scipy.optimize import linprog

load_dotenv()

# ------------------ Logging ------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise")

# ------------------ Config ------------------

APP_VERSION = "1.1.0"

# gemini-1.5-flash has been shut down; use a current model and keep older ones as fallbacks.
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_FALLBACKS = "gemini-2.5-flash"
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))

app = FastAPI(
    title="GridWise LLM",
    description="BUP CSE Fest 2026 - Smart Campus Energy Optimization Engine",
    version=APP_VERSION,
    swagger_ui_parameters={
        "syntaxHighlight.theme": "obsidian",
        "defaultModelsExpandDepth": -1,
    },
)


class InfeasibleScheduleError(Exception):
    """Raised when the LP has no feasible schedule for the given scenario."""


# ------------------ Models ------------------


class HourData(BaseModel):
    hour: int = Field(..., ge=0, le=23)
    demand_kwh: float = Field(..., ge=0)
    solar_kwh: float = Field(..., ge=0)
    tariff_bdt_per_kwh: float = Field(..., ge=0)


class BatteryData(BaseModel):
    capacity_kwh: float = Field(..., gt=0)
    initial_energy_kwh: float = Field(..., ge=0)
    minimum_energy_kwh: float = Field(..., ge=0)
    max_charge_kwh_per_hour: float = Field(..., ge=0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0)

    @model_validator(mode="after")
    def check_consistency(self):
        # Without these checks the LP is infeasible (E_23 must equal initial energy)
        # and the API used to answer with a confusing 500.
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if not (
            self.minimum_energy_kwh <= self.initial_energy_kwh <= self.capacity_kwh
        ):
            raise ValueError(
                "initial_energy_kwh must be between minimum_energy_kwh and capacity_kwh"
            )
        return self


class OptimizeRequest(BaseModel):
    scenario_id: str
    operator_notes: List[str] = Field(..., min_length=1, max_length=3)
    hours: List[HourData]
    battery: BatteryData

    @field_validator("hours")
    @classmethod
    def validate_hours(cls, v):
        if len(v) != 24:
            raise ValueError("hours must contain exactly 24 entries")
        hour_set = {h.hour for h in v}
        if hour_set != set(range(24)):
            raise ValueError("hours must contain unique hours 0 through 23")
        return sorted(v, key=lambda x: x.hour)


class StructuredAdjustment(BaseModel):
    hours: List[int] = Field(default_factory=list)
    factor: Optional[float] = None
    minimum_energy_kwh: Optional[float] = None
    max_grid_kwh: Optional[float] = None

    @model_serializer(mode="wrap")
    def serialize_adjustment(self, handler):
        data = handler(self)
        # Exclude null fields inside structured_adjustment
        return {k: v for k, v in data.items() if v is not None}


class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: Literal[
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op",
    ]
    structured_adjustment: Optional[StructuredAdjustment] = None
    explanation: str


class HourlyPlan(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlan]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


# ---- New (v1.1) response models used only by /api/analyze ----


class HourlyInsight(BaseModel):
    hour: int
    tariff_bdt_per_kwh: float
    demand_kwh: float
    effective_solar_kwh: float
    min_reserve_kwh: float
    max_grid_kwh: Optional[float] = None  # None = unlimited
    charge_allowed: bool
    discharge_allowed: bool
    baseline_grid_kwh: float
    cost_bdt: float


class Analytics(BaseModel):
    interpreter: str
    pipeline_time_ms: float
    baseline_cost_bdt: float
    savings_bdt: float
    savings_pct: float
    baseline_peak_grid_kwh: float
    peak_reduction_kwh: float
    solar_utilization_pct: float
    solar_curtailed_kwh: float
    battery_discharged_kwh: float
    equivalent_full_cycles: float
    warnings: List[str]
    hourly: List[HourlyInsight]


class AnalyzeResponse(BaseModel):
    optimization: OptimizeResponse
    analytics: Analytics


# ------------------ NLP Fallback & Extraction Utilities ------------------

SUPPORTED_DIRECTIVES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}

NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "noon": 12,
    "midday": 12,
    "midnight": 0,
}


def _parse_time_token(token: str, default_pm: bool = False) -> Optional[int]:
    """Parse a single time string or word into an hour 0..23."""
    t = token.strip().lower()
    if t in ("noon", "midday"):
        return 12
    if t in ("midnight",):
        return 0
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", t)
    if m:
        hr = int(m.group(1))
        period = m.group(3)
        if period == "pm" and hr < 12:
            hr += 12
        elif period == "am" and hr == 12:
            hr = 0
        elif not period and default_pm and hr < 12:
            hr += 12
        return hr
    if t in NUMBER_WORDS:
        val = NUMBER_WORDS[t]
        if default_pm and val < 12:
            val += 12
        return val
    return None


def _hours_between(h1: int, h2: int, allow_wrap: bool = False) -> List[int]:
    """
    Whole-hour list for the window [h1, h2) (start inclusive, end exclusive).

    * h2 == 0 with h1 > 0 is read as midnight (24), e.g. "10 PM until midnight".
    * allow_wrap=True lets an overnight window such as 22 -> 2 return [0, 1, 22, 23].
    """
    if h2 == 0 and h1 > 0:
        h2 = 24
    if 0 <= h1 < h2 <= 24:
        return list(range(h1, h2))
    if allow_wrap and 0 <= h2 < h1 <= 23:
        return sorted(list(range(h1, 24)) + list(range(0, h2)))
    return []


def _is_explicit_overnight(s1: str, s2: str) -> bool:
    """
    Only treat "start > end" as an overnight window when the text makes it unambiguous:
    "10 pm ... 2 am" or 24-hour clock times like "22:00 ... 02:00".
    Ambiguous text such as "12 to 3" must NOT wrap around.
    """
    if "pm" in s1 and "am" in s2:
        return True
    return ":" in s1 and ":" in s2 and "am" not in s1 + s2 and "pm" not in s1 + s2


def extract_hours_window(text: str) -> List[int]:
    """
    Extract start-inclusive, end-exclusive hours from natural language.
    Examples:
      'from noon until 2 PM'      -> [12, 13]
      '1 PM to 3 PM'              -> [13, 14]
      '1-3 PM'                    -> [13, 14]
      'between 13:00 and 15:00'   -> [13, 14]
      'from one until three'      -> [13, 14]
      'from 10 PM to 2 AM'        -> [0, 1, 22, 23]   (overnight)
      'from 10 PM until midnight' -> [22, 23]
    """
    t = text.lower()

    # Pattern A: '1-3 PM' or '1–3 PM'
    m_range = re.search(r"(\d{1,2})\s*[-–]\s*(\d{1,2})\s*(am|pm)", t)
    if m_range:
        period = m_range.group(3)
        h1 = int(m_range.group(1))
        h2 = int(m_range.group(2))
        if period == "pm":
            if h1 < 12:
                h1 += 12
            if h2 < 12:
                h2 += 12
        hrs = _hours_between(h1, h2)
        if hrs:
            return hrs

    # Pattern B: 'from X until/to Y' or 'between X and Y'
    m = re.search(
        r"(?:from|between)\s+([a-z0-9:]+(?:\s*(?:am|pm))?)\s+(?:until|to|and)\s+([a-z0-9:]+(?:\s*(?:am|pm))?)",
        t,
    )
    if m:
        s1, s2 = m.group(1).strip(), m.group(2).strip()
        has_pm = (
            ("pm" in s2)
            or ("noon" in s1)
            or ("evening" in t)
            or ("afternoon" in t)
            or ("solar" in t)
            or ("panel" in t)
        )
        h1 = _parse_time_token(s1, default_pm=has_pm)
        h2 = _parse_time_token(s2, default_pm=has_pm)
        if h1 is not None and h2 is not None:
            hrs = _hours_between(h1, h2, allow_wrap=_is_explicit_overnight(s1, s2))
            if hrs:
                return hrs

    # Pattern C: 'X until Y' or 'X to Y'
    m_to = re.search(
        r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:until|to)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
        t,
    )
    if m_to:
        s1, s2 = m_to.group(1).strip(), m_to.group(2).strip()
        p2 = "pm" if "pm" in s2 else ("am" if "am" in s2 else None)
        h2 = _parse_time_token(s2, default_pm=(p2 == "pm"))
        h1 = _parse_time_token(s1, default_pm=(p2 == "pm"))
        if h1 is not None and h2 is not None:
            hrs = _hours_between(h1, h2, allow_wrap=_is_explicit_overnight(s1, s2))
            if hrs:
                return hrs

    return []


_PCT = r"(\d+(?:\.\d+)?)\s*(?:%|percent)"


def _extract_solar_factor(t: str) -> float:
    """
    Return the USABLE solar fraction (0..1) described by a lower-cased note.
      '80% reduction'          -> 0.20
      'reduced by 60%'         -> 0.40   (NEW: previously fell back to 0.50)
      'reduced to 25%'         -> 0.25
      'roughly one-fourth'     -> 0.25
    """
    m_red = re.search(_PCT + r"\s*(?:reduction|drop|decline|loss|cut)", t)
    if m_red:
        return round(max(0.0, min(1.0, (100.0 - float(m_red.group(1))) / 100.0)), 4)

    m_by = re.search(
        r"(?:reduc\w*|drop\w*|declin\w*|fall\w*|lower\w*|cut|down|los[et]\w*)\s*(?:by|of)?\s*"
        + _PCT,
        t,
    )
    if m_by:
        return round(max(0.0, min(1.0, (100.0 - float(m_by.group(1))) / 100.0)), 4)

    m_pct = re.search(r"(?:to|roughly|about|leaving|leave|at)\s*" + _PCT, t)
    if m_pct:
        return round(max(0.0, min(1.0, float(m_pct.group(1)) / 100.0)), 4)

    if "half" in t:
        return 0.5
    if "one-fifth" in t:
        return 0.2
    if "one-fourth" in t or "quarter" in t:
        return 0.25
    if "one-third" in t:
        return round(1.0 / 3.0, 4)
    return 0.5


def extract_directive_fallback(
    note: str, battery_capacity: float = 200.0
) -> Dict[str, Any]:
    """
    Deterministic rule-based NLP extractor that accurately extracts directives
    across all supported types and paraphrased variations.
    """
    t = note.lower()
    hours = extract_hours_window(t)

    # 1. Solar reduction
    if any(k in t for k in ["solar", "rooftop", "panel", "pv"]):
        if any(
            k in t
            for k in [
                "reduc",
                "drop",
                "wash",
                "clean",
                "cloud",
                "outage",
                "inverter",
                "curtail",
            ]
        ):
            factor = _extract_solar_factor(t)
            return {
                "applies": True,
                "directive_type": "solar_reduction",
                "structured_adjustment": {"hours": hours, "factor": factor},
                "explanation": f"Solar availability reduced during hours {hours} (usable factor: {factor}).",
            }

    # 2. No discharge window (Check discharge before charge to avoid substring match)
    if ("discharge" in t or "discharging" in t) and any(
        k in t
        for k in [
            "do not",
            "must not",
            "isolated",
            "unavailable",
            "disabled",
            "no discharge",
            "cannot",
            "testing",
            "relay",
            "prevent",
        ]
    ):
        return {
            "applies": True,
            "directive_type": "no_discharge_window",
            "structured_adjustment": {"hours": hours},
            "explanation": f"Battery discharge is disabled during hours {hours}.",
        }

    # 3. No charge window
    if ("charge" in t or "charging" in t) and any(
        k in t
        for k in [
            "do not",
            "must not",
            "isolated",
            "unavailable",
            "disabled",
            "no charge",
            "cannot",
            "prevent",
        ]
    ):
        return {
            "applies": True,
            "directive_type": "no_charge_window",
            "structured_adjustment": {"hours": hours},
            "explanation": f"Battery charge is disabled during hours {hours}.",
        }

    # 4. Minimum battery reserve
    if any(
        k in t
        for k in [
            "reserve",
            "remain in the battery",
            "stored in the battery",
            "minimum energy",
            "keep at least",
        ]
    ):
        m_cap_pct = re.search(
            r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s*(?:the\s*)?battery\s*capacity)?", t
        )
        m_kwh = re.search(r"(\d+(?:\.\d+)?)\s*kwh", t)
        if "%" in t and m_cap_pct:
            pct = float(m_cap_pct.group(1))
            val = round(battery_capacity * (pct / 100.0), 2)
        elif m_kwh:
            val = float(m_kwh.group(1))
        else:
            val = round(battery_capacity * 0.5, 2)
        val = min(val, battery_capacity)
        return {
            "applies": True,
            "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": hours, "minimum_energy_kwh": val},
            "explanation": f"Battery reserve raised to {val} kWh during hours {hours}.",
        }

    # 5. Max grid window
    if any(
        k in t
        for k in [
            "grid import",
            "grid intake",
            "feeder",
            "transformer limit",
            "substation",
            "max grid",
        ]
    ):
        m_kwh = re.search(r"(\d+(?:\.\d+)?)\s*kwh", t)
        val = float(m_kwh.group(1)) if m_kwh else 150.0
        return {
            "applies": True,
            "directive_type": "max_grid_window",
            "structured_adjustment": {"hours": hours, "max_grid_kwh": val},
            "explanation": f"Grid import is capped at {val} kWh during hours {hours}.",
        }

    # 6. Default no_op for irrelevant notes / distractors
    return {
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": "This note does not affect today's 24-hour energy schedule.",
    }


# ------------------ LLM Interpretation & Guardrails ------------------


def get_llm_prompt(
    notes: List[str], battery_capacity: float, base_min_energy: float
) -> str:
    notes_text = "\n".join([f"{i}. {note}" for i, note in enumerate(notes)])
    return f"""You are an expert energy management system interpreter for a smart campus microgrid.
Analyze the following operator notes for a 24-hour campus energy schedule (hours 0 through 23).

Campus Battery Information:
- Total Battery Capacity: {battery_capacity} kWh
- Base Minimum Energy Reserve: {base_min_energy} kWh

OPERATOR NOTES:
{notes_text}

For EACH note, extract exactly one directive. Allowed directive_type values:

- solar_reduction: rooftop solar output is reduced during specific hours.
  Required structured_adjustment: {{"hours": [list of integers 0-23 in ascending order], "factor": float between 0.0 and 1.0}}
  NOTE: "factor" is the USABLE fraction remaining!
    - "80% reduction" -> factor is 0.20
    - "reduced by 60%" -> factor is 0.40
    - "reduced to 25%" -> factor is 0.25
    - "leave about half" -> factor is 0.50
    - "leave roughly one-fifth" -> factor is 0.20
    - "leave roughly one-fourth" or "quarter" -> factor is 0.25

- minimum_battery_reserve: keep battery energy at or above a required level during specific hours.
  Required structured_adjustment: {{"hours": [...], "minimum_energy_kwh": float}}
  NOTE: If note specifies a percentage of battery capacity (e.g. "50% of the battery capacity"), calculate the kWh based on total capacity of {battery_capacity} kWh (e.g. 50% = {0.5 * battery_capacity} kWh).

- no_charge_window: battery charging is disabled/unavailable during specific hours.
  Required structured_adjustment: {{"hours": [...]}}

- no_discharge_window: battery discharging is disabled/unavailable during specific hours.
  Required structured_adjustment: {{"hours": [...]}}

- max_grid_window: campus grid import/intake must not exceed a stated limit during specific hours.
  Required structured_adjustment: {{"hours": [...], "max_grid_kwh": float}}

- no_op: the note does not impose any actionable constraint on the 24-hour energy schedule (e.g. cafeteria menu, sports registration, library hours, seminar bookings).
  For no_op, applies MUST be false and structured_adjustment MUST be null.

Time Window Rules:
- Whole-hour intervals only. Start hour is INCLUDED, end hour is EXCLUDED.
- "1 PM to 3 PM" -> hours [13, 14]
- "noon until 2 PM" -> hours [12, 13]
- "10 AM until noon" -> hours [10, 11]
- "2 AM until 5 AM" -> hours [2, 3, 4]
- "6 PM until 9 PM" -> hours [18, 19, 20]
- "7 PM until 10 PM" -> hours [19, 20, 21]
- "10 PM until midnight" -> hours [22, 23]
- "10 PM until 2 AM" (overnight) -> hours [0, 1, 22, 23]
- Hours must be unique integers from 0 through 23, sorted in ascending order.

Return ONLY a valid JSON array of objects with keys:
[
  {{
    "note_index": 0,
    "applies": true,
    "directive_type": "...",
    "structured_adjustment": {{ ... }},
    "explanation": "concise explanation"
  }}
]
"""


def _candidate_models() -> List[str]:
    """Primary model from GEMINI_MODEL, then GEMINI_FALLBACK_MODELS (comma separated)."""
    primary = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACKS).split(",")
    models: List[str] = []
    for name in [primary] + [m.strip() for m in fallbacks]:
        if name and name not in models:
            models.append(name)
    return models


def _parse_llm_json(text: str) -> List[Dict[str, Any]]:
    """Turn raw LLM text into a list of dict directives (tolerant of code fences / wrappers)."""
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


def interpret_notes_with_llm(
    notes: List[str], battery: BatteryData
) -> Tuple[List[Dict[str, Any]], str]:
    """
    Call Google Gemini to interpret operator notes.
    Returns (raw_directives, interpreter_label). Always falls back to the deterministic
    parser on any problem, so the endpoint never fails because of the LLM.
    """

    def fallback(reason: str) -> Tuple[List[Dict[str, Any]], str]:
        return (
            [extract_directive_fallback(n, battery.capacity_kwh) for n in notes],
            f"rule-based ({reason})",
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info(
            "No GEMINI_API_KEY configured. Using deterministic parser for all notes."
        )
        return fallback("no API key")

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
    except Exception as e:  # SDK missing / bad configuration
        logger.warning(f"Gemini SDK unavailable: {e}. Using deterministic parser.")
        return fallback("SDK unavailable")

    prompt = get_llm_prompt(notes, battery.capacity_kwh, battery.minimum_energy_kwh)

    for model_name in _candidate_models():
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
                request_options={"timeout": LLM_TIMEOUT_SECONDS},
            )
            return _parse_llm_json(response.text), f"gemini:{model_name}"
        except Exception as e:
            logger.warning(f"LLM model '{model_name}' failed: {e}")

    logger.warning("All configured LLM models failed. Using deterministic parser.")
    return fallback("LLM failed")


def guardrail_directives(
    raw: List[Dict], notes: List[str], battery: BatteryData
) -> List[DirectiveInterpretation]:
    """
    Deterministic validation and sanitization of directive interpretations.
    Enforces all problem statement rules:
      - Exactly one entry per note, in note_index order
      - Valid directive_type
      - Unique hours in ascending order within [0, 23]
      - Bounded numerical values
      - applies=False and structured_adjustment=None strictly for no_op
      - applies=True strictly for non-no_op directives
    """
    result = []
    num_notes = len(notes)
    raw = [x for x in (raw or []) if isinstance(x, dict)]

    for i in range(num_notes):
        note_text = notes[i]

        # Match by note_index or fallback to index position
        item = next((x for x in raw if x.get("note_index") == i), None)
        if item is None and i < len(raw):
            item = raw[i]

        fallback_dir = extract_directive_fallback(note_text, battery.capacity_kwh)
        if item is None:
            item = fallback_dir

        dtype = item.get("directive_type", "no_op")
        if dtype not in SUPPORTED_DIRECTIVES:
            dtype = fallback_dir["directive_type"]
            item = fallback_dir

        if dtype == "no_op":
            result.append(
                DirectiveInterpretation(
                    note_index=i,
                    applies=False,
                    directive_type="no_op",
                    structured_adjustment=None,
                    explanation=str(
                        item.get(
                            "explanation",
                            "This note does not affect today's 24-hour energy schedule.",
                        )
                    )[:200],
                )
            )
            continue

        # For non-no_op directives, validate hours and parameters
        sa = item.get("structured_adjustment")
        if not isinstance(sa, dict):
            sa = {}
        hours = sa.get("hours", [])
        if not isinstance(hours, list):
            hours = []

        # Clean and sort hours
        clean_hours = sorted(
            set(
                int(h)
                for h in hours
                if isinstance(h, (int, float)) and 0 <= int(h) <= 23
            )
        )

        # If hours empty or invalid, try fallback hours
        if not clean_hours:
            fb_adj = fallback_dir.get("structured_adjustment")
            if fb_adj and fb_adj.get("hours"):
                clean_hours = fb_adj["hours"]
                dtype = fallback_dir["directive_type"]
                sa = fb_adj

        if not clean_hours:
            # Cannot form a valid window; must be treated as no_op
            result.append(
                DirectiveInterpretation(
                    note_index=i,
                    applies=False,
                    directive_type="no_op",
                    structured_adjustment=None,
                    explanation="No valid time window identified for directive.",
                )
            )
            continue

        adj = StructuredAdjustment(hours=clean_hours)
        valid = True

        if dtype == "solar_reduction":
            factor = sa.get("factor")
            if factor is None:
                factor = (fallback_dir.get("structured_adjustment") or {}).get(
                    "factor", 0.5
                )
            try:
                f_val = float(factor)
                adj.factor = round(max(0.0, min(1.0, f_val)), 4)
            except (ValueError, TypeError):
                valid = False

        elif dtype == "minimum_battery_reserve":
            val = sa.get("minimum_energy_kwh")
            if val is None:
                val = (fallback_dir.get("structured_adjustment") or {}).get(
                    "minimum_energy_kwh", battery.minimum_energy_kwh
                )
            try:
                v_val = float(val)
                adj.minimum_energy_kwh = round(
                    max(0.0, min(battery.capacity_kwh, v_val)), 2
                )
            except (ValueError, TypeError):
                valid = False

        elif dtype == "max_grid_window":
            val = sa.get("max_grid_kwh")
            if val is None:
                val = (fallback_dir.get("structured_adjustment") or {}).get(
                    "max_grid_kwh", 150.0
                )
            try:
                v_val = float(val)
                adj.max_grid_kwh = round(max(0.0, v_val), 2)
            except (ValueError, TypeError):
                valid = False

        elif dtype in ("no_charge_window", "no_discharge_window"):
            # Only hours needed
            pass

        if not valid:
            result.append(
                DirectiveInterpretation(
                    note_index=i,
                    applies=False,
                    directive_type="no_op",
                    structured_adjustment=None,
                    explanation="Invalid directive parameters.",
                )
            )
        else:
            explanation = str(
                item.get("explanation", fallback_dir.get("explanation", ""))
            )[:200]
            result.append(
                DirectiveInterpretation(
                    note_index=i,
                    applies=True,
                    directive_type=dtype,
                    structured_adjustment=adj,
                    explanation=explanation,
                )
            )

    return result


# ------------------ Constraints + Exact LP Optimizer (HiGHS) ------------------


def build_constraints(
    hours: List[HourData],
    battery: BatteryData,
    directives: List[DirectiveInterpretation],
) -> Dict[str, Any]:
    """Turn validated directives into the per-hour limits the LP (and the dashboard) use."""
    n = 24
    effective_solar = [float(h.solar_kwh) for h in hours]
    no_charge_hours: set = set()
    no_discharge_hours: set = set()
    min_reserve = [float(battery.minimum_energy_kwh)] * n
    max_grid = [float("inf")] * n

    for d in directives:
        if not d.applies or not d.structured_adjustment:
            continue
        adj = d.structured_adjustment
        hs = adj.hours
        dtype = d.directive_type

        if dtype == "solar_reduction":
            factor = adj.factor if adj.factor is not None else 1.0
            for h in hs:
                effective_solar[h] *= factor
        elif dtype == "no_charge_window":
            no_charge_hours.update(hs)
        elif dtype == "no_discharge_window":
            no_discharge_hours.update(hs)
        elif dtype == "minimum_battery_reserve":
            if adj.minimum_energy_kwh is not None:
                for h in hs:
                    min_reserve[h] = max(min_reserve[h], adj.minimum_energy_kwh)
        elif dtype == "max_grid_window":
            if adj.max_grid_kwh is not None:
                for h in hs:
                    max_grid[h] = min(max_grid[h], adj.max_grid_kwh)

    return {
        "effective_solar": effective_solar,
        "no_charge_hours": no_charge_hours,
        "no_discharge_hours": no_discharge_hours,
        "min_reserve": min_reserve,
        "max_grid": max_grid,
    }


def _solve_lp(
    hours: List[HourData],
    battery: BatteryData,
    cons: Dict[str, Any],
    relax_grid: bool = False,
) -> np.ndarray:
    """
    Solve the 24-hour scheduling LP to global optimality (HiGHS via scipy.optimize.linprog).

    Decision Variables per hour h in 0..23:
      g_h: Grid energy purchased (kWh)     s_h: Solar energy utilized (kWh)
      c_h: Battery energy charged (kWh)    d_h: Battery energy discharged (kWh)
      E_h: Battery energy level after hour h (kWh)

    Constraints:
      1. Energy Balance:   g_h + s_h + d_h - c_h = demand_h
      2. SoC Dynamics:     E_h - E_{h-1} - c_h + d_h = 0   (E_{-1} = initial_energy)
      3. Neutrality:       E_23 = initial_energy
      4. Solar:            0 <= s_h <= effective_solar_h
      5. Grid:             0 <= g_h <= max_grid_h
      6/7. Rate limits:    0 <= c_h <= max_charge, 0 <= d_h <= max_discharge
      8. Storage bounds:   min_reserve_h <= E_h <= capacity
    """
    n = 24
    demand = [h.demand_kwh for h in hours]
    tariff = [h.tariff_bdt_per_kwh for h in hours]
    effective_solar = cons["effective_solar"]
    max_grid = cons["max_grid"]
    min_reserve = cons["min_reserve"]

    num_vars = 120  # g:0..23  s:24..47  c:48..71  d:72..95  E:96..119
    c_obj = np.zeros(num_vars)
    for h in range(n):
        c_obj[h] = tariff[h]  # Minimize total grid cost
        c_obj[24 + h] = -1e-6  # Secondary objective: prefer using available solar

    bounds = []
    for h in range(n):  # g_h
        cap = max_grid[h]
        ub = None if (relax_grid or cap == float("inf")) else cap
        bounds.append((0.0, ub))
    for h in range(n):  # s_h
        bounds.append((0.0, max(0.0, effective_solar[h])))
    for h in range(n):  # c_h
        ub = 0.0 if h in cons["no_charge_hours"] else battery.max_charge_kwh_per_hour
        bounds.append((0.0, ub))
    for h in range(n):  # d_h
        ub = (
            0.0
            if h in cons["no_discharge_hours"]
            else battery.max_discharge_kwh_per_hour
        )
        bounds.append((0.0, ub))
    for h in range(n):  # E_h
        bounds.append((min_reserve[h], battery.capacity_kwh))

    num_eq = 49  # 24 energy balance + 24 SoC transitions + 1 neutrality
    A_eq = np.zeros((num_eq, num_vars))
    b_eq = np.zeros(num_eq)

    for h in range(n):  # Energy balance
        A_eq[h, h] = 1.0
        A_eq[h, 24 + h] = 1.0
        A_eq[h, 48 + h] = -1.0
        A_eq[h, 72 + h] = 1.0
        b_eq[h] = demand[h]

    for h in range(n):  # Battery dynamics
        row = 24 + h
        A_eq[row, 96 + h] = 1.0
        A_eq[row, 48 + h] = -1.0
        A_eq[row, 72 + h] = 1.0
        if h == 0:
            b_eq[row] = battery.initial_energy_kwh
        else:
            A_eq[row, 96 + h - 1] = -1.0
            b_eq[row] = 0.0

    A_eq[48, 96 + 23] = 1.0  # End-of-day neutrality
    b_eq[48] = battery.initial_energy_kwh

    res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        logger.error(f"Linear program failed to find optimal schedule: {res.message}")
        raise InfeasibleScheduleError(str(res.message))
    return res.x


def optimize_energy(
    hours: List[HourData],
    battery: BatteryData,
    directives: List[DirectiveInterpretation],
    constraints: Optional[Dict[str, Any]] = None,
) -> Tuple[List[HourlyPlan], List[str]]:
    """
    Returns (hourly_plan, warnings).

    If the schedule is infeasible ONLY because of max_grid_window caps (e.g. cap 100 kWh
    while demand is 190 kWh and the battery/solar cannot cover the gap), the caps are
    relaxed, the cheapest feasible schedule is returned, and a warning is reported
    instead of crashing with a 500.
    """
    n = 24
    cons = constraints or build_constraints(hours, battery, directives)
    demand = [h.demand_kwh for h in hours]
    effective_solar = cons["effective_solar"]
    warnings: List[str] = []

    try:
        x = _solve_lp(hours, battery, cons, relax_grid=False)
    except InfeasibleScheduleError:
        capped = [h for h in range(n) if cons["max_grid"][h] != float("inf")]
        if not capped:
            raise
        x = _solve_lp(hours, battery, cons, relax_grid=True)  # may still raise
        warnings.append(
            "max_grid_window limit could not be satisfied for hours "
            f"{capped}; grid cap was relaxed to keep the schedule feasible."
        )
        logger.warning(warnings[-1])

    s_arr = x[24:48]
    c_arr = x[48:72]
    d_arr = x[72:96]

    # Post-process: cancel simultaneous charge/discharge (mutual exclusivity)
    plan: List[HourlyPlan] = []
    current_energy = battery.initial_energy_kwh
    for h in range(n):
        net = c_arr[h] - d_arr[h]
        if net > 1e-4:
            action, action_kwh = "charge", float(net)
            current_energy += action_kwh
        elif net < -1e-4:
            action, action_kwh = "discharge", float(-net)
            current_energy -= action_kwh
        else:
            action, action_kwh = "idle", 0.0

        solar_val = float(max(0.0, min(effective_solar[h], s_arr[h])))

        # Re-derive grid_kwh exactly from the energy balance: g = demand + c - s - d
        c_kwh = action_kwh if action == "charge" else 0.0
        d_kwh = action_kwh if action == "discharge" else 0.0
        grid_val = float(max(0.0, demand[h] + c_kwh - solar_val - d_kwh))

        plan.append(
            HourlyPlan(
                hour=h,
                grid_kwh=round(grid_val, 4),
                solar_used_kwh=round(solar_val, 4),
                battery_action=action,
                battery_kwh=round(action_kwh, 4),
                battery_energy_after_kwh=round(current_energy, 4),
            )
        )

    plan[-1].battery_energy_after_kwh = round(battery.initial_energy_kwh, 4)
    return plan, warnings


# ------------------ Pipeline + Analytics ------------------


def run_pipeline(req: OptimizeRequest) -> Dict[str, Any]:
    """LLM interpretation -> guardrails -> constraints -> LP -> aggregates."""
    started = time.perf_counter()

    raw_interp, interpreter = interpret_notes_with_llm(req.operator_notes, req.battery)
    directives = guardrail_directives(raw_interp, req.operator_notes, req.battery)
    cons = build_constraints(req.hours, req.battery, directives)
    plan, warnings = optimize_energy(req.hours, req.battery, directives, cons)

    total_grid = sum(p.grid_kwh for p in plan)
    total_cost = sum(p.grid_kwh * req.hours[p.hour].tariff_bdt_per_kwh for p in plan)
    peak_grid = max(p.grid_kwh for p in plan) if plan else 0.0

    applied_types = [d.directive_type for d in directives if d.applies]
    summary = (
        f"Successfully optimized 24-hour schedule. "
        f"Directives applied: {', '.join(applied_types) if applied_types else 'none'}. "
        f"Total grid import: {total_grid:.2f} kWh, Total cost: {total_cost:.2f} BDT, "
        f"Peak grid: {peak_grid:.2f} kWh."
    )
    if warnings:
        summary += " Warnings: " + " ".join(warnings)

    response = OptimizeResponse(
        scenario_id=req.scenario_id,
        directive_interpretation=directives,
        hourly_plan=plan,
        total_grid_kwh=round(total_grid, 4),
        total_cost_bdt=round(total_cost, 4),
        peak_grid_kwh=round(peak_grid, 4),
        plan_summary=summary,
    )
    return {
        "response": response,
        "constraints": cons,
        "warnings": warnings,
        "interpreter": interpreter,
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
    }


def compute_analytics(req: OptimizeRequest, result: Dict[str, Any]) -> Analytics:
    """Savings vs. a 'no battery' baseline plus per-hour constraint details for the charts."""
    resp: OptimizeResponse = result["response"]
    cons = result["constraints"]
    plan = resp.hourly_plan

    baseline_grid = [
        max(0.0, req.hours[h].demand_kwh - cons["effective_solar"][h])
        for h in range(24)
    ]
    baseline_cost = sum(
        baseline_grid[h] * req.hours[h].tariff_bdt_per_kwh for h in range(24)
    )
    baseline_peak = max(baseline_grid) if baseline_grid else 0.0

    savings = baseline_cost - resp.total_cost_bdt
    savings_pct = (savings / baseline_cost * 100.0) if baseline_cost > 1e-9 else 0.0

    total_eff_solar = sum(cons["effective_solar"])
    total_solar_used = sum(p.solar_used_kwh for p in plan)
    solar_util = (
        (total_solar_used / total_eff_solar * 100.0) if total_eff_solar > 1e-9 else 0.0
    )
    discharged = sum(p.battery_kwh for p in plan if p.battery_action == "discharge")

    hourly = []
    for h in range(24):
        cap = cons["max_grid"][h]
        hourly.append(
            HourlyInsight(
                hour=h,
                tariff_bdt_per_kwh=req.hours[h].tariff_bdt_per_kwh,
                demand_kwh=req.hours[h].demand_kwh,
                effective_solar_kwh=round(cons["effective_solar"][h], 4),
                min_reserve_kwh=round(cons["min_reserve"][h], 4),
                max_grid_kwh=None if cap == float("inf") else round(cap, 4),
                charge_allowed=h not in cons["no_charge_hours"],
                discharge_allowed=h not in cons["no_discharge_hours"],
                baseline_grid_kwh=round(baseline_grid[h], 4),
                cost_bdt=round(plan[h].grid_kwh * req.hours[h].tariff_bdt_per_kwh, 4),
            )
        )

    return Analytics(
        interpreter=result["interpreter"],
        pipeline_time_ms=round(result["elapsed_ms"], 2),
        baseline_cost_bdt=round(baseline_cost, 4),
        savings_bdt=round(savings, 4),
        savings_pct=round(savings_pct, 2),
        baseline_peak_grid_kwh=round(baseline_peak, 4),
        peak_reduction_kwh=round(baseline_peak - resp.peak_grid_kwh, 4),
        solar_utilization_pct=round(solar_util, 2),
        solar_curtailed_kwh=round(total_eff_solar - total_solar_used, 4),
        battery_discharged_kwh=round(discharged, 4),
        equivalent_full_cycles=round(discharged / req.battery.capacity_kwh, 3),
        warnings=result["warnings"],
        hourly=hourly,
    )


def _execute(req: OptimizeRequest) -> Dict[str, Any]:
    """Run the pipeline and translate failures into safe HTTP errors."""
    try:
        return run_pipeline(req)
    except HTTPException:
        raise
    except InfeasibleScheduleError as e:
        logger.warning(f"Infeasible scenario '{req.scenario_id}': {e}")
        raise HTTPException(
            status_code=422,
            detail="Infeasible scenario: no schedule satisfies all constraints "
            "(check battery limits, reserve and grid caps).",
        )
    except Exception:
        logger.exception("Error processing energy optimization request")
        # Ensure safe failure without leaking credentials or raw system stack traces
        raise HTTPException(
            status_code=500,
            detail="Controlled internal server error during energy optimization processing.",
        )


# ------------------ API Endpoints ------------------


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Serve the GridWise LLM interactive dashboard."""
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>GridWise LLM Service Online</h1>")


@app.get("/api/presets")
def get_presets():
    """Return the 10 official BUP sample scenarios for the interactive UI."""
    sample_file = os.path.join(
        os.path.dirname(__file__),
        "Question",
        "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
    )
    if os.path.exists(sample_file):
        with open(sample_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            c["id"]: {"label": c["label"], "input": c["input"]}
            for c in data.get("cases", [])
        }
    fallback_file = os.path.join(os.path.dirname(__file__), "static_presets.json")
    if os.path.exists(fallback_file):
        with open(fallback_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


@app.get("/health")
def health():
    """Readiness endpoint required by BUP Hackathon specification."""
    return {"status": "ok"}


@app.get("/api/status")
def status():
    """Non-secret runtime info for the dashboard header badges."""
    return {
        "status": "ok",
        "version": APP_VERSION,
        "solver": "HiGHS (scipy.optimize.linprog)",
        "llm_configured": bool(os.getenv("GEMINI_API_KEY")),
        "llm_models": _candidate_models(),
    }


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    """
    Main energy optimization endpoint (contract unchanged):
      1. Interprets operator notes via LLM (Gemini) with deterministic NLP fallback.
      2. Validates directives through deterministic guardrails.
      3. Solves the 24-hour scheduling problem to global optimality via HiGHS.
      4. Computes aggregates and returns a validated structured response.
    """
    return _execute(req)["response"]


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: OptimizeRequest):
    """Same as /optimize-energy, plus savings vs. baseline, warnings and per-hour constraints."""
    result = _execute(req)
    return AnalyzeResponse(
        optimization=result["response"], analytics=compute_analytics(req, result)
    )


@app.post("/api/export-csv")
def export_csv(req: OptimizeRequest):
    """Download the optimized hourly schedule as a CSV file."""
    result = _execute(req)
    plan = result["response"].hourly_plan

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "hour",
            "tariff_bdt_per_kwh",
            "demand_kwh",
            "solar_used_kwh",
            "grid_kwh",
            "battery_action",
            "battery_kwh",
            "battery_energy_after_kwh",
            "cost_bdt",
        ]
    )
    for p in plan:
        tariff = req.hours[p.hour].tariff_bdt_per_kwh
        writer.writerow(
            [
                p.hour,
                tariff,
                req.hours[p.hour].demand_kwh,
                p.solar_used_kwh,
                p.grid_kwh,
                p.battery_action,
                p.battery_kwh,
                p.battery_energy_after_kwh,
                round(p.grid_kwh * tariff, 4),
            ]
        )

    safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", req.scenario_id) or "scenario"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_id}_schedule.csv"'
        },
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
