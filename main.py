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
from fastapi.openapi.docs import get_swagger_ui_html
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise")

APP_VERSION = "1.1.0"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_GEMINI_FALLBACKS = "gemini-2.5-flash"
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "12"))

app = FastAPI(
    title="GridWise LLM",
    description="BUP CSE Fest 2026 - Smart Campus Energy Optimization Engine",
    version=APP_VERSION,
    docs_url=None,
    redoc_url=None,
    swagger_ui_parameters={
        "syntaxHighlight.theme": "obsidian",
        "defaultModelsExpandDepth": -1,
    },
)


class InfeasibleScheduleError(Exception):
    pass


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


class HourlyInsight(BaseModel):
    hour: int
    tariff_bdt_per_kwh: float
    demand_kwh: float
    effective_solar_kwh: float
    min_reserve_kwh: float
    max_grid_kwh: Optional[float] = None
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
    if h2 == 0 and h1 > 0:
        h2 = 24
    if 0 <= h1 < h2 <= 24:
        return list(range(h1, h2))
    if allow_wrap and 0 <= h2 < h1 <= 23:
        return sorted(list(range(h1, 24)) + list(range(0, h2)))
    return []


def _is_explicit_overnight(s1: str, s2: str) -> bool:
    if "pm" in s1 and "am" in s2:
        return True
    return ":" in s1 and ":" in s2 and "am" not in s1 + s2 and "pm" not in s1 + s2


def extract_hours_window(text: str) -> List[int]:
    t = text.lower()
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
    t = note.lower()
    hours = extract_hours_window(t)
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
    return {
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": "This note does not affect today's 24-hour energy schedule.",
    }


def _candidate_models() -> List[str]:
    primary = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip()
    fallbacks = os.getenv("GEMINI_FALLBACK_MODELS", DEFAULT_GEMINI_FALLBACKS).split(",")
    models: List[str] = []
    for name in [primary] + [m.strip() for m in fallbacks]:
        if name and name not in models:
            models.append(name)
    return models


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


def _parse_llm_json(text: str) -> List[Dict[str, Any]]:
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
    except Exception as e:
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
    result = []
    num_notes = len(notes)
    raw = [x for x in (raw or []) if isinstance(x, dict)]

    for i in range(num_notes):
        note_text = notes[i]
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
        sa = item.get("structured_adjustment")
        if not isinstance(sa, dict):
            sa = {}
        hours = sa.get("hours", [])
        if not isinstance(hours, list):
            hours = []
        clean_hours = sorted(
            set(
                int(h)
                for h in hours
                if isinstance(h, (int, float)) and 0 <= int(h) <= 23
            )
        )
        if not clean_hours:
            fb_adj = fallback_dir.get("structured_adjustment")
            if fb_adj and fb_adj.get("hours"):
                clean_hours = fb_adj["hours"]
                dtype = fallback_dir["directive_type"]
                sa = fb_adj

        if not clean_hours:
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


def build_constraints(
    hours: List[HourData],
    battery: BatteryData,
    directives: List[DirectiveInterpretation],
) -> Dict[str, Any]:
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
    n = 24
    demand = [h.demand_kwh for h in hours]
    tariff = [h.tariff_bdt_per_kwh for h in hours]
    effective_solar = cons["effective_solar"]
    max_grid = cons["max_grid"]
    min_reserve = cons["min_reserve"]

    num_vars = 120
    c_obj = np.zeros(num_vars)
    for h in range(n):
        c_obj[h] = tariff[h]
        c_obj[24 + h] = -1e-6

    bounds = []
    for h in range(n):
        cap = max_grid[h]
        ub = None if (relax_grid or cap == float("inf")) else cap
        bounds.append((0.0, ub))
    for h in range(n):
        bounds.append((0.0, max(0.0, effective_solar[h])))
    for h in range(n):
        ub = 0.0 if h in cons["no_charge_hours"] else battery.max_charge_kwh_per_hour
        bounds.append((0.0, ub))
    for h in range(n):
        ub = (
            0.0
            if h in cons["no_discharge_hours"]
            else battery.max_discharge_kwh_per_hour
        )
        bounds.append((0.0, ub))
    for h in range(n):
        bounds.append((min_reserve[h], battery.capacity_kwh))

    num_eq = 49
    A_eq = np.zeros((num_eq, num_vars))
    b_eq = np.zeros(num_eq)

    for h in range(n):
        A_eq[h, h] = 1.0
        A_eq[h, 24 + h] = 1.0
        A_eq[h, 48 + h] = -1.0
        A_eq[h, 72 + h] = 1.0
        b_eq[h] = demand[h]

    for h in range(n):
        row = 24 + h
        A_eq[row, 96 + h] = 1.0
        A_eq[row, 48 + h] = -1.0
        A_eq[row, 72 + h] = 1.0
        if h == 0:
            b_eq[row] = battery.initial_energy_kwh
        else:
            A_eq[row, 96 + h - 1] = -1.0
            b_eq[row] = 0.0

    A_eq[48, 96 + 23] = 1.0
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
        x = _solve_lp(hours, battery, cons, relax_grid=True)
        warnings.append(
            "max_grid_window limit could not be satisfied for hours "
            f"{capped}; grid cap was relaxed to keep the schedule feasible."
        )
        logger.warning(warnings[-1])

    s_arr = x[24:48]
    c_arr = x[48:72]
    d_arr = x[72:96]
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


def run_pipeline(req: OptimizeRequest) -> Dict[str, Any]:
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
        raise HTTPException(
            status_code=500,
            detail="Controlled internal server error during energy optimization processing.",
        )


@app.get("/", response_class=HTMLResponse)
def dashboard():
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(content="<h1>GridWise LLM Service Online</h1>")


@app.get("/api/presets")
def get_presets():
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


@app.get("/docs", include_in_schema=False)
def swagger_docs():
    response = get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title="GridWise API · Swagger",
        swagger_ui_parameters=app.swagger_ui_parameters,
    )
    content = response.body.decode("utf-8")
    theme = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;500;600;700&family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600&display=swap" rel="stylesheet">
<style>
:root {
  color-scheme: dark;
  --bg-deep: #04060f;
  --bg-base: #070b14;
  --bg-surface: rgba(13, 19, 33, 0.72);
  --bg-elevated: rgba(20, 28, 46, 0.78);
  --bg-inset: rgba(6, 11, 22, 0.55);
  --border-soft: rgba(255, 255, 255, 0.06);
  --border-mid: rgba(255, 255, 255, 0.1);
  --border: rgba(255, 255, 255, 0.08);
  --primary: #10b981;
  --primary-bright: #34d399;
  --secondary: #06b6d4;
  --secondary-bright: #22d3ee;
  --text: #f8fafc;
  --text-muted: #94a3b8;
  --text-dim: #64748b;
  --text-faint: #475569;
  --font-main: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
  --font-display: 'Fraunces', 'Plus Jakarta Sans', serif;
  --font-mono: 'JetBrains Mono', 'SF Mono', monospace;
}
* { box-sizing: border-box; }
html, body {
  margin: 0;
  min-width: 320px;
  background: var(--bg-deep);
  color: var(--text);
  font-family: var(--font-main);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  background-image:
    radial-gradient(ellipse 70% 50% at 50% -10%, rgba(6, 182, 212, 0.18), transparent 65%),
    radial-gradient(ellipse 50% 40% at 90% 10%, rgba(16, 185, 129, 0.12), transparent 60%),
    radial-gradient(circle 900px at 5% 100%, rgba(196, 181, 253, 0.08), transparent 65%);
  background-attachment: fixed;
}
body::after {
  content: '';
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image:
    linear-gradient(rgba(255, 255, 255, 0.012) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.012) 1px, transparent 1px);
  background-size: 60px 60px;
  -webkit-mask-image: radial-gradient(ellipse at center, black 0%, transparent 80%);
  mask-image: radial-gradient(ellipse at center, black 0%, transparent 80%);
  z-index: 0;
}
.gw-brand-header {
  position: sticky;
  top: 0;
  z-index: 50;
  backdrop-filter: blur(28px) saturate(180%);
  -webkit-backdrop-filter: blur(28px) saturate(180%);
  background: rgba(4, 8, 18, 0.65);
  border-bottom: 1px solid var(--border-soft);
  padding: 1rem 2rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.gw-brand-header::after {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  bottom: -1px;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(52, 211, 153, 0.4), rgba(34, 211, 238, 0.4), transparent);
  opacity: 0.6;
  pointer-events: none;
}
.gw-brand-wrap { display: flex; align-items: center; gap: 1rem; }
.gw-brand-icon {
  width: 46px;
  height: 46px;
  border-radius: 13px;
  background: linear-gradient(135deg, #10b981 0%, #06b6d4 50%, #0ea5e9 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow:
    0 0 30px rgba(16, 185, 129, 0.4),
    inset 0 1px 0 rgba(255, 255, 255, 0.3),
    inset 0 -1px 0 rgba(0, 0, 0, 0.2);
  color: #fff;
}
.gw-brand-icon svg { width: 24px; height: 24px; filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.3)); }
.gw-brand-title {
  font-family: var(--font-display);
  font-size: 1.45rem;
  font-weight: 500;
  letter-spacing: -0.025em;
  color: #fff;
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  line-height: 1.1;
}
.gw-brand-title .gw-badge {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  font-weight: 700;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.18), rgba(6, 182, 212, 0.18));
  color: #5eead4;
  padding: 3px 9px;
  border-radius: 6px;
  border: 1px solid rgba(94, 234, 212, 0.25);
}
.gw-brand-sub {
  font-size: 0.72rem;
  color: var(--text-muted);
  font-weight: 500;
  letter-spacing: 0.04em;
  margin-top: 3px;
  text-transform: uppercase;
}
.gw-nav { display: flex; align-items: center; gap: 0.65rem; }
.gw-status-pill {
  display: flex;
  align-items: center;
  gap: 0.55rem;
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(16, 185, 129, 0.06));
  border: 1px solid rgba(52, 211, 153, 0.28);
  color: #6ee7b7;
  font-size: 0.74rem;
  font-weight: 600;
  padding: 6px 13px;
  border-radius: 999px;
}
.gw-status-pill .gw-pulse {
  position: relative;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #34d399;
}
.gw-status-pill .gw-pulse::before {
  content: '';
  position: absolute;
  inset: -4px;
  border-radius: 50%;
  background: #34d399;
  opacity: 0.4;
  animation: gw-pulse 2.2s ease-in-out infinite;
}
@keyframes gw-pulse {
  0%, 100% { transform: scale(0.8); opacity: 0.5; }
  50% { transform: scale(1.4); opacity: 0; }
}
.gw-nav a {
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border-soft);
  color: var(--text-muted);
  font-size: 0.78rem;
  font-weight: 600;
  padding: 7px 14px;
  border-radius: 10px;
  text-decoration: none;
  transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
  display: inline-flex;
  align-items: center;
  gap: 0.45rem;
}
.gw-nav a:hover {
  background: rgba(255, 255, 255, 0.08);
  color: var(--text);
  border-color: rgba(34, 211, 238, 0.4);
  transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(6, 182, 212, 0.15);
}
.gw-nav a svg { width: 14px; height: 14px; }
.swagger-ui {
  position: relative;
  z-index: 1;
  max-width: 1380px;
  margin: 0 auto;
  padding: 32px clamp(20px, 4vw, 56px) 80px;
}
.swagger-ui .topbar { display: none; }
.swagger-ui .info { margin: 0 0 32px; }
.swagger-ui .info .title {
  color: var(--text);
  font-family: var(--font-display);
  font-weight: 500;
  font-size: 36px;
  letter-spacing: -0.035em;
}
.swagger-ui .info p, .swagger-ui .info li, .swagger-ui .opblock-description-wrapper p { color: var(--text-muted); }
.swagger-ui .scheme-container, .swagger-ui .opblock-tag-section {
  background: transparent;
  box-shadow: none;
}
.swagger-ui .opblock-tag {
  color: var(--text);
  border-bottom-color: var(--border-soft);
  font-family: var(--font-display);
  font-weight: 500;
  font-size: 22px;
  letter-spacing: -0.02em;
}
.swagger-ui .opblock {
  overflow: hidden;
  border: 1px solid var(--border-soft);
  border-radius: 16px;
  background: var(--bg-surface);
  backdrop-filter: blur(24px) saturate(150%);
  -webkit-backdrop-filter: blur(24px) saturate(150%);
  box-shadow: 0 16px 40px rgba(0, 0, 0, .35);
}
.swagger-ui .opblock.is-open { box-shadow: 0 20px 50px rgba(0, 0, 0, .45); }
.swagger-ui .opblock-summary {
  border-bottom-color: var(--border-soft);
  padding: 14px 20px;
  background: transparent;
}
.swagger-ui .opblock-summary:hover { background: rgba(255, 255, 255, 0.02); }
.swagger-ui .opblock-summary-method {
  min-width: 88px;
  border-radius: 9px;
  font: 800 11px var(--font-mono);
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 7px 14px;
  text-shadow: 0 1px 0 rgba(0, 0, 0, 0.15);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.18),
    inset 0 -1px 0 rgba(0, 0, 0, 0.15),
    0 1px 2px rgba(0, 0, 0, 0.25);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.swagger-ui .opblock-summary:hover .opblock-summary-method {
  transform: translateY(-1px);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.25),
    inset 0 -1px 0 rgba(0, 0, 0, 0.18),
    0 4px 12px rgba(0, 0, 0, 0.35);
}
.swagger-ui .opblock-summary-method-get {
  background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
  color: #f0f9ff;
  border: 1px solid rgba(56, 189, 248, 0.45);
}
.swagger-ui .opblock-summary-method-post {
  background: linear-gradient(135deg, #10b981 0%, #059669 100%);
  color: #ecfdf5;
  border: 1px solid rgba(52, 211, 153, 0.5);
}
.swagger-ui .opblock-summary-method-put {
  background: linear-gradient(135deg, #fbbf24 0%, #d97706 100%);
  color: #fffbeb;
  border: 1px solid rgba(251, 191, 36, 0.5);
}
.swagger-ui .opblock-summary-method-delete {
  background: linear-gradient(135deg, #f87171 0%, #dc2626 100%);
  color: #fef2f2;
  border: 1px solid rgba(248, 113, 113, 0.5);
}
.swagger-ui .opblock-summary-method-head,
.swagger-ui .opblock-summary-method-options {
  background: linear-gradient(135deg, #a78bfa 0%, #7c3aed 100%);
  color: #f5f3ff;
  border: 1px solid rgba(167, 139, 250, 0.5);
}
.swagger-ui .opblock-summary-method-patch {
  background: linear-gradient(135deg, #c084fc 0%, #9333ea 100%);
  color: #faf5ff;
  border: 1px solid rgba(192, 132, 252, 0.5);
}
.swagger-ui .opblock-summary-path,
.swagger-ui .opblock-summary-description,
.swagger-ui .parameter__name,
.swagger-ui .parameter__type,
.swagger-ui label,
.swagger-ui table thead tr th,
.swagger-ui .response-col_status,
.swagger-ui .responses-table .response { color: var(--text); }
.swagger-ui .opblock-summary-path {
  font-family: var(--font-mono);
  font-weight: 600;
  font-size: 14px;
}
.swagger-ui .opblock-description-wrapper,
.swagger-ui .opblock-section-header,
.swagger-ui .responses-inner,
.swagger-ui .model-box {
  background: rgba(4, 8, 16, 0.5);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
}
.swagger-ui .opblock-section-header {
  border-color: var(--border-soft);
  padding: 14px 20px;
}
.swagger-ui .opblock-section-header .opblock-title { font-weight: 600; }
.swagger-ui .btn,
.swagger-ui select,
.swagger-ui input,
.swagger-ui textarea {
  border-radius: 10px;
  border-color: var(--border-soft);
  background: rgba(4, 8, 16, 0.72);
  color: var(--text);
  transition: all 0.2s ease;
  font-family: inherit;
}
.swagger-ui input,
.swagger-ui textarea,
.swagger-ui select {
  outline: none;
}
.swagger-ui input:focus,
.swagger-ui textarea:focus,
.swagger-ui select:focus {
  border-color: var(--secondary-bright);
  box-shadow: 0 0 0 3px rgba(6, 182, 212, 0.15);
}
.swagger-ui .btn:hover,
.swagger-ui select:hover,
.swagger-ui input:hover,
.swagger-ui textarea:hover {
  border-color: rgba(34, 211, 238, 0.4);
}
.swagger-ui .btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  padding: 7px 16px;
  font-family: var(--font-main);
  font-size: 13px;
  font-weight: 600;
  letter-spacing: 0.01em;
  cursor: pointer;
  background: rgba(255, 255, 255, 0.04);
  border: 1px solid var(--border-soft);
  color: var(--text-muted);
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04);
}
.swagger-ui .btn:hover {
  background: rgba(255, 255, 255, 0.07);
  color: var(--text);
  border-color: rgba(34, 211, 238, 0.45);
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.06),
    0 2px 10px rgba(6, 182, 212, 0.15);
}
.swagger-ui .btn:active { transform: translateY(1px); }
.swagger-ui .try-out__btn {
  font-family: var(--font-main);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: 0.02em;
  padding: 6px 14px;
  border-radius: 9px;
  background: rgba(34, 211, 238, 0.08);
  border: 1px solid rgba(34, 211, 238, 0.3);
  color: #5eead4;
  text-transform: uppercase;
  transition: all 0.2s ease;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.05),
    0 0 0 0 rgba(34, 211, 238, 0.3);
}
.swagger-ui .try-out__btn:hover {
  background: rgba(34, 211, 238, 0.15);
  border-color: rgba(34, 211, 238, 0.55);
  color: #ecfeff;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.08),
    0 0 18px rgba(34, 211, 238, 0.3);
  transform: translateY(-1px);
}
.swagger-ui .try-out__btn.cancel {
  background: rgba(248, 113, 113, 0.08);
  border-color: rgba(248, 113, 113, 0.3);
  color: #fca5a5;
}
.swagger-ui .try-out__btn.cancel:hover {
  background: rgba(248, 113, 113, 0.15);
  border-color: rgba(248, 113, 113, 0.55);
  color: #fee2e2;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.08),
    0 0 18px rgba(248, 113, 113, 0.3);
}
.swagger-ui .btn.cancel,
.swagger-ui button.btn-clear,
.swagger-ui button.btn-clear-filter {
  border-color: var(--border-soft);
  color: var(--text-muted);
  background: rgba(255, 255, 255, 0.04);
  font-weight: 600;
}
.swagger-ui .btn.cancel:hover,
.swagger-ui button.btn-clear:hover,
.swagger-ui button.btn-clear-filter:hover {
  color: var(--text);
  background: rgba(255, 255, 255, 0.07);
}
.swagger-ui .btn.execute {
  border: 0;
  background: linear-gradient(135deg, #10b981 0%, #06b6d4 100%);
  color: #021014;
  font-weight: 700;
  letter-spacing: 0.02em;
  padding: 9px 22px;
  font-size: 13px;
  border-radius: 10px;
  box-shadow:
    0 6px 20px rgba(16, 185, 129, 0.35),
    inset 0 1px 0 rgba(255, 255, 255, 0.35),
    inset 0 -1px 0 rgba(0, 0, 0, 0.15);
  position: relative;
  overflow: hidden;
}
.swagger-ui .btn.execute::before {
  content: '';
  position: absolute;
  top: 0;
  left: -100%;
  width: 100%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.35), transparent);
  transition: left 0.6s ease;
  pointer-events: none;
}
.swagger-ui .btn.execute:hover {
  filter: brightness(1.08);
  box-shadow:
    0 10px 30px rgba(16, 185, 129, 0.5),
    inset 0 1px 0 rgba(255, 255, 255, 0.4),
    inset 0 -1px 0 rgba(0, 0, 0, 0.18);
  transform: translateY(-1px);
}
.swagger-ui .btn.execute:hover::before { left: 100%; }
.swagger-ui .btn.execute:active { transform: translateY(0); }
.swagger-ui .execute-wrapper { padding-top: 10px; }
.swagger-ui .responses-inner h4,
.swagger-ui .responses-inner h5 { color: var(--text); }
.swagger-ui .response .response-col_status code {
  font-family: var(--font-mono);
  font-weight: 700;
  font-size: 13px;
  padding: 2px 8px;
  border-radius: 6px;
}
.swagger-ui .response .response-col_status .response-success {
  background: rgba(16, 185, 129, 0.15);
  color: #34d399;
  border: 1px solid rgba(16, 185, 129, 0.35);
}
.swagger-ui .response .response-col_status .response-other {
  background: rgba(148, 163, 184, 0.1);
  color: var(--text-muted);
  border: 1px solid var(--border-soft);
}
.swagger-ui .highlight-code,
.swagger-ui .microlight {
  background: #040810 !important;
  color: #7dd3fc !important;
  border-radius: 8px;
  border: 1px solid var(--border-soft);
}
.swagger-ui .model,
.swagger-ui .model-title,
.swagger-ui .prop-type,
.swagger-ui .prop-format,
.swagger-ui .renderedMarkdown p { color: var(--text-muted); }
.swagger-ui table thead tr td,
.swagger-ui table thead tr th { border-bottom-color: var(--border-soft); }
.swagger-ui table tbody tr td { padding: 10px 12px; border-bottom-color: rgba(255, 255, 255, 0.03); }
.swagger-ui .response-col_description { color: var(--text-muted); }
.swagger-ui .markdown p, .swagger-ui .markdown li, .swagger-ui .renderedMarkdown p { color: var(--text-muted); }
.swagger-ui .scheme-container .schemes > label { color: var(--text-muted); }
.swagger-ui .filter input { color: var(--text); }
.swagger-ui .filter input::placeholder { color: var(--text-faint); }
.swagger-ui .opblock-tag-section h3,
.swagger-ui .opblock-tag small { color: var(--text-muted); }
.swagger-ui .parameter__type { font-family: var(--font-mono); font-size: 0.78rem; }
.swagger-ui .parameter__name { font-family: var(--font-mono); font-weight: 600; }
.swagger-ui .response-col_status { font-family: var(--font-mono); font-weight: 700; }
.swagger-ui .response.unauthorized .response-col_status { color: #fbbf24; }
.swagger-ui .response.internal .response-col_status,
.swagger-ui .response.default .response-col_status { color: #f87171; }
.swagger-ui .expand-collapse-operation,
.swagger-ui .expand-collapse-methods,
.swagger-ui .expand-collapse { color: var(--text-muted); }
.swagger-ui a { color: #22d3ee; }
.swagger-ui a:hover { color: #5eead4; }
.swagger-ui .dialog-ux .modal-ux { background: var(--bg-elevated); border-color: var(--border-soft); }
.swagger-ui .dialog-ux .modal-ux-header { background: transparent; border-bottom-color: var(--border-soft); color: var(--text); }
.swagger-ui .dialog-ux .modal-ux-content { color: var(--text-muted); background: transparent; }
.swagger-ui .info__extdocs { color: var(--text-muted); }
.gw-footer {
  text-align: center;
  padding: 2rem 1rem 1rem;
  font-size: 0.74rem;
  color: var(--text-faint);
  letter-spacing: 0.08em;
  text-transform: uppercase;
  position: relative;
  z-index: 1;
}
.gw-footer span {
  background: linear-gradient(135deg, #34d399, #22d3ee);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  font-weight: 700;
}
@media (max-width: 720px) {
  .gw-brand-header { padding: 0.85rem 1rem; }
  .gw-brand-title { font-size: 1.15rem; }
  .gw-brand-sub { font-size: 0.66rem; }
  .gw-nav a span { display: none; }
}
</style>
<header class="gw-brand-header">
  <div class="gw-brand-wrap">
    <div class="gw-brand-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
    </div>
    <div>
      <div class="gw-brand-title">GridWise<span class="gw-badge">LLM</span></div>
      <div class="gw-brand-sub">Campus Energy Optimization · API Reference</div>
    </div>
  </div>
  <div class="gw-nav">
    <div class="gw-status-pill"><span class="gw-pulse"></span><span>OpenAPI 3 · Live</span></div>
    <a href="/" target="_blank">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12l9-9 9 9"></path><path d="M5 10v10a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1V10"></path></svg>
      <span>Dashboard</span>
    </a>
    <a href="/health?ui=1" target="_blank">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>
      <span>Health</span>
    </a>
  </div>
</header>
<div class="gw-footer">Crafted for <span>BUP CSE Fest 2026</span> · GridWise LLM Optimization Engine</div>
"""
    return HTMLResponse(content=content.replace("</head>", f"{theme}</head>"))


@app.get("/health")
def health(ui: bool = False):
    if ui:
        return HTMLResponse(
            content="""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>GridWise · Health Probe</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&family=Fraunces:opsz,wght@9..144,400;9..144,500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg-deep: #04060f;
      --bg-base: #070b14;
      --surface: rgba(13,19,33,.72);
      --border: rgba(255,255,255,.08);
      --text: #f8fafc;
      --muted: #94a3b8;
      --dim: #64748b;
      --green: #10b981;
      --cyan: #06b6d4;
      --violet: #c4b5fd;
      --font-main: 'Plus Jakarta Sans', sans-serif;
      --font-display: 'Fraunces', serif;
      --font-mono: 'JetBrains Mono', monospace;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100vh; }
    body {
      display: grid;
      place-items: center;
      padding: clamp(20px, 4vw, 48px);
      color: var(--text);
      font-family: var(--font-main);
      background: var(--bg-deep);
      background-image:
        radial-gradient(ellipse 70% 50% at 50% -10%, rgba(6,182,212,.18), transparent 65%),
        radial-gradient(ellipse 50% 40% at 90% 10%, rgba(16,185,129,.12), transparent 60%),
        radial-gradient(circle 700px at 5% 100%, rgba(196,181,253,.08), transparent 65%);
      background-attachment: fixed;
      -webkit-font-smoothing: antialiased;
    }
    body::before {
      content: '';
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255,255,255,.012) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,.012) 1px, transparent 1px);
      background-size: 60px 60px;
      mask-image: radial-gradient(ellipse at center, black 0%, transparent 80%);
    }
    .shell { width: min(820px, 100%); position: relative; z-index: 1; }
    .brand { display: flex; align-items: center; gap: 16px; margin-bottom: 26px; }
    .mark {
      width: 48px; height: 48px;
      display: grid; place-items: center;
      border-radius: 14px;
      background: linear-gradient(135deg, #10b981 0%, #06b6d4 60%, #0ea5e9 100%);
      box-shadow: 0 0 30px rgba(16,185,129,.4), inset 0 1px 0 rgba(255,255,255,.3);
      color: #fff;
      font-size: 22px;
    }
    .mark svg { width: 24px; height: 24px; }
    h1 {
      margin: 0;
      font-family: var(--font-display);
      font-weight: 500;
      font-size: clamp(1.85rem, 4vw, 2.6rem);
      letter-spacing: -0.03em;
      color: #fff;
      line-height: 1.1;
    }
    .eyebrow {
      margin: 6px 0 0;
      color: var(--muted);
      font-size: 0.84rem;
      letter-spacing: 0.02em;
    }
    .card {
      padding: clamp(28px, 5vw, 48px);
      border: 1px solid var(--border);
      border-radius: 22px;
      background: var(--surface);
      box-shadow: 0 24px 60px -12px rgba(0,0,0,.6);
      backdrop-filter: blur(28px) saturate(150%);
      -webkit-backdrop-filter: blur(28px) saturate(150%);
      position: relative;
      overflow: hidden;
    }
    .card::before {
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255,255,255,.12), transparent);
    }
    .status {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 18px 22px;
      border: 1px solid rgba(16,185,129,.28);
      border-radius: 14px;
      background: linear-gradient(135deg, rgba(16,185,129,.12), rgba(16,185,129,.04));
      color: #6ee7b7;
      font-weight: 600;
      letter-spacing: 0.01em;
    }
    .dot {
      position: relative;
      width: 11px; height: 11px;
      border-radius: 50%;
      background: #34d399;
      flex-shrink: 0;
    }
    .dot::before {
      content: '';
      position: absolute;
      inset: -5px;
      border-radius: 50%;
      background: #34d399;
      opacity: 0.4;
      animation: pulse 2.2s ease-in-out infinite;
    }
    @keyframes pulse { 0%, 100% { transform: scale(.8); opacity: .5; } 50% { transform: scale(1.4); opacity: 0; } }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 14px;
      margin-top: 22px;
    }
    .metric {
      padding: 18px 20px;
      border: 1px solid var(--border);
      border-radius: 14px;
      background: rgba(6,11,22,.55);
      position: relative;
      overflow: hidden;
    }
    .metric::before {
      content: '';
      position: absolute;
      left: 0; top: 0; bottom: 0;
      width: 2px;
      background: linear-gradient(180deg, var(--cyan), var(--violet));
      opacity: 0.5;
    }
    .label {
      color: var(--dim);
      font-family: var(--font-mono);
      font-size: 0.65rem;
      font-weight: 700;
      letter-spacing: 0.16em;
      text-transform: uppercase;
    }
    .value {
      margin-top: 8px;
      color: var(--text);
      font-family: var(--font-mono);
      font-weight: 700;
      font-size: 1.02rem;
      overflow-wrap: anywhere;
    }
    .links {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 28px;
    }
    a {
      padding: 10px 16px;
      border: 1px solid var(--border);
      border-radius: 10px;
      color: #cbd5e1;
      text-decoration: none;
      font-size: 0.82rem;
      font-weight: 600;
      background: rgba(255,255,255,.04);
      transition: all 0.25s ease;
      letter-spacing: 0.01em;
    }
    a:hover {
      border-color: rgba(34,211,238,.4);
      color: #fff;
      background: rgba(34,211,238,.1);
      transform: translateY(-1px);
    }
    @media (max-width: 520px) { .grid { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main class="shell">
    <div class="brand">
      <div class="mark">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>
      </div>
      <div>
        <h1>GridWise Health Probe</h1>
        <p class="eyebrow">Live service readiness and runtime status</p>
      </div>
    </div>
    <section class="card">
      <div class="status"><span class="dot"></span><span id="health-status">Checking service health…</span></div>
      <div class="grid">
        <div class="metric"><div class="label">API Status</div><div class="value" id="api-status">—</div></div>
        <div class="metric"><div class="label">Version</div><div class="value" id="version">—</div></div>
        <div class="metric"><div class="label">Solver</div><div class="value" id="solver">—</div></div>
        <div class="metric"><div class="label">LLM Configured</div><div class="value" id="llm">—</div></div>
      </div>
      <nav class="links">
        <a href="/">← Live Dashboard</a>
        <a href="/docs">Swagger Docs</a>
        <a href="/health">JSON Response</a>
      </nav>
    </section>
  </main>
  <script>
    Promise.all([fetch('/health'), fetch('/api/status')]).then(async ([healthResponse, statusResponse]) => {
      const health = await healthResponse.json();
      const status = await statusResponse.json();
      document.getElementById('health-status').textContent = health.status === 'ok' ? 'Service is healthy and ready' : 'Service reported an issue';
      document.getElementById('api-status').textContent = health.status.toUpperCase();
      document.getElementById('version').textContent = status.version;
      document.getElementById('solver').textContent = status.solver;
      document.getElementById('llm').textContent = status.llm_configured ? 'Configured' : 'Offline fallback';
    }).catch(() => { document.getElementById('health-status').textContent = 'Unable to reach service'; });
  </script>
</body>
</html>""",
        )
    return {"status": "ok"}


@app.get("/api/status")
def status():
    return {
        "status": "ok",
        "version": APP_VERSION,
        "solver": "HiGHS (scipy.optimize.linprog)",
        "llm_configured": bool(os.getenv("GEMINI_API_KEY")),
        "llm_models": _candidate_models(),
    }


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    return _execute(req)["response"]


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: OptimizeRequest):
    result = _execute(req)
    return AnalyzeResponse(
        optimization=result["response"], analytics=compute_analytics(req, result)
    )


@app.post("/api/export-csv")
def export_csv(req: OptimizeRequest):
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
