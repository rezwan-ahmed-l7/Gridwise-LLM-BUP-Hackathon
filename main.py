"""
GridWise LLM - BUP CSE Fest 2026 Hackathon Preliminary
Smart Campus Energy Optimization with LLM-assisted Operator Directive Interpretation
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_serializer
from typing import List, Optional, Dict, Any, Literal
import os
import re
import json
import logging
import numpy as np
from scipy.optimize import linprog
from dotenv import load_dotenv

load_dotenv()

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise")

app = FastAPI(
    title="GridWise LLM",
    description="BUP CSE Fest 2026 - Smart Campus Energy Optimization",
    version="1.0.0",
)

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


def extract_hours_window(text: str) -> List[int]:
    """
    Extract start-inclusive, end-exclusive hours from natural language.
    Examples:
      'from noon until 2 PM' -> [12, 13]
      '1 PM to 3 PM' -> [13, 14]
      '1-3 PM' -> [13, 14]
      'between 13:00 and 15:00' -> [13, 14]
      'from one until three' -> [13, 14]
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
        if 0 <= h1 < h2 <= 24:
            return list(range(h1, h2))

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
        if h1 is not None and h2 is not None and 0 <= h1 < h2 <= 24:
            return list(range(h1, h2))

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
        if h1 is not None and h2 is not None and 0 <= h1 < h2 <= 24:
            return list(range(h1, h2))

    return []


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
            # Check explicit reduction percentage: e.g. "80% reduction" -> factor 0.2
            m_red = re.search(r"(\d+(?:\.\d+)?)\s*%\s*reduction", t)
            if m_red:
                pct = float(m_red.group(1))
                factor = round(max(0.0, min(1.0, (100.0 - pct) / 100.0)), 4)
            else:
                m_pct = re.search(
                    r"(?:to|roughly|about|leaving|leave)\s*(\d+(?:\.\d+)?)\s*%", t
                )
                if m_pct:
                    factor = round(float(m_pct.group(1)) / 100.0, 4)
                elif "half" in t:
                    factor = 0.5
                elif "one-fifth" in t:
                    factor = 0.2
                elif "one-fourth" in t or "quarter" in t:
                    factor = 0.25
                elif "one-third" in t:
                    factor = round(1.0 / 3.0, 4)
                else:
                    factor = 0.5

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


def interpret_notes_with_llm(notes: List[str], battery: BatteryData) -> List[Dict]:
    """Call Google Gemini to interpret operator notes, with graceful fallback on error."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.info(
            "No GEMINI_API_KEY configured. Using deterministic parser for all notes."
        )
        return [
            extract_directive_fallback(note, battery.capacity_kwh) for note in notes
        ]

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)

        prompt = get_llm_prompt(notes, battery.capacity_kwh, battery.minimum_energy_kwh)
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "response_mime_type": "application/json",
            },
        )
        text = response.text.strip()
        # Clean markdown code blocks if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text.strip())
        if not isinstance(data, list):
            raise ValueError("LLM response is not a JSON array")
        return data
    except Exception as e:
        logger.warning(
            f"LLM interpretation encountered issue: {e}. Utilizing deterministic fallback parser."
        )
        return [
            extract_directive_fallback(note, battery.capacity_kwh) for note in notes
        ]


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
        sa = item.get("structured_adjustment") or {}
        hours = sa.get("hours", [])

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
                factor = fallback_dir.get("structured_adjustment", {}).get(
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
                val = fallback_dir.get("structured_adjustment", {}).get(
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
                val = fallback_dir.get("structured_adjustment", {}).get(
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


# ------------------ Exact LP Optimizer (HiGHS) ------------------


def optimize_energy(
    hours: List[HourData],
    battery: BatteryData,
    directives: List[DirectiveInterpretation],
) -> List[HourlyPlan]:
    """
    Solves the 24-hour campus energy scheduling problem to global mathematical
    optimality using Linear Programming (HiGHS solver via scipy.optimize.linprog).

    Decision Variables per hour h in 0..23:
      - g_h: Grid energy purchased (kWh)
      - s_h: Solar energy utilized (kWh)
      - c_h: Battery energy charged (kWh)
      - d_h: Battery energy discharged (kWh)
      - E_h: Battery energy level after hour h (kWh)

    Constraints:
      1. Energy Balance: g_h + s_h + d_h - c_h = demand_h  (for all h)
      2. State of Charge Dynamics: E_h - E_{h-1} - c_h + d_h = 0  (with E_{-1} = initial_energy)
      3. End-of-day Neutrality: E_23 = initial_energy
      4. Solar Availability: 0 <= s_h <= effective_solar_h
      5. Grid Import: 0 <= g_h <= max_grid_h
      6. Charge Rate: 0 <= c_h <= max_charge_rate_h
      7. Discharge Rate: 0 <= d_h <= max_discharge_rate_h
      8. Storage Bounds: min_reserve_h <= E_h <= capacity
    """
    n = 24
    demand = [h.demand_kwh for h in hours]
    solar = [h.solar_kwh for h in hours]
    tariff = [h.tariff_bdt_per_kwh for h in hours]

    # 1. Apply solar_reduction directives to determine effective available solar
    effective_solar = solar[:]
    for d in directives:
        if (
            d.applies
            and d.directive_type == "solar_reduction"
            and d.structured_adjustment
        ):
            factor = (
                d.structured_adjustment.factor
                if d.structured_adjustment.factor is not None
                else 1.0
            )
            for h in d.structured_adjustment.hours:
                effective_solar[h] *= factor

    # 2. Extract constraint maps from directives
    no_charge_hours = set()
    no_discharge_hours = set()
    min_reserve = [battery.minimum_energy_kwh] * n
    max_grid = [float("inf")] * n

    for d in directives:
        if not d.applies or not d.structured_adjustment:
            continue
        hs = d.structured_adjustment.hours
        dtype = d.directive_type
        if dtype == "no_charge_window":
            no_charge_hours.update(hs)
        elif dtype == "no_discharge_window":
            no_discharge_hours.update(hs)
        elif dtype == "minimum_battery_reserve":
            val = d.structured_adjustment.minimum_energy_kwh
            if val is not None:
                for h in hs:
                    min_reserve[h] = max(min_reserve[h], val)
        elif dtype == "max_grid_window":
            val = d.structured_adjustment.max_grid_kwh
            if val is not None:
                for h in hs:
                    max_grid[h] = min(max_grid[h], val)

    # 3. Setup Linear Program
    # Variables indexing (120 total):
    #   g_h: 0..23
    #   s_h: 24..47
    #   c_h: 48..71
    #   d_h: 72..95
    #   E_h: 96..119
    num_vars = 120
    c_obj = np.zeros(num_vars)
    for h in range(n):
        c_obj[h] = tariff[h]  # Minimize total grid cost
        c_obj[24 + h] = (
            -1e-6
        )  # Secondary objective: strictly maximize solar utilization

    # Variable bounds
    bounds = []
    # g_h: [0, max_grid[h]]
    for h in range(n):
        ub = max_grid[h] if max_grid[h] != float("inf") else None
        bounds.append((0.0, ub))
    # s_h: [0, effective_solar[h]]
    for h in range(n):
        bounds.append((0.0, max(0.0, effective_solar[h])))
    # c_h: [0, max_charge]
    for h in range(n):
        ub = 0.0 if h in no_charge_hours else battery.max_charge_kwh_per_hour
        bounds.append((0.0, ub))
    # d_h: [0, max_discharge]
    for h in range(n):
        ub = 0.0 if h in no_discharge_hours else battery.max_discharge_kwh_per_hour
        bounds.append((0.0, ub))
    # E_h: [min_reserve[h], capacity]
    for h in range(n):
        bounds.append((min_reserve[h], battery.capacity_kwh))

    # Equalities A_eq * x = b_eq
    # 24 energy balance + 24 battery state transitions + 1 end-of-day neutrality = 49 rows
    num_eq = 49
    A_eq = np.zeros((num_eq, num_vars))
    b_eq = np.zeros(num_eq)

    # Row 0..23: Energy balance: g_h + s_h + d_h - c_h = demand[h]
    for h in range(n):
        row = h
        A_eq[row, h] = 1.0  # g_h
        A_eq[row, 24 + h] = 1.0  # s_h
        A_eq[row, 48 + h] = -1.0  # -c_h
        A_eq[row, 72 + h] = 1.0  # d_h
        b_eq[row] = demand[h]

    # Row 24..47: Battery dynamics: E_h - E_{h-1} - c_h + d_h = 0
    for h in range(n):
        row = 24 + h
        A_eq[row, 96 + h] = 1.0  # E_h
        A_eq[row, 48 + h] = -1.0  # -c_h
        A_eq[row, 72 + h] = 1.0  # d_h
        if h == 0:
            b_eq[row] = battery.initial_energy_kwh
        else:
            A_eq[row, 96 + h - 1] = -1.0
            b_eq[row] = 0.0

    # Row 48: End-of-day neutrality: E_23 = initial_energy_kwh
    A_eq[48, 96 + 23] = 1.0
    b_eq[48] = battery.initial_energy_kwh

    res = linprog(c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        logger.error(f"Linear program failed to find optimal schedule: {res.message}")
        raise RuntimeError(f"Optimization solver failed: {res.message}")

    x = res.x
    g_arr = x[0:24]
    s_arr = x[24:48]
    c_arr = x[48:72]
    d_arr = x[72:96]
    e_arr = x[96:120]

    # Post-process: Cancel any simultaneous charge and discharge, enforcing mutual exclusivity
    plan = []
    current_energy = battery.initial_energy_kwh

    for h in range(n):
        net = c_arr[h] - d_arr[h]
        if net > 1e-4:
            action = "charge"
            action_kwh = float(net)
            current_energy += action_kwh
        elif net < -1e-4:
            action = "discharge"
            action_kwh = float(-net)
            current_energy -= action_kwh
        else:
            action = "idle"
            action_kwh = 0.0

        solar_val = float(max(0.0, min(effective_solar[h], s_arr[h])))

        # Re-derive grid_kwh exactly from energy balance to guarantee 0 numeric error
        # g_h = demand_h + c_h - s_h - d_h
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

    # Ensure exact end-of-day matching within tolerance
    plan[-1].battery_energy_after_kwh = round(battery.initial_energy_kwh, 4)

    return plan


# ------------------ API Endpoints ------------------


@app.get("/health")
def health():
    """Readiness endpoint required by BUP Hackathon specification."""
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    """
    Main energy optimization endpoint:
    1. Interprets operator notes via LLM (Gemini) with deterministic NLP fallback.
    2. Validates directives through deterministic guardrails.
    3. Solves the 24-hour campus energy scheduling problem to global optimality via HiGHS.
    4. Computes aggregates and returns validated structured response.
    """
    try:
        # Step 1: LLM Interpretation
        raw_interp = interpret_notes_with_llm(req.operator_notes, req.battery)

        # Step 2: Guardrails & Deterministic Normalization
        directives = guardrail_directives(raw_interp, req.operator_notes, req.battery)

        # Step 3: Global Linear Programming Optimization
        plan = optimize_energy(req.hours, req.battery, directives)

        # Step 4: Compute Aggregates
        total_grid = sum(p.grid_kwh for p in plan)
        total_cost = sum(
            p.grid_kwh * req.hours[p.hour].tariff_bdt_per_kwh for p in plan
        )
        peak_grid = max(p.grid_kwh for p in plan) if plan else 0.0

        applied_types = [d.directive_type for d in directives if d.applies]
        summary = (
            f"Successfully optimized 24-hour schedule. "
            f"Directives applied: {', '.join(applied_types) if applied_types else 'none'}. "
            f"Total grid import: {total_grid:.2f} kWh, Total cost: {total_cost:.2f} BDT, "
            f"Peak grid: {peak_grid:.2f} kWh."
        )

        return OptimizeResponse(
            scenario_id=req.scenario_id,
            directive_interpretation=directives,
            hourly_plan=plan,
            total_grid_kwh=round(total_grid, 4),
            total_cost_bdt=round(total_cost, 4),
            peak_grid_kwh=round(peak_grid, 4),
            plan_summary=summary,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error processing energy optimization request")
        # Ensure safe failure without leaking credentials or raw system stack traces
        raise HTTPException(
            status_code=500,
            detail="Controlled internal server error during energy optimization processing.",
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
