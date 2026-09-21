"""Deterministic NLP fallback for directive extraction.

This is used both when no LLM API key is configured and when the LLM fails
or returns an invalid result.  The patterns here are tightly aligned with
the LLM prompt so that the two paths produce equivalent semantics.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Time / window parsing
# ---------------------------------------------------------------------------
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
    """Parse a single time token like ``"3pm"``, ``"noon"``, ``"14:00"``."""
    t = token.strip().lower()
    if t in ("noon", "midday"):
        return 12
    if t == "midnight":
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
    """Return inclusive-exclusive list of hours between ``h1`` and ``h2``."""
    if h2 == 0 and h1 > 0:
        h2 = 24
    if 0 <= h1 < h2 <= 24:
        return list(range(h1, h2))
    if allow_wrap and 0 <= h2 < h1 <= 23:
        return sorted(list(range(h1, 24)) + list(range(0, h2)))
    return []


def _is_explicit_overnight(s1: str, s2: str) -> bool:
    """Detect overnight windows (e.g. '10pm to 2am', '22:00 to 02:00')."""
    if "pm" in s1 and "am" in s2:
        return True
    return ":" in s1 and ":" in s2 and "am" not in s1 + s2 and "pm" not in s1 + s2


def extract_hours_window(text: str) -> List[int]:
    """Extract a list of integer hours (0..23) from a free-text window."""
    t = text.lower()
    # Pattern 1: "6 PM – 9 PM" or "6-9 pm"
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

    # Pattern 2: "from X to Y"
    m = re.search(
        r"(?:from|between)\s+([a-z0-9:]+(?:\s*(?:am|pm))?)\s+"
        r"(?:until|to|and)\s+([a-z0-9:]+(?:\s*(?:am|pm))?)",
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

    # Pattern 3: "X to Y" bare
    m_to = re.search(
        r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\s*(?:until|to)\s*"
        r"(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)",
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


# ---------------------------------------------------------------------------
# Solar reduction factor
# ---------------------------------------------------------------------------
_PCT = r"(\d+(?:\.\d+)?)\s*(?:%|percent)"


def _extract_solar_factor(t: str) -> float:
    """Return the *usable* fraction of solar (1.0 = full, 0.0 = nothing)."""
    m_red = re.search(_PCT + r"\s*(?:reduction|drop|decline|loss|cut)", t)
    if m_red:
        return round(max(0.0, min(1.0, (100.0 - float(m_red.group(1))) / 100.0)), 4)

    m_by = re.search(
        r"(?:reduc\w*|drop\w*|declin\w*|fall\w*|lower\w*|cut|down|los[et]\w*)"
        r"\s*(?:by|of)?\s*" + _PCT,
        t,
    )
    if m_by:
        return round(max(0.0, min(1.0, (100.0 - float(m_by.group(1))) / 100.0)), 4)

    m_pct = re.search(r"(?:to|roughly|about|leaving|leave|at)\s*" + _PCT, t)
    if m_pct:
        return round(max(0.0, min(1.0, float(m_pct.group(1)) / 100.0)), 4)

    # Fractional language
    if "half" in t:
        return 0.5
    if "one-fifth" in t:
        return 0.2
    if "one-fourth" in t or "quarter" in t:
        return 0.25
    if "one-third" in t:
        return round(1.0 / 3.0, 4)
    return 0.5


# ---------------------------------------------------------------------------
# Top-level directive extractor
# ---------------------------------------------------------------------------
def extract_directive_fallback(
    note: str, battery_capacity: float = 200.0
) -> Dict[str, Any]:
    """Deterministic, rule-based directive extractor.

    Returns a directive dict in the same shape as the LLM output so the rest
    of the pipeline does not care which path produced it.
    """
    t = note.lower()
    hours = extract_hours_window(t)

    # 1. Solar reduction ---------------------------------------------------
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
                "explanation": (
                    f"Solar availability reduced during hours {hours} "
                    f"(usable factor: {factor})."
                ),
            }

    # 2. No discharge ------------------------------------------------------
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

    # 3. No charge ---------------------------------------------------------
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

    # 4. Minimum battery reserve ------------------------------------------
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
            "structured_adjustment": {
                "hours": hours,
                "minimum_energy_kwh": val,
            },
            "explanation": (
                f"Battery reserve raised to {val} kWh during hours {hours}."
            ),
        }

    # 5. Max grid window ---------------------------------------------------
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
            "explanation": (
                f"Grid import is capped at {val} kWh during hours {hours}."
            ),
        }

    # 6. Default: no-op ----------------------------------------------------
    return {
        "applies": False,
        "directive_type": "no_op",
        "structured_adjustment": None,
        "explanation": (
            "This note does not affect today's 24-hour energy schedule."
        ),
    }
