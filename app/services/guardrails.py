"""Directive guardrails.

Validates and cleans raw LLM output into the canonical
``DirectiveInterpretation`` shape.  When LLM output is malformed or
semantically off, deterministic fallbacks are used per-note so the
optimizer never receives an unsafe payload.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from app.models.schemas import (
    BatteryData,
    DirectiveInterpretation,
    StructuredAdjustment,
    SUPPORTED_DIRECTIVES,
)
from app.utils.parser import extract_directive_fallback


logger = logging.getLogger("gridwise.guardrails")


def guardrail_directives(
    raw: List[Dict[str, Any]],
    notes: List[str],
    battery: BatteryData,
) -> List[DirectiveInterpretation]:
    """Validate and normalize raw directive interpretation output.

    ``raw`` may be from the LLM (preferred) or ``[]``.  The returned list
    contains exactly ``len(notes)`` items in input order.
    """
    result: List[DirectiveInterpretation] = []
    num_notes = len(notes)
    raw = [x for x in (raw or []) if isinstance(x, dict)]

    for i in range(num_notes):
        note_text = notes[i]
        item = next(
            (x for x in raw if x.get("note_index") == i),
            raw[i] if i < len(raw) else None,
        )

        fallback_dir = extract_directive_fallback(note_text, battery.capacity_kwh)
        if item is None:
            item = fallback_dir

        # ---- Type validation -------------------------------------------------
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
                            "This note does not affect today's 24-hour "
                            "energy schedule.",
                        )
                    )[:200],
                )
            )
            continue

        # ---- Hour list cleaning ---------------------------------------------
        sa = item.get("structured_adjustment")
        if not isinstance(sa, dict):
            sa = {}
        hours = sa.get("hours", [])
        if not isinstance(hours, list):
            hours = []
        clean_hours = sorted(
            {
                int(h)
                for h in hours
                if isinstance(h, (int, float)) and 0 <= int(h) <= 23
            }
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

        # ---- Per-type numeric fields ----------------------------------------
        adj = StructuredAdjustment(hours=clean_hours)
        valid = True

        if dtype == "solar_reduction":
            factor = sa.get("factor")
            if factor is None:
                factor = (fallback_dir.get("structured_adjustment") or {}).get(
                    "factor", 0.5
                )
            try:
                adj.factor = round(max(0.0, min(1.0, float(factor))), 4)
            except (ValueError, TypeError):
                valid = False

        elif dtype == "minimum_battery_reserve":
            val = sa.get("minimum_energy_kwh")
            if val is None:
                val = (fallback_dir.get("structured_adjustment") or {}).get(
                    "minimum_energy_kwh", battery.minimum_energy_kwh
                )
            try:
                adj.minimum_energy_kwh = round(
                    max(0.0, min(battery.capacity_kwh, float(val))), 2
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
                adj.max_grid_kwh = round(max(0.0, float(val)), 2)
            except (ValueError, TypeError):
                valid = False

        elif dtype in ("no_charge_window", "no_discharge_window"):
            pass  # hours only

        # ---- Emit ----------------------------------------------------------
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
