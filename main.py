"""
GridWise LLM - BUP CSE Fest 2026 Hackathon Preliminary
Smart Campus Energy Optimization with LLM-assisted Operator Directive Interpretation
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict, Any, Literal
import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

# ------------------ Logging ------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise")

app = FastAPI(
    title="GridWise LLM",
    description="BUP CSE Fest 2026 - Smart Campus Energy Optimization",
    version="1.0.0"
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
    hours: List[int] = []
    factor: Optional[float] = None
    minimum_energy_kwh: Optional[float] = None
    max_grid_kwh: Optional[float] = None


class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: Literal[
        "solar_reduction",
        "minimum_battery_reserve",
        "no_charge_window",
        "no_discharge_window",
        "max_grid_window",
        "no_op"
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


# ------------------ LLM Interpretation ------------------

SUPPORTED_DIRECTIVES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op"
}


def get_llm_prompt(notes: List[str]) -> str:
    notes_text = "\n".join([f"{i}. {note}" for i, note in enumerate(notes)])
    return f"""You are an expert energy management system interpreter.

Analyze the following operator notes for a 24-hour campus energy schedule (hours 0-23).

OPERATOR NOTES:
{notes_text}

For EACH note, extract exactly one directive. Allowed directive_type values:
- solar_reduction: solar output is reduced. Need "hours" (list of hours) and "factor" (remaining fraction, e.g. 0.2 means 80% reduction)
- minimum_battery_reserve: raise minimum battery energy for certain hours. Need "hours" and "minimum_energy_kwh"
- no_charge_window: battery cannot charge in these hours. Need "hours"
- no_discharge_window: battery cannot discharge in these hours. Need "hours"
- max_grid_window: limit grid import. Need "hours" and "max_grid_kwh"
- no_op: the note does not impose any actionable constraint

Rules:
- Time windows are start-inclusive, end-exclusive. "1 PM to 3 PM" → hours [13, 14]
- Hours must be integers 0-23, unique, sorted ascending
- For solar_reduction: factor is remaining fraction (80% reduction → factor=0.2)
- If note is unclear or has no actionable constraint → no_op

Return ONLY a valid JSON array with one object per note (in order):
[
  {{
    "note_index": 0,
    "applies": true/false,
    "directive_type": "...",
    "structured_adjustment": {{ "hours": [...], "factor": 0.2 }} or null,
    "explanation": "short reason"
  }}
]
"""


def interpret_notes_with_llm(notes: List[str]) -> List[Dict]:
    """Call Gemini to interpret operator notes. Falls back to no_op on failure."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("No GEMINI_API_KEY found. Using no_op for all notes.")
        return [
            {
                "note_index": i,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": "No LLM key configured"
            }
            for i in range(len(notes))
        ]

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = get_llm_prompt(notes)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
        )
        text = response.text.strip()
        # Clean markdown if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("LLM did not return a list")
        return data
    except Exception as e:
        logger.error(f"LLM interpretation failed: {e}")
        return [
            {
                "note_index": i,
                "applies": False,
                "directive_type": "no_op",
                "structured_adjustment": None,
                "explanation": f"LLM failed: {str(e)[:80]}"
            }
            for i in range(len(notes))
        ]


def guardrail_directives(raw: List[Dict], num_notes: int) -> List[DirectiveInterpretation]:
    """Deterministic validation & cleaning of LLM output."""
    result = []
    for i in range(num_notes):
        item = next((x for x in raw if x.get("note_index") == i), None)
        if item is None:
            result.append(DirectiveInterpretation(
                note_index=i, applies=False, directive_type="no_op",
                structured_adjustment=None, explanation="Missing interpretation"
            ))
            continue

        dtype = item.get("directive_type", "no_op")
        if dtype not in SUPPORTED_DIRECTIVES:
            dtype = "no_op"

        applies = bool(item.get("applies", False)) and dtype != "no_op"
        adj = None

        if applies:
            sa = item.get("structured_adjustment") or {}
            hours = sa.get("hours", [])
            # Clean hours
            clean_hours = sorted(set(
                h for h in hours
                if isinstance(h, (int, float)) and 0 <= int(h) <= 23
            ))
            clean_hours = [int(h) for h in clean_hours]

            if not clean_hours and dtype != "no_op":
                applies = False
                dtype = "no_op"
            else:
                adj = StructuredAdjustment(hours=clean_hours)
                if dtype == "solar_reduction":
                    factor = sa.get("factor")
                    if factor is None or not (0 < float(factor) <= 1):
                        applies = False
                        dtype = "no_op"
                        adj = None
                    else:
                        adj.factor = float(factor)
                elif dtype == "minimum_battery_reserve":
                    val = sa.get("minimum_energy_kwh")
                    if val is None or float(val) < 0:
                        applies = False
                        dtype = "no_op"
                        adj = None
                    else:
                        adj.minimum_energy_kwh = float(val)
                elif dtype == "max_grid_window":
                    val = sa.get("max_grid_kwh")
                    if val is None or float(val) < 0:
                        applies = False
                        dtype = "no_op"
                        adj = None
                    else:
                        adj.max_grid_kwh = float(val)

        if not applies:
            dtype = "no_op"
            adj = None

        result.append(DirectiveInterpretation(
            note_index=i,
            applies=applies,
            directive_type=dtype,
            structured_adjustment=adj,
            explanation=str(item.get("explanation", ""))[:200]
        ))
    return result


# ------------------ Optimizer ------------------

def optimize_energy(
    hours: List[HourData],
    battery: BatteryData,
    directives: List[DirectiveInterpretation]
) -> List[HourlyPlan]:
    """
    Simple but correct rule-based optimizer with directive support.
    Priority: satisfy demand → use solar → use battery (when cheap) → grid.
    """
    n = 24
    demand = [h.demand_kwh for h in hours]
    solar = [h.solar_kwh for h in hours]
    tariff = [h.tariff_bdt_per_kwh for h in hours]

    # Apply solar_reduction
    effective_solar = solar[:]
    for d in directives:
        if d.applies and d.directive_type == "solar_reduction" and d.structured_adjustment:
            factor = d.structured_adjustment.factor or 1.0
            for h in d.structured_adjustment.hours:
                effective_solar[h] *= factor

    # Build constraint maps
    no_charge = set()
    no_discharge = set()
    min_reserve = [battery.minimum_energy_kwh] * n
    max_grid = [float("inf")] * n

    for d in directives:
        if not d.applies or not d.structured_adjustment:
            continue
        hs = d.structured_adjustment.hours
        if d.directive_type == "no_charge_window":
            no_charge.update(hs)
        elif d.directive_type == "no_discharge_window":
            no_discharge.update(hs)
        elif d.directive_type == "minimum_battery_reserve":
            val = d.structured_adjustment.minimum_energy_kwh or battery.minimum_energy_kwh
            for h in hs:
                min_reserve[h] = max(min_reserve[h], val)
        elif d.directive_type == "max_grid_window":
            val = d.structured_adjustment.max_grid_kwh
            if val is not None:
                for h in hs:
                    max_grid[h] = min(max_grid[h], val)

    # Greedy optimization with end-of-day neutrality
    energy = battery.initial_energy_kwh
    plan = []
    grid_usage = [0.0] * n
    solar_used = [0.0] * n
    batt_action = ["idle"] * n
    batt_kwh = [0.0] * n
    energy_after = [0.0] * n

    # First pass: satisfy demand with solar + battery + grid
    for h in range(n):
        rem = demand[h]
        su = min(effective_solar[h], rem)
        rem -= su
        solar_used[h] = su

        # Try discharge if beneficial or needed
        can_discharge = (
            h not in no_discharge
            and energy > min_reserve[h]
            and rem > 0
        )
        if can_discharge:
            max_dis = min(
                battery.max_discharge_kwh_per_hour,
                energy - min_reserve[h],
                rem
            )
            if max_dis > 0.001:
                batt_action[h] = "discharge"
                batt_kwh[h] = max_dis
                energy -= max_dis
                rem -= max_dis

        # Grid for remaining (respect max_grid)
        g = min(rem, max_grid[h])
        grid_usage[h] = g
        rem -= g

        # If still remaining (max_grid hit), try more discharge already handled
        energy_after[h] = energy

    # Second pass: charge battery when tariff is low and possible
    # Simple heuristic: charge in lowest tariff hours if room
    charge_candidates = sorted(
        [h for h in range(n) if h not in no_charge],
        key=lambda x: tariff[x]
    )

    for h in charge_candidates:
        room = battery.capacity_kwh - energy_after[h]
        if room < 0.01:
            continue
        # Only charge if we can do it without violating later min_reserve too badly
        max_ch = min(
            battery.max_charge_kwh_per_hour,
            room,
            max_grid[h] - grid_usage[h] + 1e-6  # can use extra grid to charge
        )
        if max_ch > 0.01 and tariff[h] < sorted(tariff)[n // 3]:  # charge only in cheaper 1/3
            # Add charge
            if batt_action[h] == "idle":
                batt_action[h] = "charge"
                batt_kwh[h] = max_ch
                grid_usage[h] += max_ch
                # Update energy from this hour onward
                for t in range(h, n):
                    energy_after[t] += max_ch
                energy = energy_after[n-1]

    # Enforce end-of-day neutrality approximately
    final_diff = energy_after[n-1] - battery.initial_energy_kwh
    if abs(final_diff) > 0.05:
        # Try to correct in last hours
        for h in range(n-1, -1, -1):
            if abs(final_diff) < 0.05:
                break
            if final_diff > 0 and h not in no_discharge and batt_action[h] != "charge":
                # discharge a bit more
                extra = min(final_diff, battery.max_discharge_kwh_per_hour - batt_kwh[h], energy_after[h] - min_reserve[h])
                if extra > 0.01:
                    if batt_action[h] == "idle":
                        batt_action[h] = "discharge"
                    batt_kwh[h] += extra
                    energy_after[h] -= extra
                    for t in range(h+1, n):
                        energy_after[t] -= extra
                    final_diff -= extra
            elif final_diff < 0 and h not in no_charge and batt_action[h] != "discharge":
                extra = min(-final_diff, battery.max_charge_kwh_per_hour - batt_kwh[h], battery.capacity_kwh - energy_after[h])
                if extra > 0.01:
                    if batt_action[h] == "idle":
                        batt_action[h] = "charge"
                    batt_kwh[h] += extra
                    grid_usage[h] += extra
                    energy_after[h] += extra
                    for t in range(h+1, n):
                        energy_after[t] += extra
                    final_diff += extra

    # Build final plan
    for h in range(n):
        plan.append(HourlyPlan(
            hour=h,
            grid_kwh=round(max(0, grid_usage[h]), 4),
            solar_used_kwh=round(solar_used[h], 4),
            battery_action=batt_action[h],
            battery_kwh=round(batt_kwh[h], 4),
            battery_energy_after_kwh=round(max(min_reserve[h], min(battery.capacity_kwh, energy_after[h])), 4)
        ))
    return plan


# ------------------ API Endpoints ------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize(req: OptimizeRequest):
    try:
        # 1. LLM Interpretation
        raw_interp = interpret_notes_with_llm(req.operator_notes)
        directives = guardrail_directives(raw_interp, len(req.operator_notes))

        # 2. Optimize
        plan = optimize_energy(req.hours, req.battery, directives)

        # 3. Aggregates
        total_grid = sum(p.grid_kwh for p in plan)
        total_cost = sum(p.grid_kwh * req.hours[p.hour].tariff_bdt_per_kwh for p in plan)
        peak_grid = max(p.grid_kwh for p in plan) if plan else 0.0

        # Summary
        applied = [d.directive_type for d in directives if d.applies]
        summary = f"Applied {len(applied)} directive(s): {', '.join(applied) if applied else 'none'}. " \
                  f"Total grid {total_grid:.2f} kWh, cost {total_cost:.2f} BDT."

        return OptimizeResponse(
            scenario_id=req.scenario_id,
            directive_interpretation=directives,
            hourly_plan=plan,
            total_grid_kwh=round(total_grid, 4),
            total_cost_bdt=round(total_cost, 4),
            peak_grid_kwh=round(peak_grid, 4),
            plan_summary=summary
        )
    except Exception as e:
        logger.exception("Optimization failed")
        raise HTTPException(status_code=500, detail=str(e)[:200])


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
