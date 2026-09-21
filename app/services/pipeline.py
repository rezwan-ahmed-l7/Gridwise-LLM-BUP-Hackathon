"""End-to-end processing pipeline.

Combines interpretation → guardrails → optimization → analytics.  Keeps
the FastAPI route handlers thin and side-effect-free.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict

from app.core.llm import interpret_notes_with_llm
from app.models.schemas import (
    Analytics,
    AnalyzeResponse,
    HourlyInsight,
    OptimizeRequest,
    OptimizeResponse,
)
from app.services.guardrails import guardrail_directives
from app.services.optimizer import (
    HOURS_PER_DAY,
    build_constraints,
    optimize_energy,
)


logger = logging.getLogger("gridwise.pipeline")


def run_pipeline(req: OptimizeRequest) -> Dict[str, Any]:
    """Execute the full LLM → guardrail → optimization pipeline.

    Returns a dict containing the :class:`OptimizeResponse`, intermediate
    constraints, warnings, interpreter label, and elapsed milliseconds.
    """
    started = time.perf_counter()

    raw_interp, interpreter = interpret_notes_with_llm(req.operator_notes, req.battery)
    directives = guardrail_directives(raw_interp, req.operator_notes, req.battery)
    cons = build_constraints(req.hours, req.battery, directives)
    plan, warnings = optimize_energy(req.hours, req.battery, directives, cons)

    total_grid = sum(p.grid_kwh for p in plan)
    total_cost = sum(
        p.grid_kwh * req.hours[p.hour].tariff_bdt_per_kwh for p in plan
    )
    peak_grid = max((p.grid_kwh for p in plan), default=0.0)

    applied_types = [d.directive_type for d in directives if d.applies]
    summary = (
        "Successfully optimized 24-hour schedule. "
        f"Directives applied: {', '.join(applied_types) if applied_types else 'none'}. "
        f"Total grid import: {total_grid:.2f} kWh, "
        f"Total cost: {total_cost:.2f} BDT, "
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
    """Compute baseline-vs-optimized analytics from a pipeline result."""
    resp: OptimizeResponse = result["response"]
    cons = result["constraints"]
    plan = resp.hourly_plan

    # Baseline = grid-only, no battery use, after solar
    baseline_grid = [
        max(0.0, req.hours[h].demand_kwh - cons["effective_solar"][h])
        for h in range(HOURS_PER_DAY)
    ]
    baseline_cost = sum(
        baseline_grid[h] * req.hours[h].tariff_bdt_per_kwh
        for h in range(HOURS_PER_DAY)
    )
    baseline_peak = max(baseline_grid) if baseline_grid else 0.0

    savings = baseline_cost - resp.total_cost_bdt
    savings_pct = (savings / baseline_cost * 100.0) if baseline_cost > 1e-9 else 0.0

    total_eff_solar = sum(cons["effective_solar"])
    total_solar_used = sum(p.solar_used_kwh for p in plan)
    solar_util = (
        (total_solar_used / total_eff_solar * 100.0)
        if total_eff_solar > 1e-9
        else 0.0
    )
    discharged = sum(
        p.battery_kwh for p in plan if p.battery_action == "discharge"
    )

    hourly = []
    for h in range(HOURS_PER_DAY):
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
                cost_bdt=round(
                    plan[h].grid_kwh * req.hours[h].tariff_bdt_per_kwh, 4
                ),
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
        equivalent_full_cycles=round(
            discharged / req.battery.capacity_kwh, 3
        ),
        warnings=result["warnings"],
        hourly=hourly,
    )
