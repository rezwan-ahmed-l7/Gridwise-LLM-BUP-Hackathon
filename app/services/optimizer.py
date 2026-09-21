"""Energy optimization via linear programming.

Implements the GridWise 24-hour campus microgrid LP using ``scipy.optimize``.
The decision variables are laid out as a flat 120-element vector covering:
    grid_kwh[h]          for h in 0..23   (indices  0-23)
    solar_used_kwh[h]    for h in 0..23   (indices 24-47)
    battery_charge[h]    for h in 0..23   (indices 48-71)
    battery_discharge[h] for h in 0..23   (indices 72-95)
    battery_energy[h]    for h in 0..23   (indices 96-119)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.optimize import linprog

from app.models.schemas import (
    BatteryData,
    DirectiveInterpretation,
    HourData,
    HourlyPlan,
)


logger = logging.getLogger("gridwise.optimizer")

HOURS_PER_DAY = 24


class InfeasibleScheduleError(Exception):
    """Raised when the LP solver cannot find a feasible schedule."""


# ---------------------------------------------------------------------------
# Constraint construction
# ---------------------------------------------------------------------------
def build_constraints(
    hours: List[HourData],
    battery: BatteryData,
    directives: List[DirectiveInterpretation],
) -> Dict[str, Any]:
    """Translate directives into per-hour constraint arrays.

    Returns a dict with parallel arrays indexed by hour.
    """
    n = HOURS_PER_DAY
    effective_solar = [float(h.solar_kwh) for h in hours]
    no_charge_hours: Set[int] = set()
    no_discharge_hours: Set[int] = set()
    min_reserve = [float(battery.minimum_energy_kwh)] * n
    max_grid = [float("inf")] * n

    for d in directives:
        if not d.applies or not d.structured_adjustment:
            continue
        adj = d.structured_adjustment
        dtype = d.directive_type

        if dtype == "solar_reduction":
            factor = adj.factor if adj.factor is not None else 1.0
            for h in adj.hours:
                effective_solar[h] *= factor
        elif dtype == "no_charge_window":
            no_charge_hours.update(adj.hours)
        elif dtype == "no_discharge_window":
            no_discharge_hours.update(adj.hours)
        elif dtype == "minimum_battery_reserve":
            if adj.minimum_energy_kwh is not None:
                for h in adj.hours:
                    min_reserve[h] = max(min_reserve[h], adj.minimum_energy_kwh)
        elif dtype == "max_grid_window":
            if adj.max_grid_kwh is not None:
                for h in adj.hours:
                    max_grid[h] = min(max_grid[h], adj.max_grid_kwh)

    return {
        "effective_solar": effective_solar,
        "no_charge_hours": no_charge_hours,
        "no_discharge_hours": no_discharge_hours,
        "min_reserve": min_reserve,
        "max_grid": max_grid,
    }


# ---------------------------------------------------------------------------
# Linear program
# ---------------------------------------------------------------------------
def _solve_lp(
    hours: List[HourData],
    battery: BatteryData,
    cons: Dict[str, Any],
    relax_grid: bool = False,
) -> np.ndarray:
    """Solve the 24-hour scheduling LP. Returns the optimal x vector."""
    n = HOURS_PER_DAY
    num_vars = 5 * n  # 120

    demand = [h.demand_kwh for h in hours]
    tariff = [h.tariff_bdt_per_kwh for h in hours]
    effective_solar = cons["effective_solar"]
    max_grid = cons["max_grid"]
    min_reserve = cons["min_reserve"]

    # ---- Objective: minimize grid cost + tiny anti-idle incentive ---------
    c_obj = np.zeros(num_vars)
    for h in range(n):
        c_obj[h] = tariff[h]              # cost per grid kWh
        c_obj[24 + h] = -1e-6             # tiny nudge to use available solar

    # ---- Variable bounds --------------------------------------------------
    bounds: List[Tuple[float, Optional[float]]] = []

    # grid[h]: capped by directive, optionally relaxed
    for h in range(n):
        cap = max_grid[h]
        ub = None if (relax_grid or cap == float("inf")) else cap
        bounds.append((0.0, ub))

    # solar_used[h]: 0..effective_solar[h]
    for h in range(n):
        bounds.append((0.0, max(0.0, effective_solar[h])))

    # battery_charge[h]
    for h in range(n):
        ub = 0.0 if h in cons["no_charge_hours"] else battery.max_charge_kwh_per_hour
        bounds.append((0.0, ub))

    # battery_discharge[h]
    for h in range(n):
        ub = (
            0.0
            if h in cons["no_discharge_hours"]
            else battery.max_discharge_kwh_per_hour
        )
        bounds.append((0.0, ub))

    # battery_energy[h]: enforces reserve
    for h in range(n):
        bounds.append((min_reserve[h], battery.capacity_kwh))

    # ---- Equality constraints --------------------------------------------
    # 24 hourly energy-balance rows + 24 battery-transition rows +
    # 1 end-of-day neutrality row = 49 rows
    num_eq = 2 * n + 1
    A_eq = np.zeros((num_eq, num_vars))
    b_eq = np.zeros(num_eq)

    # Row h (0..23): demand[h] = grid[h] + solar_used[h] - charge[h] + discharge[h]
    for h in range(n):
        A_eq[h, h] = 1.0            # grid
        A_eq[h, 24 + h] = 1.0       # solar
        A_eq[h, 48 + h] = -1.0      # charge
        A_eq[h, 72 + h] = 1.0       # discharge
        b_eq[h] = demand[h]

    # Row 24+h: battery_energy[h] = battery_energy[h-1] + charge[h] - discharge[h]
    for h in range(n):
        row = n + h
        A_eq[row, 96 + h] = 1.0     # battery_energy[h]
        A_eq[row, 48 + h] = -1.0    # charge[h]
        A_eq[row, 72 + h] = 1.0     # discharge[h]
        if h == 0:
            b_eq[row] = battery.initial_energy_kwh
        else:
            A_eq[row, 96 + h - 1] = -1.0
            b_eq[row] = 0.0

    # End-of-day: battery_energy[23] = initial_energy_kwh  (neutrality)
    A_eq[2 * n, 96 + (n - 1)] = 1.0
    b_eq[2 * n] = battery.initial_energy_kwh

    result = linprog(
        c_obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if not result.success:
        logger.error("LP failed: %s", result.message)
        raise InfeasibleScheduleError(str(result.message))
    return result.x


# ---------------------------------------------------------------------------
# Plan reconstruction
# ---------------------------------------------------------------------------
def optimize_energy(
    hours: List[HourData],
    battery: BatteryData,
    directives: List[DirectiveInterpretation],
    constraints: Optional[Dict[str, Any]] = None,
) -> Tuple[List[HourlyPlan], List[str]]:
    """Run the LP and reconstruct a friendly :class:`HourlyPlan` list."""
    n = HOURS_PER_DAY
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
        # Relax grid caps once to salvage feasibility.
        x = _solve_lp(hours, battery, cons, relax_grid=True)
        msg = (
            "max_grid_window limit could not be satisfied for hours "
            f"{capped}; grid cap was relaxed to keep the schedule feasible."
        )
        warnings.append(msg)
        logger.warning(msg)

    # Unpack the flat decision vector.
    grid_arr = x[0:24]
    solar_arr = x[24:48]
    charge_arr = x[48:72]
    discharge_arr = x[72:96]

    plan: List[HourlyPlan] = []
    for h in range(n):
        net = charge_arr[h] - discharge_arr[h]
        if net > 1e-4:
            action: str = "charge"
            action_kwh = float(net)
        elif net < -1e-4:
            action = "discharge"
            action_kwh = float(-net)
        else:
            action = "idle"
            action_kwh = 0.0

        solar_used = float(max(0.0, min(effective_solar[h], solar_arr[h])))
        charge_kwh = action_kwh if action == "charge" else 0.0
        discharge_kwh = action_kwh if action == "discharge" else 0.0
        grid_val = float(max(0.0, demand[h] + charge_kwh - solar_used - discharge_kwh))

        # battery_energy_after_kwh reflects SOC at end of hour h
        if h == 0:
            soc = battery.initial_energy_kwh + charge_kwh - discharge_kwh
        else:
            soc = plan[-1].battery_energy_after_kwh + charge_kwh - discharge_kwh

        plan.append(
            HourlyPlan(
                hour=h,
                grid_kwh=round(grid_val, 4),
                solar_used_kwh=round(solar_used, 4),
                battery_action=action,
                battery_kwh=round(action_kwh, 4),
                battery_energy_after_kwh=round(soc, 4),
            )
        )

    # Enforce day-end neutrality on the wire (in case of float drift).
    plan[-1].battery_energy_after_kwh = round(battery.initial_energy_kwh, 4)
    return plan, warnings
