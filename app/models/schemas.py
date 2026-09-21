"""Pydantic schemas for request/response payloads.

These match the official GridWise LLM hackathon API contract exactly.
Field names, types, and aliases are not negotiable.
"""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)


# ---------------------------------------------------------------------------
# Input models
# ---------------------------------------------------------------------------
class HourData(BaseModel):
    """One hour of campus energy data."""

    hour: int = Field(..., ge=0, le=23)
    demand_kwh: float = Field(..., ge=0)
    solar_kwh: float = Field(..., ge=0)
    tariff_bdt_per_kwh: float = Field(..., ge=0)


class BatteryData(BaseModel):
    """Campus battery storage parameters."""

    capacity_kwh: float = Field(..., gt=0)
    initial_energy_kwh: float = Field(..., ge=0)
    minimum_energy_kwh: float = Field(..., ge=0)
    max_charge_kwh_per_hour: float = Field(..., ge=0)
    max_discharge_kwh_per_hour: float = Field(..., ge=0)

    @model_validator(mode="after")
    def _check_consistency(self) -> "BatteryData":
        if self.minimum_energy_kwh > self.capacity_kwh:
            raise ValueError("minimum_energy_kwh cannot exceed capacity_kwh")
        if not (
            self.minimum_energy_kwh
            <= self.initial_energy_kwh
            <= self.capacity_kwh
        ):
            raise ValueError(
                "initial_energy_kwh must be between minimum_energy_kwh and "
                "capacity_kwh"
            )
        return self


class OptimizeRequest(BaseModel):
    """Top-level request body for /optimize-energy."""

    scenario_id: str
    operator_notes: List[str] = Field(..., min_length=1, max_length=3)
    hours: List[HourData]
    battery: BatteryData

    @field_validator("hours")
    @classmethod
    def _validate_hours(cls, v: List[HourData]) -> List[HourData]:
        if len(v) != 24:
            raise ValueError("hours must contain exactly 24 entries")
        hour_set = {h.hour for h in v}
        if hour_set != set(range(24)):
            raise ValueError("hours must contain unique hours 0 through 23")
        return sorted(v, key=lambda x: x.hour)


# ---------------------------------------------------------------------------
# Directive models
# ---------------------------------------------------------------------------
class StructuredAdjustment(BaseModel):
    """Structured fields attached to a directive (hours + numeric fields)."""

    hours: List[int] = Field(default_factory=list)
    factor: Optional[float] = None
    minimum_energy_kwh: Optional[float] = None
    max_grid_kwh: Optional[float] = None

    @model_serializer(mode="wrap")
    def _serialize(self, handler):  # type: ignore[no-untyped-def]
        """Drop None fields so the wire format stays compact."""
        data = handler(self)
        return {k: v for k, v in data.items() if v is not None}


class DirectiveInterpretation(BaseModel):
    """Per-note interpretation result."""

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
    """One row of the produced 24-hour schedule."""

    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: Literal["charge", "discharge", "idle"]
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    """Top-level response body for /optimize-energy."""

    scenario_id: str
    directive_interpretation: List[DirectiveInterpretation]
    hourly_plan: List[HourlyPlan]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str


# ---------------------------------------------------------------------------
# Analytics models (extended UI response)
# ---------------------------------------------------------------------------
class HourlyInsight(BaseModel):
    """Per-hour diagnostic row used by the analytics payload."""

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
    """Extended metrics computed from the optimization output."""

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
    """Combined optimization + analytics payload."""

    optimization: OptimizeResponse
    analytics: Analytics


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SUPPORTED_DIRECTIVES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}
