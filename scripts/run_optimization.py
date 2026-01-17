from typing import Dict, List
from src.optimization.milp.model import optimize_inventory_and_procurement


def run_optimization(
    periods: List[str],
    materials: List[str],
    suppliers: List[str],

    forecast_demand: Dict[str, Dict[str, float]],

    initial_inventory: Dict[str, float],
    safety_stock_months: int,

    supplier_materials: Dict[str, List[str]],
    purchase_price: Dict[str, Dict[str, float]],
    lead_time: Dict[str, Dict[str, int]],
    supplier_capacity: Dict[str, float],

    holding_cost: Dict[str, float],
    penalty_cost: float,
    supplier_risk: Dict[str, float],
    payment_adjustment: Dict[str, float],
    budget_buffer: float,
    reference_price: Dict[str, float],
):
    """
    High-level orchestration layer.
    """

    if not periods:
        raise ValueError("Planning periods cannot be empty")

    # -------------------------------------------------
    # VALIDATE FORECAST
    # -------------------------------------------------
    for t in periods:
        for m in materials:
            if m not in forecast_demand.get(t, {}):
                raise ValueError(f"Missing forecast for material '{m}' in period '{t}'")

    # -------------------------------------------------
    # SAFETY STOCK
    # -------------------------------------------------
    safety_stock = {}
    for m in materials:
        avg_demand = sum(forecast_demand[t][m] for t in periods) / len(periods)
        safety_stock[m] = safety_stock_months * avg_demand

    # -------------------------------------------------
    # MONTHLY BUDGET
    # -------------------------------------------------
    monthly_budget = {}
    for t in periods:
        monthly_budget[t] = (
            1 + budget_buffer
        ) * sum(
            forecast_demand[t][m] * reference_price[m]
            for m in materials
        )

    # -------------------------------------------------
    # OPTIMIZATION
    # -------------------------------------------------
    results = optimize_inventory_and_procurement(
        periods=periods,
        materials=materials,
        suppliers=suppliers,
        demand=forecast_demand,
        initial_inventory=initial_inventory,
        safety_stock=safety_stock,
        supplier_materials=supplier_materials,
        purchase_price=purchase_price,
        lead_time=lead_time,
        supplier_capacity=supplier_capacity,
        holding_cost=holding_cost,
        penalty_cost=penalty_cost,
        supplier_risk=supplier_risk,
        payment_adjustment=payment_adjustment,
        monthly_budget=monthly_budget
    )

    if results["status"] != "Optimal":
        raise RuntimeError(f"Optimization failed: {results['status']}")

    results["meta"] = {
        "planning_horizon": len(periods),
        "safety_stock_months": safety_stock_months,
        "budget_buffer": budget_buffer
    }

    return results
