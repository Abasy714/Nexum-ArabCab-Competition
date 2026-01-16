from typing import Dict, List, Tuple
import pulp


def optimize_inventory_and_procurement(
    periods: List[str],
    materials: List[str],
    suppliers: List[str],

    # -----------------------------
    # Forecasting output (input)
    # -----------------------------
    demand: Dict[str, Dict[str, float]],  
    # demand[period][material]

    # -----------------------------
    # User inputs
    # -----------------------------
    initial_inventory: Dict[str, float],
    safety_stock: Dict[str, float],

    supplier_materials: Dict[str, List[str]],
    purchase_price: Dict[str, Dict[str, float]],
    lead_time: Dict[str, int],
    supplier_capacity: Dict[str, float],

    # -----------------------------
    # Cost & policy parameters
    # -----------------------------
    holding_cost: Dict[str, float],
    penalty_cost: float,
    supplier_risk: Dict[str, float],
    payment_adjustment: Dict[str, float],
    monthly_budget: Dict[str, float],

):
    """
    MILP-based inventory & procurement optimization.

    This function is intentionally UI-agnostic and forecasting-agnostic.
    It can be called from:
    - Streamlit
    - React (via API)
    - Batch pipeline
    """

    # =========================================================
    # MODEL
    # =========================================================

    model = pulp.LpProblem("Inventory_Procurement_Optimization", pulp.LpMinimize)

    # =========================================================
    # DECISION VARIABLES
    # =========================================================

    # Order quantity
    order = pulp.LpVariable.dicts(
        "Order",
        [(t, s, m) for t in periods for s in suppliers for m in supplier_materials[s]],
        lowBound=0,
        cat="Continuous"
    )

    # Inventory level
    inventory = pulp.LpVariable.dicts(
        "Inventory",
        [(t, m) for t in periods for m in materials],
        lowBound=0
    )

    # Safety stock shortfall (soft constraint)
    shortage = pulp.LpVariable.dicts(
        "SafetyShortage",
        [(t, m) for t in periods for m in materials],
        lowBound=0
    )

    # =========================================================
    # INVENTORY BALANCE CONSTRAINTS
    # =========================================================

    for i, t in enumerate(periods):
        for m in materials:

            arrivals = []
            for s in suppliers:
                if m in supplier_materials[s]:
                    lt = lead_time[s]
                    if i - lt >= 0:
                        arrivals.append(order[(periods[i - lt], s, m)])

            if i == 0:
                model += (
                    inventory[(t, m)]
                    == initial_inventory.get(m, 0)
                    + pulp.lpSum(arrivals)
                    - demand[t][m]
                )
            else:
                prev_t = periods[i - 1]
                model += (
                    inventory[(t, m)]
                    == inventory[(prev_t, m)]
                    + pulp.lpSum(arrivals)
                    - demand[t][m]
                )

    # =========================================================
    # SUPPLIER CAPACITY CONSTRAINTS
    # =========================================================

    for t in periods:
        for s in suppliers:
            model += (
                pulp.lpSum(
                    order[(t, s, m)]
                    for m in supplier_materials[s]
                )
                <= supplier_capacity[s]
            )

    # =========================================================
    # SAFETY STOCK (SOFT)
    # =========================================================

    for t in periods:
        for m in materials:
            model += inventory[(t, m)] + shortage[(t, m)] >= safety_stock[m]

    # =========================================================
    # BUDGET CONSTRAINT
    # =========================================================

    for t in periods:
        model += (
            pulp.lpSum(
                (purchase_price[s][m] + supplier_risk[s] + payment_adjustment[s])
                * order[(t, s, m)]
                for s in suppliers
                for m in supplier_materials[s]
            )
            <= monthly_budget[t]
        )

    # =========================================================
    # OBJECTIVE FUNCTION
    # =========================================================

    model += (
        pulp.lpSum(
            (purchase_price[s][m] + supplier_risk[s] + payment_adjustment[s])
            * order[(t, s, m)]
            for t in periods
            for s in suppliers
            for m in supplier_materials[s]
        )
        + pulp.lpSum(
            holding_cost[m] * inventory[(t, m)]
            + penalty_cost * shortage[(t, m)]
            for t in periods
            for m in materials
        )
    )

    # =========================================================
    # SOLVE
    # =========================================================

    model.solve(pulp.PULP_CBC_CMD(msg=False))

    # =========================================================
    # OUTPUT (JSON-FRIENDLY)
    # =========================================================

    results = {
        "status": pulp.LpStatus[model.status],
        "orders": [],
        "inventory": [],
        "shortages": [],
        "kpis": {}
    }

    # Orders
    for t, s, m in order:
        q = order[(t, s, m)].value()
        if q and q > 0:
            results["orders"].append({
                "period": t,
                "supplier": s,
                "material": m,
                "quantity": round(q, 2),
                "unit_cost": purchase_price[s][m]
            })

    # Inventory & shortages
    for t in periods:
        for m in materials:
            results["inventory"].append({
                "period": t,
                "material": m,
                "inventory": round(inventory[(t, m)].value(), 2)
            })
            results["shortages"].append({
                "period": t,
                "material": m,
                "shortage": round(shortage[(t, m)].value(), 2)
            })

    # KPIs
    total_cost = sum(
        o["quantity"] * o["unit_cost"]
        for o in results["orders"]
    )

    total_quantity = sum(o["quantity"] for o in results["orders"])

    risky_periods = len({
        s["period"] for s in results["shortages"] if s["shortage"] > 0
    })

    results["kpis"] = {
        "total_cost": round(total_cost, 2),
        "total_quantity": round(total_quantity, 2),
        "risky_periods": risky_periods,
        "average_cost_per_ton": round(
            total_cost / total_quantity, 2
        ) if total_quantity > 0 else 0
    }

    return results
