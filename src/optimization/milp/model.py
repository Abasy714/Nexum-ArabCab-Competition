import pulp


def build_inventory_model(
    periods,
    materials,
    suppliers,
    supplier_materials,
    demand,
    initial_inventory,
    safety_stock,
    purchase_price,
    holding_cost,
    penalty_cost,
    lead_time,
    capacity,
    supplier_risk,
    payment_adjustment,
    monthly_budget,
):
    """
    Industrial-grade multi-material inventory & procurement optimization model
    """

    model = pulp.LpProblem("Inventory_Planning", pulp.LpMinimize)

    # -----------------------
    # Decision variables
    # -----------------------

    inventory = pulp.LpVariable.dicts(
        "Inventory",
        [(t, m) for t in periods for m in materials],
        lowBound=0
    )

    order = pulp.LpVariable.dicts(
        "Order",
        [
            (t, s, m)
            for t in periods
            for s in suppliers
            for m in supplier_materials[s]
        ],
        lowBound=0
    )

    shortage = pulp.LpVariable.dicts(
        "SafetyShortage",
        [(t, m) for t in periods for m in materials],
        lowBound=0
    )

    # -----------------------
    # Inventory balance (with lead time)
    # -----------------------

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
                    == initial_inventory[m]
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

    # -----------------------
    # Supplier capacity (shared across materials)
    # -----------------------

    for t in periods:
        for s in suppliers:
            model += (
                pulp.lpSum(
                    order[(t, s, m)]
                    for m in supplier_materials[s]
                )
                <= capacity[s]
            )

    # -----------------------
    # Monthly budget constraint (demand-indexed)
    # -----------------------

    for t in periods:
        model += (
            pulp.lpSum(
                (purchase_price[s][m]
                 + supplier_risk[s]
                 + payment_adjustment[s])
                * order[(t, s, m)]
                for s in suppliers
                for m in supplier_materials[s]
            )
            <= monthly_budget[t]
        )

    # -----------------------
    # Soft safety stock
    # -----------------------

    for t in periods:
        for m in materials:
            model += inventory[(t, m)] + shortage[(t, m)] >= safety_stock[m]

    # -----------------------
    # Objective function
    # -----------------------

    model += (
        pulp.lpSum(
            (purchase_price[s][m]
             + supplier_risk[s]
             + payment_adjustment[s])
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

    return model, inventory, order, shortage
