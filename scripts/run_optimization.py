import pulp
from src.optimization.milp.model import build_inventory_model

# -----------------------
# TIME PERIODS
# -----------------------

periods = ["Nov", "Dec", "Jan"]

# -----------------------
# MATERIALS
# -----------------------

materials = ["PVC", "XLPE", "PE", "LSF"]

# -----------------------
# DEMAND (FORECAST INPUT)
# -----------------------

demand = {
    "Nov": {"PVC": 120, "XLPE": 80, "PE": 60, "LSF": 40},
    "Dec": {"PVC": 130, "XLPE": 90, "PE": 70, "LSF": 50},
    "Jan": {"PVC": 140, "XLPE": 100, "PE": 80, "LSF": 60},
}

# -----------------------
# INITIAL INVENTORY
# -----------------------

initial_inventory = {
    "PVC": 300,
    "XLPE": 200,
    "PE": 180,
    "LSF": 150
}

# -----------------------
# SAFETY STOCK (2 months)
# -----------------------

safety_stock = {
    m: 2 * sum(demand[t][m] for t in periods) / len(periods)
    for m in materials
}

# -----------------------
# SUPPLIERS
# -----------------------

suppliers = ["Local", "Imported"]

supplier_materials = {
    "Local": ["PVC", "PE", "LSF"],
    "Imported": ["PVC", "XLPE", "PE", "LSF"]
}

# -----------------------
# PRICES
# -----------------------

purchase_price = {
    "Local": {
        "PVC": 850, "PE": 900, "LSF": 950
    },
    "Imported": {
        "PVC": 900, "XLPE": 1200, "PE": 950, "LSF": 1000
    }
}

# -----------------------
# LEAD TIMES
# -----------------------

lead_time = {
    "Local": 0,
    "Imported": 2
}

# -----------------------
# CAPACITY
# -----------------------

capacity = {
    "Local": 400,
    "Imported": 500
}

# -----------------------
# SUPPLIER RISK SCORING (EXPLICIT)
# -----------------------

w = {
    "lead_time": 0.35,
    "reliability": 0.30,
    "quality": 0.20,
    "financial": 0.15
}

risk_inputs = {
    "Local": {
        "lead_time": 1,
        "reliability": 2,
        "quality": 3,
        "financial": 8
    },
    "Imported": {
        "lead_time": 4,
        "reliability": 6,
        "quality": 4,
        "financial": 2
    }
}

base_risk = {
    s: sum(w[k] * risk_inputs[s][k] for k in w)
    for s in suppliers
}

RISK_SCALING_FACTOR = 5

supplier_risk = {
    s: base_risk[s] * RISK_SCALING_FACTOR
    for s in suppliers
}

# -----------------------
# PAYMENT TERMS
# -----------------------

payment_adjustment = {
    "Local": 0,
    "Imported": -40
}

# -----------------------
# HOLDING COST (SHELF LIFE)
# -----------------------

holding_cost = {
    "PVC": 20,
    "XLPE": 60,
    "PE": 25,
    "LSF": 30
}

penalty_cost = 5000

# -----------------------
# DEMAND-INDEXED MONTHLY BUDGET
# -----------------------

reference_price = {
    "PVC": 900,
    "XLPE": 1200,
    "PE": 950,
    "LSF": 1000
}

ALPHA = 1.2  # 20% buffer

monthly_budget = {
    t: ALPHA * sum(demand[t][m] * reference_price[m] for m in materials)
    for t in periods
}

# -----------------------
# BUILD & SOLVE
# -----------------------

model, inventory, order, shortage = build_inventory_model(
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
)

model.solve(pulp.PULP_CBC_CMD(msg=False))

# -----------------------
# OUTPUT
# -----------------------

print("Status:", pulp.LpStatus[model.status])

for t in periods:
    print(f"\n=== {t} ===")
    for m in materials:
        print(
            f"{m}: Inventory={inventory[(t, m)].value():.1f}, "
            f"Shortage={shortage[(t, m)].value():.1f}"
        )
    for s in suppliers:
        for m in supplier_materials[s]:
            q = order[(t, s, m)].value()
            if q and q > 0:
                print(f"Order {q:.1f} tons of {m} from {s}")
