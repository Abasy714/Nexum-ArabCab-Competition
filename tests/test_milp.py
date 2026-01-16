from src.services.optimization_runner import run_optimization

periods = ["M1", "M2"]
materials = ["PVC"]
suppliers = ["A"]

forecast_demand = {
    "M1": {"PVC": 100},
    "M2": {"PVC": 120}
}

result = run_optimization(
    periods=periods,
    materials=materials,
    suppliers=suppliers,
    forecast_demand=forecast_demand,
    initial_inventory={"PVC": 50},
    safety_stock_months=1,
    supplier_materials={"A": ["PVC"]},
    purchase_price={"A": {"PVC": 1000}},
    lead_time={"A": 1},
    supplier_capacity={"A": 200},
    holding_cost={"PVC": 5},
    penalty_cost=500,
    supplier_risk={"A": 0.02},
    payment_adjustment={"A": 0},
    budget_buffer=0.1,
    reference_price={"PVC": 1000}
)

print(result)
