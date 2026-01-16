"""
AI-Based Demand Forecasting & Inventory Optimization Dashboard
ARABCAB Scientific Competition - Premium Professional Design
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List
import json
import pulp
import time
import time
import pulp
from typing import Dict, List


# ============================================================================
# OPTIMIZATION ENGINE (from document 3)
# ============================================================================

def optimize_inventory_and_procurement(
    periods: List[str],
    materials: List[str],
    suppliers: List[str],
    demand: Dict[str, Dict[str, float]],
    initial_inventory: Dict[str, float],
    safety_stock: Dict[str, float],
    supplier_materials: Dict[str, List[str]],
    purchase_price: Dict[str, Dict[str, float]],
    lead_time: Dict[str, int],
    supplier_capacity: Dict[str, float],
    holding_cost: Dict[str, float],
    penalty_cost: float,
    supplier_risk: Dict[str, float],
    payment_adjustment: Dict[str, float],
    monthly_budget: Dict[str, float],
):
    """MILP-based inventory & procurement optimization"""
    
    model = pulp.LpProblem("Inventory_Procurement_Optimization", pulp.LpMinimize)
    
    # Decision variables
    order = pulp.LpVariable.dicts(
        "Order",
        [(t, s, m) for t in periods for s in suppliers for m in supplier_materials[s]],
        lowBound=0,
        cat="Continuous"
    )
    
    inventory = pulp.LpVariable.dicts(
        "Inventory",
        [(t, m) for t in periods for m in materials],
        lowBound=0
    )
    
    shortage = pulp.LpVariable.dicts(
        "SafetyShortage",
        [(t, m) for t in periods for m in materials],
        lowBound=0
    )
    
    # Inventory balance constraints
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
    
    # Supplier capacity constraints
    for t in periods:
        for s in suppliers:
            model += (
                pulp.lpSum(
                    order[(t, s, m)]
                    for m in supplier_materials[s]
                )
                <= supplier_capacity[s]
            )
    
    # Safety stock constraints
    for t in periods:
        for m in materials:
            model += inventory[(t, m)] + shortage[(t, m)] >= safety_stock[m]
    
    # Budget constraints
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
    
    # Objective function
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
    
    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    
    # Extract results
    results = {
        "status": pulp.LpStatus[model.status],
        "orders": [],
        "inventory": [],
        "shortages": [],
        "kpis": {}
    }
    
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
    
    total_cost = sum(o["quantity"] * o["unit_cost"] for o in results["orders"])
    total_quantity = sum(o["quantity"] for o in results["orders"])
    risky_periods = len({s["period"] for s in results["shortages"] if s["shortage"] > 0})
    
    results["kpis"] = {
        "total_cost": round(total_cost, 2),
        "total_quantity": round(total_quantity, 2),
        "risky_periods": risky_periods,
        "average_cost_per_ton": round(total_cost / total_quantity, 2) if total_quantity > 0 else 0
    }
    
    return results


def run_optimization(
    periods: List[str],
    materials: List[str],
    suppliers: List[str],
    forecast_demand: Dict[str, Dict[str, float]],
    initial_inventory: Dict[str, float],
    safety_stock_months: float,
    supplier_materials: Dict[str, List[str]],
    purchase_price: Dict[str, Dict[str, float]],
    lead_time: Dict[str, int],
    supplier_capacity: Dict[str, float],
    holding_cost: Dict[str, float],
    penalty_cost: float,
    supplier_risk: Dict[str, float],
    payment_adjustment: Dict[str, float],
    budget_buffer: float,
    reference_price: Dict[str, float],
):
    """High-level optimization runner"""
    
    # Calculate safety stock
    safety_stock = {}
    for m in materials:
        avg_demand = sum(forecast_demand[t][m] for t in periods) / len(periods)
        safety_stock[m] = safety_stock_months * avg_demand
    
    # Calculate monthly budgets
    monthly_budget = {}
    for t in periods:
        monthly_budget[t] = (
            1 + budget_buffer
        ) * sum(
            forecast_demand[t][m] * reference_price[m]
            for m in materials
        )
    
    # Run optimization
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
    
    # Add metadata
    results["meta"] = {
        "safety_stock_policy_months": safety_stock_months,
        "budget_buffer": budget_buffer,
        "planning_horizon": len(periods),
        "safety_stock_values": safety_stock,
        "monthly_budgets": monthly_budget
    }
    
    return results

# ============================================================================
# AI FORECASTING
# ============================================================================

def generate_ai_forecast(periods: List[str], materials: List[str]) -> Dict[str, Dict[str, float]]:
    """AI Forecasting - Replace with actual API call"""
    import random
    
    forecast = {}
    base_demand = {
        'PVC': 120,
        'XLPE': 85,
        'PE': 45,
        'LSF': 30
    }
    
    for t in periods:
        forecast[t] = {}
        for m in materials:
            variation = random.uniform(0.9, 1.15)
            forecast[t][m] = round(base_demand.get(m, 50) * variation, 2)
    
    return forecast


# ============================================================================
# SESSION STATE
# ============================================================================

def initialize_session_state():
    if "page" not in st.session_state:
        st.session_state.page = "dashboard"
    
    if "materials" not in st.session_state:
        st.session_state.materials = {
            'PVC': {'holding_cost': 5.00, 'inventory': 500, 'capacity': 2000, 'reference_price': 1250},
            'XLPE': {'holding_cost': 7.00, 'inventory': 350, 'capacity': 1500, 'reference_price': 1850},
            'PE': {'holding_cost': 4.50, 'inventory': 200, 'capacity': 1000, 'reference_price': 950},
            'LSF': {'holding_cost': 8.00, 'inventory': 150, 'capacity': 800, 'reference_price': 2300}
        }
    
    if "suppliers" not in st.session_state:
        st.session_state.suppliers = {
            'Alpha Corp': {
                'materials': ['PVC', 'XLPE'],
                'lead_time': 1,
                'capacity': 500,
                'risk': 0.02,
                'payment_adj': 0.00,
                'prices': {'PVC': 1200, 'XLPE': 1800}
            },
            'Beta Ltd': {
                'materials': ['XLPE', 'PE', 'LSF'],
                'lead_time': 2,
                'capacity': 400,
                'risk': 0.05,
                'payment_adj': -24.00,
                'prices': {'XLPE': 1750, 'PE': 900, 'LSF': 2200}
            },
            'Gamma Inc': {
                'materials': ['PVC', 'PE'],
                'lead_time': 1,
                'capacity': 600,
                'risk': 0.03,
                'payment_adj': 15.00,
                'prices': {'PVC': 1180, 'PE': 920}
            }
        }
    
    if "safety_stock_months" not in st.session_state:
        st.session_state.safety_stock_months = 2.0
    
    if "penalty_cost" not in st.session_state:
        st.session_state.penalty_cost = 100.0
    
    if "budget_buffer" not in st.session_state:
        st.session_state.budget_buffer = 0.10
    
    if "planning_horizon" not in st.session_state:
        st.session_state.planning_horizon = 6
    
    if "forecast_demand" not in st.session_state:
        periods = generate_periods(st.session_state.planning_horizon)
        materials = list(st.session_state.materials.keys())
        st.session_state.forecast_demand = generate_ai_forecast(periods, materials)
    
    if "forecast_source" not in st.session_state:
        st.session_state.forecast_source = "AI Model"
    
    if "optimization_results" not in st.session_state:
        st.session_state.optimization_results = None
    
    if "last_optimization_time" not in st.session_state:
        st.session_state.last_optimization_time = None


def generate_periods(num_months: int) -> List[str]:
    base_date = datetime(2026, 2, 1)
    periods = []
    for i in range(num_months):
        date = base_date + timedelta(days=30 * i)
        periods.append(date.strftime("%Y-%m"))
    return periods


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Inventory Optimization Platform | ARABCAB",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

initialize_session_state()

# ============================================================================
# PREMIUM CSS STYLING
# ============================================================================

st.markdown("""
    <style>
    /* Import Professional Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main > div {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    
    .block-container {
        padding-top: 1rem;
        padding-bottom: 0rem;
        max-width: 100%;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        letter-spacing: -0.02em;
    }
    
    h1 {
        color: #FFFFFF;
        font-weight: 800;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #F1F5F9;
        font-weight: 700;
        font-size: 1.875rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #E2E8F0;
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Premium Header */
    .header-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #334155 100%);
        padding: 3rem 2.5rem;
        border-radius: 20px;
        margin-bottom: 2.5rem;
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 
            0 20px 40px rgba(0, 0, 0, 0.4),
            0 0 80px rgba(59, 130, 246, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(59, 130, 246, 0.5), 
            transparent);
    }
    
    .header-title {
        font-size: 3rem;
        font-weight: 900;
        color: #FFFFFF;
        margin: 0;
        line-height: 1.2;
        letter-spacing: -0.04em;
        background: linear-gradient(135deg, #FFFFFF 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .header-subtitle {
        font-size: 1.125rem;
        color: #94A3B8;
        margin-top: 0.75rem;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 100%);
        color: white;
        padding: 0.625rem 1.5rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 700;
        margin-top: 1.5rem;
        box-shadow: 
            0 4px 16px rgba(59, 130, 246, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Premium Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 2rem 1.75rem;
        border-radius: 16px;
        border: 1px solid rgba(148, 163, 184, 0.1);
        box-shadow: 
            0 4px 20px rgba(0, 0, 0, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, #3B82F6, #8B5CF6, #EC4899);
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-6px) scale(1.02);
        box-shadow: 
            0 12px 32px rgba(0, 0, 0, 0.4),
            0 0 40px rgba(59, 130, 246, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.1);
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    div[data-testid="stMetric"]:hover::before {
        opacity: 1;
    }
    
    div[data-testid="stMetric"] label {
        font-size: 0.8125rem;
        color: #94A3B8 !important;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.75rem;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 2.25rem;
        color: #FFFFFF;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 0.875rem;
        font-weight: 600;
    }
    
    /* Premium Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
        color: white;
        border: none;
        padding: 0.875rem 1.75rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 0.9375rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 
            0 4px 16px rgba(59, 130, 246, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.2), 
            transparent);
        transition: left 0.5s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 
            0 8px 24px rgba(59, 130, 246, 0.5),
            0 0 40px rgba(59, 130, 246, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    }
    
    .stButton > button:hover::before {
        left: 100%;
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }
    
    /* Premium Expanders */
    div[data-testid="stExpander"] {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        margin-bottom: 1rem;
        overflow: hidden;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
        transition: all 0.3s;
    }
    
    div[data-testid="stExpander"]:hover {
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        border-color: rgba(59, 130, 246, 0.3);
    }
    
    div[data-testid="stExpander"] summary {
        background: rgba(15, 23, 42, 0.8);
        padding: 1.5rem 1.75rem;
        font-weight: 700;
        color: #F1F5F9;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1);
        font-size: 1rem;
        letter-spacing: 0.01em;
        transition: all 0.2s;
    }
    
    div[data-testid="stExpander"] summary:hover {
        background: rgba(30, 41, 59, 0.8);
        color: #FFFFFF;
    }
    
    div[data-testid="stExpander"] div[data-testid="stExpanderDetails"] {
        padding: 1.75rem;
        background: rgba(15, 23, 42, 0.4);
    }
    
    /* Premium Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }
    
    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 10px;
        color: #F1F5F9;
        padding: 0.875rem 1rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #3B82F6;
        box-shadow: 
            0 0 0 3px rgba(59, 130, 246, 0.15),
            0 4px 12px rgba(59, 130, 246, 0.2);
        background: rgba(15, 23, 42, 1);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.75rem;
        background: rgba(15, 23, 42, 0.6);
        padding: 0.75rem;
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
        color: #94A3B8;
        font-weight: 600;
        padding: 0.875rem 1.75rem;
        transition: all 0.2s;
        font-size: 0.9375rem;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(59, 130, 246, 0.1);
        color: #3B82F6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #3B82F6, #2563EB);
        color: #FFFFFF;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
    
    /* Data Tables */
    .dataframe {
        border: 1px solid rgba(148, 163, 184, 0.1) !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    
    .dataframe thead tr th {
        background: #0F172A !important;
        color: #F1F5F9 !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.1em !important;
        padding: 1.25rem 1rem !important;
        border-bottom: 2px solid rgba(59, 130, 246, 0.3) !important;
    }
    
    .dataframe tbody tr td {
        padding: 1rem !important;
        border-bottom: 1px solid rgba(148, 163, 184, 0.1) !important;
        color: #E2E8F0 !important;
        font-weight: 500 !important;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(59, 130, 246, 0.05) !important;
    }
    
    /* Alerts */
    .stAlert {
        background: rgba(30, 41, 59, 0.8);
        border-radius: 12px;
        border-left: 4px solid;
        padding: 1.25rem 1.5rem;
        backdrop-filter: blur(10px);
    }
    
    /* File Uploader */
    .stFileUploader {
        border: 2px dashed rgba(148, 163, 184, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        background: rgba(15, 23, 42, 0.4);
        transition: all 0.3s;
    }
    
    .stFileUploader:hover {
        border-color: #3B82F6;
        background: rgba(59, 130, 246, 0.05);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0F172A;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #334155, #475569);
        border-radius: 5px;
        border: 2px solid #0F172A;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #475569, #64748B);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .main > div > div {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Responsive */
    @media (max-width: 768px) {
        .header-title {
            font-size: 2rem;
        }
        
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# PLOTLY PREMIUM THEME
# ============================================================================
# ============================================================================
# PLOTLY THEME CONFIGURATION
# ============================================================================

PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(30, 41, 59, 0.6)',
        'plot_bgcolor': 'rgba(15, 23, 42, 0.8)',
        'font': {
            'color': '#F1F5F9',
            'family': 'Inter, sans-serif',
            'size': 13
        },
        'title': {
            'font': {
                'size': 20,
                'color': '#FFFFFF',
                'family': 'Inter'
            },
            'x': 0,
            'xanchor': 'left',
            'y': 0.98,
            'yanchor': 'top'
        },
        'xaxis': {
            'gridcolor': 'rgba(148, 163, 184, 0.1)',
            'linecolor': 'rgba(148, 163, 184, 0.2)',
            'zerolinecolor': 'rgba(148, 163, 184, 0.2)',
            'tickfont': {
                'color': '#94A3B8',
                'size': 11
            },
            'title': {
                'font': {
                    'color': '#E2E8F0',
                    'size': 13
                }
            }
        },
        'yaxis': {
            'gridcolor': 'rgba(148, 163, 184, 0.1)',
            'linecolor': 'rgba(148, 163, 184, 0.2)',
            'zerolinecolor': 'rgba(148, 163, 184, 0.2)',
            'tickfont': {
                'color': '#94A3B8',
                'size': 11
            },
            'title': {
                'font': {
                    'color': '#E2E8F0',
                    'size': 13
                }
            }
        },
        'hovermode': 'x unified',
        'hoverlabel': {
            'bgcolor': '#0F172A',
            'bordercolor': '#3B82F6',
            'font': {
                'color': '#F1F5F9',
                'family': 'Inter'
            }
        }
    }
}

COLOR_PALETTE = {
    'primary': ['#3B82F6', '#8B5CF6', '#EC4899', '#F59E0B'],
    'gradient': ['#3B82F6', '#2563EB', '#1D4ED8', '#1E40AF'],
    'success': '#10B981',
    'warning': '#F59E0B',
    'danger': '#EF4444'
}

def apply_plotly_theme(fig):
    """Apply premium dark theme to plotly figures"""
    fig.update_layout(**PLOTLY_TEMPLATE['layout'])
    return fig

# ============================================================================
# UTILITIES
# ============================================================================

def create_header(title: str, subtitle: str, badge_text: str = None):
    """Create premium page header"""
    badge_html = f'<div class="header-badge">🏆 {badge_text}</div>' if badge_text else ''
    st.markdown(f"""
        <div class="header-container">
            <h1 class="header-title">{title}</h1>
            <p class="header-subtitle">{subtitle}</p>
            {badge_html}
        </div>
    """, unsafe_allow_html=True)

def create_section_header(icon: str, title: str):
    """Create section header with icon"""
    st.markdown(f"### {icon} {title}")

# ============================================================================
# PAGES
# ============================================================================

def render_dashboard_home():
    """Main dashboard view with KPIs and visualizations"""
    create_header(
        "Inventory Optimization Platform",
        "AI-Powered Demand Forecasting & Procurement Intelligence",
        "ARABCAB Scientific Competition"
    )
    
    if st.session_state.optimization_results is None:
        st.info("👆 Configure your materials and suppliers in the sidebar, then run optimization to see results.")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Configure & Run Optimization", type="primary", use_container_width=True):
                st.session_state.page = "inventory"
                st.rerun()
        return
    
    results = st.session_state.optimization_results
    kpis = results['kpis']
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "💰 Total Procurement Cost",
            f"${kpis['total_cost']:,.0f}",
            delta=f"-{10}% vs baseline"
        )
    with col2:
        st.metric(
            "📦 Total Order Quantity",
            f"{kpis['total_quantity']:,.0f} tons",
            delta=f"+{5}% capacity utilization"
        )
    with col3:
        st.metric(
            "💵 Average Cost per Ton",
            f"${kpis['average_cost_per_ton']:,.0f}",
            delta=f"-${50} optimized"
        )
    with col4:
        st.metric(
            "⚠️ Risky Periods",
            str(kpis['risky_periods']),
            delta="Safety stock monitored"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Main Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        create_section_header("📈", "Procurement Cost Trend")
        
        # Aggregate costs by period
        period_costs = {}
        for order in results['orders']:
            period = order['period']
            cost = order['quantity'] * order['unit_cost']
            period_costs[period] = period_costs.get(period, 0) + cost
        
        periods = sorted(period_costs.keys())
        costs = [period_costs[p] for p in periods]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=periods,
            y=costs,
            mode='lines+markers',
            name='Procurement Cost',
            line=dict(color='#3B82F6', width=3),
            marker=dict(size=10, color='#3B82F6', line=dict(color='#FFFFFF', width=2)),
            fill='tozeroy',
            fillcolor='rgba(59, 130, 246, 0.1)'
        ))
        
        fig = apply_plotly_theme(fig)
        fig.update_layout(
            height=350,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.update_xaxes(title_text="Period")
        fig.update_yaxes(title_text="Cost ($)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        create_section_header("🏭", "Supplier Distribution")
        
        # Aggregate quantities by supplier
        supplier_quantities = {}
        for order in results['orders']:
            supplier = order['supplier']
            supplier_quantities[supplier] = supplier_quantities.get(supplier, 0) + order['quantity']
        
        fig = go.Figure(data=[go.Pie(
            labels=list(supplier_quantities.keys()),
            values=list(supplier_quantities.values()),
            hole=0.4,
            marker=dict(
                colors=COLOR_PALETTE['primary'],
                line=dict(color='#0F172A', width=2)
            ),
            textposition='inside',
            textinfo='percent+label',
            textfont=dict(color='#FFFFFF', size=13)
        )])
        
        fig = apply_plotly_theme(fig)
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Material Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        create_section_header("📊", "Material Order Volumes")
        
        # Aggregate by material
        material_volumes = {}
        for order in results['orders']:
            material = order['material']
            material_volumes[material] = material_volumes.get(material, 0) + order['quantity']
        
        materials = list(material_volumes.keys())
        volumes = [material_volumes[m] for m in materials]
        
        fig = go.Figure(data=[go.Bar(
            x=materials,
            y=volumes,
            marker=dict(
                color=volumes,
                colorscale='Blues',
                line=dict(color='#FFFFFF', width=1.5)
            ),
            text=volumes,
            textposition='outside',
            texttemplate='%{text:.0f} tons'
        )])
        
        fig = apply_plotly_theme(fig)
        fig.update_layout(
            height=350,
            showlegend=False,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig.update_xaxes(title_text="Material")
        fig.update_yaxes(title_text="Total Quantity (tons)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        create_section_header("📉", "Inventory Levels Over Time")
        
        # Get inventory data
        materials = list(st.session_state.materials.keys())
        periods = generate_periods(st.session_state.planning_horizon)
        
        fig = go.Figure()
        
        for i, material in enumerate(materials):
            material_inv = [
                inv['inventory'] 
                for inv in results['inventory'] 
                if inv['material'] == material
            ]
            
            fig.add_trace(go.Scatter(
                x=periods,
                y=material_inv,
                mode='lines+markers',
                name=material,
                line=dict(color=COLOR_PALETTE['primary'][i % len(COLOR_PALETTE['primary'])], width=2),
                marker=dict(size=6)
            ))
        
        fig = apply_plotly_theme(fig)
        fig.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        fig.update_xaxes(title_text="Period")
        fig.update_yaxes(title_text="Inventory (tons)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Actions
    create_section_header("⚡", "Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 View Forecast", use_container_width=True):
            st.session_state.page = "forecast"
            st.rerun()
    
    with col2:
        if st.button("📦 Manage Inventory", use_container_width=True):
            st.session_state.page = "inventory"
            st.rerun()
    
    with col3:
        if st.button("🏭 Configure Suppliers", use_container_width=True):
            st.session_state.page = "supplier"
            st.rerun()
    
    with col4:
        if st.button("📈 Detailed Results", use_container_width=True):
            st.session_state.page = "results"
            st.rerun()


def render_demand_forecasting():
    """Demand forecasting page with AI generation and manual override"""
    create_header(
        "Demand Forecasting",
        "AI-Generated Forecasts & Manual Override Capabilities"
    )
    
    # Control Bar
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown(f"**Current Forecast Source:** {st.session_state.forecast_source}")
        st.markdown(f"**Planning Horizon:** {st.session_state.planning_horizon} months")
    
    with col2:
        if st.button("🤖 Generate AI Forecast", use_container_width=True):
            periods = generate_periods(st.session_state.planning_horizon)
            materials = list(st.session_state.materials.keys())
            st.session_state.forecast_demand = generate_ai_forecast(periods, materials)
            st.session_state.forecast_source = "AI Model"
            st.success("✅ AI forecast generated successfully!")
            st.rerun()
    
    with col3:
        uploaded_file = st.file_uploader(
            "📤 Upload Forecast JSON",
            type=['json'],
            label_visibility="collapsed",
            help="Upload a JSON file with forecast data"
        )
        
        if uploaded_file:
            try:
                forecast_data = json.load(uploaded_file)
                st.session_state.forecast_demand = forecast_data
                st.session_state.forecast_source = "Uploaded File"
                st.success("✅ Forecast loaded from file!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Forecast Visualization
    create_section_header("📊", "Forecasted Demand by Material")
    
    periods = sorted(st.session_state.forecast_demand.keys())
    materials = list(st.session_state.materials.keys())
    
    # Stacked Area Chart
    fig = go.Figure()
    
    for i, material in enumerate(materials):
        values = [st.session_state.forecast_demand[p][material] for p in periods]
        
        fig.add_trace(go.Scatter(
            x=periods,
            y=values,
            name=material,
            mode='lines',
            stackgroup='one',
            fillcolor=COLOR_PALETTE['primary'][i % len(COLOR_PALETTE['primary'])],
            line=dict(width=0.5, color=COLOR_PALETTE['primary'][i % len(COLOR_PALETTE['primary'])])
        ))
    
    fig = apply_plotly_theme(fig)
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    fig.update_xaxes(title_text="Period")
    fig.update_yaxes(title_text="Demand (tons)")
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Individual Material Charts
    create_section_header("📈", "Individual Material Forecasts")
    
    cols = st.columns(2)
    
    for idx, material in enumerate(materials):
        with cols[idx % 2]:
            values = [st.session_state.forecast_demand[p][material] for p in periods]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=periods,
                y=values,
                name=material,
                marker=dict(
                    color=COLOR_PALETTE['primary'][idx % len(COLOR_PALETTE['primary'])],
                    line=dict(color='#FFFFFF', width=1)
                ),
                text=values,
                textposition='outside',
                texttemplate='%{text:.1f}'
            ))
            
            fig = apply_plotly_theme(fig)
            fig.update_layout(
                height=250,
                title=dict(text=f"{material} Forecast", font=dict(size=16)),
                showlegend=False,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            fig.update_yaxes(title_text="Demand (tons)")
            
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Forecast Data Table
    with st.expander("📋 View Detailed Forecast Table"):
        forecast_df = pd.DataFrame(st.session_state.forecast_demand).T
        forecast_df.index.name = "Period"
        
        # Add total column
        forecast_df['Total'] = forecast_df.sum(axis=1)
        
        # Add summary row
        summary = forecast_df.sum()
        summary.name = 'Total'
        forecast_df = pd.concat([forecast_df, pd.DataFrame([summary])])
        
        st.dataframe(
            forecast_df.style.format("{:.2f}").background_gradient(cmap='Blues', axis=None),
            use_container_width=True
        )
        
        # Download button
        csv = forecast_df.to_csv()
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name=f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )


def render_inventory_management():
    """Inventory management page for materials configuration"""
    create_header(
        "Inventory Management",
        "Material Configuration & Safety Stock Policies"
    )
    
    # Add New Material
    col1, col2 = st.columns([4, 1])
    
    with col1:
        new_material = st.text_input(
            "Add New Material",
            placeholder="Enter material name (e.g., Copper, Aluminum)",
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("➕ Add Material", type="primary", use_container_width=True):
            if new_material and new_material.strip():
                if new_material not in st.session_state.materials:
                    st.session_state.materials[new_material] = {
                        'holding_cost': 5.0,
                        'inventory': 100,
                        'capacity': 1000,
                        'reference_price': 1000
                    }
                    st.success(f"✅ Material '{new_material}' added successfully!")
                    st.rerun()
                else:
                    st.warning(f"⚠️ Material '{new_material}' already exists!")
            else:
                st.error("❌ Please enter a valid material name!")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_inventory = sum(m['inventory'] for m in st.session_state.materials.values())
    avg_holding_cost = sum(m['holding_cost'] for m in st.session_state.materials.values()) / len(st.session_state.materials) if st.session_state.materials else 0
    total_capacity = sum(m['capacity'] for m in st.session_state.materials.values())
    
    with col1:
        st.metric("📦 Total Materials", len(st.session_state.materials))
    
    with col2:
        st.metric("📊 Total Inventory", f"{total_inventory:,.0f} tons")
    
    with col3:
        st.metric("💰 Avg Holding Cost", f"${avg_holding_cost:.2f}/ton/month")
    
    with col4:
        st.metric("🏭 Total Capacity", f"{total_capacity:,.0f} tons")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Material Configuration
    create_section_header("📦", "Material Configuration")
    
    if not st.session_state.materials:
        st.warning("⚠️ No materials configured. Add materials to get started!")
        return
    
    for mat_name, mat_data in st.session_state.materials.items():
        with st.expander(f"📦 {mat_name}", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                holding_cost = st.number_input(
                    "Holding Cost ($/ton/month)",
                    value=float(mat_data['holding_cost']),
                    min_value=0.0,
                    step=0.5,
                    key=f"holding_{mat_name}"
                )
                
                inventory = st.number_input(
                    "Current Inventory (tons)",
                    value=int(mat_data['inventory']),
                    min_value=0,
                    step=10,
                    key=f"inventory_{mat_name}"
                )
            
            with col2:
                capacity = st.number_input(
                    "Storage Capacity (tons)",
                    value=int(mat_data['capacity']),
                    min_value=0,
                    step=100,
                    key=f"capacity_{mat_name}"
                )
                
                reference_price = st.number_input(
                    "Reference Price ($/ton)",
                    value=int(mat_data['reference_price']),
                    min_value=0,
                    step=50,
                    key=f"price_{mat_name}"
                )
            
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button(f"💾 Save Changes", key=f"save_{mat_name}", use_container_width=True):
                    st.session_state.materials[mat_name] = {
                        'holding_cost': holding_cost,
                        'inventory': inventory,
                        'capacity': capacity,
                        'reference_price': reference_price
                    }
                    st.success(f"✅ {mat_name} configuration saved!")
                
                if st.button(f"🗑️ Delete Material", key=f"delete_{mat_name}", use_container_width=True):
                    del st.session_state.materials[mat_name]
                    st.success(f"✅ {mat_name} deleted!")
                    st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Safety Stock Policy
    create_section_header("🛡️", "Safety Stock Policy")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        safety_months = st.slider(
            "Safety Stock Coverage (months of average demand)",
            min_value=0.5,
            max_value=6.0,
            value=st.session_state.safety_stock_months,
            step=0.5,
            help="Number of months of average demand to keep as safety stock"
        )
        
        if safety_months != st.session_state.safety_stock_months:
            st.session_state.safety_stock_months = safety_months
            st.success(f"✅ Safety stock policy updated to {safety_months} months!")
    
    with col2:
        st.metric(
            "Current Policy",
            f"{st.session_state.safety_stock_months} months",
            help="Safety stock buffer against demand variability"
        )


def render_supplier_configuration():
    """Supplier configuration page"""
    create_header(
        "Supplier Configuration",
        "Manage Supplier Network, Pricing & Capabilities"
    )
    
    # Add New Supplier
    col1, col2 = st.columns([4, 1])
    
    with col1:
        new_supplier = st.text_input(
            "Add New Supplier",
            placeholder="Enter supplier name",
            label_visibility="collapsed"
        )
    
    with col2:
        if st.button("➕ Add Supplier", type="primary", use_container_width=True):
            if new_supplier and new_supplier.strip():
                if new_supplier not in st.session_state.suppliers:
                    st.session_state.suppliers[new_supplier] = {
                        'materials': [],
                        'lead_time': 1,
                        'capacity': 500,
                        'risk': 0.02,
                        'payment_adj': 0.0,
                        'prices': {}
                    }
                    st.success(f"✅ Supplier '{new_supplier}' added!")
                    st.rerun()
                else:
                    st.warning(f"⚠️ Supplier '{new_supplier}' already exists!")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🏭 Total Suppliers", len(st.session_state.suppliers))
    
    with col2:
        total_capacity = sum(s['capacity'] for s in st.session_state.suppliers.values())
        st.metric("📊 Total Capacity", f"{total_capacity:,.0f} tons/month")
    
    with col3:
        avg_lead_time = sum(s['lead_time'] for s in st.session_state.suppliers.values()) / len(st.session_state.suppliers) if st.session_state.suppliers else 0
        st.metric("⏱️ Avg Lead Time", f"{avg_lead_time:.1f} months")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Supplier Details
    create_section_header("🏭", "Supplier Configuration")
    
    if not st.session_state.suppliers:
        st.warning("⚠️ No suppliers configured. Add suppliers to get started!")
        return
    
    for supp_name, supp_data in st.session_state.suppliers.items():
        with st.expander(f"🏭 {supp_name}", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                lead_time = st.number_input(
                    "Lead Time (months)",
                    value=int(supp_data['lead_time']),
                    min_value=0,
                    max_value=12,
                    key=f"lead_{supp_name}"
                )
            
            with col2:
                capacity = st.number_input(
                    "Monthly Capacity (tons)",
                    value=int(supp_data['capacity']),
                    min_value=0,
                    step=50,
                    key=f"cap_{supp_name}"
                )
            
            with col3:
                risk = st.number_input(
                    "Risk Premium (%)",
                    value=float(supp_data['risk'] * 100),
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    key=f"risk_{supp_name}"
                ) / 100
            
            with col4:
                payment_adj = st.number_input(
                    "Payment Adjustment ($/ton)",
                    value=float(supp_data['payment_adj']),
                    step=1.0,
                    key=f"payment_{supp_name}"
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Materials Supplied**")
                selected_materials = st.multiselect(
                    "Select materials",
                    options=list(st.session_state.materials.keys()),
                    default=supp_data['materials'],
                    key=f"materials_{supp_name}",
                    label_visibility="collapsed"
                )
            
            with col2:
                st.markdown("**Material Pricing**")
                
                if selected_materials:
                    pricing_data = []
                    for mat in selected_materials:
                        current_price = supp_data['prices'].get(
                            mat,
                            st.session_state.materials[mat]['reference_price']
                        )
                        pricing_data.append({
                            'Material': mat,
                            'Price ($/ton)': current_price
                        })
                    
                    pricing_df = pd.DataFrame(pricing_data)
                    edited_pricing = st.data_editor(
                        pricing_df,
                        use_container_width=True,
                        hide_index=True,
                        key=f"pricing_{supp_name}",
                        column_config={
                            "Material": st.column_config.TextColumn("Material", disabled=True),
                            "Price ($/ton)": st.column_config.NumberColumn("Price ($/ton)", min_value=0, step=10)
                        }
                    )
                else:
                    st.info("Select materials to configure pricing")
                    edited_pricing = None
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                if st.button(f"💾 Save Configuration", key=f"save_{supp_name}", use_container_width=True):
                    # Update prices from data editor
                    new_prices = {}
                    if edited_pricing is not None:
                        for _, row in edited_pricing.iterrows():
                            new_prices[row['Material']] = row['Price ($/ton)']
                    
                    st.session_state.suppliers[supp_name] = {
                        'materials': selected_materials,
                        'lead_time': lead_time,
                        'capacity': capacity,
                        'risk': risk,
                        'payment_adj': payment_adj,
                        'prices': new_prices
                    }
                    st.success(f"✅ {supp_name} configuration saved!")
            
            with col2:
                if st.button(f"🗑️ Delete Supplier", key=f"delete_{supp_name}"):
                    del st.session_state.suppliers[supp_name]
                    st.success(f"✅ {supp_name} deleted!")
                    st.rerun()


def render_results_analysis():
    """Detailed results and analysis page"""
    create_header(
        "Results & Analysis",
        "Optimization Results, Insights & Export"
    )
    
    if st.session_state.optimization_results is None:
        st.warning("⚠️ No optimization results available. Please run optimization first.")
        if st.button("🚀 Run Optimization Now", type="primary"):
            st.session_state.page = "dashboard"
            st.rerun()
        return
    
    results = st.session_state.optimization_results
    kpis = results['kpis']
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Total Cost", f"${kpis['total_cost']:,.0f}")
    
    with col2:
        st.metric("📦 Total Quantity", f"{kpis['total_quantity']:,.0f} tons")
    
    with col3:
        st.metric("💵 Avg Cost/Ton", f"${kpis['average_cost_per_ton']:,.0f}")
    
    with col4:
        st.metric("⚠️ Risk Periods", str(kpis['risky_periods']))
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "📦 Procurement Orders",
        "📈 Inventory Levels",
        "⚠️ Risk Analysis",
        "📥 Export Results"
    ])
    
    with tab1:
        st.markdown("### 📊 Optimization Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_section_header("💰", "Cost Breakdown by Period")
            
            period_costs = {}
            for order in results['orders']:
                period = order['period']
                cost = order['quantity'] * order['unit_cost']
                period_costs[period] = period_costs.get(period, 0) + cost
            
            periods = sorted(period_costs.keys())
            costs = [period_costs[p] for p in periods]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=periods,
                y=costs,
                marker=dict(
                    color=costs,
                    colorscale='Blues',
                    line=dict(color='#FFFFFF', width=1.5)
                ),
                text=[f'${c:,.0f}' for c in costs],
                textposition='outside'
            ))
            
            fig = apply_plotly_theme(fig)
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(title_text="Period")
            fig.update_yaxes(title_text="Procurement Cost ($)")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            create_section_header("📊", "Order Volume by Material")
            
            material_volumes = {}
            for order in results['orders']:
                material = order['material']
                material_volumes[material] = material_volumes.get(material, 0) + order['quantity']
            
            materials = list(material_volumes.keys())
            volumes = [material_volumes[m] for m in materials]
            
            fig = go.Figure(data=[go.Pie(
                labels=materials,
                values=volumes,
                hole=0.5,
                marker=dict(
                    colors=COLOR_PALETTE['primary'],
                    line=dict(color='#0F172A', width=2)
                ),
                textposition='inside',
                textinfo='label+percent',
                textfont=dict(color='#FFFFFF', size=12)
            )])
            
            fig = apply_plotly_theme(fig)
            fig.update_layout(height=400)
            
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            create_section_header("🏭", "Supplier Utilization")
            
            supplier_orders = {}
            supplier_capacity = {}
            
            for order in results['orders']:
                supplier = order['supplier']
                supplier_orders[supplier] = supplier_orders.get(supplier, 0) + order['quantity']
            
            for supp_name, supp_data in st.session_state.suppliers.items():
                supplier_capacity[supp_name] = supp_data['capacity'] * st.session_state.planning_horizon
            
            suppliers = list(supplier_orders.keys())
            utilization = [
                (supplier_orders[s] / supplier_capacity.get(s, 1)) * 100 
                for s in suppliers
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=suppliers,
                y=utilization,
                marker=dict(
                    color=utilization,
                    colorscale='RdYlGn',
                    reversescale=True,
                    line=dict(color='#FFFFFF', width=1.5)
                ),
                text=[f'{u:.1f}%' for u in utilization],
                textposition='outside'
            ))
            
            fig.add_hline(
                y=100,
                line_dash="dash",
                line_color="#EF4444",
                annotation_text="Capacity Limit"
            )
            
            fig = apply_plotly_theme(fig)
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(title_text="Supplier")
            fig.update_yaxes(title_text="Utilization (%)")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            create_section_header("📊", "Cost vs Quantity Trade-off")
            
            order_data = []
            for order in results['orders']:
                order_data.append({
                    'supplier': order['supplier'],
                    'material': order['material'],
                    'quantity': order['quantity'],
                    'cost': order['quantity'] * order['unit_cost']
                })
            
            df = pd.DataFrame(order_data)
            
            fig = go.Figure()
            
            for supplier in df['supplier'].unique():
                supplier_df = df[df['supplier'] == supplier]
                fig.add_trace(go.Scatter(
                    x=supplier_df['quantity'],
                    y=supplier_df['cost'],
                    mode='markers',
                    name=supplier,
                    marker=dict(size=12, line=dict(width=1, color='#FFFFFF'))
                ))
            
            fig = apply_plotly_theme(fig)
            fig.update_layout(
                height=400,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            fig.update_xaxes(title_text="Order Quantity (tons)")
            fig.update_yaxes(title_text="Order Cost ($)")
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("### 📦 Procurement Order Schedule")
        
        # Convert orders to DataFrame
        orders_df = pd.DataFrame(results['orders'])
        
        if not orders_df.empty:
            orders_df['total_cost'] = orders_df['quantity'] * orders_df['unit_cost']
            
            # Summary by period
            st.markdown("#### 📅 Orders by Period")
            
            period_summary = orders_df.groupby('period').agg({
                'quantity': 'sum',
                'total_cost': 'sum'
            }).reset_index()
            period_summary.columns = ['Period', 'Total Quantity (tons)', 'Total Cost ($)']
            
            st.dataframe(
                period_summary.style.format({
                    'Total Quantity (tons)': '{:.2f}',
                    'Total Cost ($)': '${:,.2f}'
                }).background_gradient(cmap='Blues', subset=['Total Quantity (tons)', 'Total Cost ($)']),
                use_container_width=True,
                hide_index=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Detailed order list
            st.markdown("#### 📋 Detailed Order List")
            
            display_df = orders_df.copy()
            display_df = display_df.rename(columns={
                'period': 'Period',
                'supplier': 'Supplier',
                'material': 'Material',
                'quantity': 'Quantity (tons)',
                'unit_cost': 'Unit Cost ($/ton)',
                'total_cost': 'Total Cost ($)'
            })
            
            st.dataframe(
                display_df.style.format({
                    'Quantity (tons)': '{:.2f}',
                    'Unit Cost ($/ton)': '${:,.2f}',
                    'Total Cost ($)': '${:,.2f}'
                }).background_gradient(cmap='YlOrRd', subset=['Total Cost ($)']),
                use_container_width=True,
                hide_index=True
            )
            
            # Download orders
            csv = orders_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Orders CSV",
                data=csv,
                file_name=f"orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=False
            )
        else:
            st.info("No orders generated in the optimization.")
    
    with tab3:
        st.markdown("### 📈 Inventory Level Analysis")
        
        # Inventory over time
        materials = list(st.session_state.materials.keys())
        periods = generate_periods(st.session_state.planning_horizon)
        
        create_section_header("📊", "Inventory Trajectory by Material")
        
        fig = go.Figure()
        
        for i, material in enumerate(materials):
            material_inv = [
                inv['inventory'] 
                for inv in results['inventory'] 
                if inv['material'] == material
            ]
            
            fig.add_trace(go.Scatter(
                x=periods,
                y=material_inv,
                mode='lines+markers',
                name=material,
                line=dict(
                    color=COLOR_PALETTE['primary'][i % len(COLOR_PALETTE['primary'])],
                    width=3
                ),
                marker=dict(size=8)
            ))
        
        fig = apply_plotly_theme(fig)
        fig.update_layout(
            height=450,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        fig.update_xaxes(title_text="Period")
        fig.update_yaxes(title_text="Inventory Level (tons)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Inventory table
        st.markdown("#### 📋 Inventory Data Table")
        
        inv_df = pd.DataFrame(results['inventory'])
        inv_pivot = inv_df.pivot(index='period', columns='material', values='inventory')
        
        st.dataframe(
            inv_pivot.style.format('{:.2f}').background_gradient(cmap='Greens', axis=None),
            use_container_width=True
        )
        
        # Safety stock comparison
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🛡️ Safety Stock Status")
        
        meta = results.get('meta', {})
        safety_stock = meta.get('safety_stock_values', {})
        
        if safety_stock:
            safety_data = []
            for material in materials:
                avg_inv = inv_pivot[material].mean()
                required_safety = safety_stock.get(material, 0)
                
                safety_data.append({
                    'Material': material,
                    'Avg Inventory': avg_inv,
                    'Safety Stock Required': required_safety,
                    'Buffer': avg_inv - required_safety,
                    'Status': '✅ OK' if avg_inv >= required_safety else '⚠️ Below'
                })
            
            safety_df = pd.DataFrame(safety_data)
            
            st.dataframe(
                safety_df.style.format({
                    'Avg Inventory': '{:.2f}',
                    'Safety Stock Required': '{:.2f}',
                    'Buffer': '{:.2f}'
                }).apply(
                    lambda x: ['background-color: #10B981' if v == '✅ OK' else 'background-color: #EF4444' for v in x],
                    subset=['Status']
                ),
                use_container_width=True,
                hide_index=True
            )
    
    with tab4:
        st.markdown("### ⚠️ Risk Analysis")
        
        # Shortage analysis
        shortages_df = pd.DataFrame(results['shortages'])
        
        create_section_header("⚠️", "Safety Stock Shortages")
        
        # Filter periods with shortages
        risky_shortages = shortages_df[shortages_df['shortage'] > 0]
        
        if not risky_shortages.empty:
            st.warning(f"⚠️ Found {len(risky_shortages)} instances of safety stock shortages")
            
            # Shortage heatmap
            shortage_pivot = shortages_df.pivot(index='period', columns='material', values='shortage')
            
            fig = go.Figure(data=go.Heatmap(
                z=shortage_pivot.values,
                x=shortage_pivot.columns,
                y=shortage_pivot.index,
                colorscale='Reds',
                text=shortage_pivot.values,
                texttemplate='%{text:.1f}',
                textfont={"size": 10},
                colorbar=dict(title="Shortage (tons)")
            ))
            
            fig = apply_plotly_theme(fig)
            fig.update_layout(
                height=400,
                xaxis_title="Material",
                yaxis_title="Period"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Shortage details
            st.markdown("#### 📋 Shortage Details")
            
            display_shortages = risky_shortages.copy()
            display_shortages = display_shortages.rename(columns={
                'period': 'Period',
                'material': 'Material',
                'shortage': 'Shortage (tons)'
            })
            
            st.dataframe(
                display_shortages.style.format({
                    'Shortage (tons)': '{:.2f}'
                }).background_gradient(cmap='OrRd', subset=['Shortage (tons)']),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("✅ No safety stock shortages detected! All periods meet safety requirements.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Budget utilization
        create_section_header("💰", "Budget Utilization by Period")
        
        meta = results.get('meta', {})
        monthly_budgets = meta.get('monthly_budgets', {})
        
        if monthly_budgets:
            period_costs = {}
            for order in results['orders']:
                period = order['period']
                cost = order['quantity'] * order['unit_cost']
                period_costs[period] = period_costs.get(period, 0) + cost
            
            budget_data = []
            for period in sorted(monthly_budgets.keys()):
                budget = monthly_budgets[period]
                actual = period_costs.get(period, 0)
                utilization = (actual / budget * 100) if budget > 0 else 0
                
                budget_data.append({
                    'Period': period,
                    'Budget': budget,
                    'Actual Cost': actual,
                    'Utilization (%)': utilization,
                    'Remaining': budget - actual
                })
            
            budget_df = pd.DataFrame(budget_data)
            
            # Budget chart
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=budget_df['Period'],
                y=budget_df['Budget'],
                name='Budget',
                marker=dict(color='#94A3B8')
            ))
            
            fig.add_trace(go.Bar(
                x=budget_df['Period'],
                y=budget_df['Actual Cost'],
                name='Actual Cost',
                marker=dict(color='#3B82F6')
            ))
            
            fig = apply_plotly_theme(fig)
            fig.update_layout(
                height=400,
                barmode='group',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            fig.update_xaxes(title_text="Period")
            fig.update_yaxes(title_text="Amount ($)")
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Budget table
            st.markdown("#### 📊 Budget Details")
            
            st.dataframe(
                budget_df.style.format({
                    'Budget': '${:,.2f}',
                    'Actual Cost': '${:,.2f}',
                    'Utilization (%)': '{:.1f}%',
                    'Remaining': '${:,.2f}'
                }).apply(
                    lambda x: ['background-color: #EF4444' if v > 100 else 'background-color: #10B981' for v in x],
                    subset=['Utilization (%)']
                ),
                use_container_width=True,
                hide_index=True
            )
    
    with tab5:
        st.markdown("### 📥 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📄 JSON Export")
            st.markdown("Export complete optimization results in JSON format")
            
            json_data = json.dumps(results, indent=2)
            
            st.download_button(
                label="📥 Download Results JSON",
                data=json_data,
                file_name=f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("#### 📊 Orders CSV")
            st.markdown("Export procurement orders as CSV")
            
            orders_df = pd.DataFrame(results['orders'])
            if not orders_df.empty:
                orders_df['total_cost'] = orders_df['quantity'] * orders_df['unit_cost']
                csv_orders = orders_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Orders CSV",
                    data=csv_orders,
                    file_name=f"orders_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col2:
            st.markdown("#### 📈 Inventory CSV")
            st.markdown("Export inventory levels as CSV")
            
            inv_df = pd.DataFrame(results['inventory'])
            if not inv_df.empty:
                csv_inv = inv_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Inventory CSV",
                    data=csv_inv,
                    file_name=f"inventory_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            st.markdown("#### 📊 Summary Report")
            st.markdown("Export executive summary as text")
            
            summary_text = f"""
ARABCAB Inventory Optimization Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

========================================
KEY PERFORMANCE INDICATORS
========================================
Total Procurement Cost: ${kpis['total_cost']:,.2f}
Total Order Quantity: {kpis['total_quantity']:,.2f} tons
Average Cost per Ton: ${kpis['average_cost_per_ton']:,.2f}
Periods with Risk: {kpis['risky_periods']}

========================================
OPTIMIZATION PARAMETERS
========================================
Planning Horizon: {st.session_state.planning_horizon} months
Safety Stock Policy: {st.session_state.safety_stock_months} months
Budget Buffer: {st.session_state.budget_buffer * 100}%
Penalty Cost: ${st.session_state.penalty_cost}/ton

========================================
MATERIALS CONFIGURED
========================================
"""
            for mat_name, mat_data in st.session_state.materials.items():
                summary_text += f"\n{mat_name}:"
                summary_text += f"\n  - Holding Cost: ${mat_data['holding_cost']}/ton/month"
                summary_text += f"\n  - Current Inventory: {mat_data['inventory']} tons"
                summary_text += f"\n  - Capacity: {mat_data['capacity']} tons"
            
            summary_text += "\n\n========================================\n"
            summary_text += "SUPPLIERS CONFIGURED\n"
            summary_text += "========================================\n"
            
            for supp_name, supp_data in st.session_state.suppliers.items():
                summary_text += f"\n{supp_name}:"
                summary_text += f"\n  - Lead Time: {supp_data['lead_time']} months"
                summary_text += f"\n  - Capacity: {supp_data['capacity']} tons/month"
                summary_text += f"\n  - Materials: {', '.join(supp_data['materials'])}"
            
            st.download_button(
                label="📥 Download Summary Report",
                data=summary_text,
                file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("#### 🔄 Re-run Optimization")
        st.markdown("Make changes to parameters and re-run the optimization")
        
        if st.button("🚀 Back to Dashboard", type="primary", use_container_width=False):
            st.session_state.page = "dashboard"
            st.rerun()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.markdown("""
        <div style="padding: 2rem 1.5rem; text-align: center; border-bottom: 1px solid #334155; margin-bottom: 2rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">📊</div>
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFFFFF; margin-bottom: 0.25rem;">ARABCAB</div>
            <div style="font-size: 0.875rem; color: #94A3B8; margin-bottom: 0.5rem;">Inventory Optimization</div>
            <div style="font-size: 0.75rem; color: #64748B;">Scientific Competition 2026</div>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### 🧭 Navigation")
    
    pages = {
        "dashboard": ("🏠", "Dashboard"),
        "forecast": ("📊", "Forecasting"),
        "inventory": ("📦", "Inventory"),
        "supplier": ("🏭", "Suppliers"),
        "results": ("📈", "Results")
    }
    
    for key, (icon, label) in pages.items():
        is_current = st.session_state.page == key
        button_type = "primary" if is_current else "secondary"
        
        if st.button(
            f"{icon} {label}",
            key=f"nav_{key}",
            use_container_width=True,
            type=button_type
        ):
            if not is_current:
                st.session_state.page = key
                st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()
    
    # Optimization Controls
    st.markdown("### ⚙️ Optimization Controls")
    
    with st.expander("📅 Planning Horizon", expanded=True):
        planning_horizon = st.number_input(
            "Months to plan ahead",
            min_value=3,
            max_value=24,
            value=st.session_state.planning_horizon,
            step=1,
            help="Number of months to optimize",
            key="planning_input"
        )
        
        if planning_horizon != st.session_state.planning_horizon:
            st.session_state.planning_horizon = planning_horizon
            periods = generate_periods(planning_horizon)
            materials = list(st.session_state.materials.keys())
            if materials:
                st.session_state.forecast_demand = generate_ai_forecast(periods, materials)
            st.success(f"✅ Updated to {planning_horizon} months")
    
    with st.expander("💰 Budget Settings", expanded=True):
        budget_buffer = st.slider(
            "Budget buffer (%)",
            min_value=0,
            max_value=50,
            value=int(st.session_state.budget_buffer * 100),
            step=5,
            help="Additional budget headroom",
            key="budget_slider"
        )
        
        if budget_buffer / 100 != st.session_state.budget_buffer:
            st.session_state.budget_buffer = budget_buffer / 100
            st.success(f"✅ Buffer set to {budget_buffer}%")
    
    with st.expander("⚠️ Risk Parameters", expanded=True):
        penalty_cost = st.number_input(
            "Shortage penalty ($/ton)",
            min_value=0.0,
            value=float(st.session_state.penalty_cost),
            step=10.0,
            help="Cost of safety stock shortage",
            key="penalty_input"
        )
        
        if penalty_cost != st.session_state.penalty_cost:
            st.session_state.penalty_cost = penalty_cost
            st.success(f"✅ Penalty set to ${penalty_cost}")
        
        safety_months = st.number_input(
            "Safety stock (months)",
            min_value=0.5,
            max_value=6.0,
            value=float(st.session_state.safety_stock_months),
            step=0.5,
            help="Months of demand coverage",
            key="safety_input"
        )
        
        if safety_months != st.session_state.safety_stock_months:
            st.session_state.safety_stock_months = safety_months
            st.success(f"✅ Safety stock: {safety_months}mo")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Run Optimization
    if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
        with st.spinner("🔄 Running optimization..."):
            try:
                periods = generate_periods(st.session_state.planning_horizon)
                materials = list(st.session_state.materials.keys())
                suppliers = list(st.session_state.suppliers.keys())
                
                if not materials:
                    st.error("⚠️ Please add at least one material")
                elif not suppliers:
                    st.error("⚠️ Please add at least one supplier")
                elif not st.session_state.forecast_demand:
                    st.error("⚠️ No forecast data available. Generate forecast first.")
                else:
                    # Validate supplier materials and pricing
                    valid_suppliers = []
                    for supp_name, supp_data in st.session_state.suppliers.items():
                        if supp_data['materials'] and supp_data['prices']:
                            valid_suppliers.append(supp_name)
                        else:
                            st.warning(f"⚠️ Supplier '{supp_name}' has no materials or prices configured")
                    
                    if not valid_suppliers:
                        st.error("⚠️ No valid suppliers configured. Please add materials and prices to suppliers.")
                    else:
                        suppliers = valid_suppliers
                        
                        # Prepare optimization inputs
                        initial_inventory = {
                            m: st.session_state.materials[m]['inventory'] 
                            for m in materials
                        }
                        
                        holding_cost = {
                            m: st.session_state.materials[m]['holding_cost'] 
                            for m in materials
                        }
                        
                        reference_price = {
                            m: st.session_state.materials[m]['reference_price'] 
                            for m in materials
                        }
                        
                        supplier_materials = {
                            s: st.session_state.suppliers[s]['materials'] 
                            for s in suppliers
                        }
                        
                        purchase_price = {
                            s: st.session_state.suppliers[s]['prices'] 
                            for s in suppliers
                        }
                        
                        lead_time = {
                            s: st.session_state.suppliers[s]['lead_time'] 
                            for s in suppliers
                        }
                        
                        supplier_capacity = {
                            s: st.session_state.suppliers[s]['capacity'] 
                            for s in suppliers
                        }
                        
                        supplier_risk = {
                            s: st.session_state.suppliers[s]['risk'] 
                            for s in suppliers
                        }
                        
                        payment_adjustment = {
                            s: st.session_state.suppliers[s]['payment_adj'] 
                            for s in suppliers
                        }
                        
                        # Run optimization
                        st.info("⏳ Solving optimization model...")
                        results = run_optimization(
                            periods=periods,
                            materials=materials,
                            suppliers=suppliers,
                            forecast_demand=st.session_state.forecast_demand,
                            initial_inventory=initial_inventory,
                            safety_stock_months=st.session_state.safety_stock_months,
                            supplier_materials=supplier_materials,
                            purchase_price=purchase_price,
                            lead_time=lead_time,
                            supplier_capacity=supplier_capacity,
                            holding_cost=holding_cost,
                            penalty_cost=st.session_state.penalty_cost,
                            supplier_risk=supplier_risk,
                            payment_adjustment=payment_adjustment,
                            budget_buffer=st.session_state.budget_buffer,
                            reference_price=reference_price
                        )
                        
                        # Store results
                        st.session_state.optimization_results = results
                        st.session_state.last_optimization_time = datetime.now()
                        
                        # Show success message
                        st.success("✅ Optimization completed successfully!")
                        st.balloons()
                        
                        # Navigate to dashboard to see results
                        st.session_state.page = "dashboard"
                        st.rerun()
            
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
            
            except Exception as e:
                st.error(f"❌ Optimization failed: {str(e)}")
                with st.expander("🔍 Error Details"):
                    import traceback
                    st.code(traceback.format_exc())
    
    # Quick Stats
    if st.session_state.optimization_results:
        st.divider()
        st.markdown("### 📊 Quick Stats")
        
        kpis = st.session_state.optimization_results['kpis']
        
        st.metric(
            "💰 Total Cost",
            f"${kpis['total_cost']:,.0f}",
            help="Total procurement cost"
        )
        
        st.metric(
            "📦 Total Quantity",
            f"{kpis['total_quantity']:,.0f} tons",
            help="Total order quantity"
        )
        
        st.metric(
            "⚠️ Risk Periods",
            str(kpis['risky_periods']),
            help="Periods with safety shortages"
        )
        
        if st.session_state.last_optimization_time:
            st.caption(
                f"Last run: {st.session_state.last_optimization_time.strftime('%H:%M:%S')}"
            )
    
    st.divider()
    
    # System Info
    st.markdown("### ℹ️ System Info")
    
    st.caption(f"**Materials:** {len(st.session_state.materials)}")
    st.caption(f"**Suppliers:** {len(st.session_state.suppliers)}")
    st.caption(f"**Horizon:** {st.session_state.planning_horizon} months")
    st.caption(f"**Forecast:** {st.session_state.forecast_source}")


# ============================================================================
# PAGE ROUTING
# ============================================================================

page = st.session_state.page

if page == "dashboard":
    render_dashboard_home()
elif page == "forecast":
    render_demand_forecasting()
elif page == "inventory":
    render_inventory_management()
elif page == "supplier":
    render_supplier_configuration()
elif page == "results":
    render_results_analysis()
else:
    render_dashboard_home()


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
        <div style="text-align: center; color: #64748B; font-size: 0.875rem;">
            <strong>ARABCAB Scientific Competition 2026</strong><br>
            AI-Powered Supply Chain Optimization
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
        <div style="text-align: center; color: #64748B; font-size: 0.875rem;">
            Built with Streamlit • PuLP • Plotly<br>
            Mixed-Integer Linear Programming
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
        <div style="text-align: center; color: #64748B; font-size: 0.875rem;">
            Version 1.0.0<br>
            © 2026 All Rights Reserved
        </div>
    """, unsafe_allow_html=True)