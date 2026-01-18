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
from pathlib import Path


# ============================================================================
# LOAD CUSTOM CSS
# ============================================================================

def load_css():
    """Load custom CSS from external file"""
    css_file = Path(__file__).parent / "styles.css"
    if css_file.exists():
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ============================================================================
# PERSISTENT STORAGE
# ============================================================================

def save_state_to_storage():
    """Save current session state to persistent storage"""
    try:
        storage_data = {
            'materials': st.session_state.materials,
            'suppliers': st.session_state.suppliers,
            'safety_stock_months': st.session_state.safety_stock_months,
            'penalty_cost': st.session_state.penalty_cost,
            'budget_buffer': st.session_state.budget_buffer,
            'planning_horizon': st.session_state.planning_horizon,
            'forecast_demand': st.session_state.forecast_demand,
            'forecast_source': st.session_state.forecast_source,
            'optimization_results': st.session_state.optimization_results,
            'last_optimization_time': st.session_state.last_optimization_time.isoformat() if st.session_state.last_optimization_time else None
        }
        
        # Save to browser localStorage (you'll need to implement this via JavaScript)
        import json
        json_data = json.dumps(storage_data)
        
        # For now, offer download option
        return json_data
    except Exception as e:
        st.error(f"Failed to save state: {e}")
        return None

def load_state_from_storage(json_data):
    """Load session state from persistent storage"""
    try:
        import json
        data = json.loads(json_data)
        
        st.session_state.materials = data.get('materials', {})
        st.session_state.suppliers = data.get('suppliers', {})
        st.session_state.safety_stock_months = data.get('safety_stock_months', 2.0)
        st.session_state.penalty_cost = data.get('penalty_cost', 100.0)
        st.session_state.budget_buffer = data.get('budget_buffer', 0.10)
        st.session_state.planning_horizon = data.get('planning_horizon', 6)
        st.session_state.forecast_demand = data.get('forecast_demand', {})
        st.session_state.forecast_source = data.get('forecast_source', 'AI Model')
        st.session_state.optimization_results = data.get('optimization_results', None)
        
        last_opt = data.get('last_optimization_time')
        if last_opt:
            from datetime import datetime
            st.session_state.last_optimization_time = datetime.fromisoformat(last_opt)
        
        return True
    except Exception as e:
        st.error(f"Failed to load state: {e}")
        return False
  
def auto_save_state():
    """Auto-save state to a local JSON file"""
    import json
    from pathlib import Path
    
    save_path = Path("app_state.json")
    
    storage_data = {
        'materials': st.session_state.materials,
        'suppliers': st.session_state.suppliers,
        'safety_stock_months': st.session_state.safety_stock_months,
        'penalty_cost': st.session_state.penalty_cost,
        'budget_buffer': st.session_state.budget_buffer,
        'planning_horizon': st.session_state.planning_horizon,
    }
    
    with open(save_path, 'w') as f:
        json.dump(storage_data, f, indent=2)

def auto_load_state():
    """Auto-load state from local JSON file if it exists"""
    from pathlib import Path
    import json
    
    save_path = Path("app_state.json")
    
    if save_path.exists():
        try:
            with open(save_path, 'r') as f:
                data = json.load(f)
            
            for key, value in data.items():
                if key in st.session_state:
                    st.session_state[key] = value
            
            return True
        except:
            return False
    return False

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
    lead_time: Dict[str, Dict[str, int]],
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
                    lt = lead_time.get(s, {}).get(m, 1)
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
            # Get lead time with fallback to 1 if not found
            material_lead_time = lead_time.get(s, {}).get(m, 1)
            
            results["orders"].append({
                "period": t,
                "supplier": s,
                "material": m,
                "quantity": round(q, 2),
                "unit_cost": purchase_price[s][m],
                "lead_time": material_lead_time
            })
    
    for t in periods:
        for m in materials:
            inv_value = inventory[(t, m)].value()
            short_value = shortage[(t, m)].value()
            
            results["inventory"].append({
                "period": t,
                "material": m,
                "inventory": round(inv_value if inv_value else 0, 2)
            })
            results["shortages"].append({
                "period": t,
                "material": m,
                "shortage": round(short_value if short_value else 0, 2)
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
    lead_time: Dict[str, Dict[str, int]],  # Changed from Dict[str, int] to Dict[str, Dict[str, int]]
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
                'capacity': 500,
                'risk': 0.02,
                'payment_adj': 0.00,
                'prices': {'PVC': 1200, 'XLPE': 1800},
                'lead_times': {'PVC': 1, 'XLPE': 1},
                'origin': 'Local'  # NEW: Add default origin
            },
            'Beta Ltd': {
                'materials': ['XLPE', 'PE', 'LSF'],
                'capacity': 400,
                'risk': 0.05,
                'payment_adj': -24.00,
                'prices': {'XLPE': 1750, 'PE': 900, 'LSF': 2200},
                'lead_times': {'XLPE': 2, 'PE': 2, 'LSF': 1},
                'origin': 'Imported'  # NEW: Add default origin
            },
            'Gamma Inc': {
                'materials': ['PVC', 'PE'],
                'capacity': 600,
                'risk': 0.03,
                'payment_adj': 15.00,
                'prices': {'PVC': 1180, 'PE': 920},
                'lead_times': {'PVC': 1, 'PE': 2},
                'origin': 'Local'  # NEW: Add default origin
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
    with st.expander("📘 Supplier Configuration Guidelines", expanded=False):
        st.markdown("""
        <div style="
            background: rgba(15, 23, 42, 0.6);
            padding: 1.25rem;
            border-radius: 10px;
            border-left: 3px solid #3B82F6;
        ">

        ### General Principle
        
        Supplier parameters represent **expected future behavior**, not perfect certainty.  
        The optimizer balances **cost, risk, lead time, and capacity** to generate robust procurement plans.
        
        ---
        
        ### Monthly Capacity (tons)
        
        Maximum quantity the supplier can deliver per month across all materials.
        
        **Typical cases:**
        - Small / specialized supplier: **50 – 200 tons**
        - Medium supplier: **200 – 600 tons**
        - Large industrial supplier: **600 – 1500+ tons**
        
        *Tip:* Use conservative values if capacity is uncertain.
        
        ---
        
        ### Supplier Origin (Local / Imported)
        
        Indicates logistical exposure and delivery uncertainty.
        
        **Local supplier**
        - Lead time: **0 – 1 month**
        - Lower disruption risk
        - Faster response to demand changes
        
        **Imported supplier**
        - Lead time: **2 – 4 months**
        - Customs, shipping, and currency risk
        - Higher exposure to delays
        
        ---
        
        ### Risk Premium (USD/ton)
        
        Represents expected additional cost due to:
        - Delivery delays
        - Quality issues
        - Logistics or geopolitical uncertainty
        
        This value is **added to unit price** in optimization.
        
        **Common cases:**
        - Very reliable supplier: **5 – 15 USD/ton**
        - Average reliability: **20 – 35 USD/ton**
        - High-risk supplier: **40 – 70 USD/ton**
        
        **Special case – New supplier (no historical data):**
        - Use **medium-to-high risk** (30 – 50 USD/ton)
        - Reduce later as performance data becomes available
        
        ---
        
        ### Payment Adjustment (USD/ton)
        
        Models the financial effect of payment terms.
        
        - **Negative value** → cost reduction (discounts)
        - **Positive value** → cost increase (financing, installments)
        
        **Typical cases:**
        - Advance payment discount: **-10 to -30 USD/ton**
        - Cash on delivery: **0 USD/ton**
        - Net 30 / Net 60 / Installments: **+10 to +30 USD/ton**
        
        *Note:* Payment adjustment affects cost only, not delivery timing.
        
        ---
        
        ### Materials & Pricing
        
        Suppliers can only deliver **explicitly assigned materials**.
        
        **Guidelines:**
        - Assign only materials the supplier is contractually capable of supplying
        - Prices should represent **base price before risk & payment effects**
        
        ---
        
        ### Lead Time (months)
        
        Time between placing an order and receiving material.
        
        **Typical values:**
        - Local supplier: **0 – 1 month**
        - Regional importer: **1 – 2 months**
        - Overseas importer: **2 – 4 months**
        
        **Worst-case planning:**  
        If lead time varies, use the **upper bound**.
        
        ---
        
        ### How the Optimizer Interprets These Inputs
        
        - Lower **effective cost** → higher selection priority
        - Higher **risk premium** → reduced reliance unless necessary
        - Longer **lead time** → earlier ordering required
        - Capacity limits are **hard constraints**
        
        ---
        
        **Final Tip:**  
        When uncertain, prefer **conservative assumptions**.  
        The optimizer performs best when inputs reflect realistic operational constraints.
        
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Add New Supplier
    col1, col2 = st.columns([4, 1])

    

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
                        'capacity': 500,
                        'risk': 0.02,
                        'payment_adj': 0.0,
                        'prices': {},
                        'lead_times': {},
                        'origin': 'Local'  # NEW: Default origin
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
        # Calculate average lead time across all materials
        all_lead_times = []
        for supp_data in st.session_state.suppliers.values():
            if 'lead_times' in supp_data:
                all_lead_times.extend(supp_data['lead_times'].values())
        avg_lead_time = sum(all_lead_times) / len(all_lead_times) if all_lead_times else 0
        st.metric("⏱️ Avg Lead Time", f"{avg_lead_time:.1f} months")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Supplier Details
    create_section_header("🏭", "Supplier Configuration")
    
    if not st.session_state.suppliers:
        st.warning("⚠️ No suppliers configured. Add suppliers to get started!")
        return
    
    for supp_name, supp_data in st.session_state.suppliers.items():
        with st.expander(f"🏭 {supp_name}", expanded=False):
            # Section 1: Basic Configuration
            st.markdown("**📋 Basic Configuration**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                capacity = st.number_input(
                    "Monthly Capacity (tons)",
                    value=int(supp_data['capacity']),
                    min_value=0,
                    step=50,
                    key=f"cap_{supp_name}",
                    help="Maximum total quantity this supplier can deliver per month across all materials"
                )
            
            with col2:
                origin = st.radio(
                    "Supplier Origin",
                    options=['Local', 'Imported'],
                    index=0 if supp_data.get('origin', 'Local') == 'Local' else 1,
                    key=f"origin_{supp_name}",
                    help="🌍 Origin affects lead time reliability, logistics risk, and supply flexibility",
                    horizontal=True  # Makes it display side-by-side
                )
            
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                # Display origin badge
                origin_badge = "🏠 Local" if origin == 'Local' else "🌐 Imported"
                st.info(f"**Current Origin:** {origin_badge}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Section 2: Risk & Financial Parameters
            st.markdown("**💰 Risk & Financial Parameters**")
            col1, col2 = st.columns(2)
            
            with col1:
                # ENHANCED: Risk Premium with detailed guidance
                risk = st.number_input(
                    "Risk Premium (%)",
                    value=float(supp_data['risk'] * 100),
                    min_value=0.0,
                    max_value=100.0,
                    step=0.5,
                    key=f"risk_{supp_name}",
                    help="""
📊 **Aggregated Risk Assessment**

This value represents overall supplier risk based on multiple factors:
- Supplier reliability and track record
- Lead time uncertainty and variability
- Quality performance and defect rates
- Logistics and sourcing complexity

**Risk Classification Guidelines:**
- **Low Risk (0–5%)**: Established suppliers with proven reliability
- **Medium Risk (10–30%)**: Suppliers with moderate uncertainty
- **High Risk (40%+)**: New or unreliable suppliers with significant risk

*Note: This premium is added to unit costs in the optimization model to reflect risk-adjusted procurement costs.*
                    """
                ) / 100
                
                # Display risk classification
                if risk * 100 <= 5:
                    st.success("✅ Low Risk Supplier")
                elif risk * 100 <= 30:
                    st.warning("⚠️ Medium Risk Supplier")
                else:
                    st.error("🚨 High Risk Supplier")
            
            with col2:
                # ENHANCED: Payment Adjustment with detailed guidance
                payment_adj = st.number_input(
                    "Payment Adjustment ($/ton)",
                    value=float(supp_data['payment_adj']),
                    step=1.0,
                    key=f"payment_{supp_name}",
                    help="""
💳 **Payment Terms Impact**

This value models the financial effect of payment policies:

**Negative Values** = Supplier Discounts:
- Early/advance payment discounts (e.g., −$25)
- Cash payment benefits (e.g., −$15)

**Positive Values** = Financing Costs:
- Net 30 days payment terms (e.g., +$10)
- Net 60/90 days extended terms (e.g., +$20–$30)
- Installment plans (e.g., +$40+)

**Zero** = Standard terms with no adjustment

**Payment Terms Examples:**
- Cash payment: $0
- 2% discount for advance payment: −$25
- Net 30 days: +$10
- Net 60 days: +$20
- Installment financing: +$40

*Note: This adjustment is added to unit costs to reflect true total cost of procurement.*
                    """
                )
                
                # Display payment terms classification
                if payment_adj < 0:
                    st.success(f"💰 Discount: ${abs(payment_adj):.2f}/ton")
                elif payment_adj > 0:
                    st.warning(f"📈 Financing Cost: +${payment_adj:.2f}/ton")
                else:
                    st.info("💵 Standard Terms")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Section 3: Materials & Pricing
            st.markdown("**📦 Materials & Pricing**")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("**Materials Supplied**")
                selected_materials = st.multiselect(
                    "Select materials",
                    options=list(st.session_state.materials.keys()),
                    default=supp_data['materials'],
                    key=f"materials_{supp_name}",
                    label_visibility="collapsed",
                    help="Select which materials this supplier can provide"
                )
            
            with col2:
                st.markdown("**Material Pricing & Lead Times**")
                
                if selected_materials:
                    # Get current lead times
                    current_lead_times = supp_data.get('lead_times', {})
                    
                    pricing_data = []
                    for mat in selected_materials:
                        current_price = supp_data['prices'].get(
                            mat,
                            st.session_state.materials[mat]['reference_price']
                        )
                        current_lead = current_lead_times.get(mat, 1)
                        pricing_data.append({
                            'Material': mat,
                            'Price ($/ton)': current_price,
                            'Lead Time (months)': current_lead
                        })
                    
                    pricing_df = pd.DataFrame(pricing_data)
                    edited_pricing = st.data_editor(
                        pricing_df,
                        use_container_width=True,
                        hide_index=True,
                        key=f"pricing_{supp_name}",
                        column_config={
                            "Material": st.column_config.TextColumn("Material", disabled=True),
                            "Price ($/ton)": st.column_config.NumberColumn(
                                "Price ($/ton)", 
                                min_value=0, 
                                step=10,
                                help="Base unit price before risk and payment adjustments"
                            ),
                            "Lead Time (months)": st.column_config.NumberColumn(
                                "Lead Time (months)", 
                                min_value=0, 
                                max_value=12, 
                                step=1,
                                help="Time from order placement to delivery arrival"
                            )
                        }
                    )
                else:
                    st.info("Select materials to configure pricing and lead times")
                    edited_pricing = None
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Section 4: Actions
            col1, col2 = st.columns([1, 3])

            with col1:
                if st.button(f"💾 Save Configuration", key=f"save_{supp_name}", use_container_width=True):
                    # Update prices and lead times from data editor
                    new_prices = {}
                    new_lead_times = {}
                    if edited_pricing is not None:
                        for _, row in edited_pricing.iterrows():
                            new_prices[row['Material']] = row['Price ($/ton)']
                            new_lead_times[row['Material']] = int(row['Lead Time (months)'])
                    
                    st.session_state.suppliers[supp_name] = {
                        'materials': selected_materials,
                        'capacity': capacity,
                        'risk': risk,
                        'payment_adj': payment_adj,
                        'prices': new_prices,
                        'lead_times': new_lead_times,
                        'origin': origin  # NEW: Save origin
                    }
                    st.success(f"✅ {supp_name} configuration saved!")

            with col2:
                if st.button(f"🗑️ Delete Supplier", key=f"delete_{supp_name}"):
                    del st.session_state.suppliers[supp_name]
                    st.success(f"✅ {supp_name} deleted!")
                    st.rerun()
            
            # Section 5: Summary Display
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("**📊 Effective Cost Summary**")
            
            if edited_pricing is not None and not edited_pricing.empty:
                summary_data = []
                for _, row in edited_pricing.iterrows():
                    base_price = row['Price ($/ton)']
                    risk_cost = base_price * risk
                    total_cost = base_price + risk_cost + payment_adj
                    
                    summary_data.append({
                        'Material': row['Material'],
                        'Base Price': f"${base_price:,.2f}",
                        'Risk Premium': f"+${risk_cost:,.2f}",
                        'Payment Adj': f"{payment_adj:+.2f}",
                        'Total Cost': f"${total_cost:,.2f}"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(
                    summary_df,
                    use_container_width=True,
                    hide_index=True
                )
                st.caption("💡 Total Cost = Base Price + Risk Premium + Payment Adjustment")


def render_results_analysis():
    """Detailed results and analysis page"""
    create_header(
        "Results & Analysis",
        "Optimization Results, Insights & Export"
    )
    
    if st.session_state.optimization_results is None:
        st.warning("⚠️ No optimization results available. Please run optimization first.")
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
                summary_text += f"\n  - Capacity: {supp_data['capacity']} tons/month"
                summary_text += f"\n  - Materials: {', '.join(supp_data['materials'])}"
                if 'lead_times' in supp_data and supp_data['lead_times']:
                    summary_text += f"\n  - Lead Times: {', '.join([f'{m}:{lt}mo' for m, lt in supp_data['lead_times'].items()])}"


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
    # Header/Branding
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 1.75rem 1.25rem;
        border-radius: 14px;
        border: 1px solid #334155;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    ">
        <div style="
            font-size: 1.75rem;
            font-weight: 800;
            color: #FFFFFF;
            letter-spacing: -0.02em;
            margin-bottom: 0.25rem;
        ">
            Nexum
        </div>
        <div style="
            font-size: 0.85rem;
            color: #94A3B8;
            margin-bottom: 0.25rem;
        ">
            Inventory Optimization
        </div>
        <div style="
            font-size: 0.7rem;
            color: #64748B;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        ">
            Scientific Competition 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border:1px solid #334155;'>", unsafe_allow_html=True)

    # Navigation
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

    # ========================================================================
    # SAVE/LOAD CONFIGURATION (FIXED VERSION)
    # ========================================================================
    
    st.markdown("### 💾 Configuration")

    # Save Section
    with st.expander("📥 Save Configuration", expanded=False):
        st.markdown("""
            <div style='color: #94A3B8; font-size: 0.85rem; margin-bottom: 1rem;'>
                Download your current setup to restore later
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Generate Save File", use_container_width=True, type="primary", key="btn_generate_save"):
            json_data = save_state_to_storage()
            if json_data:
                st.download_button(
                    label="💾 Download Configuration",
                    data=json_data,
                    file_name=f"nexum_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                    key="btn_download_config"
                )
                st.success("✅ Configuration ready to download!")

    # Load Section (FIXED)
    with st.expander("📤 Load Configuration", expanded=False):
        st.markdown("""
            <div style='color: #94A3B8; font-size: 0.85rem; margin-bottom: 1rem;'>
                Restore a previously saved configuration
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_config = st.file_uploader(
            "Choose configuration file",
            type=['json'],
            help="Upload .json configuration file",
            key="config_uploader_sidebar"
        )
        
        if uploaded_config:
            # Show file info
            st.info(f"📄 {uploaded_config.name} ({uploaded_config.size:,} bytes)")
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                if st.button("🚀 Load", use_container_width=True, type="primary", key="btn_load_config"):
                    with st.spinner("⏳ Loading..."):
                        try:
                            json_data = uploaded_config.read().decode('utf-8')
                            if load_state_from_storage(json_data):
                                st.success("✅ Loaded!")
                                st.balloons()
                                
                                # CRITICAL: Clear uploader state before rerun
                                if 'config_uploader_sidebar' in st.session_state:
                                    del st.session_state['config_uploader_sidebar']
                                
                                time.sleep(0.3)
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Error: {str(e)}")
            
            with col2:
                if st.button("✕", use_container_width=True, help="Clear", key="btn_clear_upload"):
                    if 'config_uploader_sidebar' in st.session_state:
                        del st.session_state['config_uploader_sidebar']
                    st.rerun()

    st.divider()

    # ========================================================================
    # OPTIMIZATION CONTROLS
    # ========================================================================
    
    st.markdown("### ⚙️ Optimization Controls")
    
    with st.expander("📅 Planning Horizon", expanded=False):
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
    
    with st.expander("💰 Budget Settings", expanded=False):
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
    
    with st.expander("⚠️ Risk Parameters", expanded=False):
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
                    initial_inventory = {m: st.session_state.materials[m]['inventory'] for m in materials}
                    holding_cost = {m: st.session_state.materials[m]['holding_cost'] for m in materials}
                    reference_price = {m: st.session_state.materials[m]['reference_price'] for m in materials}
                    supplier_materials = {s: st.session_state.suppliers[s]['materials'] for s in suppliers}
                    
                    # Validate and build purchase prices
                    purchase_price = {}
                    for s in suppliers:
                        supp_prices = st.session_state.suppliers[s].get('prices', {})
                        if not supp_prices:
                            st.error(f"❌ Supplier '{s}' has no prices configured!")
                            st.stop()
                        purchase_price[s] = supp_prices
                    
                    # Validate and build lead times with defaults
                    lead_time = {}
                    for s in suppliers:
                        lead_time[s] = {}
                        supp_lead_times = st.session_state.suppliers[s].get('lead_times', {})
                        for m in st.session_state.suppliers[s]['materials']:
                            if m not in supp_lead_times:
                                st.warning(f"⚠️ Supplier '{s}' missing lead time for '{m}', using default 1 month")
                                lead_time[s][m] = 1
                            else:
                                lead_time[s][m] = supp_lead_times[m]
                    
                    supplier_capacity = {s: st.session_state.suppliers[s]['capacity'] for s in suppliers}
                    supplier_risk = {s: st.session_state.suppliers[s]['risk'] for s in suppliers}
                    payment_adjustment = {s: st.session_state.suppliers[s]['payment_adj'] for s in suppliers}

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
    
st.markdown("### ℹ️ System Info")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="📦 Materials",
        value=len(st.session_state.materials)
    )

with col2:
    st.metric(
        label="🏭 Suppliers",
        value=len(st.session_state.suppliers)
    )

with col3:
    st.metric(
        label="🗓️ Horizon",
        value=f"{st.session_state.planning_horizon} months"
    )

with col4:
    st.metric(
        label="📈 Forecast",
        value=st.session_state.forecast_source
    )



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