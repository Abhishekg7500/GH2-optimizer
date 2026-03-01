"""
app.py
------
GH2 Optimiser — Streamlit Web Application
Green Hydrogen Plant Capacity Sizing & CAPEX Optimization Engine

Run with:  streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from simulator.sizing   import run_sizing_optimization
from simulator.dispatch import run_dispatch
from economics.capex    import calculate_capex

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title   = "GH2 Optimiser",
    page_icon    = "⚗",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* Dark background */
  .stApp { background-color: #060c14; color: #c0d8e8; }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #091422;
    border-right: 1px solid #183048;
  }

  /* Headers */
  h1, h2, h3 { color: #00bfa8 !important; font-family: 'Exo 2', sans-serif; }
  h4, h5     { color: #7fb0cc !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #0c1c2e;
    border: 1px solid #183048;
    padding: 12px 16px;
    border-radius: 2px;
  }
  [data-testid="metric-container"] label { color: #3a6080 !important; font-size: 11px; letter-spacing: 1px; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #00bfa8 !important; font-size: 22px; }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 11px; }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, rgba(0,191,168,0.12), rgba(41,171,255,0.06));
    border: 1px solid #00bfa8;
    color: #00bfa8;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    width: 100%;
    padding: 14px;
    font-size: 14px;
  }
  .stButton > button:hover {
    background: rgba(0,191,168,0.2);
    border-color: #00bfa8;
    color: #00bfa8;
  }

  /* Number inputs */
  .stNumberInput input { background: #0c1c2e; color: #c0d8e8; border: 1px solid #183048; font-family: monospace; }

  /* Section dividers */
  hr { border-color: #183048; }

  /* Tables */
  .stDataFrame { border: 1px solid #183048; }

  /* Info boxes */
  .derived-box {
    background: rgba(0,191,168,0.06);
    border: 1px solid rgba(0,191,168,0.3);
    border-radius: 2px;
    padding: 10px 14px;
    font-family: monospace;
    font-size: 13px;
    color: #00bfa8;
    margin: 8px 0;
  }

  /* Status badge */
  .badge-ok   { background:rgba(94,240,144,0.12); color:#5ef090; border:1px solid rgba(94,240,144,0.3); padding:2px 8px; font-size:11px; }
  .badge-warn { background:rgba(255,181,32,0.12);  color:#ffb520; border:1px solid rgba(255,181,32,0.3);  padding:2px 8px; font-size:11px; }
  .badge-err  { background:rgba(255,77,106,0.12);  color:#ff4d6a; border:1px solid rgba(255,77,106,0.3);  padding:2px 8px; font-size:11px; }

  /* Plotly chart background */
  .js-plotly-plot .plotly { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor = "#091422",
    plot_bgcolor  = "#0c1c2e",
    font          = dict(color="#7fb0cc", family="JetBrains Mono, monospace", size=11),
    xaxis         = dict(gridcolor="#183048", linecolor="#183048", zerolinecolor="#183048"),
    yaxis         = dict(gridcolor="#183048", linecolor="#183048", zerolinecolor="#183048"),
    legend        = dict(bgcolor="rgba(0,0,0,0)", bordercolor="#183048", borderwidth=1),
    margin        = dict(l=50, r=30, t=40, b=50),
)

COLORS = {
    "solar":    "#ffd060",
    "wind":     "#29abff",
    "elec":     "#c084fc",
    "storage":  "#5ef090",
    "demand":   "#ff5578",
    "teal":     "#00bfa8",
    "gold":     "#ffb520",
    "rose":     "#ff4d6a",
    "bess":     "#ff80a0",
}

MONTHS     = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_DAYS = [31,28,31,30,31,30,31,31,30,31,30,31]

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ─────────────────────────────────────────────
#  SIDEBAR — ALL INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗ GH2 Optimiser")
    st.markdown("*Green Hydrogen Plant Sizing*")
    st.divider()

    # ── RE Profile Upload ──
    st.markdown("### ☀ RE Generation Profiles")
    st.caption("Upload hourly profiles — 8760 rows, kWh per MW per hour")

    solar_file = st.file_uploader("Solar Profile (CSV)", type=["csv"], key="solar_upload")
    wind_file  = st.file_uploader("Wind Profile (CSV)",  type=["csv"], key="wind_upload")

    st.divider()

    # ── Demand Inputs ──
    st.markdown("### ⚗ H₂ Demand Specification")

    h2_annual_t = st.number_input(
        "Annual H₂ Production (tonnes/yr)",
        min_value   = 100,
        max_value   = 500000,
        value       = 10000,
        step        = 500,
        help        = "Total green hydrogen to be delivered to consumer per year"
    )

    col1, col2 = st.columns(2)
    with col1:
        min_flow = st.number_input(
            "Min H₂ Flow (kg/hr)",
            min_value = 0,
            value     = 500,
            step      = 50,
            help      = "Minimum offtake rate consumer will accept"
        )
    with col2:
        max_flow = st.number_input(
            "Max H₂ Flow (kg/hr)",
            min_value = 1,
            value     = 2000,
            step      = 50,
            help      = "Maximum offtake rate — pipe/equipment limit"
        )

    col3, col4 = st.columns(2)
    with col3:
        op_days = st.number_input(
            "Operating Days (days/yr)",
            min_value = 1,
            max_value = 365,
            value     = 333,
            step      = 1,
        )
    with col4:
        hrs_per_day = st.number_input(
            "Hours Per Day (hrs/day)",
            min_value = 1,
            max_value = 24,
            value     = 24,
            step      = 1,
        )

    # Derived values display
    op_hours  = op_days * hrs_per_day
    avg_flow  = h2_annual_t * 1000 / op_hours if op_hours > 0 else 0
    daily_t   = h2_annual_t / op_days if op_days > 0 else 0

    st.markdown(f"""
    <div class="derived-box">
    Op Hours = {op_hours:,} hrs/yr<br>
    Avg Flow = {avg_flow:.0f} kg/hr<br>
    Daily Target = {daily_t:.1f} t/day<br>
    Plant Factor = {op_hours/8760*100:.1f}%
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # ── Electrolyzer Config ──
    st.markdown("### ⚡ Electrolyzer")

    col5, col6 = st.columns(2)
    with col5:
        stack_mw = st.number_input(
            "Stack Size (MW)",
            min_value = 1,
            max_value = 100,
            value     = 10,
            step      = 1,
        )
    with col6:
        efficiency = st.number_input(
            "Efficiency (kWh/kgH₂)",
            min_value = 38.0,
            max_value = 80.0,
            value     = 52.0,
            step      = 0.5,
        )

    col7, col8 = st.columns(2)
    with col7:
        min_load_pct = st.number_input(
            "Min Load (%)",
            min_value = 5,
            max_value = 50,
            value     = 30,
            step      = 5,
            help      = "30% of one stack = technical minimum"
        ) / 100.0
    with col8:
        availability = st.number_input(
            "Availability (%)",
            min_value = 70,
            max_value = 100,
            value     = 95,
            step      = 1,
        ) / 100.0

    st.divider()

    # ── Search Bounds ──
    st.markdown("### 🔍 Optimization Bounds")
    st.caption("Electrolyzer sweep range")

    col9, col10 = st.columns(2)
    with col9:
        elec_min = st.number_input(
            "Elec Min (MW)",
            min_value = 1,
            value     = stack_mw,
            step      = stack_mw,
        )
    with col10:
        elec_max = st.number_input(
            "Elec Max (MW)",
            min_value = elec_min + stack_mw,
            value     = max(elec_min + stack_mw, 300),
            step      = stack_mw,
        )

    st.divider()

    # ── Unit Costs ──
    st.markdown("### ₹ Unit Costs (₹ Crores)")

    cost_solar  = st.number_input("Solar (₹ Cr/MWp)",    value=3.5, step=0.1, format="%.2f")
    cost_wind   = st.number_input("Wind (₹ Cr/MW)",       value=7.0, step=0.1, format="%.2f")
    cost_elec   = st.number_input("Electrolyzer (₹ Cr/MW)",value=7.0, step=0.1, format="%.2f")
    cost_stor   = st.number_input("H₂ Storage (₹ Cr/tH₂)",value=0.55,step=0.05,format="%.2f")
    cost_comp   = st.number_input("Compressor (₹ Cr/MW)", value=4.2, step=0.1, format="%.2f")

    st.markdown("**Project Adders (%)**")
    col11, col12, col13 = st.columns(3)
    with col11: bop_pct   = st.number_input("BOP%",   value=12, step=1) / 100.0
    with col12: epc_pct   = st.number_input("EPC%",   value=8,  step=1) / 100.0
    with col13: cont_pct  = st.number_input("Cont%",  value=5,  step=1) / 100.0

    st.divider()

    # ── Run Button ──
    run_btn = st.button("▶  OPTIMISE CAPEX", use_container_width=True)

# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#091422,#0c1c2e);
            border-bottom:1px solid #183048;
            padding:20px 28px 16px;
            margin-bottom:24px;">
  <span style="font-size:26px;font-weight:900;color:#00bfa8;
               letter-spacing:4px;font-family:'Exo 2',sans-serif;">
    GH2 OPTIMISER
  </span>
  <span style="font-size:12px;color:#3a6080;margin-left:16px;
               font-family:monospace;letter-spacing:1px;">
    Green Hydrogen Plant · CAPEX Optimization Engine · ₹ Crores
  </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  VALIDATION & PROFILE LOADING
# ─────────────────────────────────────────────
def load_profile(uploaded_file, name: str) -> np.ndarray:
    """Load a CSV profile file and return 8760 numpy array."""
    try:
        df = pd.read_csv(uploaded_file, header=None)
        # Take first numeric column
        for col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(series) >= 8760:
                return series.values[:8760].astype(np.float32)
            elif len(series) > 100:
                # Interpolate to 8760 if close
                x_old = np.linspace(0, 1, len(series))
                x_new = np.linspace(0, 1, 8760)
                return np.interp(x_new, x_old, series.values).astype(np.float32)
        st.error(f"❌ {name}: Could not find 8760 numeric values in file.")
        return None
    except Exception as e:
        st.error(f"❌ {name} load error: {e}")
        return None


# ─────────────────────────────────────────────
#  HELPER: Build hourly index
# ─────────────────────────────────────────────
def build_hourly_df(result: dict, solar_mw, wind_mw) -> pd.DataFrame:
    hours = np.arange(8760)
    month_arr, day_arr, hour_arr = [], [], []
    h = 0
    for m, days in enumerate(MONTH_DAYS):
        for d in range(days):
            for hr in range(24):
                month_arr.append(m)
                day_arr.append(d + 1)
                hour_arr.append(hr)
                h += 1
    return pd.DataFrame({
        "hour":          hours,
        "month_idx":     month_arr,
        "hour_of_day":   hour_arr,
        "re_power_mw":   result["re_power"],
        "elec_power_mw": result["elec_power"],
        "h2_produced_kg":result["h2_produced"],
        "h2_delivered_kg":result["h2_delivered"],
        "h2_storage_kg": result["h2_storage"],
        "curtailment_mw":result["curtailment_mw"],
        "deficit_kg":    result["deficit"],
    })


# ─────────────────────────────────────────────
#  RUN OPTIMIZATION
# ─────────────────────────────────────────────
if run_btn:
    # ── Validate inputs ──
    errors = []
    if solar_file is None:
        errors.append("Solar profile CSV not uploaded.")
    if wind_file is None:
        errors.append("Wind profile CSV not uploaded.")
    if min_flow > max_flow:
        errors.append("Min H₂ Flow must be ≤ Max H₂ Flow.")
    if avg_flow > max_flow:
        errors.append(f"Average flow ({avg_flow:.0f} kg/hr) exceeds Max Flow ({max_flow} kg/hr).")
    if avg_flow < min_flow:
        errors.append(f"Average flow ({avg_flow:.0f} kg/hr) is below Min Flow ({min_flow} kg/hr).")

    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    solar_profile = load_profile(solar_file, "Solar")
    wind_profile  = load_profile(wind_file,  "Wind")

    if solar_profile is None or wind_profile is None:
        st.stop()

    # ── Progress UI ──
    progress_bar   = st.progress(0.0)
    status_text    = st.empty()
    log_placeholder= st.empty()

    log_lines = []

    def progress_cb(step, total, elec_mw, capex_cr):
        pct = step / max(total, 1)
        progress_bar.progress(min(pct, 1.0))
        status_text.markdown(
            f"**Electrolyzer: {elec_mw:.0f} MW** | "
            f"CAPEX so far: ₹ {capex_cr:.1f} Cr | "
            f"Step {step}/{total}"
        )
        log_lines.append(f"[{step:>3}/{total}] Elec={elec_mw:.0f} MW → ₹ {capex_cr:.2f} Cr")
        log_placeholder.code("\n".join(log_lines[-15:]), language="")

    try:
        with st.spinner("Running GH2 CAPEX Optimization..."):
            results = run_sizing_optimization(
                solar_profile             = solar_profile,
                wind_profile              = wind_profile,
                annual_h2_target_t        = h2_annual_t,
                min_flow_kg_hr            = min_flow,
                max_flow_kg_hr            = max_flow,
                op_days                   = op_days,
                hrs_per_day               = hrs_per_day,
                stack_mw                  = stack_mw,
                efficiency_kwh_per_kg     = efficiency,
                min_load_pct              = min_load_pct,
                availability              = availability,
                elec_min_mw               = elec_min,
                elec_max_mw               = elec_max,
                cost_solar_cr_per_mwp     = cost_solar,
                cost_wind_cr_per_mw       = cost_wind,
                cost_elec_cr_per_mw       = cost_elec,
                cost_storage_cr_per_t     = cost_stor,
                cost_compressor_cr_per_mw = cost_comp,
                bop_pct                   = bop_pct,
                epc_pct                   = epc_pct,
                contingency_pct           = cont_pct,
                progress_callback         = progress_cb,
            )
            # Store profiles for dispatch tab
            results["solar_profile"] = solar_profile
            results["wind_profile"]  = wind_profile
            results["inputs"] = {
                "h2_annual_t": h2_annual_t,
                "min_flow":    min_flow,
                "max_flow":    max_flow,
                "op_days":     op_days,
                "hrs_per_day": hrs_per_day,
                "op_hours":    op_hours,
                "stack_mw":    stack_mw,
                "efficiency":  efficiency,
            }
            st.session_state.results = results

        progress_bar.progress(1.0)
        status_text.success("✅ Optimization Complete!")

    except Exception as e:
        st.error(f"Optimization failed: {e}")
        st.stop()


# ─────────────────────────────────────────────
#  RESULTS — TABS
# ─────────────────────────────────────────────
res = st.session_state.results

if res is None:
    # ── Welcome screen ──
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;color:#3a6080;">
      <div style="font-size:64px;">⚗</div>
      <div style="font-size:20px;color:#7fb0cc;margin:16px 0;font-family:monospace;letter-spacing:2px;">
        GH2 CAPEX OPTIMIZER
      </div>
      <div style="font-size:13px;line-height:2;max-width:600px;margin:0 auto;">
        1. Upload Solar and Wind hourly profiles (8760 rows, kWh/MW)<br>
        2. Enter demand specification and cost inputs in the sidebar<br>
        3. Click <b style="color:#00bfa8;">OPTIMISE CAPEX</b> to run the sizing engine<br><br>
        The optimizer will find the minimum CAPEX combination of<br>
        <b style="color:#ffd060;">Solar</b> + <b style="color:#29abff;">Wind</b> +
        <b style="color:#c084fc;">Electrolyzer</b> + <b style="color:#5ef090;">H₂ Storage</b>
        that meets your annual H₂ target with zero deficit hours.
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Unpack results ──
cfg      = res["best_config"]
capex    = res["final_capex"]
sim      = res["final_sim"]
sweep    = res["sweep_results"]
inp      = res["inputs"]
solar_p  = res["solar_profile"]
wind_p   = res["wind_profile"]

n_stacks = int(np.ceil(cfg["electrolyzer_mw"] / inp["stack_mw"]))

# Build hourly dataframe
hourly_df = build_hourly_df(sim, cfg["solar_mw"], cfg["wind_mw"])

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Results",
    "⚡ Dispatch",
    "₹ CAPEX Breakdown",
    "📈 Optimization Curve",
    "📋 Report",
])


# ══════════════════════════════════════════════
#  TAB 1 — RESULTS OVERVIEW
# ══════════════════════════════════════════════
with tab1:

    # ── Total CAPEX banner ──
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(255,181,32,0.08),rgba(0,191,168,0.05));
                border:1px solid rgba(255,181,32,0.35);
                padding:20px 28px; margin-bottom:24px;">
      <div style="font-size:11px;color:#ffb520;letter-spacing:3px;font-family:monospace;">
        MINIMUM PROJECT CAPEX
      </div>
      <div style="font-size:42px;font-weight:700;color:#ffb520;
                  text-shadow:0 0 24px rgba(255,181,32,0.4);font-family:monospace;">
        ₹ {capex['total_cr']:,.2f} <span style="font-size:18px;">Crores</span>
      </div>
      <div style="font-size:11px;color:#3a6080;margin-top:4px;">
        Includes Equipment + BOP + EPC + Contingency &nbsp;|&nbsp;
        Unit: ₹ {capex['total_cr']/inp['h2_annual_t']:.3f} Cr / (tH₂/yr)
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Optimized Capacities ──
    st.markdown("#### ⚙ Optimized Capacities")
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "☀ Solar PV",
        f"{cfg['solar_mw']:.0f} MWp",
        f"₹ {capex['cap_solar_cr']:.1f} Cr",
    )
    c2.metric(
        "💨 Wind",
        f"{cfg['wind_mw']:.0f} MW",
        f"₹ {capex['cap_wind_cr']:.1f} Cr",
    )
    c3.metric(
        "⚡ Electrolyzer",
        f"{cfg['electrolyzer_mw']:.0f} MW",
        f"{n_stacks} × {inp['stack_mw']} MW stacks",
    )
    c4.metric(
        "🔵 H₂ Storage",
        f"{cfg['storage_t']:.0f} tH₂",
        f"₹ {capex['cap_storage_cr']:.1f} Cr",
    )
    c5.metric(
        "⚙ Compressor",
        f"{cfg['compressor_mw']:.1f} MW",
        f"₹ {capex['cap_compressor_cr']:.1f} Cr",
    )

    st.divider()

    # ── Performance KPIs ──
    st.markdown("#### 📊 Performance Metrics")
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    k1.metric("H₂ Delivered",    f"{sim['annual_h2_delivered_t']:,.0f} t/yr",
              f"Target: {inp['h2_annual_t']:,} t")
    k2.metric("Deficit Hours",    f"{sim['deficit_hours']}",
              "✅ Zero" if sim['deficit_hours'] == 0 else "⚠ Non-zero")
    k3.metric("Elec. Utilization", f"{sim['elec_utilization_pct']:.1f}%")
    k4.metric("RE Self-Consump.", f"{sim['re_self_consumption_pct']:.1f}%")
    k5.metric("Curtailment",      f"{sim['annual_curtailment_mwh']:,.0f} MWh")
    k6.metric("Solar Gen.",       f"{sim['solar_gen_mwh']:,.0f} MWh",
              f"Wind: {sim['wind_gen_mwh']:,.0f} MWh")

    st.divider()

    # ── Constraint check ──
    st.markdown("#### ✓ Constraint Verification")

    constraints = [
        ("Annual H₂ Target Met",
         sim["annual_h2_delivered_t"] >= inp["h2_annual_t"] * 0.99,
         f"{sim['annual_h2_delivered_t']:,.0f} t vs {inp['h2_annual_t']:,} t target"),
        ("Zero Deficit Hours",
         sim["deficit_hours"] == 0,
         f"{sim['deficit_hours']} deficit hours"),
        ("Max Flow Respected",
         cfg["electrolyzer_mw"] * 1000 / inp["efficiency"] >= inp["max_flow"],
         f"Elec capacity: {cfg['electrolyzer_mw']*1000/inp['efficiency']:.0f} kg/hr"),
        ("Wind ≥ 50% of Solar",
         cfg["wind_mw"] >= cfg["solar_mw"] * 0.50,
         f"Wind {cfg['wind_mw']:.0f} MW ≥ {cfg['solar_mw']*0.5:.0f} MW floor"),
        ("Solar = Electrolyzer",
         abs(cfg["solar_mw"] - cfg["electrolyzer_mw"]) < 0.5,
         f"Solar {cfg['solar_mw']:.0f} MWp = Elec {cfg['electrolyzer_mw']:.0f} MW"),
        ("Electrolyzer Util. > 40%",
         sim["elec_utilization_pct"] > 40,
         f"{sim['elec_utilization_pct']:.1f}%"),
    ]

    cols = st.columns(3)
    for i, (label, ok, val) in enumerate(constraints):
        badge = "✅" if ok else "⚠️"
        with cols[i % 3]:
            st.markdown(f"""
            <div style="background:#0c1c2e;border:1px solid {'#183048' if ok else 'rgba(255,181,32,0.3)'};
                        padding:10px 14px;margin-bottom:8px;">
              <div style="font-size:10px;color:#3a6080;font-family:monospace;letter-spacing:1px;">{label}</div>
              <div style="font-size:13px;color:{'#5ef090' if ok else '#ffb520'};margin-top:4px;">
                {badge} {val}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Monthly H2 production chart ──
    st.divider()
    st.markdown("#### 📅 Monthly H₂ Production vs Demand")

    monthly_prod, monthly_del, monthly_solar, monthly_wind = [], [], [], []
    h = 0
    for days in MONTH_DAYS:
        hrs = days * 24
        monthly_prod.append(sum(sim["h2_produced"][h:h+hrs]) / 1000)
        monthly_del.append(sum(sim["h2_delivered"][h:h+hrs]) / 1000)
        monthly_solar.append(sum(solar_p[h:h+hrs] * cfg["solar_mw"]) / 1000)
        monthly_wind.append(sum(wind_p[h:h+hrs] * cfg["wind_mw"]) / 1000)
        h += hrs

    fig_monthly = make_subplots(specs=[[{"secondary_y": True}]])
    fig_monthly.add_trace(go.Bar(name="Solar Gen (GWh)",  x=MONTHS, y=monthly_solar,
                                  marker_color=COLORS["solar"], opacity=0.7))
    fig_monthly.add_trace(go.Bar(name="Wind Gen (GWh)",   x=MONTHS, y=monthly_wind,
                                  marker_color=COLORS["wind"],  opacity=0.7))
    fig_monthly.add_trace(go.Scatter(name="H₂ Produced (tH₂)", x=MONTHS,
                                      y=monthly_prod, mode="lines+markers",
                                      line=dict(color=COLORS["teal"], width=2),
                                      marker=dict(size=6)),
                           secondary_y=True)
    fig_monthly.add_trace(go.Scatter(name="H₂ Delivered (tH₂)", x=MONTHS,
                                      y=monthly_del, mode="lines+markers",
                                      line=dict(color=COLORS["demand"], width=2, dash="dash"),
                                      marker=dict(size=6)),
                           secondary_y=True)
    fig_monthly.update_layout(barmode="stack", **PLOTLY_LAYOUT, height=340,
                               title="Monthly RE Generation & H₂ Output")
    fig_monthly.update_yaxes(title_text="Generation (GWh)", secondary_y=False)
    fig_monthly.update_yaxes(title_text="H₂ (tonnes)", secondary_y=True)
    st.plotly_chart(fig_monthly, use_container_width=True)


# ══════════════════════════════════════════════
#  TAB 2 — DISPATCH
# ══════════════════════════════════════════════
with tab2:
    st.markdown("#### ⚡ Hourly Dispatch — Select Week")

    week_labels = {
        "Week 1  — Jan (Winter)":  0,
        "Week 14 — Apr (Spring)":  13,
        "Week 27 — Jul (Monsoon)": 26,
        "Week 40 — Oct (Autumn)":  39,
        "Week 52 — Dec (Winter)":  51,
    }
    selected_week_label = st.selectbox("Select Week", list(week_labels.keys()))
    week_start_h = week_labels[selected_week_label] * 168

    hours_168 = np.arange(168)
    labels_168 = [f"D{h//24+1} {h%24:02d}:00" for h in hours_168]

    sl = slice(week_start_h, week_start_h + 168)
    sp = solar_p[sl] * cfg["solar_mw"]
    wp = wind_p[sl]  * cfg["wind_mw"]

    # RE + Dispatch chart
    fig_dis = make_subplots(rows=3, cols=1, shared_xaxes=True,
                             row_heights=[0.4, 0.35, 0.25],
                             subplot_titles=[
                                 "RE Power & Electrolyzer Dispatch (MW)",
                                 "H₂ Produced vs Delivered (kg/hr)",
                                 "H₂ Storage Level (tH₂)",
                             ])

    fig_dis.add_trace(go.Bar(name="Solar MW", x=labels_168, y=sp,
                              marker_color=COLORS["solar"], opacity=0.7), row=1, col=1)
    fig_dis.add_trace(go.Bar(name="Wind MW",  x=labels_168, y=wp,
                              marker_color=COLORS["wind"],  opacity=0.6), row=1, col=1)
    fig_dis.add_trace(go.Scatter(name="Electrolyzer MW", x=labels_168,
                                  y=sim["elec_power"][sl],
                                  line=dict(color=COLORS["elec"], width=2),
                                  fill="tozeroy", fillcolor="rgba(192,132,252,0.1)"),
                       row=1, col=1)
    # Min load line
    min_load_mw = inp["stack_mw"] * min_load_pct
    fig_dis.add_hline(y=min_load_mw, line_dash="dot",
                       line_color="rgba(255,77,106,0.5)",
                       annotation_text=f"Min Load {min_load_mw:.1f} MW",
                       row=1, col=1)

    fig_dis.add_trace(go.Scatter(name="H₂ Produced", x=labels_168,
                                  y=sim["h2_produced"][sl],
                                  line=dict(color=COLORS["teal"], width=2)),
                       row=2, col=1)
    fig_dis.add_trace(go.Scatter(name="H₂ Delivered", x=labels_168,
                                  y=sim["h2_delivered"][sl],
                                  line=dict(color=COLORS["storage"], width=2, dash="dash")),
                       row=2, col=1)
    fig_dis.add_hline(y=inp["min_flow"], line_dash="dot",
                       line_color="rgba(255,77,106,0.4)",
                       annotation_text=f"Min Flow {inp['min_flow']} kg/hr",
                       row=2, col=1)
    fig_dis.add_hline(y=inp["max_flow"], line_dash="dot",
                       line_color="rgba(255,208,96,0.4)",
                       annotation_text=f"Max Flow {inp['max_flow']} kg/hr",
                       row=2, col=1)

    fig_dis.add_trace(go.Scatter(name="Storage (tH₂)", x=labels_168,
                                  y=sim["h2_storage"][sl] / 1000,
                                  line=dict(color=COLORS["storage"], width=2),
                                  fill="tozeroy", fillcolor="rgba(94,240,144,0.08)"),
                       row=3, col=1)
    fig_dis.add_hline(y=cfg["storage_t"] * 0.1, line_dash="dot",
                       line_color="rgba(255,77,106,0.4)",
                       annotation_text="Min Reserve",
                       row=3, col=1)

    fig_dis.update_layout(**PLOTLY_LAYOUT, height=680, barmode="stack",
                           showlegend=True,
                           title=f"Dispatch — {selected_week_label}")
    fig_dis.update_xaxes(tickangle=45, tickfont_size=9, nticks=28)
    st.plotly_chart(fig_dis, use_container_width=True)

    # Deficit check for this week
    week_deficit = np.sum(sim["deficit"][sl])
    if week_deficit > 0:
        st.warning(f"⚠ Deficit in this week: {week_deficit/1000:.2f} tH₂ across "
                   f"{np.sum(sim['deficit'][sl]>0)} hours")
    else:
        st.success("✅ No deficit in this week — consumer demand fully met")


# ══════════════════════════════════════════════
#  TAB 3 — CAPEX BREAKDOWN
# ══════════════════════════════════════════════
with tab3:
    st.markdown("#### ₹ CAPEX Breakdown — ₹ Crores")

    col_pie, col_bar = st.columns(2)

    labels  = ["Solar PV", "Wind", "Electrolyzer", "H₂ Storage", "Compressor", "BOP+Civil", "EPC", "Contingency"]
    values  = [
        capex["cap_solar_cr"],
        capex["cap_wind_cr"],
        capex["cap_elec_cr"],
        capex["cap_storage_cr"],
        capex["cap_compressor_cr"],
        capex["cap_bop_cr"],
        capex["cap_epc_cr"],
        capex["cap_contingency_cr"],
    ]
    colors_pie = [COLORS["solar"], COLORS["wind"], COLORS["elec"], COLORS["storage"],
                  "#60b0d0", "#3a5570", "#2a4560", "#1a3050"]

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=[round(v,2) for v in values],
            hole=0.52,
            marker=dict(colors=colors_pie, line=dict(color="#091422", width=2)),
            textinfo="label+percent",
            textfont=dict(size=10),
        ))
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=380,
                               title=f"Total: ₹ {capex['total_cr']:,.2f} Crores",
                               showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_bar = go.Figure(go.Bar(
            x=[round(v,2) for v in values], y=labels,
            orientation="h",
            marker=dict(color=colors_pie),
            text=[f"₹ {v:.2f} Cr" for v in values],
            textposition="outside",
            textfont=dict(size=10, color="#7fb0cc"),
        ))
        fig_bar.update_layout(**PLOTLY_LAYOUT, height=380,
                               title="CAPEX Waterfall (₹ Crores)",
                               xaxis_title="₹ Crores")
        st.plotly_chart(fig_bar, use_container_width=True)

    # Detailed table
    st.divider()
    st.markdown("#### Detailed CAPEX Table")

    capex_rows = [
        ["Solar PV",       f"{cfg['solar_mw']:.0f} MWp",        f"₹ {cost_solar} Cr/MWp",    f"₹ {capex['cap_solar_cr']:.2f}"],
        ["Wind Farm",      f"{cfg['wind_mw']:.0f} MW",           f"₹ {cost_wind} Cr/MW",      f"₹ {capex['cap_wind_cr']:.2f}"],
        ["Electrolyzer",   f"{cfg['electrolyzer_mw']:.0f} MW",   f"₹ {cost_elec} Cr/MW",      f"₹ {capex['cap_elec_cr']:.2f}"],
        ["H₂ Storage",     f"{cfg['storage_t']:.0f} tH₂",        f"₹ {cost_stor} Cr/tH₂",     f"₹ {capex['cap_storage_cr']:.2f}"],
        ["Compressor",     f"{cfg['compressor_mw']:.1f} MW",     f"₹ {cost_comp} Cr/MW",      f"₹ {capex['cap_compressor_cr']:.2f}"],
        ["Equipment Total","—",                                   "—",                          f"₹ {capex['equip_total_cr']:.2f}"],
        ["BOP + Civil",    f"{bop_pct*100:.0f}% of Equipment",   "—",                          f"₹ {capex['cap_bop_cr']:.2f}"],
        ["EPC Margin",     f"{epc_pct*100:.0f}% of Equipment",   "—",                          f"₹ {capex['cap_epc_cr']:.2f}"],
        ["Contingency",    f"{cont_pct*100:.0f}% of Equipment",  "—",                          f"₹ {capex['cap_contingency_cr']:.2f}"],
        ["TOTAL PROJECT",  "—",                                   "—",                          f"₹ {capex['total_cr']:.2f}"],
    ]
    df_capex = pd.DataFrame(capex_rows, columns=["Component", "Capacity", "Unit Rate", "CAPEX (₹ Crores)"])
    st.dataframe(df_capex, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
#  TAB 4 — OPTIMIZATION CURVE
# ══════════════════════════════════════════════
with tab4:
    st.markdown("#### 📈 CAPEX Optimization Curve — Inflection Point")
    st.caption("Total CAPEX vs Electrolyzer (= Solar) MW — shows the minimum CAPEX inflection point")

    if len(sweep) > 1:
        sw_df = pd.DataFrame(sweep)

        # Find optimal point
        opt_idx  = sw_df["total_capex_cr"].idxmin()
        opt_elec = sw_df.loc[opt_idx, "electrolyzer_mw"]
        opt_cap  = sw_df.loc[opt_idx, "total_capex_cr"]

        fig_sweep = go.Figure()
        fig_sweep.add_trace(go.Scatter(
            x=sw_df["electrolyzer_mw"], y=sw_df["total_capex_cr"],
            mode="lines+markers",
            name="Total CAPEX (₹ Cr)",
            line=dict(color=COLORS["teal"], width=2),
            marker=dict(size=5, color=COLORS["teal"]),
        ))
        # Highlight optimal
        fig_sweep.add_trace(go.Scatter(
            x=[opt_elec], y=[opt_cap],
            mode="markers",
            name=f"Optimal: {opt_elec:.0f} MW → ₹{opt_cap:.1f} Cr",
            marker=dict(size=16, color=COLORS["gold"], symbol="star"),
        ))
        fig_sweep.add_vline(x=opt_elec, line_dash="dot",
                             line_color="rgba(255,181,32,0.4)",
                             annotation_text=f"Optimal {opt_elec:.0f} MW")

        fig_sweep.update_layout(**PLOTLY_LAYOUT, height=380,
                                 title="Total CAPEX vs Electrolyzer MW (= Solar MWp)",
                                 xaxis_title="Electrolyzer MW (= Solar MWp)",
                                 yaxis_title="Total CAPEX (₹ Crores)")
        st.plotly_chart(fig_sweep, use_container_width=True)

        # Component breakdown across sweep
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Scatter(x=sw_df["electrolyzer_mw"],
                                       y=sw_df["electrolyzer_mw"] * cost_elec,
                                       name="Electrolyzer CAPEX",
                                       line=dict(color=COLORS["elec"], width=1.5, dash="dot")))
        fig_comp.add_trace(go.Scatter(x=sw_df["electrolyzer_mw"],
                                       y=sw_df["electrolyzer_mw"] * cost_solar,
                                       name="Solar CAPEX",
                                       line=dict(color=COLORS["solar"], width=1.5, dash="dot")))
        fig_comp.add_trace(go.Scatter(x=sw_df["electrolyzer_mw"],
                                       y=sw_df["wind_mw"] * cost_wind,
                                       name="Wind CAPEX",
                                       line=dict(color=COLORS["wind"], width=1.5, dash="dot")))
        fig_comp.add_trace(go.Scatter(x=sw_df["electrolyzer_mw"],
                                       y=sw_df["total_capex_cr"],
                                       name="Total CAPEX",
                                       line=dict(color=COLORS["teal"], width=2)))

        fig_comp.update_layout(**PLOTLY_LAYOUT, height=340,
                                title="Component CAPEX Breakdown Across Sweep",
                                xaxis_title="Electrolyzer MW",
                                yaxis_title="₹ Crores")
        st.plotly_chart(fig_comp, use_container_width=True)

        # Wind MW across sweep
        fig_wind = make_subplots(specs=[[{"secondary_y": True}]])
        fig_wind.add_trace(go.Scatter(x=sw_df["electrolyzer_mw"], y=sw_df["wind_mw"],
                                       name="Wind MW Required",
                                       line=dict(color=COLORS["wind"], width=2)),
                            secondary_y=False)
        fig_wind.add_trace(go.Scatter(x=sw_df["electrolyzer_mw"],
                                       y=sw_df["elec_util_pct"],
                                       name="Electrolyzer Utilization %",
                                       line=dict(color=COLORS["elec"], width=2, dash="dash")),
                            secondary_y=True)
        fig_wind.update_layout(**PLOTLY_LAYOUT, height=300,
                                title="Wind MW Required & Electrolyzer Utilization vs Electrolyzer Size")
        fig_wind.update_yaxes(title_text="Wind MW", secondary_y=False)
        fig_wind.update_yaxes(title_text="Electrolyzer Utilization (%)", secondary_y=True)
        st.plotly_chart(fig_wind, use_container_width=True)

    else:
        st.info("Only one configuration evaluated — increase the Elec MW sweep range to see the full curve.")


# ══════════════════════════════════════════════
#  TAB 5 — FULL REPORT
# ══════════════════════════════════════════════
with tab5:
    st.markdown("#### 📋 Full Sizing & CAPEX Report")

    report_data = {
        "Category": [],
        "Parameter": [],
        "Value": [],
        "Unit": [],
        "Remark": [],
    }

    def add_row(cat, param, val, unit, remark=""):
        report_data["Category"].append(cat)
        report_data["Parameter"].append(param)
        report_data["Value"].append(val)
        report_data["Unit"].append(unit)
        report_data["Remark"].append(remark)

    # Demand inputs
    add_row("Demand Inputs", "Annual H₂ Target",     f"{inp['h2_annual_t']:,}", "t/yr",     "Input")
    add_row("Demand Inputs", "Min H₂ Flow",          f"{inp['min_flow']}",      "kg/hr",    "Input")
    add_row("Demand Inputs", "Max H₂ Flow",          f"{inp['max_flow']}",      "kg/hr",    "Input")
    add_row("Demand Inputs", "Operating Days",       f"{inp['op_days']}",       "days/yr",  "Input")
    add_row("Demand Inputs", "Hours Per Day",        f"{inp['hrs_per_day']}",   "hrs/day",  "Input")
    add_row("Demand Inputs", "Yearly Op. Hours",     f"{inp['op_hours']:,}",    "hrs/yr",   "= Days × Hrs/Day")
    add_row("Demand Inputs", "Avg H₂ Flow",          f"{inp['h2_annual_t']*1000/inp['op_hours']:.0f}", "kg/hr", "= Annual ÷ Op Hours")
    add_row("Demand Inputs", "Plant Factor",         f"{inp['op_hours']/8760*100:.1f}", "%", "= Op Hours ÷ 8760")

    # Sizing logic
    add_row("Sizing Logic", "Solar = Electrolyzer", f"{cfg['solar_mw']:.0f} = {cfg['electrolyzer_mw']:.0f}", "MW", "Proportional constraint")
    add_row("Sizing Logic", "Wind Floor",           f"{cfg['solar_mw']*0.5:.0f}", "MW",   "= 50% of Solar")
    add_row("Sizing Logic", "Wind Installed",       f"{cfg['wind_mw']:.0f}",       "MW",   "Simulation-converged")
    add_row("Sizing Logic", "Min Load Threshold",   f"{inp['stack_mw']*min_load_pct:.1f}", "MW", f"= {min_load_pct*100:.0f}% of one stack")

    # Optimized capacities
    add_row("Optimized Capacities", "Solar PV",       f"{cfg['solar_mw']:.0f}",       "MWp",  f"= Electrolyzer MW")
    add_row("Optimized Capacities", "Wind Farm",      f"{cfg['wind_mw']:.0f}",        "MW",   f"≥ {cfg['solar_mw']*0.5:.0f} MW floor")
    add_row("Optimized Capacities", "Electrolyzer",   f"{cfg['electrolyzer_mw']:.0f}","MW",   f"{n_stacks} × {inp['stack_mw']} MW stacks")
    add_row("Optimized Capacities", "H₂ Storage",     f"{cfg['storage_t']:.0f}",      "tH₂",  "Min for zero deficit")
    add_row("Optimized Capacities", "Compressor",     f"{cfg['compressor_mw']:.1f}",  "MW",   "Sized for Max Flow")

    # Performance
    add_row("Performance", "H₂ Delivered",         f"{sim['annual_h2_delivered_t']:,.0f}", "t/yr",  "✅ Met" if sim['annual_h2_delivered_t'] >= inp['h2_annual_t']*0.99 else "⚠ Short")
    add_row("Performance", "Deficit Hours",         f"{sim['deficit_hours']}",              "hrs",   "✅ Zero" if sim['deficit_hours']==0 else "⚠ Check storage")
    add_row("Performance", "Electrolyzer Util.",    f"{sim['elec_utilization_pct']:.1f}",   "%",     "")
    add_row("Performance", "RE Self-Consumption",   f"{sim['re_self_consumption_pct']:.1f}","% ",    "")
    add_row("Performance", "Curtailment",           f"{sim['annual_curtailment_mwh']:,.0f}","MWh/yr","")
    add_row("Performance", "Solar Generation",      f"{sim['solar_gen_mwh']:,.0f}",         "MWh/yr","")
    add_row("Performance", "Wind Generation",       f"{sim['wind_gen_mwh']:,.0f}",          "MWh/yr","")

    # CAPEX
    add_row("CAPEX (₹ Crores)", "Solar PV",         f"{capex['cap_solar_cr']:.2f}",      "₹ Cr", f"₹{cost_solar} Cr/MWp")
    add_row("CAPEX (₹ Crores)", "Wind Farm",        f"{capex['cap_wind_cr']:.2f}",       "₹ Cr", f"₹{cost_wind} Cr/MW")
    add_row("CAPEX (₹ Crores)", "Electrolyzer",     f"{capex['cap_elec_cr']:.2f}",       "₹ Cr", f"₹{cost_elec} Cr/MW")
    add_row("CAPEX (₹ Crores)", "H₂ Storage",       f"{capex['cap_storage_cr']:.2f}",    "₹ Cr", f"₹{cost_stor} Cr/tH₂")
    add_row("CAPEX (₹ Crores)", "Compressor",       f"{capex['cap_compressor_cr']:.2f}", "₹ Cr", f"₹{cost_comp} Cr/MW")
    add_row("CAPEX (₹ Crores)", "Equipment Total",  f"{capex['equip_total_cr']:.2f}",    "₹ Cr", "")
    add_row("CAPEX (₹ Crores)", "BOP + Civil",      f"{capex['cap_bop_cr']:.2f}",        "₹ Cr", f"{bop_pct*100:.0f}% of Equipment")
    add_row("CAPEX (₹ Crores)", "EPC Margin",       f"{capex['cap_epc_cr']:.2f}",        "₹ Cr", f"{epc_pct*100:.0f}% of Equipment")
    add_row("CAPEX (₹ Crores)", "Contingency",      f"{capex['cap_contingency_cr']:.2f}","₹ Cr", f"{cont_pct*100:.0f}% of Equipment")
    add_row("CAPEX (₹ Crores)", "TOTAL PROJECT",    f"{capex['total_cr']:.2f}",          "₹ Cr", "OPTIMIZED MINIMUM")
    add_row("CAPEX (₹ Crores)", "Unit CAPEX",       f"{capex['total_cr']/inp['h2_annual_t']:.4f}", "₹ Cr/(tH₂/yr)", "")

    df_report = pd.DataFrame(report_data)
    st.dataframe(df_report, use_container_width=True, hide_index=True)

    # ── Download buttons ──
    st.divider()
    col_dl1, col_dl2 = st.columns(2)

    with col_dl1:
        csv_report = df_report.to_csv(index=False)
        st.download_button(
            label     = "⬇ Download Sizing Report (CSV)",
            data      = csv_report,
            file_name = "GH2_Optimiser_Sizing_Report.csv",
            mime      = "text/csv",
        )

    with col_dl2:
        # Hourly dispatch CSV
        dispatch_df = pd.DataFrame({
            "Hour":           np.arange(8760),
            "RE_Power_MW":    np.round(sim["re_power"], 2),
            "Elec_Power_MW":  np.round(sim["elec_power"], 2),
            "H2_Produced_kg": np.round(sim["h2_produced"], 1),
            "H2_Delivered_kg":np.round(sim["h2_delivered"], 1),
            "H2_Storage_kg":  np.round(sim["h2_storage"], 1),
            "Curtailment_MW": np.round(sim["curtailment_mw"], 2),
            "Deficit_kg":     np.round(sim["deficit"], 1),
        })
        csv_dispatch = dispatch_df.to_csv(index=False)
        st.download_button(
            label     = "⬇ Download Hourly Dispatch (CSV)",
            data      = csv_dispatch,
            file_name = "GH2_Optimiser_Dispatch.csv",
            mime      = "text/csv",
        )
