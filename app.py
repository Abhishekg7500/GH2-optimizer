"""
GH2 Optimiser v2 — Single File Streamlit App
Green Hydrogen Plant CAPEX Sizing & Optimization

Algorithm:
  1. Min Electrolyzer = Annual Target x Efficiency / Op Hours / 1000
  2. Solar = Electrolyzer (locked)
  3. Wind  = max(Energy Gap / Wind Profile Sum, Solar x 50%)
  4. Run 8760h simulation -> check annual H2 produced >= target
  5. If not met -> Electrolyzer += 1 MW -> repeat
  6. If met -> record CAPEX -> continue until CAPEX rises 5 steps in a row
  7. Pick minimum CAPEX configuration
  8. Storage = Avg Flow x Storage Days x 24 (deterministic, user defined)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="GH2 Optimiser",
    page_icon="⚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  LIGHT MODE CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  .stApp { background-color: #f0f4f8; color: #1a2b3c; }
  section[data-testid="stSidebar"] {
    background-color: #ffffff;
    border-right: 1px solid #dde3ea;
  }
  h1,h2,h3 { color: #0d6e5f !important; font-weight: 700; }
  h4,h5    { color: #2a5a7a !important; }

  [data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #dde3ea;
    border-top: 3px solid #0d6e5f;
    padding: 14px 16px;
    border-radius: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }
  [data-testid="metric-container"] label {
    color: #5a7a90 !important; font-size: 11px;
    letter-spacing: 0.5px; text-transform: uppercase;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #0d6e5f !important; font-size: 22px; font-weight: 700;
  }

  .stButton > button {
    background: linear-gradient(135deg,#0d6e5f,#0a5a4f);
    border: none; color: #ffffff; font-weight: 700;
    letter-spacing: 2px; text-transform: uppercase;
    width: 100%; padding: 14px; font-size: 13px;
    border-radius: 4px;
  }
  .stButton > button:hover { background: linear-gradient(135deg,#0a5a4f,#085040); }

  .stNumberInput input {
    background: #f8fafb; color: #1a2b3c;
    border: 1px solid #c8d8e8; border-radius: 4px;
  }
  hr { border-color: #dde3ea; }
  .stDataFrame { border: 1px solid #dde3ea; border-radius: 4px; }

  .derived-box {
    background: #e8f5f2; border: 1px solid #a0d4c8;
    border-left: 4px solid #0d6e5f; border-radius: 4px;
    padding: 12px 16px; font-family: monospace;
    font-size: 12px; color: #0d6e5f; margin: 10px 0; line-height: 1.9;
  }
  .sidebar-section {
    font-size: 11px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; color: #0d6e5f;
    margin: 16px 0 8px 0; padding-bottom: 4px;
    border-bottom: 2px solid #0d6e5f;
  }
  .capex-banner {
    background: linear-gradient(135deg,#fff8e6,#fffdf5);
    border: 1px solid #e8c84a; border-left: 5px solid #c89a00;
    border-radius: 6px; padding: 20px 28px; margin-bottom: 24px;
    box-shadow: 0 2px 8px rgba(200,154,0,0.1);
  }
  .constraint-card {
    background: #ffffff; border: 1px solid #dde3ea;
    border-radius: 4px; padding: 12px 14px; margin-bottom: 8px;
  }
  .c-ok   { border-left: 4px solid #0d6e5f; }
  .c-warn { border-left: 4px solid #c89a00; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PLOTLY THEME
# ─────────────────────────────────────────────
PL = dict(
    paper_bgcolor="#ffffff", plot_bgcolor="#f8fafb",
    font=dict(color="#3a5a70", family="Arial, sans-serif", size=11),
    xaxis=dict(gridcolor="#e8eef4", linecolor="#c8d8e8"),
    yaxis=dict(gridcolor="#e8eef4", linecolor="#c8d8e8"),
    legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor="#dde3ea", borderwidth=1),
    margin=dict(l=50, r=30, t=40, b=50),
)

C = dict(
    solar="#f5a623", wind="#2196f3", elec="#9c27b0",
    storage="#4caf50", demand="#f44336", teal="#0d6e5f", gold="#c89a00",
)

MONTHS     = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
MONTH_DAYS = [31,28,31,30,31,30,31,31,30,31,30,31]

if "results" not in st.session_state:
    st.session_state.results = None


# ══════════════════════════════════════════
#  CORE ENGINE FUNCTIONS
# ══════════════════════════════════════════

def calculate_capex(solar_mw, wind_mw, elec_mw, storage_t, comp_mw,
                    cs, cw, ce, cst, cc, bop, epc, cont):
    cap_s = solar_mw   * cs
    cap_w = wind_mw    * cw
    cap_e = elec_mw    * ce
    cap_t = storage_t  * cst
    cap_c = comp_mw    * cc
    eq    = cap_s + cap_w + cap_e + cap_t + cap_c
    total = eq * (1 + bop + epc + cont)
    return {
        "cap_solar_cr": cap_s, "cap_wind_cr": cap_w,
        "cap_elec_cr": cap_e,  "cap_storage_cr": cap_t,
        "cap_compressor_cr": cap_c, "equip_total_cr": eq,
        "cap_bop_cr": eq*bop, "cap_epc_cr": eq*epc,
        "cap_contingency_cr": eq*cont, "total_cr": total,
    }


def run_dispatch(sp, wp, solar_mw, wind_mw, elec_mw,
                 stack_mw, min_load_pct, eff,
                 min_flow, max_flow, stor_kg):
    N            = 8760
    min_load_mw  = stack_mw * min_load_pct
    min_flow     = float(min_flow)
    max_flow     = float(max_flow)
    stor_kg      = float(stor_kg)
    stor_level   = stor_kg * 0.5

    re_pw   = np.zeros(N); ep     = np.zeros(N)
    h2_pr   = np.zeros(N); h2_dl  = np.zeros(N)
    h2_st   = np.zeros(N); curt   = np.zeros(N)
    defic   = np.zeros(N); s_draw = np.zeros(N)
    s_chrg  = np.zeros(N)

    for h in range(N):
        re = float(sp[h]) * solar_mw + float(wp[h]) * wind_mw
        re_pw[h] = re

        if re >= min_load_mw:
            p         = min(re, elec_mw)
            ep[h]     = p
            h2_pr[h]  = p * 1000.0 / eff
            curt[h]   = max(0.0, re - elec_mw)
        else:
            curt[h]   = re

        # Direct supply capped at max flow
        direct = min(h2_pr[h], max_flow)

        # Excess beyond max flow goes to storage
        excess = h2_pr[h] - direct
        if excess > 0:
            space       = stor_kg - stor_level
            charged     = min(excess, space)
            stor_level += charged
            s_chrg[h]   = charged

        # If below min flow draw from storage
        if direct < min_flow:
            shortfall   = min_flow - direct
            draw        = min(shortfall, stor_level)
            direct     += draw
            stor_level  -= draw
            s_draw[h]   = draw

        # Record deficit if still below min flow after storage draw
        if direct < min_flow:
            defic[h] = min_flow - direct

        h2_dl[h] = direct
        h2_st[h] = stor_level

    sg = float(np.sum(sp * solar_mw))
    wg = float(np.sum(wp * wind_mw))
    rc = float(np.sum(ep))

    return {
        "re_power":       re_pw,  "elec_power":   ep,
        "h2_produced":    h2_pr,  "h2_delivered": h2_dl,
        "h2_storage":     h2_st,  "curtailment":  curt,
        "deficit":        defic,  "storage_draw": s_draw,
        "storage_charge": s_chrg,
        "annual_h2_produced_t":  float(np.sum(h2_pr)) / 1000.0,
        "annual_h2_delivered_t": float(np.sum(h2_dl)) / 1000.0,
        "deficit_hours":    int(np.sum(defic > 0)),
        "storage_draw_hrs": int(np.sum(s_draw > 0)),
        "elec_util_pct":    float(np.sum(ep > 0)) / N * 100.0,
        "curtailment_mwh":  float(np.sum(curt)),
        "solar_gen_mwh": sg, "wind_gen_mwh": wg,
        "re_self_consump": rc/(sg+wg)*100 if (sg+wg)>0 else 0.0,
    }


def get_wind_mw(h2_target_t, eff, solar_mw, sp, wp):
    energy_mwh  = h2_target_t * 1000.0 * eff / 1000.0
    solar_gen   = solar_mw * float(np.sum(sp))
    gap         = max(0.0, energy_mwh - solar_gen)
    wind_annual = float(np.sum(wp))
    w_gap       = gap / wind_annual if wind_annual > 0 else 0.0
    return max(1.0, round(max(w_gap, solar_mw * 0.5)))


def run_optimization(sp, wp, h2_target_t, min_flow, max_flow,
                     op_hours, stack_mw, eff, min_load_pct, stor_t,
                     cs, cw, ce, cst, cc, bop, epc, cont,
                     progress_cb=None):

    h2_kg    = h2_target_t * 1000.0
    avg_flow = h2_kg / op_hours
    stor_kg  = stor_t * 1000.0   # direct from user input in tH2
    comp_mw  = max(1.0, round(max_flow / 1000.0 * 0.055, 1))

    # Step 1 — minimum electrolyzer from annual math
    min_elec = np.ceil((h2_kg * eff) / (op_hours * 1000.0) / stack_mw) * stack_mw

    sweep        = []
    best_capex   = np.inf
    best_config  = None
    rising       = 0
    MAX_RISING   = 5
    elec_mw      = min_elec
    step         = 0

    while step < 500:
        step   += 1
        solar_mw = elec_mw
        wind_mw  = get_wind_mw(h2_target_t, eff, solar_mw, sp, wp)

        sim = run_dispatch(sp, wp, solar_mw, wind_mw, elec_mw,
                           stack_mw, min_load_pct, eff,
                           min_flow, max_flow, stor_kg)

        if sim["annual_h2_produced_t"] >= h2_target_t * 0.99:
            cap = calculate_capex(solar_mw, wind_mw, elec_mw,
                                  stor_t, comp_mw,
                                  cs, cw, ce, cst, cc, bop, epc, cont)
            sweep.append({
                "electrolyzer_mw": elec_mw, "solar_mw": solar_mw,
                "wind_mw": wind_mw, "total_capex_cr": cap["total_cr"],
                "h2_produced_t": sim["annual_h2_produced_t"],
                "elec_util_pct": sim["elec_util_pct"],
                "curtailment_mwh": sim["curtailment_mwh"],
                "deficit_hours": sim["deficit_hours"],
            })

            if progress_cb:
                progress_cb(step, elec_mw, wind_mw,
                            cap["total_cr"], sim["annual_h2_produced_t"])

            if cap["total_cr"] < best_capex:
                best_capex   = cap["total_cr"]
                best_config  = dict(electrolyzer_mw=elec_mw, solar_mw=solar_mw,
                                    wind_mw=wind_mw, storage_t=stor_t,
                                    compressor_mw=comp_mw)
                rising = 0
            else:
                rising += 1
                if rising >= MAX_RISING:
                    break

        elec_mw += 1.0

    if best_config is None:
        raise ValueError("No valid configuration found. Check RE profiles and demand inputs.")

    final_sim = run_dispatch(sp, wp,
                              best_config["solar_mw"], best_config["wind_mw"],
                              best_config["electrolyzer_mw"],
                              stack_mw, min_load_pct, eff,
                              min_flow, max_flow, stor_kg)

    final_cap = calculate_capex(best_config["solar_mw"], best_config["wind_mw"],
                                 best_config["electrolyzer_mw"], stor_t, comp_mw,
                                 cs, cw, ce, cst, cc, bop, epc, cont)

    return dict(best_config=best_config, final_capex=final_cap,
                final_sim=final_sim, sweep=sweep,
                min_elec_mw=min_elec, storage_t=stor_t, avg_flow=avg_flow)


def load_profile(f, name):
    try:
        df = pd.read_csv(f, header=None)
        for col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(s) >= 8760:
                arr = s.values[:8760].astype(np.float32)
                if arr.max() > 10:
                    arr = arr / arr.max()
                return arr
        st.error(f"❌ {name}: Need at least 8760 numeric rows.")
        return None
    except Exception as e:
        st.error(f"❌ {name}: {e}")
        return None


# ─────────────────────────────────────────────
#  SIDEBAR INPUTS
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗ GH2 Optimiser")
    st.caption("Green Hydrogen Plant Sizing · v2.0")
    st.divider()

    st.markdown('<div class="sidebar-section">☀ RE Profiles</div>', unsafe_allow_html=True)
    st.caption("8760 rows · single column · capacity factor (0 to 1)")
    solar_file = st.file_uploader("Solar Profile CSV", type=["csv"], key="solar")
    wind_file  = st.file_uploader("Wind Profile CSV",  type=["csv"], key="wind")

    st.divider()
    st.markdown('<div class="sidebar-section">⚗ H₂ Demand</div>', unsafe_allow_html=True)

    h2_annual = st.number_input("Annual H₂ Target (t/yr)", min_value=100, value=10000, step=500)
    c1, c2 = st.columns(2)
    with c1: min_flow = st.number_input("Min Flow (kg/hr)", min_value=0,  value=350,  step=50)
    with c2: max_flow = st.number_input("Max Flow (kg/hr)", min_value=1,  value=1900, step=50)
    c3, c4 = st.columns(2)
    with c3: op_days     = st.number_input("Op Days/yr",  min_value=1, max_value=365, value=345)
    with c4: hrs_per_day = st.number_input("Hrs/Day",     min_value=1, max_value=24,  value=24)

    op_hours = op_days * hrs_per_day
    avg_flow = h2_annual * 1000 / op_hours if op_hours > 0 else 0

    st.markdown(f"""<div class="derived-box">
    Op Hours     = {op_hours:,} hrs/yr<br>
    Avg Flow     = {avg_flow:.0f} kg/hr<br>
    Daily Target = {h2_annual/op_days:.1f} t/day<br>
    Plant Factor = {op_hours/8760*100:.1f}%
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sidebar-section">⚡ Electrolyzer</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5: stack_mw   = st.number_input("Stack MW",    min_value=1,    value=10,   step=1)
    with c6: efficiency = st.number_input("Eff kWh/kg",  min_value=38.0, value=52.0, step=0.5)
    c7, c8 = st.columns(2)
    with c7: min_load_pct = st.number_input("Min Load %",    min_value=5,  value=30, step=5) / 100.0
    with c8: availability = st.number_input("Availability %",min_value=70, value=95, step=1) / 100.0

    min_elec_calc    = h2_annual * 1000 * efficiency / (op_hours * 1000) if op_hours > 0 else 0
    min_elec_rounded = np.ceil(min_elec_calc / stack_mw) * stack_mw if stack_mw > 0 else 0

    st.markdown(f"""<div class="derived-box">
    Min Elec (calc)    = {min_elec_calc:.1f} MW<br>
    Min Elec (rounded) = {min_elec_rounded:.0f} MW<br>
    = {int(min_elec_rounded/stack_mw) if stack_mw>0 else 0} × {stack_mw} MW stacks<br>
    Sweep starts here ↑
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sidebar-section">🔵 H₂ Storage</div>', unsafe_allow_html=True)
    storage_t_input = st.number_input(
        "Storage Capacity (tH₂)",
        min_value=1.0, max_value=10000.0, value=100.0, step=10.0,
        help="Total H₂ storage tank capacity in tonnes"
    )
    st.markdown(f"""<div class="derived-box">
    Storage = {storage_t_input:.0f} tH₂<br>
    = {storage_t_input*1000:.0f} kg<br>
    ≈ {storage_t_input/(avg_flow*24/1000):.1f} days of avg demand
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sidebar-section">₹ Unit Costs (Crores)</div>', unsafe_allow_html=True)
    cost_solar = st.number_input("Solar (₹ Cr/MWp)",      value=3.5,  step=0.1, format="%.2f")
    cost_wind  = st.number_input("Wind (₹ Cr/MW)",         value=7.0,  step=0.1, format="%.2f")
    cost_elec  = st.number_input("Electrolyzer (₹ Cr/MW)", value=7.0,  step=0.1, format="%.2f")
    cost_stor  = st.number_input("Storage (₹ Cr/tH₂)",     value=0.55, step=0.05,format="%.2f")
    cost_comp  = st.number_input("Compressor (₹ Cr/MW)",   value=4.2,  step=0.1, format="%.2f")

    st.markdown("**Project Adders**")
    ca, cb, cc = st.columns(3)
    with ca: bop_pct  = st.number_input("BOP%",  value=12, step=1) / 100.0
    with cb: epc_pct  = st.number_input("EPC%",  value=8,  step=1) / 100.0
    with cc: cont_pct = st.number_input("Cont%", value=5,  step=1) / 100.0

    st.divider()
    run_btn = st.button("▶  OPTIMISE CAPEX", use_container_width=True)


# ─────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0d6e5f,#0a5a4f);
            border-radius:6px;padding:20px 28px 16px;
            margin-bottom:24px;box-shadow:0 2px 12px rgba(13,110,95,0.2);">
  <span style="font-size:26px;font-weight:900;color:#ffffff;letter-spacing:3px;">
    GH2 OPTIMISER
  </span>
  <span style="font-size:12px;color:rgba(255,255,255,0.7);margin-left:16px;
               font-family:monospace;letter-spacing:1px;">
    Green Hydrogen Plant · CAPEX Optimization Engine · ₹ Crores · v2.0
  </span>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────
if run_btn:
    errors = []
    if solar_file is None: errors.append("Solar profile CSV not uploaded.")
    if wind_file  is None: errors.append("Wind profile CSV not uploaded.")
    if min_flow > max_flow: errors.append("Min Flow must be ≤ Max Flow.")
    if avg_flow > max_flow: errors.append(f"Avg Flow ({avg_flow:.0f}) > Max Flow ({max_flow}). Reduce target or increase max flow.")
    if avg_flow < min_flow: errors.append(f"Avg Flow ({avg_flow:.0f}) < Min Flow ({min_flow}). Check inputs.")
    for e in errors: st.error(e)
    if errors: st.stop()

    sp_arr = load_profile(solar_file, "Solar")
    wp_arr = load_profile(wind_file,  "Wind")
    if sp_arr is None or wp_arr is None: st.stop()

    prog = st.progress(0.0)
    stat = st.empty()
    logc = st.empty()
    logs = []

    def pcb(step, elec_mw, wind_mw, capex_cr, h2_t):
        prog.progress(min(step/150, 1.0))
        stat.markdown(
            f"**Step {step}** | Elec = **{elec_mw:.0f} MW** | "
            f"Solar = **{elec_mw:.0f} MWp** | Wind = **{wind_mw:.0f} MW** | "
            f"H₂ = {h2_t:.0f} t | CAPEX = ₹ {capex_cr:.1f} Cr"
        )
        logs.append(f"[{step:>3}] Elec={elec_mw:.0f} Solar={elec_mw:.0f} "
                    f"Wind={wind_mw:.0f} H2={h2_t:.0f}t → ₹{capex_cr:.2f}Cr")
        logc.code("\n".join(logs[-10:]), language="")

    try:
        with st.spinner("Running optimization..."):
            res = run_optimization(
                sp=sp_arr, wp=wp_arr,
                h2_target_t=h2_annual, min_flow=float(min_flow),
                max_flow=float(max_flow), op_hours=float(op_hours),
                stack_mw=float(stack_mw), eff=float(efficiency),
                min_load_pct=float(min_load_pct), stor_t=float(storage_t_input),
                cs=float(cost_solar), cw=float(cost_wind), ce=float(cost_elec),
                cst=float(cost_stor), cc=float(cost_comp),
                bop=float(bop_pct), epc=float(epc_pct), cont=float(cont_pct),
                progress_cb=pcb,
            )
            res["sp"] = sp_arr; res["wp"] = wp_arr
            res["inp"] = dict(
                h2_annual=h2_annual, min_flow=min_flow, max_flow=max_flow,
                op_days=op_days, hrs_per_day=hrs_per_day, op_hours=op_hours,
                stack_mw=stack_mw, efficiency=efficiency,
                storage_t=storage_t_input,
            )
            st.session_state.results = res

        prog.progress(1.0)
        stat.success("✅ Optimization Complete — Minimum CAPEX Found!")
        logc.empty()

    except Exception as e:
        st.error(f"Optimization failed: {e}")
        st.stop()


# ─────────────────────────────────────────────
#  RESULTS TABS
# ─────────────────────────────────────────────
res = st.session_state.results

if res is None:
    st.markdown("""
    <div style="text-align:center;padding:80px 40px;background:#ffffff;
                border-radius:8px;border:1px solid #dde3ea;">
      <div style="font-size:56px;">⚗</div>
      <div style="font-size:22px;color:#0d6e5f;margin:16px 0;font-weight:700;">
        GH2 CAPEX OPTIMIZER
      </div>
      <div style="font-size:14px;color:#5a7a90;line-height:2.4;max-width:560px;margin:0 auto;">
        1. Upload Solar and Wind hourly profiles (CSV · 8760 rows · capacity factor 0–1)<br>
        2. Enter H₂ demand, electrolyzer specs and costs in the sidebar<br>
        3. Click <b style="color:#0d6e5f;">▶ OPTIMISE CAPEX</b><br><br>
        Finds minimum CAPEX combination of<br>
        <b style="color:#f5a623;">Solar</b> +
        <b style="color:#2196f3;">Wind</b> +
        <b style="color:#9c27b0;">Electrolyzer</b> +
        <b style="color:#4caf50;">H₂ Storage</b>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

cfg   = res["best_config"]
cap   = res["final_capex"]
sim   = res["final_sim"]
sweep = res["sweep"]
inp   = res["inp"]
sp    = res["sp"]
wp    = res["wp"]
n_st  = int(np.ceil(cfg["electrolyzer_mw"] / inp["stack_mw"]))

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Results", "⚡ Dispatch", "₹ CAPEX Breakdown",
    "📈 Optimization Curve", "📋 Report", "🗃 Hourly Data",
])


# ════════════════════════════════
#  TAB 1 — RESULTS
# ════════════════════════════════
with tab1:

    st.markdown(f"""
    <div class="capex-banner">
      <div style="font-size:11px;color:#8a6800;letter-spacing:3px;font-weight:700;">
        MINIMUM PROJECT CAPEX
      </div>
      <div style="font-size:44px;font-weight:800;color:#c89a00;font-family:monospace;margin:6px 0;">
        ₹ {cap['total_cr']:,.2f}
        <span style="font-size:20px;font-weight:500;">Crores</span>
      </div>
      <div style="font-size:12px;color:#8a6800;">
        Equipment + BOP + EPC + Contingency &nbsp;|&nbsp;
        Unit: ₹ {cap['total_cr']/inp['h2_annual']:.3f} Cr / (tH₂/yr)
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Sizing logic trace
    st.markdown("#### 🔢 How the Electrolyzer Was Sized")
    avg_f = inp["h2_annual"] * 1000 / inp["op_hours"]
    m_elec = inp["h2_annual"] * 1000 * inp["efficiency"] / (inp["op_hours"] * 1000)

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Avg H₂ Flow",       f"{avg_f:.0f} kg/hr",
              f"{inp['h2_annual']:,}t ÷ {inp['op_hours']:,}h")
    s2.metric("Min Elec (calc)",    f"{m_elec:.1f} MW",
              f"Avg Flow × {inp['efficiency']} ÷ 1000")
    s3.metric("Min Elec (rounded)", f"{res['min_elec_mw']:.0f} MW",
              f"⌈{m_elec:.1f} ÷ {inp['stack_mw']}⌉ × {inp['stack_mw']}")
    s4.metric("Optimal (min CAPEX)",f"{cfg['electrolyzer_mw']:.0f} MW",
              "After CAPEX sweep ↓")

    st.divider()
    st.markdown("#### ⚙ Optimized Capacities")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("☀ Solar PV",     f"{cfg['solar_mw']:.0f} MWp",  f"₹ {cap['cap_solar_cr']:.1f} Cr")
    c2.metric("💨 Wind",         f"{cfg['wind_mw']:.0f} MW",    f"₹ {cap['cap_wind_cr']:.1f} Cr")
    c3.metric("⚡ Electrolyzer", f"{cfg['electrolyzer_mw']:.0f} MW", f"{n_st} × {inp['stack_mw']} MW")
    c4.metric("🔵 H₂ Storage",   f"{cfg['storage_t']:.0f} tH₂", f"₹ {cap['cap_storage_cr']:.1f} Cr")
    c5.metric("⚙ Compressor",   f"{cfg['compressor_mw']:.1f} MW", f"₹ {cap['cap_compressor_cr']:.1f} Cr")

    st.divider()
    st.markdown("#### 📊 Performance")
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("H₂ Produced",     f"{sim['annual_h2_produced_t']:,.0f} t/yr",
              f"Target {inp['h2_annual']:,} t")
    k2.metric("H₂ Delivered",    f"{sim['annual_h2_delivered_t']:,.0f} t/yr")
    k3.metric("Deficit Hours",   f"{sim['deficit_hours']} hrs",
              "✅ Zero" if sim['deficit_hours']==0 else "⚠ Increase storage days")
    k4.metric("Storage Draw Hrs",f"{sim.get('storage_draw_hrs',0)} hrs",
              "Hours storage supplied consumer")
    k5.metric("Elec. Util.",     f"{sim['elec_util_pct']:.1f}%")
    k6.metric("Curtailment",     f"{sim['curtailment_mwh']:,.0f} MWh")

    st.divider()
    st.markdown("#### ✓ Constraint Checks")
    chk = [
        ("Annual H₂ Target Met",
         sim["annual_h2_produced_t"] >= inp["h2_annual"]*0.99,
         f"{sim['annual_h2_produced_t']:,.0f} t vs {inp['h2_annual']:,} t"),
        ("Zero Deficit Hours",
         sim["deficit_hours"]==0,
         f"{sim['deficit_hours']} hrs/yr"),
        ("Solar = Electrolyzer",
         abs(cfg["solar_mw"]-cfg["electrolyzer_mw"])<0.5,
         f"{cfg['solar_mw']:.0f} MWp = {cfg['electrolyzer_mw']:.0f} MW"),
        ("Wind ≥ 50% of Solar",
         cfg["wind_mw"] >= cfg["solar_mw"]*0.5,
         f"{cfg['wind_mw']:.0f} ≥ {cfg['solar_mw']*0.5:.0f} MW"),
        ("Elec capacity ≥ Max Flow",
         cfg["electrolyzer_mw"]*1000/inp["efficiency"] >= inp["max_flow"],
         f"{cfg['electrolyzer_mw']*1000/inp['efficiency']:.0f} kg/hr cap"),
        ("Elec Utilization > 40%",
         sim["elec_util_pct"]>40,
         f"{sim['elec_util_pct']:.1f}%"),
    ]
    cols = st.columns(3)
    for i,(lbl,ok,val) in enumerate(chk):
        with cols[i%3]:
            st.markdown(f"""
            <div class="constraint-card {'c-ok' if ok else 'c-warn'}">
              <div style="font-size:10px;color:#5a7a90;text-transform:uppercase;">{lbl}</div>
              <div style="font-size:13px;color:{'#0d6e5f' if ok else '#c07800'};
                          margin-top:4px;font-weight:600;">
                {'✅' if ok else '⚠️'} {val}
              </div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown("#### 📅 Monthly H₂ Production")
    mp,md,ms,mw = [],[],[],[]
    h = 0
    for days in MONTH_DAYS:
        hrs = days*24
        mp.append(np.sum(sim["h2_produced"][h:h+hrs])/1000)
        md.append(np.sum(sim["h2_delivered"][h:h+hrs])/1000)
        ms.append(np.sum(sp[h:h+hrs]*cfg["solar_mw"])/1000)
        mw.append(np.sum(wp[h:h+hrs]*cfg["wind_mw"])/1000)
        h += hrs

    fig_m = make_subplots(specs=[[{"secondary_y":True}]])
    fig_m.add_trace(go.Bar(name="Solar Gen (GWh)", x=MONTHS, y=ms,
                            marker_color=C["solar"], opacity=0.75))
    fig_m.add_trace(go.Bar(name="Wind Gen (GWh)",  x=MONTHS, y=mw,
                            marker_color=C["wind"],  opacity=0.75))
    fig_m.add_trace(go.Scatter(name="H₂ Produced (t)", x=MONTHS, y=mp,
                                mode="lines+markers",
                                line=dict(color=C["teal"],   width=2.5),
                                marker=dict(size=7)), secondary_y=True)
    fig_m.add_trace(go.Scatter(name="H₂ Delivered (t)", x=MONTHS, y=md,
                                mode="lines+markers",
                                line=dict(color=C["demand"], width=2, dash="dash"),
                                marker=dict(size=6)), secondary_y=True)
    fig_m.update_layout(**PL, barmode="stack", height=340,
                         title="Monthly RE Generation & H₂ Output")
    fig_m.update_yaxes(title_text="Generation (GWh)", secondary_y=False)
    fig_m.update_yaxes(title_text="H₂ (tonnes)",      secondary_y=True)
    st.plotly_chart(fig_m, use_container_width=True)


# ════════════════════════════════
#  TAB 2 — DISPATCH
# ════════════════════════════════
with tab2:
    st.markdown("#### ⚡ Hourly Dispatch")
    wk_opts = {
        "Week 1  — January":   0,
        "Week 14 — April":    13,
        "Week 27 — July":     26,
        "Week 40 — October":  39,
        "Week 52 — December": 51,
    }
    sel = st.selectbox("Select Week", list(wk_opts.keys()))
    wh  = wk_opts[sel]*168
    sl  = slice(wh, wh+168)
    lbl = [f"D{h//24+1} {h%24:02d}h" for h in range(168)]

    fig_d = make_subplots(rows=3, cols=1, shared_xaxes=True,
                           row_heights=[0.4,0.35,0.25],
                           subplot_titles=[
                               "RE Power & Electrolyzer (MW)",
                               "H₂ Produced vs Delivered (kg/hr)",
                               "H₂ Storage Level (tH₂)",
                           ])

    fig_d.add_trace(go.Bar(name="Solar MW", x=lbl, y=sp[sl]*cfg["solar_mw"],
                            marker_color=C["solar"], opacity=0.8), row=1,col=1)
    fig_d.add_trace(go.Bar(name="Wind MW",  x=lbl, y=wp[sl]*cfg["wind_mw"],
                            marker_color=C["wind"],  opacity=0.7), row=1,col=1)
    fig_d.add_trace(go.Scatter(name="Electrolyzer MW", x=lbl,
                                y=sim["elec_power"][sl],
                                line=dict(color=C["elec"],width=2),
                                fill="tozeroy",
                                fillcolor="rgba(156,39,176,0.08)"), row=1,col=1)
    fig_d.add_hline(y=inp["stack_mw"]*min_load_pct, line_dash="dot",
                     line_color="rgba(244,67,54,0.5)",
                     annotation_text=f"Min Load", row=1,col=1)

    fig_d.add_trace(go.Scatter(name="H₂ Produced", x=lbl,
                                y=sim["h2_produced"][sl],
                                line=dict(color=C["teal"],width=2)), row=2,col=1)
    fig_d.add_trace(go.Scatter(name="H₂ Delivered", x=lbl,
                                y=sim["h2_delivered"][sl],
                                line=dict(color=C["storage"],width=2,dash="dash")),
                     row=2,col=1)
    fig_d.add_hline(y=inp["min_flow"], line_dash="dot",
                     line_color="rgba(244,67,54,0.4)",
                     annotation_text=f"Min {inp['min_flow']} kg/hr", row=2,col=1)
    fig_d.add_hline(y=inp["max_flow"], line_dash="dot",
                     line_color="rgba(245,166,35,0.4)",
                     annotation_text=f"Max {inp['max_flow']} kg/hr", row=2,col=1)

    fig_d.add_trace(go.Scatter(name="Storage tH₂", x=lbl,
                                y=sim["h2_storage"][sl]/1000,
                                line=dict(color=C["storage"],width=2),
                                fill="tozeroy",
                                fillcolor="rgba(76,175,80,0.08)"), row=3,col=1)
    fig_d.add_hline(y=cfg["storage_t"], line_dash="dot",
                     line_color="rgba(200,150,0,0.4)",
                     annotation_text=f"Max {cfg['storage_t']:.0f} tH₂", row=3,col=1)

    fig_d.update_layout(**PL, height=680, barmode="stack", title=f"Dispatch — {sel}")
    fig_d.update_xaxes(tickangle=45, tickfont_size=8, nticks=28)
    st.plotly_chart(fig_d, use_container_width=True)

    wd = np.sum(sim["deficit"][sl])
    if wd > 0:
        st.warning(f"⚠ Deficit this week: {wd/1000:.2f} tH₂ across "
                   f"{int(np.sum(sim['deficit'][sl]>0))} hours. "
                   f"Increase Storage Days in sidebar.")
    else:
        st.success("✅ No deficit this week")


# ════════════════════════════════
#  TAB 3 — CAPEX BREAKDOWN
# ════════════════════════════════
with tab3:
    st.markdown("#### ₹ CAPEX Breakdown (₹ Crores)")
    cp1, cp2 = st.columns(2)

    lp  = ["Solar PV","Wind","Electrolyzer","H₂ Storage",
           "Compressor","BOP+Civil","EPC","Contingency"]
    vp  = [cap["cap_solar_cr"],cap["cap_wind_cr"],cap["cap_elec_cr"],
           cap["cap_storage_cr"],cap["cap_compressor_cr"],
           cap["cap_bop_cr"],cap["cap_epc_cr"],cap["cap_contingency_cr"]]
    clp = [C["solar"],C["wind"],C["elec"],C["storage"],
           "#00bcd4","#78909c","#546e7a","#37474f"]

    with cp1:
        fig_pie = go.Figure(go.Pie(
            labels=lp, values=[round(v,2) for v in vp], hole=0.52,
            marker=dict(colors=clp, line=dict(color="#ffffff",width=2)),
            textinfo="label+percent", textfont=dict(size=10),
        ))
        fig_pie.update_layout(**PL, height=380,
                               title=f"Total: ₹ {cap['total_cr']:,.2f} Cr",
                               showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with cp2:
        fig_bar = go.Figure(go.Bar(
            x=[round(v,2) for v in vp], y=lp, orientation="h",
            marker=dict(color=clp),
            text=[f"₹ {v:.1f} Cr" for v in vp], textposition="outside",
        ))
        fig_bar.update_layout(**PL, height=380,
                               title="CAPEX by Component",
                               xaxis_title="₹ Crores")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    rows_t = [
        ["Solar PV",     f"{cfg['solar_mw']:.0f} MWp",
         f"₹{cost_solar}/MWp",   f"₹ {cap['cap_solar_cr']:.2f} Cr"],
        ["Wind Farm",    f"{cfg['wind_mw']:.0f} MW",
         f"₹{cost_wind}/MW",     f"₹ {cap['cap_wind_cr']:.2f} Cr"],
        ["Electrolyzer", f"{cfg['electrolyzer_mw']:.0f} MW",
         f"₹{cost_elec}/MW",     f"₹ {cap['cap_elec_cr']:.2f} Cr"],
        ["H₂ Storage",   f"{cfg['storage_t']:.0f} tH₂",
         f"₹{cost_stor}/tH₂",    f"₹ {cap['cap_storage_cr']:.2f} Cr"],
        ["Compressor",   f"{cfg['compressor_mw']:.1f} MW",
         f"₹{cost_comp}/MW",     f"₹ {cap['cap_compressor_cr']:.2f} Cr"],
        ["Equipment Total","—","—",f"₹ {cap['equip_total_cr']:.2f} Cr"],
        ["BOP + Civil",  f"{int(bop_pct*100)}%","—",
         f"₹ {cap['cap_bop_cr']:.2f} Cr"],
        ["EPC Margin",   f"{int(epc_pct*100)}%","—",
         f"₹ {cap['cap_epc_cr']:.2f} Cr"],
        ["Contingency",  f"{int(cont_pct*100)}%","—",
         f"₹ {cap['cap_contingency_cr']:.2f} Cr"],
        ["TOTAL PROJECT","—","—",f"₹ {cap['total_cr']:.2f} Cr"],
    ]
    st.dataframe(pd.DataFrame(rows_t,
                               columns=["Component","Capacity","Rate","CAPEX"]),
                  use_container_width=True, hide_index=True)


# ════════════════════════════════
#  TAB 4 — OPTIMIZATION CURVE
# ════════════════════════════════
with tab4:
    st.markdown("#### 📈 CAPEX vs Electrolyzer MW — Inflection Point")

    if len(sweep) > 1:
        sw  = pd.DataFrame(sweep)
        opt = sw.loc[sw["total_capex_cr"].idxmin()]

        fig_sw = go.Figure()
        fig_sw.add_trace(go.Scatter(
            x=sw["electrolyzer_mw"], y=sw["total_capex_cr"],
            mode="lines+markers", name="Total CAPEX",
            line=dict(color=C["teal"],width=2.5), marker=dict(size=5),
        ))
        fig_sw.add_trace(go.Scatter(
            x=[opt["electrolyzer_mw"]], y=[opt["total_capex_cr"]],
            mode="markers",
            name=f"Optimal {opt['electrolyzer_mw']:.0f} MW → ₹{opt['total_capex_cr']:.1f} Cr",
            marker=dict(size=18,color=C["gold"],symbol="star"),
        ))
        fig_sw.add_vline(x=opt["electrolyzer_mw"], line_dash="dot",
                          line_color=C["gold"],
                          annotation_text=f"Optimal {opt['electrolyzer_mw']:.0f} MW")
        fig_sw.update_layout(**PL, height=380,
                              title="Total CAPEX vs Electrolyzer MW (= Solar MWp)",
                              xaxis_title="Electrolyzer MW",
                              yaxis_title="Total CAPEX (₹ Cr)")
        st.plotly_chart(fig_sw, use_container_width=True)

        fig_c2 = go.Figure()
        fig_c2.add_trace(go.Scatter(
            x=sw["electrolyzer_mw"],
            y=sw["electrolyzer_mw"]*(cost_elec+cost_solar),
            name="Electrolyzer + Solar",
            line=dict(color=C["elec"],width=2,dash="dot")))
        fig_c2.add_trace(go.Scatter(
            x=sw["electrolyzer_mw"],
            y=sw["wind_mw"]*cost_wind,
            name="Wind",
            line=dict(color=C["wind"],width=2,dash="dot")))
        fig_c2.add_trace(go.Scatter(
            x=sw["electrolyzer_mw"],
            y=sw["total_capex_cr"],
            name="Total CAPEX",
            line=dict(color=C["teal"],width=2.5)))
        fig_c2.update_layout(**PL, height=320,
                              title="Component CAPEX Breakdown",
                              xaxis_title="Electrolyzer MW",
                              yaxis_title="₹ Crores")
        st.plotly_chart(fig_c2, use_container_width=True)

        fig_w = make_subplots(specs=[[{"secondary_y":True}]])
        fig_w.add_trace(go.Scatter(x=sw["electrolyzer_mw"], y=sw["wind_mw"],
                                    name="Wind MW",
                                    line=dict(color=C["wind"],width=2)),
                         secondary_y=False)
        fig_w.add_trace(go.Scatter(x=sw["electrolyzer_mw"], y=sw["elec_util_pct"],
                                    name="Elec Util %",
                                    line=dict(color=C["elec"],width=2,dash="dash")),
                         secondary_y=True)
        fig_w.update_layout(**PL, height=300,
                             title="Wind MW & Electrolyzer Utilization vs Electrolyzer Size")
        fig_w.update_yaxes(title_text="Wind MW",      secondary_y=False)
        fig_w.update_yaxes(title_text="Util. (%)",    secondary_y=True)
        st.plotly_chart(fig_w, use_container_width=True)
    else:
        st.info("Only one configuration evaluated.")


# ════════════════════════════════
#  TAB 5 — REPORT
# ════════════════════════════════
with tab5:
    st.markdown("#### 📋 Full Sizing Report")
    avg_f2  = inp["h2_annual"]*1000/inp["op_hours"]
    m_elec2 = inp["h2_annual"]*1000*inp["efficiency"]/(inp["op_hours"]*1000)

    rw = []
    def r(cat,param,val,unit,rem=""):
        rw.append({"Category":cat,"Parameter":param,
                   "Value":val,"Unit":unit,"Remark":rem})

    r("Demand","Annual H₂ Target",   f"{inp['h2_annual']:,}",     "t/yr",  "Input")
    r("Demand","Min Flow",           f"{inp['min_flow']}",         "kg/hr", "Input")
    r("Demand","Max Flow",           f"{inp['max_flow']}",         "kg/hr", "Input")
    r("Demand","Op Days",            f"{inp['op_days']}",          "d/yr",  "Input")
    r("Demand","Hrs/Day",            f"{inp['hrs_per_day']}",      "h/d",   "Input")
    r("Demand","Op Hours",           f"{inp['op_hours']:,}",       "hrs/yr","= Days × Hrs")
    r("Demand","Avg Flow",           f"{avg_f2:.0f}",              "kg/hr", "= Annual ÷ Op Hours")

    r("Sizing","Min Elec (calc)",    f"{m_elec2:.1f}",             "MW",    "= Annual×Eff÷OpHrs÷1000")
    r("Sizing","Min Elec (rounded)", f"{res['min_elec_mw']:.0f}",  "MW",    "Rounded to stack size")
    r("Sizing","Optimal Elec",       f"{cfg['electrolyzer_mw']:.0f}","MW",  "Min CAPEX from sweep")
    r("Sizing","Solar = Elec",       f"{cfg['solar_mw']:.0f}",     "MWp",   "Locked equal")
    r("Sizing","Wind",               f"{cfg['wind_mw']:.0f}",      "MW",    "From energy gap")
    r("Sizing","Storage Capacity",    f"{cfg['storage_t']:.0f}",    "tH₂",   "User defined direct input")

    r("Performance","H₂ Produced",   f"{sim['annual_h2_produced_t']:,.0f}",  "t/yr","")
    r("Performance","H₂ Delivered",  f"{sim['annual_h2_delivered_t']:,.0f}", "t/yr","")
    r("Performance","Deficit Hours", f"{sim['deficit_hours']}",               "h/yr","")
    r("Performance","Elec Util",     f"{sim['elec_util_pct']:.1f}",           "%",  "")
    r("Performance","RE Self-Consump",f"{sim['re_self_consump']:.1f}",         "%",  "")
    r("Performance","Curtailment",   f"{sim['curtailment_mwh']:,.0f}",         "MWh/yr","")

    r("CAPEX","Solar",      f"{cap['cap_solar_cr']:.2f}",      "₹ Cr","")
    r("CAPEX","Wind",       f"{cap['cap_wind_cr']:.2f}",       "₹ Cr","")
    r("CAPEX","Electrolyzer",f"{cap['cap_elec_cr']:.2f}",      "₹ Cr","")
    r("CAPEX","H₂ Storage", f"{cap['cap_storage_cr']:.2f}",    "₹ Cr","")
    r("CAPEX","Compressor", f"{cap['cap_compressor_cr']:.2f}", "₹ Cr","")
    r("CAPEX","Equipment",  f"{cap['equip_total_cr']:.2f}",    "₹ Cr","")
    r("CAPEX","BOP",        f"{cap['cap_bop_cr']:.2f}",        "₹ Cr","")
    r("CAPEX","EPC",        f"{cap['cap_epc_cr']:.2f}",        "₹ Cr","")
    r("CAPEX","Contingency",f"{cap['cap_contingency_cr']:.2f}","₹ Cr","")
    r("CAPEX","TOTAL",      f"{cap['total_cr']:.2f}",          "₹ Cr","MINIMUM")
    r("CAPEX","Unit CAPEX", f"{cap['total_cr']/inp['h2_annual']:.4f}",
      "₹Cr/(tH₂/yr)","")

    df_r = pd.DataFrame(rw)
    st.dataframe(df_r, use_container_width=True, hide_index=True)

    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇ Download Sizing Report (CSV)",
                            df_r.to_csv(index=False),
                            "GH2_Report.csv","text/csv")
    with d2:
        # Build month/day/hour labels
        month_col, day_col, hour_col = [], [], []
        month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]
        month_days_list = [31,28,31,30,31,30,31,31,30,31,30,31]
        for mi, days in enumerate(month_days_list):
            for d in range(1, days+1):
                for hr in range(24):
                    month_col.append(month_names[mi])
                    day_col.append(d)
                    hour_col.append(hr)

        solar_gen_hr = np.round(sp * cfg["solar_mw"], 2)
        wind_gen_hr  = np.round(wp * cfg["wind_mw"],  2)
        total_gen_hr = np.round(solar_gen_hr + wind_gen_hr, 2)

        dd = pd.DataFrame({
            "Hour":              np.arange(8760),
            "Month":             month_col,
            "Day":               day_col,
            "Hour_of_Day":       hour_col,
            "Solar_Gen_MW":      solar_gen_hr,
            "Wind_Gen_MW":       wind_gen_hr,
            "Total_RE_Gen_MW":   total_gen_hr,
            "Elec_Power_MW":     np.round(sim["elec_power"],     2),
            "H2_Produced_kg":    np.round(sim["h2_produced"],    1),
            "H2_Delivered_kg":   np.round(sim["h2_delivered"],   1),
            "Storage_Draw_kg":   np.round(sim["storage_draw"],   1),
            "Storage_Charge_kg": np.round(sim["storage_charge"], 1),
            "H2_Storage_kg":     np.round(sim["h2_storage"],     1),
            "Curtailment_MW":    np.round(sim["curtailment"],     2),
            "Deficit_kg":        np.round(sim["deficit"],         1),
        })
        st.download_button("⬇ Download Hourly Dispatch (CSV)",
                            dd.to_csv(index=False),
                            "GH2_Dispatch.csv","text/csv")


# ════════════════════════════════
#  TAB 6 — HOURLY DATA VIEWER
# ════════════════════════════════
with tab6:
    st.markdown("#### 🗃 Hourly Data Viewer")
    st.caption("Browse all 8760 hours of simulation data directly in the app")

    # Build the full hourly dataframe
    month_names_hd   = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"]
    month_days_hd    = [31,28,31,30,31,30,31,31,30,31,30,31]
    month_col_hd, day_col_hd, hour_col_hd = [], [], []
    for mi, days in enumerate(month_days_hd):
        for d in range(1, days+1):
            for hr in range(24):
                month_col_hd.append(month_names_hd[mi])
                day_col_hd.append(d)
                hour_col_hd.append(hr)

    solar_gen_hd = np.round(sp * cfg["solar_mw"], 2)
    wind_gen_hd  = np.round(wp * cfg["wind_mw"],  2)
    total_gen_hd = np.round(solar_gen_hd + wind_gen_hd, 2)

    hourly_df = pd.DataFrame({
        "Hour":              np.arange(8760),
        "Month":             month_col_hd,
        "Day":               day_col_hd,
        "Hour_of_Day":       hour_col_hd,
        "Solar_Gen_MW":      solar_gen_hd,
        "Wind_Gen_MW":       wind_gen_hd,
        "Total_RE_Gen_MW":   total_gen_hd,
        "Elec_Power_MW":     np.round(sim["elec_power"],     2),
        "H2_Produced_kg":    np.round(sim["h2_produced"],    1),
        "H2_Delivered_kg":   np.round(sim["h2_delivered"],   1),
        "Storage_Draw_kg":   np.round(sim["storage_draw"],   1),
        "Storage_Charge_kg": np.round(sim["storage_charge"], 1),
        "H2_Storage_kg":     np.round(sim["h2_storage"],     1),
        "Curtailment_MW":    np.round(sim["curtailment"],     2),
        "Deficit_kg":        np.round(sim["deficit"],         1),
    })

    # ── Filter controls ──
    st.markdown("##### 🔍 Filter Data")
    fc1, fc2, fc3 = st.columns(3)

    with fc1:
        sel_months = st.multiselect(
            "Filter by Month",
            options=month_names_hd,
            default=month_names_hd,
        )

    with fc2:
        hour_range = st.slider(
            "Hour of Day",
            min_value=0, max_value=23, value=(0, 23),
        )

    with fc3:
        show_filter = st.selectbox(
            "Show Only",
            options=[
                "All Hours",
                "Storage Drawing Hours",
                "Deficit Hours",
                "Curtailment Hours",
                "Zero Production Hours",
                "Below Min Flow Hours",
            ]
        )

    # Apply filters
    filtered = hourly_df[hourly_df["Month"].isin(sel_months)]
    filtered = filtered[
        (filtered["Hour_of_Day"] >= hour_range[0]) &
        (filtered["Hour_of_Day"] <= hour_range[1])
    ]

    if show_filter == "Storage Drawing Hours":
        filtered = filtered[filtered["Storage_Draw_kg"] > 0]
    elif show_filter == "Deficit Hours":
        filtered = filtered[filtered["Deficit_kg"] > 0]
    elif show_filter == "Curtailment Hours":
        filtered = filtered[filtered["Curtailment_MW"] > 0]
    elif show_filter == "Zero Production Hours":
        filtered = filtered[filtered["H2_Produced_kg"] == 0]
    elif show_filter == "Below Min Flow Hours":
        filtered = filtered[filtered["H2_Delivered_kg"] < inp["min_flow"]]

    # ── Summary stats for filtered data ──
    st.markdown(f"**Showing {len(filtered):,} of 8,760 hours**")

    sm1, sm2, sm3, sm4, sm5 = st.columns(5)
    sm1.metric("Avg Solar Gen",    f"{filtered['Solar_Gen_MW'].mean():.1f} MW")
    sm2.metric("Avg Wind Gen",     f"{filtered['Wind_Gen_MW'].mean():.1f} MW")
    sm3.metric("Avg H₂ Produced",  f"{filtered['H2_Produced_kg'].mean():.0f} kg/hr")
    sm4.metric("Avg H₂ Delivered", f"{filtered['H2_Delivered_kg'].mean():.0f} kg/hr")
    sm5.metric("Total Deficit",    f"{filtered['Deficit_kg'].sum()/1000:.1f} tH₂")

    st.divider()

    # ── Column selector ──
    all_cols = hourly_df.columns.tolist()
    default_cols = ["Hour","Month","Day","Hour_of_Day",
                    "Solar_Gen_MW","Wind_Gen_MW","Total_RE_Gen_MW",
                    "Elec_Power_MW","H2_Produced_kg","H2_Delivered_kg",
                    "Storage_Draw_kg","H2_Storage_kg","Deficit_kg"]

    selected_cols = st.multiselect(
        "Select Columns to Display",
        options=all_cols,
        default=default_cols,
    )

    # ── Data table ──
    st.dataframe(
        filtered[selected_cols].reset_index(drop=True),
        use_container_width=True,
        height=500,
    )

    st.divider()

    # ── Quick charts from filtered data ──
    st.markdown("##### 📊 Quick Charts")
    cc1, cc2 = st.columns(2)

    with cc1:
        # RE generation breakdown for filtered hours
        fig_re = go.Figure()
        fig_re.add_trace(go.Scatter(
            x=filtered["Hour"], y=filtered["Solar_Gen_MW"],
            name="Solar MW", mode="lines",
            line=dict(color=C["solar"], width=1),
            fill="tozeroy", fillcolor="rgba(245,166,35,0.15)",
        ))
        fig_re.add_trace(go.Scatter(
            x=filtered["Hour"],
            y=filtered["Solar_Gen_MW"] + filtered["Wind_Gen_MW"],
            name="Solar + Wind MW", mode="lines",
            line=dict(color=C["wind"], width=1),
            fill="tonexty", fillcolor="rgba(33,150,243,0.15)",
        ))
        fig_re.update_layout(**PL, height=280,
                              title="Solar & Wind Generation (MW)",
                              xaxis_title="Hour", yaxis_title="MW")
        st.plotly_chart(fig_re, use_container_width=True)

    with cc2:
        # H2 produced vs delivered
        fig_h2 = go.Figure()
        fig_h2.add_trace(go.Scatter(
            x=filtered["Hour"], y=filtered["H2_Produced_kg"],
            name="H₂ Produced", mode="lines",
            line=dict(color=C["teal"], width=1),
        ))
        fig_h2.add_trace(go.Scatter(
            x=filtered["Hour"], y=filtered["H2_Delivered_kg"],
            name="H₂ Delivered", mode="lines",
            line=dict(color=C["demand"], width=1, dash="dash"),
        ))
        fig_h2.add_hline(y=inp["min_flow"], line_dash="dot",
                          line_color="rgba(244,67,54,0.5)",
                          annotation_text=f"Min {inp['min_flow']} kg/hr")
        fig_h2.update_layout(**PL, height=280,
                              title="H₂ Produced vs Delivered (kg/hr)",
                              xaxis_title="Hour", yaxis_title="kg/hr")
        st.plotly_chart(fig_h2, use_container_width=True)

    # Storage level chart
    fig_stor = go.Figure()
    fig_stor.add_trace(go.Scatter(
        x=filtered["Hour"], y=filtered["H2_Storage_kg"]/1000,
        name="Storage Level (tH₂)", mode="lines",
        line=dict(color=C["storage"], width=1.5),
        fill="tozeroy", fillcolor="rgba(76,175,80,0.10)",
    ))
    fig_stor.add_hline(y=cfg["storage_t"], line_dash="dot",
                        line_color="rgba(200,150,0,0.5)",
                        annotation_text=f"Capacity {cfg['storage_t']:.0f} tH₂")
    fig_stor.update_layout(**PL, height=250,
                            title="H₂ Storage Level (tH₂)",
                            xaxis_title="Hour", yaxis_title="tH₂")
    st.plotly_chart(fig_stor, use_container_width=True)
