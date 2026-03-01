"""
sizing.py
---------
Iterative sizing engine for the GH2 Optimizer.

Algorithm:
  OUTER LOOP: Electrolyzer MW (1 MW increments)
    Solar_MW = Electrolyzer_MW  (proportional, locked)
    Wind_floor = Solar_MW × 0.50

    Wind_start = max(energy_gap_calc, wind_floor)

    INNER LOOP: Wind MW from Wind_start upward (1 MW increments)
      Run 8760h dispatch simulation
      Check: Annual H2 Delivered >= Target AND deficit == 0
      If yes → record CAPEX, break inner loop
      If no  → Wind += 1 MW

  After outer loop → Pick (Electrolyzer, Solar, Wind) with minimum CAPEX
  Then run storage sizing pass → find minimum storage for zero deficit
"""

import numpy as np
from simulator.dispatch import run_dispatch
from economics.capex import calculate_capex


def calculate_wind_start(
    annual_h2_target_kg: float,
    efficiency_kwh_per_kg: float,
    solar_mw: float,
    solar_profile: np.ndarray,
    wind_profile: np.ndarray,
    wind_floor_mw: float,
) -> float:
    """
    Calculate the starting point for Wind MW based on energy gap.

    Wind_start = max(
        (Total Energy Needed - Solar Generation) / Wind Annual kWh per MW,
        Wind floor (50% of Solar)
    )
    """
    # Energy in MWh (profiles are in MWh per MW per hour, so sum = MWh/MW/yr)
    total_energy_needed_mwh = annual_h2_target_kg * efficiency_kwh_per_kg / 1000.0

    annual_solar_mwh_per_mw = float(np.sum(solar_profile))  # MWh/MW/yr
    annual_wind_mwh_per_mw  = float(np.sum(wind_profile))   # MWh/MW/yr

    solar_generation_mwh = solar_mw * annual_solar_mwh_per_mw

    energy_gap_mwh = max(0.0, total_energy_needed_mwh - solar_generation_mwh)

    if annual_wind_mwh_per_mw > 0:
        wind_from_gap = energy_gap_mwh / annual_wind_mwh_per_mw
    else:
        wind_from_gap = 0.0

    wind_start = max(wind_from_gap, wind_floor_mw)

    return max(1.0, round(wind_start))  # At least 1 MW


def find_minimum_storage(
    solar_profile: np.ndarray,
    wind_profile: np.ndarray,
    solar_mw: float,
    wind_mw: float,
    electrolyzer_mw: float,
    stack_mw: float,
    min_load_pct: float,
    efficiency_kwh_per_kg: float,
    min_flow_kg_hr: float,
    max_flow_kg_hr: float,
    annual_h2_target_kg: float,
    storage_step_t: float = 10.0,   # Increment in tonnes
    max_storage_t: float = 50000.0, # Upper bound safety cap
    progress_callback=None,
) -> tuple:
    """
    Find the minimum H2 storage (tH2) that results in:
      1. Zero deficit hours
      2. Annual H2 delivered >= target

    Increments storage by storage_step_t tonnes at a time.

    Returns (storage_tonnes, simulation_result)
    """
    storage_t = 0.0

    while storage_t <= max_storage_t:
        storage_kg = storage_t * 1000.0

        result = run_dispatch(
            solar_profile        = solar_profile,
            wind_profile         = wind_profile,
            solar_mw             = solar_mw,
            wind_mw              = wind_mw,
            electrolyzer_mw      = electrolyzer_mw,
            stack_mw             = stack_mw,
            min_load_pct         = min_load_pct,
            efficiency_kwh_per_kg= efficiency_kwh_per_kg,
            min_flow_kg_hr       = min_flow_kg_hr,
            max_flow_kg_hr       = max_flow_kg_hr,
            storage_capacity_kg  = storage_kg,
        )

        if (
            result["deficit_hours"] == 0
            and result["annual_h2_delivered_t"] >= annual_h2_target_kg / 1000.0 * 0.99
        ):
            return storage_t, result

        storage_t += storage_step_t

        if progress_callback:
            progress_callback(storage_t)

    # Return best found even if not perfect
    return storage_t, result


def run_sizing_optimization(
    solar_profile: np.ndarray,
    wind_profile: np.ndarray,
    # Demand inputs
    annual_h2_target_t: float,      # tonnes/year
    min_flow_kg_hr: float,          # kg/hr
    max_flow_kg_hr: float,          # kg/hr
    op_days: int,                   # days/year
    hrs_per_day: float,             # hours/day
    # Electrolyzer config
    stack_mw: float,                # MW per stack
    efficiency_kwh_per_kg: float,   # kWh/kgH2
    min_load_pct: float,            # 0.30
    availability: float,            # 0.95
    # Search bounds
    elec_min_mw: float,             # Start of outer loop
    elec_max_mw: float,             # End of outer loop
    wind_step_mw: float = 1.0,      # Wind inner loop step
    # Cost inputs (₹ Crores)
    cost_solar_cr_per_mwp: float    = 3.5,
    cost_wind_cr_per_mw: float      = 7.0,
    cost_elec_cr_per_mw: float      = 7.0,
    cost_storage_cr_per_t: float    = 0.55,
    cost_compressor_cr_per_mw: float= 4.2,
    bop_pct: float                  = 0.12,
    epc_pct: float                  = 0.08,
    contingency_pct: float          = 0.05,
    # Progress callback
    progress_callback=None,
) -> dict:
    """
    Main sizing optimization loop.

    Returns dictionary with:
      - best configuration (Elec MW, Solar MW, Wind MW)
      - storage tonnes (from separate pass)
      - CAPEX breakdown
      - Full simulation result
      - All sweep results for plotting
    """

    annual_h2_target_kg = annual_h2_target_t * 1000.0
    op_hours            = op_days * hrs_per_day

    sweep_results = []   # All valid configurations for plotting
    best_capex    = np.inf
    best_config   = None

    # Outer loop step = 1 stack at a time
    elec_step = stack_mw

    total_steps = int((elec_max_mw - elec_min_mw) / elec_step) + 1
    step_count  = 0

    elec_mw = elec_min_mw

    while elec_mw <= elec_max_mw:

        step_count += 1

        # ── Solar = Electrolyzer (proportional, locked) ──
        solar_mw   = elec_mw
        wind_floor = solar_mw * 0.50   # 50% of solar

        # ── Wind starting point ──
        wind_start = calculate_wind_start(
            annual_h2_target_kg   = annual_h2_target_kg,
            efficiency_kwh_per_kg = efficiency_kwh_per_kg,
            solar_mw              = solar_mw,
            solar_profile         = solar_profile,
            wind_profile          = wind_profile,
            wind_floor_mw         = wind_floor,
        )

        # ── Inner loop: Wind MW ──
        wind_mw         = wind_start
        inner_converged = False
        max_wind_tries  = 500   # Safety cap

        for _ in range(max_wind_tries):

            # Quick check without storage to see if annual target is reachable
            # Use a large storage so storage never limits production
            result = run_dispatch(
                solar_profile         = solar_profile,
                wind_profile          = wind_profile,
                solar_mw              = solar_mw,
                wind_mw               = wind_mw,
                electrolyzer_mw       = elec_mw,
                stack_mw              = stack_mw,
                min_load_pct          = min_load_pct,
                efficiency_kwh_per_kg = efficiency_kwh_per_kg,
                min_flow_kg_hr        = min_flow_kg_hr,
                max_flow_kg_hr        = max_flow_kg_hr,
                storage_capacity_kg   = 1e9,  # Unlimited storage for production check
            )

            if result["annual_h2_delivered_t"] >= annual_h2_target_t * 0.99:
                inner_converged = True
                break

            wind_mw += wind_step_mw

        if not inner_converged:
            elec_mw += elec_step
            continue

        # ── Compressor MW: sized for max H2 flow ──
        compressor_mw = max(1.0, round(max_flow_kg_hr / 1000.0 * 0.055, 1))

        # ── Estimate storage (rough) for CAPEX sweep ──
        # Use 2-day buffer as placeholder — exact storage found after best config selected
        avg_flow_kg_hr   = annual_h2_target_kg / op_hours
        rough_storage_t  = max(10.0, round(avg_flow_kg_hr * 48 / 1000.0 / 10) * 10)

        # ── Calculate CAPEX for this configuration ──
        capex = calculate_capex(
            solar_mw             = solar_mw,
            wind_mw              = wind_mw,
            electrolyzer_mw      = elec_mw,
            storage_t            = rough_storage_t,
            compressor_mw        = compressor_mw,
            cost_solar           = cost_solar_cr_per_mwp,
            cost_wind            = cost_wind_cr_per_mw,
            cost_elec            = cost_elec_cr_per_mw,
            cost_storage         = cost_storage_cr_per_t,
            cost_compressor      = cost_compressor_cr_per_mw,
            bop_pct              = bop_pct,
            epc_pct              = epc_pct,
            contingency_pct      = contingency_pct,
        )

        sweep_results.append({
            "electrolyzer_mw":  elec_mw,
            "solar_mw":         solar_mw,
            "wind_mw":          wind_mw,
            "storage_t":        rough_storage_t,
            "compressor_mw":    compressor_mw,
            "total_capex_cr":   capex["total_cr"],
            "capex_breakdown":  capex,
            "h2_delivered_t":   result["annual_h2_delivered_t"],
            "elec_util_pct":    result["elec_utilization_pct"],
            "curtailment_mwh":  result["annual_curtailment_mwh"],
        })

        if progress_callback:
            progress_callback(step_count, total_steps, elec_mw, capex["total_cr"])

        if capex["total_cr"] < best_capex:
            best_capex  = capex["total_cr"]
            best_config = {
                "electrolyzer_mw": elec_mw,
                "solar_mw":        solar_mw,
                "wind_mw":         wind_mw,
                "compressor_mw":   compressor_mw,
            }

        elec_mw += elec_step

    if best_config is None:
        raise ValueError(
            "No valid configuration found. "
            "Check demand inputs and RE profiles."
        )

    # ── Storage sizing pass ──
    # Now find exact minimum storage for zero deficit at best config
    storage_t, final_sim = find_minimum_storage(
        solar_profile         = solar_profile,
        wind_profile          = wind_profile,
        solar_mw              = best_config["solar_mw"],
        wind_mw               = best_config["wind_mw"],
        electrolyzer_mw       = best_config["electrolyzer_mw"],
        stack_mw              = stack_mw,
        min_load_pct          = min_load_pct,
        efficiency_kwh_per_kg = efficiency_kwh_per_kg,
        min_flow_kg_hr        = min_flow_kg_hr,
        max_flow_kg_hr        = max_flow_kg_hr,
        annual_h2_target_kg   = annual_h2_target_kg,
    )

    best_config["storage_t"] = storage_t

    # ── Final CAPEX with correct storage ──
    final_capex = calculate_capex(
        solar_mw         = best_config["solar_mw"],
        wind_mw          = best_config["wind_mw"],
        electrolyzer_mw  = best_config["electrolyzer_mw"],
        storage_t        = storage_t,
        compressor_mw    = best_config["compressor_mw"],
        cost_solar       = cost_solar_cr_per_mwp,
        cost_wind        = cost_wind_cr_per_mw,
        cost_elec        = cost_elec_cr_per_mw,
        cost_storage     = cost_storage_cr_per_t,
        cost_compressor  = cost_compressor_cr_per_mw,
        bop_pct          = bop_pct,
        epc_pct          = epc_pct,
        contingency_pct  = contingency_pct,
    )

    return {
        "best_config":    best_config,
        "final_capex":    final_capex,
        "final_sim":      final_sim,
        "sweep_results":  sweep_results,
        "op_hours":       op_hours,
        "annual_h2_target_t": annual_h2_target_t,
    }
