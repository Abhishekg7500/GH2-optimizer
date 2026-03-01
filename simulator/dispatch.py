"""
dispatch.py
-----------
Core 8760-hour hourly dispatch simulation engine for the GH2 Optimizer.

For a given set of installed capacities (Electrolyzer, Solar, Wind, Storage),
this module simulates every hour of the year and returns:
  - H₂ produced each hour
  - H₂ delivered to consumer each hour
  - H₂ storage level each hour
  - Deficit hours and total deficit
  - Annual H₂ delivered
  - Curtailment
"""

import numpy as np


def run_dispatch(
    solar_profile: np.ndarray,   # 8760 values, kWh per MW per hour
    wind_profile: np.ndarray,    # 8760 values, kWh per MW per hour
    solar_mw: float,             # Solar installed capacity (MWp)
    wind_mw: float,              # Wind installed capacity (MW)
    electrolyzer_mw: float,      # Electrolyzer rated capacity (MW)
    stack_mw: float,             # Single stack size (MW)
    min_load_pct: float,         # Minimum load as fraction (0.30 = 30%)
    efficiency_kwh_per_kg: float,# Electrolyzer efficiency (kWh per kgH2)
    min_flow_kg_hr: float,       # Consumer min offtake (kg/hr)
    max_flow_kg_hr: float,       # Consumer max offtake (kg/hr)
    storage_capacity_kg: float,  # H2 storage tank capacity (kg)
    initial_storage_kg: float = None,  # Starting storage level (kg), default 50%
) -> dict:
    """
    Run a full 8760-hour dispatch simulation.

    Returns a dictionary with all hourly arrays and summary metrics.
    """

    N = 8760

    # Starting storage level — default 50% of capacity
    if initial_storage_kg is None:
        initial_storage_kg = storage_capacity_kg * 0.50

    # Minimum load threshold in MW (30% of one stack)
    min_load_mw = stack_mw * min_load_pct

    # Pre-allocate output arrays
    re_power          = np.zeros(N)   # Total RE power available (MW)
    elec_power        = np.zeros(N)   # Electrolyzer power consumed (MW)
    h2_produced       = np.zeros(N)   # H2 produced (kg/hr)
    h2_delivered      = np.zeros(N)   # H2 delivered to consumer (kg/hr)
    h2_storage        = np.zeros(N)   # H2 storage level at end of hour (kg)
    curtailment_mw    = np.zeros(N)   # Curtailed RE power (MW)
    storage_draw      = np.zeros(N)   # H2 drawn from storage (kg/hr)
    storage_charge    = np.zeros(N)   # H2 added to storage (kg/hr)
    deficit           = np.zeros(N)   # Unmet consumer demand (kg/hr)

    storage_level = initial_storage_kg  # Rolling storage level (kg)

    for h in range(N):

        # ── Step 1: Calculate total RE power available this hour ──
        solar_power = solar_profile[h] * solar_mw   # MW (kWh/MW × MW = kWh = MWh for 1hr)
        wind_power  = wind_profile[h]  * wind_mw
        total_re    = solar_power + wind_power
        re_power[h] = total_re

        # ── Step 2: Electrolyzer dispatch ──
        # Only runs if RE >= minimum load threshold (30% of one stack)
        if total_re >= min_load_mw:
            ep = min(total_re, electrolyzer_mw)   # Can't exceed rated capacity
            elec_power[h]  = ep
            h2_prod        = ep * 1000.0 / efficiency_kwh_per_kg  # kg/hr
            h2_produced[h] = h2_prod

            # Curtailment = RE that electrolyzer could not consume
            curtailment_mw[h] = max(0.0, total_re - electrolyzer_mw)
        else:
            # RE below minimum load → electrolyzer OFF
            elec_power[h]     = 0.0
            h2_produced[h]    = 0.0
            curtailment_mw[h] = total_re  # All RE wasted (too low to use)

        # ── Step 3: Supply to consumer ──
        # Direct supply = H2 produced this hour, capped at Max Flow
        direct_supply = min(h2_produced[h], max_flow_kg_hr)

        # Excess H2 beyond max flow → goes to storage
        excess_to_storage = h2_produced[h] - direct_supply

        # ── Step 4: Check if direct supply meets Min Flow ──
        shortfall = max_flow_kg_hr  # We will determine actual delivery below

        if direct_supply >= min_flow_kg_hr:
            # Production covers minimum — deliver what we produced (up to max)
            h2_del         = direct_supply
            storage_charge[h] = excess_to_storage
            storage_draw[h]   = 0.0
        else:
            # Production below minimum — try to draw from storage
            needed_from_storage = min_flow_kg_hr - direct_supply
            available_in_storage = storage_level  # Current storage before this hour

            draw = min(needed_from_storage, available_in_storage)
            h2_del            = direct_supply + draw
            storage_draw[h]   = draw
            storage_charge[h] = excess_to_storage

        # ── Step 5: Update storage level ──
        storage_level = (
            storage_level
            + storage_charge[h]   # H2 added
            - storage_draw[h]     # H2 drawn
        )

        # Clamp storage to [0, capacity]
        storage_level = max(0.0, min(storage_level, storage_capacity_kg))

        h2_storage[h]    = storage_level
        h2_delivered[h]  = h2_del

        # ── Step 6: Record deficit ──
        if h2_del < min_flow_kg_hr:
            deficit[h] = min_flow_kg_hr - h2_del
        else:
            deficit[h] = 0.0

    # ── Summary Metrics ──
    annual_h2_produced_kg  = float(np.sum(h2_produced))
    annual_h2_delivered_kg = float(np.sum(h2_delivered))
    annual_curtailment_mwh = float(np.sum(curtailment_mw))
    total_deficit_kg       = float(np.sum(deficit))
    deficit_hours          = int(np.sum(deficit > 0))
    elec_utilization_pct   = float(np.sum(elec_power > 0)) / N * 100.0
    avg_storage_pct        = (
        float(np.mean(h2_storage)) / storage_capacity_kg * 100.0
        if storage_capacity_kg > 0 else 0.0
    )
    solar_gen_mwh          = float(np.sum(solar_profile * solar_mw))
    wind_gen_mwh           = float(np.sum(wind_profile  * wind_mw))
    re_consumed_mwh        = float(np.sum(elec_power))
    re_self_consumption_pct= (
        re_consumed_mwh / (solar_gen_mwh + wind_gen_mwh) * 100.0
        if (solar_gen_mwh + wind_gen_mwh) > 0 else 0.0
    )

    return {
        # Hourly arrays
        "re_power":          re_power,
        "elec_power":        elec_power,
        "h2_produced":       h2_produced,
        "h2_delivered":      h2_delivered,
        "h2_storage":        h2_storage,
        "curtailment_mw":    curtailment_mw,
        "storage_draw":      storage_draw,
        "storage_charge":    storage_charge,
        "deficit":           deficit,

        # Summary metrics
        "annual_h2_produced_kg":   annual_h2_produced_kg,
        "annual_h2_produced_t":    annual_h2_produced_kg  / 1000.0,
        "annual_h2_delivered_kg":  annual_h2_delivered_kg,
        "annual_h2_delivered_t":   annual_h2_delivered_kg / 1000.0,
        "annual_curtailment_mwh":  annual_curtailment_mwh,
        "total_deficit_kg":        total_deficit_kg,
        "deficit_hours":           deficit_hours,
        "elec_utilization_pct":    elec_utilization_pct,
        "avg_storage_pct":         avg_storage_pct,
        "solar_gen_mwh":           solar_gen_mwh,
        "wind_gen_mwh":            wind_gen_mwh,
        "re_consumed_mwh":         re_consumed_mwh,
        "re_self_consumption_pct": re_self_consumption_pct,
    }
