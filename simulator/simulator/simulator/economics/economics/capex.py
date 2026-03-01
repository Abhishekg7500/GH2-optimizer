"""
capex.py
--------
CAPEX calculation for the GH2 Optimizer.
All values in Indian Rupees — Crores (₹ Cr).

Project CAPEX = Equipment CAPEX × (1 + BOP% + EPC% + Contingency%)
"""


def calculate_capex(
    solar_mw: float,
    wind_mw: float,
    electrolyzer_mw: float,
    storage_t: float,
    compressor_mw: float,
    # Unit costs (₹ Crores)
    cost_solar: float           = 3.5,    # ₹ Cr / MWp
    cost_wind: float            = 7.0,    # ₹ Cr / MW
    cost_elec: float            = 7.0,    # ₹ Cr / MW
    cost_storage: float         = 0.55,   # ₹ Cr / tH2
    cost_compressor: float      = 4.2,    # ₹ Cr / MW
    # Project cost adders (as fraction)
    bop_pct: float              = 0.12,   # Balance of Plant + Civil
    epc_pct: float              = 0.08,   # EPC Margin
    contingency_pct: float      = 0.05,   # Contingency
) -> dict:
    """
    Calculate full project CAPEX in ₹ Crores.

    Returns detailed breakdown dictionary.
    """

    # ── Equipment CAPEX ──
    cap_solar      = solar_mw       * cost_solar
    cap_wind       = wind_mw        * cost_wind
    cap_elec       = electrolyzer_mw* cost_elec
    cap_storage    = storage_t      * cost_storage
    cap_compressor = compressor_mw  * cost_compressor

    equip_total = (
        cap_solar
        + cap_wind
        + cap_elec
        + cap_storage
        + cap_compressor
    )

    # ── Project adders ──
    cap_bop        = equip_total * bop_pct
    cap_epc        = equip_total * epc_pct
    cap_contingency= equip_total * contingency_pct

    total_cr = equip_total + cap_bop + cap_epc + cap_contingency

    return {
        # Component breakdown
        "cap_solar_cr":       cap_solar,
        "cap_wind_cr":        cap_wind,
        "cap_elec_cr":        cap_elec,
        "cap_storage_cr":     cap_storage,
        "cap_compressor_cr":  cap_compressor,
        # Subtotals
        "equip_total_cr":     equip_total,
        "cap_bop_cr":         cap_bop,
        "cap_epc_cr":         cap_epc,
        "cap_contingency_cr": cap_contingency,
        # Grand total
        "total_cr":           total_cr,
        # Unit cost
        "unit_capex_cr_per_t": total_cr / max(1.0, storage_t),  # rough — overwritten later
    }
