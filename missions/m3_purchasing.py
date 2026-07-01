"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def policy_v2_comparison(jobs, cat: dict) -> dict:
    """Extension 1 (Your Turn): re-run tier selection with the improved
    `recommend_tier()` (GPU-specific interruption rate + 1yr-vs-3yr reserved by
    job duration) and report how the mix and total spend shift vs. the original
    simple policy.
    """
    v1_total = v2_total = 0.0
    rows = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        days = int(num(j["days"]))
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])

        tier_v1 = pricing.recommend_tier(hpd, interruptible)
        cost_v1 = _tier_cost(tier_v1, gpu_hours, od, c)

        disc_1yr = 1.0 - num(c["reserved_1yr_hr"]) / od if od else 0.0
        # `days` in workloads.csv is just the observation window, not the job's
        # true commitment length. Production inference services (kind=="infer")
        # are evergreen by nature -> treat as a multi-year commitment; training /
        # dev jobs genuinely end -> their observed `days` IS the real duration.
        job_lifetime_days = 3650 if j.get("kind") == "infer" else days
        # 5%/hr is a tighter bar than the DEFAULT_INTERRUPT_RATES table's implicit
        # 10% — commodity cards (A10G ~8%, L4 ~6%) now get bumped off spot even
        # when marked interruptible, since rework churn eats the discount.
        tier_v2 = pricing.recommend_tier(
            hpd, interruptible, gpu_type=gtype, job_days=job_lifetime_days,
            reserved_discount_1yr=disc_1yr, max_acceptable_interrupt_rate=0.05,
        )
        cost_v2 = _tier_cost(tier_v2, gpu_hours, od, c)

        v1_total += cost_v1
        v2_total += cost_v2
        rows.append({"job_id": j["job_id"], "gpu_type": gtype,
                      "tier_v1": tier_v1, "cost_v1": round(cost_v1),
                      "tier_v2": tier_v2, "cost_v2": round(cost_v2)})

    savings_pct_v1 = None  # filled by caller relative to on-demand if needed
    return {
        "rows": rows, "v1_monthly": round(v1_total), "v2_monthly": round(v2_total),
        "delta_usd": round(v1_total - v2_total),
        "delta_pct": round((v1_total - v2_total) / v1_total * 100, 1) if v1_total else 0.0,
    }


def _tier_cost(tier: str, gpu_hours: float, on_demand_hr: float, cat_row: dict) -> float:
    if tier == "spot":
        return pricing.spot_checkpoint_cost(gpu_hours, num(cat_row["spot_hr"]), on_demand_hr)["spot_cost"]
    if tier in ("reserved", "reserved_3yr"):
        return gpu_hours * num(cat_row["reserved_3yr_hr"])
    if tier == "reserved_1yr":
        return gpu_hours * num(cat_row["reserved_1yr_hr"])
    return gpu_hours * on_demand_hr


def carbon_aware_scheduling(jobs, cat: dict) -> dict:
    """Extension 5 (Your Turn): for every interruptible job, compare the carbon
    (and electricity $) cost of running it in `us-east-1` vs. the cleanest grid
    (`europe-north1`), using each GPU's rated wattage x GPU-hours as the energy
    draw. Interruptible jobs are the ones that can realistically be rescheduled
    to a different region/time without breaking anything.
    """
    rows = []
    total_g_saved = 0.0
    for j in jobs:
        if not bool(int(num(j["interruptible"]))):
            continue
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        days = int(num(j["days"]))
        watts = num(cat[gtype]["watts"])
        wh = watts * ngpu * hpd * days  # Wh = W x GPU-count x hours

        best_region = min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
        g_current = sustainability.carbon_g(wh, "us-east-1")
        g_best = sustainability.carbon_g(wh, best_region)
        cost_current = sustainability.energy_cost_usd(wh, "us-east-1")
        cost_best = sustainability.energy_cost_usd(wh, best_region)
        saved_g = g_current - g_best
        total_g_saved += saved_g
        rows.append({
            "job_id": j["job_id"], "gpu_type": gtype, "wh": round(wh, 1),
            "carbon_us_east_1_g": round(g_current, 1), "carbon_best_g": round(g_best, 1),
            "best_region": best_region,
            "carbon_saved_g": round(saved_g, 1),
            "carbon_saved_pct": round(saved_g / g_current * 100, 1) if g_current else 0.0,
            "energy_cost_us_east_1_usd": round(cost_current, 2), "energy_cost_best_usd": round(cost_best, 2),
        })

    region_table = [
        {"region": r, "gco2_per_kwh": sustainability.REGION_CARBON[r],
         "usd_per_kwh": sustainability.REGION_PRICE_KWH[r]}
        for r in sustainability.REGION_CARBON
    ]
    return {
        "jobs": rows, "total_carbon_saved_g": round(total_g_saved, 1),
        "total_carbon_saved_kg": round(total_g_saved / 1000, 2),
        "region_table": sorted(region_table, key=lambda r: r["gco2_per_kwh"]),
    }


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        tier = pricing.recommend_tier(hpd, interruptible)
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            opt_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    v2 = policy_v2_comparison(jobs, cat)
    carbon = carbon_aware_scheduling(jobs, cat)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")

        print("\n-- Extension 1: policy v2 (GPU-aware interrupt rate + 1yr/3yr reserved) --")
        print(f"{'job':18}{'v1 tier':11}{'v1 $':>10}{'v2 tier':13}{'v2 $':>10}")
        for r in v2["rows"]:
            print(f"{r['job_id']:18}{r['tier_v1']:11}${r['cost_v1']:>9,}{r['tier_v2']:13}${r['cost_v2']:>9,}")
        print(f"v1 monthly ${v2['v1_monthly']:,} -> v2 monthly ${v2['v2_monthly']:,}  "
              f"(delta ${v2['delta_usd']:,}, {v2['delta_pct']}%)")

        print("\n-- Extension 5: carbon-aware scheduling for interruptible jobs --")
        print(f"{'job':18}{'Wh':>10}{'gCO2 us-east-1':>16}{'gCO2 best':>12}{'saved %':>9}")
        for r in carbon["jobs"]:
            print(f"{r['job_id']:18}{r['wh']:>10,.0f}{r['carbon_us_east_1_g']:>16,.0f}"
                  f"{r['carbon_best_g']:>12,.0f}{r['carbon_saved_pct']:>8.1f}%")
        print(f"Total carbon saved by moving interruptible jobs to the cleanest region: "
              f"{carbon['total_carbon_saved_kg']:,.1f} kg CO2e/run-window")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "policy_v2": v2, "carbon_aware": carbon}


if __name__ == "__main__":
    run()
