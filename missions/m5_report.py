"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def build_extensions_section(r1: dict, r2: dict, r3: dict) -> str:
    """Render the 5 "Your Turn" extension results into a markdown section so the
    report is self-contained (doesn't require re-reading terminal output)."""
    lines = ["## Extensions (\"Your Turn\")", ""]

    lines += ["### 1. Purchasing policy v2 (GPU-aware interrupt rate + 1yr/3yr reserved)", ""]
    v2 = r3["policy_v2"]
    lines.append(f"- v1 monthly: ${v2['v1_monthly']:,} -> v2 monthly: ${v2['v2_monthly']:,} "
                 f"(delta ${v2['delta_usd']:,}, {v2['delta_pct']}%)")
    changed = [row for row in v2["rows"] if row["tier_v1"] != row["tier_v2"]]
    if changed:
        for row in changed:
            lines.append(f"  - `{row['job_id']}`: {row['tier_v1']} (${row['cost_v1']:,}) -> "
                         f"{row['tier_v2']} (${row['cost_v2']:,}) — GPU-specific risk/duration check changed the pick")
    else:
        lines.append("  - No job's tier changed on this dataset: every GPU family in use clears the "
                     "risk bar and every inference job's assumed duration justifies its commitment.")
    lines.append("")

    lines += ["### 2. Right-sizing memory-bound GPUs by MBU/bandwidth", ""]
    rs = r1["rightsize_suggestions"]
    if rs:
        lines.append("| GPU | current | $/GB-VRAM | suggested | $/GB-VRAM | BW kept | $/month saved |")
        lines.append("|---|---|---|---|---|---|---|")
        for x in rs:
            lines.append(f"| {x['gpu_id']} | {x['current_type']} | {x['current_dollar_per_gb_vram']} | "
                         f"{x['suggested_type']} | {x['suggested_dollar_per_gb_vram']} | {x['bw_kept_pct']}% | "
                         f"${x['monthly_savings_usd']:,.0f} |")
        lines.append(f"\nTotal if all memory-bound GPUs are right-sized: **${r1['rightsize_monthly_savings']:,.0f}/month**")
    else:
        lines.append("No GPU in this telemetry window is predominantly memory-bound.")
    lines.append("")

    lines += ["### 3. `cache_is_worth_it()` — is the cache write premium repaid?", ""]
    for tier, d in r2["cache_decision"].items():
        verdict = "worth it" if d["worth_it"] else "NOT worth it"
        lines.append(f"- **{tier}**: avg cache reads/prefix = {d['avg_cache_reads']}, "
                     f"break-even = {d['breakeven_reads']} reads -> **{verdict}**")
    lines.append("")

    lines += ["### 4. Reasoning traffic budget", ""]
    rb = r2["reasoning_budget"]
    lines.append(f"- Reasoning: {rb['reasoning_requests']} requests ({rb['reasoning_traffic_pct']}% of traffic), "
                 f"${rb['reasoning_cost_usd']}/day ({rb['reasoning_cost_pct_of_total']}% of $ spend), "
                 f"{rb['reasoning_wh']} Wh/day")
    lines.append(f"- Normal: {rb['normal_requests']} requests, ${rb['normal_cost_usd']}/day, {rb['normal_wh']} Wh/day")
    lines.append(f"- Capping reasoning to {rb['cap_target_pct']:.0f}% of traffic would save "
                 f"**${rb['saved_usd_if_capped']}/day** and **{rb['saved_wh_if_capped']} Wh/day** "
                 f"({rb['saved_wh_if_capped']/rb['reasoning_wh']*100 if rb['reasoning_wh'] else 0:.0f}% of reasoning's energy draw)")
    lines.append("- Note the asymmetry: reasoning is a much bigger energy problem than a $ problem under "
                 "today's per-token pricing — routing rules justified on cost alone will under-value it.")
    lines.append("")

    lines += ["### 5. Carbon-aware scheduling for interruptible jobs", ""]
    ca = r3["carbon_aware"]
    if ca["jobs"]:
        lines.append("| Job | Wh | gCO2 us-east-1 | gCO2 best region | saved |")
        lines.append("|---|---|---|---|---|")
        for x in ca["jobs"]:
            lines.append(f"| {x['job_id']} | {x['wh']:,.0f} | {x['carbon_us_east_1_g']:,.0f} | "
                         f"{x['carbon_best_g']:,.0f} ({x['best_region']}) | {x['carbon_saved_pct']}% |")
        lines.append(f"\nTotal carbon saved by moving all interruptible jobs to the cleanest region: "
                     f"**{ca['total_carbon_saved_kg']:,.1f} kg CO2e** over their run window.")
    lines.append("")
    return "\n".join(lines)


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
    }

    md = report.build_report(baseline, optimized, levers, sustainability=sust)
    md += "\n\n" + build_extensions_section(r1, r2, r3)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
