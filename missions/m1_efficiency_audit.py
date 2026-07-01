"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

DAYS = 30


def dollars_per_gb_vram(cat_row: dict) -> float:
    """$/hr per GB of HBM — the right yardstick for memory-bound workloads
    (cheapest $/GPU-hr can still be a bad deal if it starves VRAM/bandwidth)."""
    hbm = num(cat_row.get("hbm_gb"))
    od = num(cat_row.get("on_demand_hr"))
    return od / hbm if hbm > 0 else float("inf")


def right_size_by_mbu(tel_rows, summary: list, cat: dict, bw_floor_frac: float = 0.70) -> list:
    """Extension 2 (Your Turn): for GPUs whose workload is *mostly* memory-bound
    (roofline regime from raw telemetry, not just a low MFU number), suggest a
    cheaper catalog GPU that still keeps >= `bw_floor_frac` of the current peak
    bandwidth — because a memory-bound job's throughput tracks HBM bandwidth, not
    peak FLOPs, so right-sizing on `$/GPU-hr` alone can pick a GPU that starves it.
    """
    ridge = {
        gt: (num(c["peak_tflops_fp16"]) / num(c["peak_bw_tbs"]) if num(c["peak_bw_tbs"]) > 0 else 0.0)
        for gt, c in cat.items()
    }
    regime_counts = defaultdict(lambda: {"memory-bound": 0, "compute-bound": 0})
    for r in tel_rows:
        gtype = r["gpu_type"]
        ai = metrics.arithmetic_intensity(num(r["achieved_tflops"]), num(r["achieved_bw_tbs"]))
        regime = metrics.roofline_regime(ai, ridge.get(gtype, 295.0))
        regime_counts[r["gpu_id"]][regime] += 1

    suggestions = []
    for s in summary:
        gid, cur_type = s["gpu_id"], s["gpu_type"]
        counts = regime_counts[gid]
        if counts["memory-bound"] <= counts["compute-bound"]:
            continue  # this GPU's workload is predominantly compute-bound; skip
        cur = cat[cur_type]
        cur_bw, cur_hr = num(cur["peak_bw_tbs"]), num(cur["on_demand_hr"])
        best = None
        for gtype, c in cat.items():
            if gtype == cur_type:
                continue
            bw, hr = num(c["peak_bw_tbs"]), num(c["on_demand_hr"])
            if bw >= bw_floor_frac * cur_bw and hr < cur_hr:
                if best is None or hr < best[1]:
                    best = (gtype, hr, bw)
        if best is None:
            continue
        new_type, new_hr, new_bw = best
        monthly_savings = (cur_hr - new_hr) * 24 * DAYS
        suggestions.append({
            "gpu_id": gid,
            "memory_bound_hours": counts["memory-bound"],
            "current_type": cur_type, "current_hr": cur_hr,
            "current_dollar_per_gb_vram": round(dollars_per_gb_vram(cur), 4),
            "suggested_type": new_type, "suggested_hr": new_hr,
            "suggested_dollar_per_gb_vram": round(dollars_per_gb_vram(cat[new_type]), 4),
            "bw_kept_pct": round(new_bw / cur_bw * 100, 1) if cur_bw else 0.0,
            "monthly_savings_usd": round(monthly_savings, 2),
        })
    return suggestions


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    rightsize = right_size_by_mbu(tel, summary, cat)
    rightsize_monthly_savings = round(sum(x["monthly_savings_usd"] for x in rightsize), 2)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        print("\n-- Extension 2: right-sizing memory-bound GPUs by MBU/bandwidth --")
        if rightsize:
            print(f"{'GPU':14}{'from':6}{'$/GB-VRAM':>11}{'->':>4}{'to':6}{'$/GB-VRAM':>11}{'bw kept':>9}{'$/month saved':>15}")
            for x in rightsize:
                print(f"{x['gpu_id']:14}{x['current_type']:6}{x['current_dollar_per_gb_vram']:>11}"
                      f"{'->':>4}{x['suggested_type']:6}{x['suggested_dollar_per_gb_vram']:>11}"
                      f"{x['bw_kept_pct']:>8}%{x['monthly_savings_usd']:>14,.0f}")
            print(f"Right-size all memory-bound GPUs -> ${rightsize_monthly_savings:,.0f}/month saved")
        else:
            print("No GPU is predominantly memory-bound in this telemetry window.")

    return {
        "summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
        "rightsize_suggestions": rightsize, "rightsize_monthly_savings": rightsize_monthly_savings,
    }


if __name__ == "__main__":
    run()
