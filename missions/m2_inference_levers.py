"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
# Cache-write premium: providers that bill an up-front cache-write charge (e.g.
# Anthropic) typically price it ~1.25x the normal input rate for that tier.
CACHE_WRITE_MULTIPLIER = 1.25


def cache_worth_it_by_tier(rows) -> dict:
    """Extension 3 (Your Turn): decide, per route_tier, whether caching is worth
    paying a write premium for — using the *actual* average re-read count for
    cached prefixes in this traffic (proxied by cache-hit requests per project,
    since the dataset has no explicit prefix id).
    """
    reads_by_key = defaultdict(int)
    for r in rows:
        if int(num(r["cached_input_tokens"])) > 0:
            reads_by_key[(r["route_tier"], r["team"], r["project"])] += 1

    per_tier_reads = defaultdict(list)
    for (tier, _team, _proj), n in reads_by_key.items():
        per_tier_reads[tier].append(n)

    result = {}
    for tier, (pin, _pout) in MODEL_PRICES.items():
        reads = per_tier_reads.get(tier, [])
        avg_reads = sum(reads) / len(reads) if reads else 0.0
        write_cost = pin * CACHE_WRITE_MULTIPLIER
        breakeven = pricing.cache_breakeven_reads(write_cost, price_in_per_m=pin)
        worth_it = pricing.cache_is_worth_it(avg_reads, write_cost, price_in_per_m=pin)
        result[tier] = {
            "avg_cache_reads": round(avg_reads, 2), "breakeven_reads": round(breakeven, 2),
            "worth_it": worth_it,
        }
    return result


def reasoning_budget_analysis(rows, target_reasoning_traffic_pct: float = 10.0) -> dict:
    """Extension 4 (Your Turn): split cost + energy between reasoning and normal
    traffic, and estimate $ / Wh saved if reasoning were capped to a target share
    of requests (routing rule: only escalate to reasoning below a confidence bar).
    """
    def cost_and_wh(subset):
        cost = wh = 0.0
        for r in subset:
            inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
            cached = int(num(r["cached_input_tokens"]))
            is_batch = bool(int(num(r["is_batch"])))
            pin, pout = MODEL_PRICES[r["route_tier"]]
            cost += pricing.request_cost(inp, out, pin, pout, cached_in=cached, batch=is_batch)
            wh += sustainability.wh_per_query(inp + out, is_reasoning=bool(int(num(r["is_reasoning"]))))
        return cost, wh

    reasoning_rows = [r for r in rows if bool(int(num(r["is_reasoning"])))]
    normal_rows = [r for r in rows if not bool(int(num(r["is_reasoning"])))]
    r_cost, r_wh = cost_and_wh(reasoning_rows)
    n_cost, n_wh = cost_and_wh(normal_rows)
    total_cost = r_cost + n_cost
    total_req = len(rows)
    traffic_pct = (len(reasoning_rows) / total_req * 100) if total_req else 0.0

    scale = min(target_reasoning_traffic_pct / traffic_pct, 1.0) if traffic_pct > 0 else 1.0
    saved_usd = r_cost * (1 - scale)
    saved_wh = r_wh * (1 - scale)

    return {
        "reasoning_requests": len(reasoning_rows), "normal_requests": len(normal_rows),
        "reasoning_traffic_pct": round(traffic_pct, 1),
        "reasoning_cost_usd": round(r_cost, 4), "normal_cost_usd": round(n_cost, 4),
        "reasoning_cost_pct_of_total": round(r_cost / total_cost * 100, 1) if total_cost else 0.0,
        "reasoning_wh": round(r_wh, 2), "normal_wh": round(n_wh, 2),
        "cap_target_pct": target_reasoning_traffic_pct,
        "saved_usd_if_capped": round(saved_usd, 4), "saved_wh_if_capped": round(saved_wh, 2),
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    cache_decision = cache_worth_it_by_tier(rows)
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching (only where cache_is_worth_it),
        # batch API
        pin, pout = MODEL_PRICES[r["route_tier"]]
        effective_cached = cached if cache_decision[r["route_tier"]]["worth_it"] else 0
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=effective_cached, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0
    # Default cap example is intentionally tighter (5%) than the guide's 10% so the
    # lever has a visible effect on this dataset (actual reasoning share ~8.4%).
    reasoning = reasoning_budget_analysis(rows, target_reasoning_traffic_pct=5.0)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- Extension 3: is prompt caching worth its write cost? --")
        for tier, d in cache_decision.items():
            verdict = "YES, keep caching" if d["worth_it"] else "NO, write premium not repaid"
            print(f"  {tier:6} avg_cache_reads={d['avg_cache_reads']:>5}  breakeven={d['breakeven_reads']:>6}  -> {verdict}")

        print("\n-- Extension 4: reasoning traffic budget --")
        print(f"  reasoning: {reasoning['reasoning_requests']} reqs ({reasoning['reasoning_traffic_pct']}% of traffic), "
              f"${reasoning['reasoning_cost_usd']:.2f}/day ({reasoning['reasoning_cost_pct_of_total']}% of $), "
              f"{reasoning['reasoning_wh']:.1f} Wh/day")
        print(f"  normal   : {reasoning['normal_requests']} reqs, ${reasoning['normal_cost_usd']:.2f}/day, {reasoning['normal_wh']:.1f} Wh/day")
        print(f"  cap reasoning to {reasoning['cap_target_pct']:.0f}% of traffic -> save "
              f"${reasoning['saved_usd_if_capped']:.2f}/day and {reasoning['saved_wh_if_capped']:.1f} Wh/day")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_decision": cache_decision, "reasoning_budget": reasoning,
    }


if __name__ == "__main__":
    run()
