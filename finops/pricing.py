"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


# Illustrative per-hour spot interruption probability by GPU family (deck: newer
# flagship parts get preempted less than commodity/consumer-adjacent cards because
# clouds keep more flagship spot capacity in reserve).
DEFAULT_INTERRUPT_RATES = {
    "H100": 0.02, "H200": 0.02, "B200": 0.015, "A100": 0.04,
    "A10G": 0.08, "L4": 0.06, "MI300X": 0.05,
}


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: int | None = None,
    reserved_discount_1yr: float | None = None,
    interrupt_rate_by_gpu: dict | None = None,
    max_acceptable_interrupt_rate: float = 0.10,
) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)

    Extension 1 (Your Turn): when `gpu_type` is given, spot is only recommended
    if that GPU family's interruption rate clears `max_acceptable_interrupt_rate`
    — a workload that "can" checkpoint still shouldn't ride a spot pool that gets
    reclaimed too often (rework cost eats the discount, see spot_checkpoint_cost).
    When `job_days` and `reserved_discount_1yr` are both given, short-lived jobs
    (< 365 days) are matched against a *1-year* reserved break-even instead of the
    3-year one, since committing 3 years of discount to a 60-day job is a bad trade.
    Both extensions are no-ops (identical to the original policy) unless their
    extra arguments are supplied, so existing callers/tests are unaffected.
    """
    duty = max(0.0, hours_per_day) / 24.0
    be_3yr = break_even_utilization(reserved_discount)

    if interruptible and hours_per_day < 24:
        rates = interrupt_rate_by_gpu or DEFAULT_INTERRUPT_RATES
        rate = rates.get(gpu_type, 0.05)
        if rate <= max_acceptable_interrupt_rate:
            return "spot"
        # else: this GPU family gets reclaimed too often — fall through and
        # size the commitment on duty cycle instead of chasing the spot discount.

    if job_days is not None and reserved_discount_1yr is not None:
        be_1yr = break_even_utilization(reserved_discount_1yr)
        if job_days < 365:
            return "reserved_1yr" if duty >= be_1yr else "on_demand"
        if duty >= be_3yr:
            return "reserved_3yr"
        if duty >= be_1yr:
            return "reserved_1yr"
        return "on_demand"

    if duty >= be_3yr:
        return "reserved"
    return "on_demand"


def cache_breakeven_reads(
    write_cost_per_m: float,
    read_discount: float = 0.10,
    price_in_per_m: float = 1.0,
) -> float:
    """Minimum re-reads of a cached prefix needed to offset its write cost.

    Each read saves `price_in_per_m * (1 - read_discount)` per 1M tokens vs.
    paying full input price again; the write is a one-time `write_cost_per_m`.
    """
    savings_per_read = price_in_per_m * (1.0 - read_discount)
    if savings_per_read <= 0:
        return float("inf")
    return write_cost_per_m / savings_per_read


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
    price_in_per_m: float = 1.0,
) -> bool:
    """Extension 3 (Your Turn): prompt caching only saves money once a cached
    prefix is re-read enough times to pay back its write cost.

    Providers that bill a cache *write* (e.g. an up-front indexing charge) make
    caching a bad trade for prefixes that are only read once or twice — the write
    premium can exceed the read discount. True iff `avg_cache_reads` clears the
    break-even point from `cache_breakeven_reads`.
    """
    breakeven = cache_breakeven_reads(write_cost_per_m, read_discount, price_in_per_m)
    return avg_cache_reads > breakeven


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
