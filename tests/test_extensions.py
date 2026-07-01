"""Unit tests for the "Your Turn" extensions (do not touch the graded test files)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing
from missions import m1_efficiency_audit, m3_purchasing


# --- Extension 1: recommend_tier() GPU-aware interrupt rate + 1yr/3yr reserved ---

def test_recommend_tier_backward_compatible():
    # unchanged behaviour when the new kwargs are not supplied
    assert pricing.recommend_tier(2, True) == "spot"
    assert pricing.recommend_tier(24, False) == "reserved"
    assert pricing.recommend_tier(4, False) == "on_demand"


def test_recommend_tier_blocks_high_interrupt_gpu_from_spot():
    # A10G's default interrupt rate (8%) exceeds a 5% risk tolerance -> no spot
    tier = pricing.recommend_tier(8, True, gpu_type="A10G", max_acceptable_interrupt_rate=0.05)
    assert tier != "spot"
    # H100's default interrupt rate (2%) is fine at the same tolerance
    tier_h100 = pricing.recommend_tier(8, True, gpu_type="H100", max_acceptable_interrupt_rate=0.05)
    assert tier_h100 == "spot"


def test_recommend_tier_short_job_prefers_1yr_over_3yr():
    tier = pricing.recommend_tier(20, False, gpu_type="H100", job_days=90, reserved_discount_1yr=0.20)
    assert tier == "reserved_1yr"
    tier_long = pricing.recommend_tier(20, False, gpu_type="H100", job_days=1000, reserved_discount_1yr=0.20)
    assert tier_long == "reserved_3yr"


# --- Extension 2: right-sizing memory-bound GPUs by MBU/bandwidth ---

def test_right_size_by_mbu_only_flags_memory_bound_and_saves_money():
    cat = {
        "BIG": {"peak_tflops_fp16": "1000", "peak_bw_tbs": "2.0", "on_demand_hr": "3.0", "hbm_gb": "80"},
        "SMALL": {"peak_tflops_fp16": "300", "peak_bw_tbs": "1.8", "on_demand_hr": "1.5", "hbm_gb": "40"},
    }
    # low arithmetic intensity (tflops << bw) -> memory-bound every hour
    tel = [{"gpu_id": "g1", "gpu_type": "BIG", "achieved_tflops": "5", "achieved_bw_tbs": "1.5"}] * 5
    summary = [{"gpu_id": "g1", "gpu_type": "BIG"}]
    out = m1_efficiency_audit.right_size_by_mbu(tel, summary, cat)
    assert len(out) == 1
    assert out[0]["suggested_type"] == "SMALL"
    assert out[0]["monthly_savings_usd"] > 0


def test_right_size_by_mbu_skips_compute_bound():
    cat = {
        "BIG": {"peak_tflops_fp16": "1000", "peak_bw_tbs": "2.0", "on_demand_hr": "3.0", "hbm_gb": "80"},
        "SMALL": {"peak_tflops_fp16": "300", "peak_bw_tbs": "1.8", "on_demand_hr": "1.5", "hbm_gb": "40"},
    }
    # high arithmetic intensity (tflops >> bw) -> compute-bound -> should not be flagged
    tel = [{"gpu_id": "g1", "gpu_type": "BIG", "achieved_tflops": "900", "achieved_bw_tbs": "0.5"}] * 5
    summary = [{"gpu_id": "g1", "gpu_type": "BIG"}]
    out = m1_efficiency_audit.right_size_by_mbu(tel, summary, cat)
    assert out == []


# --- Extension 3: cache_is_worth_it() ---

def test_cache_is_worth_it_break_even():
    # write costs 1.25x a $1/1M read price; each read saves 0.9x -> breakeven ~1.39 reads
    breakeven = pricing.cache_breakeven_reads(1.25, read_discount=0.10, price_in_per_m=1.0)
    assert abs(breakeven - (1.25 / 0.9)) < 1e-9
    assert pricing.cache_is_worth_it(breakeven + 0.01, 1.25, price_in_per_m=1.0) is True
    assert pricing.cache_is_worth_it(breakeven - 0.01, 1.25, price_in_per_m=1.0) is False


def test_cache_is_worth_it_single_read_not_worth_it():
    assert pricing.cache_is_worth_it(1, write_cost_per_m=1.25, price_in_per_m=1.0) is False


# --- Extension 4: reasoning_budget_analysis() ---

def test_reasoning_budget_analysis_splits_cost_and_energy():
    from missions import m2_inference_levers as m2
    rows = [
        {"input_tokens": "1000", "output_tokens": "200", "cached_input_tokens": "0",
         "is_batch": "0", "is_reasoning": "1", "route_tier": "large"},
        {"input_tokens": "1000", "output_tokens": "200", "cached_input_tokens": "0",
         "is_batch": "0", "is_reasoning": "0", "route_tier": "small"},
    ]
    out = m2.reasoning_budget_analysis(rows, target_reasoning_traffic_pct=10.0)
    assert out["reasoning_requests"] == 1
    assert out["normal_requests"] == 1
    assert out["reasoning_wh"] > out["normal_wh"]  # reasoning is ~80x more energy-hungry


# --- Extension 5: carbon_aware_scheduling() ---

def test_carbon_aware_scheduling_only_considers_interruptible_jobs():
    cat = {"H100": {"watts": "700"}}
    jobs = [
        {"job_id": "a", "gpu_type": "H100", "num_gpus": "1", "hours_per_day": "24",
         "days": "30", "interruptible": "1"},
        {"job_id": "b", "gpu_type": "H100", "num_gpus": "1", "hours_per_day": "24",
         "days": "30", "interruptible": "0"},
    ]
    out = m3_purchasing.carbon_aware_scheduling(jobs, cat)
    assert len(out["jobs"]) == 1
    assert out["jobs"][0]["job_id"] == "a"
    assert out["jobs"][0]["carbon_saved_g"] > 0
    assert out["total_carbon_saved_kg"] > 0
