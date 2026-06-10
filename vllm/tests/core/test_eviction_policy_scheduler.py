import pytest

from vllm.core.evictor import EvictionPolicyScheduler


def _scheduler(**overrides):
    config = {
        "enable_scheduler": "1",
        "scheduler_shadow_policies": "lru|rrip|fifo",
        "scheduler_min_events": "0",
        "scheduler_small_threshold": "0.10",
        "scheduler_large_threshold": "0.05",
        "model_size_b": "7",
    }
    config.update(overrides)
    return EvictionPolicyScheduler(config)


def _set_hit_rate(events, hits, total):
    events.clear()
    events.extend((0.0, 1) for _ in range(hits))
    events.extend((0.0, 0) for _ in range(total - hits))


def test_scheduler_uses_configured_shadow_policies():
    scheduler = _scheduler(scheduler_shadow_policies="lru|fifo")

    assert scheduler.shadow_policies == ("lru", "fifo")
    assert set(scheduler.shadow_caches) == {"lru", "fifo"}


def test_scheduler_rejects_empty_shadow_policy_set():
    with pytest.raises(ValueError, match="at least one non-ML policy"):
        _scheduler(scheduler_shadow_policies="ml")


def test_scheduler_rejects_unknown_shadow_policy():
    with pytest.raises(ValueError, match="Unknown scheduler shadow policies"):
        _scheduler(scheduler_shadow_policies="lru|unknown")


def test_small_model_requires_absolute_ten_point_ml_gain():
    scheduler = _scheduler(model_size_b="7", scheduler_shadow_policies="lru")
    _set_hit_rate(scheduler.ml_events, hits=40, total=100)
    _set_hit_rate(scheduler.shadow_caches["lru"].events, hits=35, total=100)

    scheduler._finalize_policy(now=0.0)

    assert scheduler.current_policy == "lru"


def test_large_model_uses_lower_ml_gain_threshold():
    scheduler = _scheduler(model_size_b="30", scheduler_shadow_policies="lru")
    _set_hit_rate(scheduler.ml_events, hits=40, total=100)
    _set_hit_rate(scheduler.shadow_caches["lru"].events, hits=35, total=100)

    scheduler._finalize_policy(now=0.0)

    assert scheduler.current_policy == "ml"
