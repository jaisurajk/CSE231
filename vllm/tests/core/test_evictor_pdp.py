# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path


_EVICTOR_PATH = Path(__file__).parents[2] / "vllm" / "core" / "evictor.py"
_SPEC = importlib.util.spec_from_file_location("evictor_under_test",
                                               _EVICTOR_PATH)
_EVICTOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVICTOR)

PDPEvictor = _EVICTOR.PDPEvictor
make_evictor = _EVICTOR.make_evictor


def test_make_evictor_returns_pdp():
    evictor = make_evictor("pdp", "pdp_initial_pd=7")

    assert isinstance(evictor, PDPEvictor)
    assert evictor.protecting_distance == 7


def test_pdp_tracks_request_level_reuse_distance():
    evictor = PDPEvictor("pdp_initial_pd=2,pdp_recompute_interval=100")

    evictor.observe_cache_accesses([11, 22], cache_hint={})
    evictor.observe_cache_accesses([11], cache_hint={})

    assert evictor.request_epoch == 2
    assert evictor.reuse_distance_counts[1] == 1
    assert evictor.last_seen_epoch[11] == 2


def test_pdp_refreshes_protection_for_free_hit():
    evictor = PDPEvictor("pdp_initial_pd=2,pdp_recompute_interval=100")
    evictor.add(1, 11, 16, 1.0, {})

    assert evictor.free_table[1].protected_until_epoch == 2

    evictor.observe_cache_accesses([11], cache_hint={}, real_hits=[True])

    assert evictor.free_table[1].protected_until_epoch == 3
    assert evictor.free_table[1].reused


def test_pdp_evicts_expired_block_before_protected_block():
    evictor = PDPEvictor("pdp_initial_pd=1,pdp_recompute_interval=100")
    evictor.add(1, 11, 16, 1.0, {})

    evictor.observe_cache_accesses([99], cache_hint={})
    evictor.add(2, 22, 16, 2.0, {})

    block_id, content_hash = evictor.evict()

    assert (block_id, content_hash) == (1, 11)


def test_pdp_fallback_evicts_inserted_block_with_highest_remaining_pd():
    evictor = PDPEvictor("pdp_initial_pd=4,pdp_recompute_interval=100")

    evictor.add(1, 11, 16, 1.0, {})
    evictor.observe_cache_accesses([99], cache_hint={})
    evictor.add(2, 22, 16, 2.0, {})

    block_id, content_hash = evictor.evict()

    assert (block_id, content_hash) == (2, 22)


def test_pdp_fallback_prefers_inserted_blocks_before_reused_blocks():
    evictor = PDPEvictor("pdp_initial_pd=4,pdp_recompute_interval=100")

    evictor.add(1, 11, 16, 1.0, {})
    evictor.add(2, 22, 16, 2.0, {})
    evictor.observe_cache_accesses([11], cache_hint={}, real_hits=[True])

    block_id, content_hash = evictor.evict()

    assert (block_id, content_hash) == (2, 22)


def test_pdp_keeps_hash_mapping_consistent_when_blocks_change():
    evictor = PDPEvictor("pdp_initial_pd=2,pdp_recompute_interval=100")

    evictor.add(1, 11, 16, 1.0, {})
    evictor.add(1, 22, 16, 2.0, {})
    evictor.add(2, 22, 16, 3.0, {})
    evictor.remove(1)

    assert 11 not in evictor.content_hash_to_block_id
    assert evictor.content_hash_to_block_id[22] == 2


def test_pdp_recomputes_protecting_distance_from_reuse_distribution():
    evictor = PDPEvictor(
        "pdp_initial_pd=4,pdp_max_distance=4,pdp_recompute_interval=2")

    evictor.observe_cache_accesses([11], cache_hint={})
    evictor.observe_cache_accesses([11], cache_hint={})

    assert evictor.protecting_distance == 1
