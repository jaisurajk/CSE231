# SPDX-License-Identifier: Apache-2.0

import enum
import heapq
import time
import statistics
import numpy as np
from collections import defaultdict, deque
from sortedcontainers import SortedDict
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

def parse_evictor_config(config: str) -> dict:
    if not config:
        return {}
    return {
        key: value
        for key, value in (
            pair.split("=", 1) for pair in config.split(",") if pair)
    }

def probability_of_future_arrival(prob_has_next, exp_scale, elapsed_time, debug=False):
    if prob_has_next == 0 or exp_scale == 0:
        return 0.0
    prob_not_accessed_till_now = np.exp(-elapsed_time / exp_scale)
    return (prob_has_next * prob_not_accessed_till_now) / (
            prob_has_next * prob_not_accessed_till_now + (1 - prob_has_next)
    )

class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed Blocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass

class CacheStat:
    def __init__(self):
        self.stat = defaultdict(list)
        self.average = {}
        self.last_log_time = 0
        self.log_interval = 10000  # Log every 5 seconds
    
    def get_average(self, key):
        if key in self.average:
            return self.average[key]
        else:
            return 1
    
    def append(self, key, value):
        if key not in self.average:
            self.average[key] = 0
        self.average[key] = (self.average[key] * len(self.stat[key]) + value) / (len(self.stat[key]) + 1)
        self.stat[key].append(value)
    
    def summary(self):
        current_time = time.time()
        if current_time - self.last_log_time < self.log_interval:
            return 0
        self.last_log_time = current_time
        has_data = 0

        for key in self.stat.keys():
            data = self.stat[key]
            if not data:
                continue
            
            mean = statistics.mean(data)
            std_dev = statistics.stdev(data) if len(data) > 1 else 0
            print(f"Summary for {key}: Mean = {mean:.2f}, Std Dev = {std_dev:.2f}")
            has_data = 1
        return has_data
        

class BlockMetaData:
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float, cache_hint: dict = None, score: float = 0):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        self.cache_hint = cache_hint
        self.score = score


class PDPBlockMetaData(BlockMetaData):
    """Metadata for PDP's request-epoch based protection window."""

    def __init__(self,
                 content_hash: int,
                 num_hashed_tokens: int,
                 last_accessed: float,
                 cache_hint: dict = None,
                 first_access_epoch: int = 0,
                 last_access_epoch: int = 0,
                 protected_until_epoch: int = 0,
                 reused: bool = False):
        super().__init__(content_hash, num_hashed_tokens, last_accessed,
                         cache_hint)
        self.first_access_epoch = first_access_epoch
        self.last_access_epoch = last_access_epoch
        self.protected_until_epoch = protected_until_epoch
        self.reused = reused


class PDPEvictor(Evictor):
    """Protecting Distance based policy adapted for prefix-cache blocks.

    PDP in the MICRO paper uses accesses to a hardware cache set as its time
    base. Here we use a request-level epoch: every prefix-cache lookup advances
    the epoch once, and reuse distance is the number of request epochs between
    two observations of the same prefix block hash.
    """

    DEFAULT_INITIAL_PD = 32
    DEFAULT_MAX_DISTANCE = 256
    DEFAULT_RECOMPUTE_INTERVAL = 512
    DEFAULT_BUCKET_SIZE = 1
    DEFAULT_EVICTION_DISTANCE = 1

    def __init__(self, config: str = ""):
        self.free_table: Dict[int, PDPBlockMetaData] = {}
        self.content_hash_to_block_id: Dict[int, int] = {}
        self.last_seen_epoch: Dict[int, int] = {}
        self.reuse_distance_counts = defaultdict(int)
        self.config = parse_evictor_config(config)
        self.protecting_distance = max(
            0, int(self.config.get("pdp_initial_pd",
                                   self.DEFAULT_INITIAL_PD)))
        self.max_distance = max(
            1, int(self.config.get("pdp_max_distance",
                                   self.DEFAULT_MAX_DISTANCE)))
        self.recompute_interval = max(
            1, int(self.config.get("pdp_recompute_interval",
                                   self.DEFAULT_RECOMPUTE_INTERVAL)))
        self.bucket_size = max(
            1, int(self.config.get("pdp_bucket_size",
                                   self.DEFAULT_BUCKET_SIZE)))
        self.eviction_distance = max(
            1, int(self.config.get("pdp_eviction_distance",
                                   self.DEFAULT_EVICTION_DISTANCE)))
        self.capacity = 0
        self.request_epoch = 0
        self.total_observed_accesses = 0
        self.stat = CacheStat()

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def set_capacity(self, capacity: int):
        # Kept for parity with other evictors. The protecting-distance formula
        # uses the separately configurable eviction distance, not total blocks.
        self.capacity = int(capacity or 0)

    def should_predict_cache_hint(self) -> bool:
        return False

    def should_observe_cache_accesses(self) -> bool:
        return True

    def _bucket_distance(self, distance: int) -> int:
        distance = max(1, min(distance, self.max_distance))
        if self.bucket_size == 1:
            return distance
        return min(self.max_distance,
                   ((distance + self.bucket_size - 1) //
                    self.bucket_size) * self.bucket_size)

    def _protect_block(self, block_id: int, reused: bool = True) -> None:
        if block_id not in self.free_table:
            return
        block = self.free_table[block_id]
        block.last_access_epoch = self.request_epoch
        block.protected_until_epoch = (
            self.request_epoch + self.protecting_distance)
        block.reused = block.reused or reused

    def _forget_hash_mapping(self, content_hash: int, block_id: int) -> None:
        if self.content_hash_to_block_id.get(content_hash) == block_id:
            self.content_hash_to_block_id.pop(content_hash, None)

    def observe_cache_accesses(self,
                               block_hashes: List[int],
                               cache_hint: Optional[dict] = None,
                               real_hits: Optional[List[bool]] = None):
        if not block_hashes:
            return

        self.request_epoch += 1
        seen_in_request = set()
        for index, content_hash in enumerate(block_hashes):
            if content_hash in seen_in_request:
                continue
            seen_in_request.add(content_hash)
            self.total_observed_accesses += 1
            real_hit = bool(real_hits[index]) if real_hits else False

            previous_epoch = self.last_seen_epoch.get(content_hash)
            if previous_epoch is not None:
                reuse_distance = self.request_epoch - previous_epoch
                self.reuse_distance_counts[
                    self._bucket_distance(reuse_distance)] += 1

            self.last_seen_epoch[content_hash] = self.request_epoch
            block_id = self.content_hash_to_block_id.get(content_hash)
            if block_id is not None:
                self._protect_block(block_id,
                                    reused=real_hit
                                    or previous_epoch is not None)

        if self.total_observed_accesses % self.recompute_interval == 0:
            self._recompute_protecting_distance()

    def observe_cache_access(self,
                             content_hash: int,
                             cache_hint: Optional[dict] = None,
                             real_hit: Optional[bool] = None):
        self.observe_cache_accesses([content_hash], cache_hint,
                                    [real_hit] if real_hit is not None else None)

    def _recompute_protecting_distance(self):
        if not self.reuse_distance_counts or self.total_observed_accesses == 0:
            return

        distances = sorted(self.reuse_distance_counts.items())
        best_distance = self.protecting_distance
        best_score = -1.0
        cumulative_hits = 0
        cumulative_cost = 0
        idx = 0

        for distance in range(1, self.max_distance + 1, self.bucket_size):
            while idx < len(distances) and distances[idx][0] <= distance:
                reuse_distance, count = distances[idx]
                cumulative_hits += count
                cumulative_cost += reuse_distance * count
                idx += 1

            misses = max(0, self.total_observed_accesses - cumulative_hits)
            denominator = (
                cumulative_cost +
                misses * (distance + self.eviction_distance))
            score = (cumulative_hits / denominator
                     if denominator > 0 else 0.0)
            if score > best_score:
                best_score = score
                best_distance = distance

        self.protecting_distance = best_distance

    def _victim_key(self, block_id: int, block: PDPBlockMetaData):
        expired = block.protected_until_epoch <= self.request_epoch
        remaining = max(0, block.protected_until_epoch - self.request_epoch)
        if expired:
            return (0, block.last_access_epoch, block.last_accessed, block_id)
        # Inclusive-cache PDP fallback: if no unprotected line exists, evict an
        # inserted line with the highest remaining PD; if every line has been
        # reused, evict the reused line with the highest remaining PD.
        return (1, 1 if block.reused else 0, -remaining,
                -block.last_access_epoch, -block.last_accessed, block_id)

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        block_id, block = min(self.free_table.items(),
                              key=lambda item: self._victim_key(
                                  item[0], item[1]))
        survival_time = time.time() - block.last_accessed
        self.stat.append("survival_times", survival_time)
        content_hash = block.content_hash
        del self.free_table[block_id]
        self._forget_hash_mapping(content_hash, block_id)
        return block_id, content_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        previous_block = self.free_table.get(block_id)
        if previous_block is not None:
            self._forget_hash_mapping(previous_block.content_hash, block_id)

        last_access_epoch = self.last_seen_epoch.get(content_hash,
                                                     self.request_epoch)
        reused = content_hash in self.last_seen_epoch
        block = PDPBlockMetaData(
            content_hash=content_hash,
            num_hashed_tokens=num_hashed_tokens,
            last_accessed=last_accessed,
            cache_hint=cache_hint,
            first_access_epoch=last_access_epoch,
            last_access_epoch=last_access_epoch,
            protected_until_epoch=(
                self.request_epoch + self.protecting_distance),
            reused=reused,
        )
        self.free_table[block_id] = block
        self.content_hash_to_block_id[content_hash] = block_id

    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        if block_id not in self.free_table:
            raise ValueError("Attempting to update block that's not in the evictor")
        block = self.free_table[block_id]
        block.last_accessed = last_accessed
        block.cache_hint = cache_hint
        self._protect_block(block_id, reused=True)

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block = self.free_table.pop(block_id)
        self._forget_hash_mapping(block.content_hash, block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)


class ShadowPolicyCache:
    RRIP_MAX_RRPV = 3
    RRIP_INSERT_RRPV = RRIP_MAX_RRPV - 1
    RRIP_HIT_RRPV = 0

    def __init__(self, policy: str, capacity: int, config: dict = None):
        self.policy = policy
        self.capacity = capacity
        self.table = {}
        self.events = deque()
        if policy == "pdp":
            cfg = config or {}
            self.pdp_epoch = 0
            self.pdp_protecting_distance = int(
                cfg.get("pdp_initial_pd", PDPEvictor.DEFAULT_INITIAL_PD))
            self.pdp_max_distance = int(
                cfg.get("pdp_max_distance", PDPEvictor.DEFAULT_MAX_DISTANCE))
            self.pdp_recompute_interval = int(
                cfg.get("pdp_recompute_interval",
                        PDPEvictor.DEFAULT_RECOMPUTE_INTERVAL))
            self.pdp_bucket_size = int(
                cfg.get("pdp_bucket_size", PDPEvictor.DEFAULT_BUCKET_SIZE))
            self.pdp_eviction_distance = int(
                cfg.get("pdp_eviction_distance",
                        PDPEvictor.DEFAULT_EVICTION_DISTANCE))
            self.pdp_last_seen_epoch: Dict[int, int] = {}
            self.pdp_reuse_distance_counts = defaultdict(int)
            self.pdp_total_accesses = 0

    def set_capacity(self, capacity: int):
        self.capacity = capacity
        while len(self.table) > self.capacity:
            self._evict()

    def hit_rate(self) -> float:
        if not self.events:
            return 0.0
        return sum(hit for _, hit in self.events) / len(self.events)

    def num_events(self) -> int:
        return len(self.events)

    def access(self, content_hash: int, now: float,
               cache_hint: Optional[dict] = None):
        hit = content_hash in self.table
        self.events.append((now, 1 if hit else 0))

        if self.policy == "pdp":
            self.pdp_epoch += 1
            self.pdp_total_accesses += 1
            prev_epoch = self.pdp_last_seen_epoch.get(content_hash)
            if prev_epoch is not None:
                dist = self._pdp_bucket_distance(self.pdp_epoch - prev_epoch)
                self.pdp_reuse_distance_counts[dist] += 1
            self.pdp_last_seen_epoch[content_hash] = self.pdp_epoch
            if self.pdp_total_accesses % self.pdp_recompute_interval == 0:
                self._pdp_recompute()

        if hit:
            entry = self.table[content_hash]
            entry["last_accessed"] = now
            entry["cache_hint"] = cache_hint
            if self.policy == "rrip":
                entry["rrpv"] = self.RRIP_HIT_RRPV
            elif self.policy == "ml":
                entry["score"] = self._ml_score(entry, cache_hint, now)
            elif self.policy == "pdp":
                entry["pdp_protected_until"] = (
                    self.pdp_epoch + self.pdp_protecting_distance)
                entry["pdp_last_access_epoch"] = self.pdp_epoch
                entry["pdp_reused"] = True
            return

        if self.capacity <= 0:
            return
        while len(self.table) >= self.capacity:
            self._evict()
        new_entry = {
            "first_accessed": now,
            "last_accessed": now,
            "rrpv": self.RRIP_INSERT_RRPV,
            "score": 0.0,
            "cache_hint": cache_hint,
        }
        if self.policy == "pdp":
            new_entry["pdp_protected_until"] = (
                self.pdp_epoch + self.pdp_protecting_distance)
            new_entry["pdp_last_access_epoch"] = self.pdp_epoch
            new_entry["pdp_reused"] = False
        elif self.policy == "ml":
            new_entry["score"] = self._ml_score(new_entry, cache_hint, now)
        self.table[content_hash] = new_entry

    def _ml_score(self, entry: dict, cache_hint: Optional[dict],
                  now: float) -> float:
        if cache_hint and "prob_has_next" in cache_hint:
            return probability_of_future_arrival(
                cache_hint["prob_has_next"], cache_hint.get("exp_scale", 1),
                now - entry["last_accessed"])
        return entry["last_accessed"]

    def _evict(self):
        if not self.table:
            return
        if self.policy == "fifo":
            victim = min(self.table,
                         key=lambda h: (self.table[h]["first_accessed"], h))
        elif self.policy == "rrip":
            victim = self._rrip_victim()
        elif self.policy == "pdp":
            victim = min(self.table, key=lambda h: self._pdp_victim_key(h))
        elif self.policy == "ml":
            now = time.time()
            victim = min(
                self.table,
                key=lambda h: (
                    self._ml_score(self.table[h],
                                   self.table[h].get("cache_hint"), now),
                    self.table[h]["last_accessed"],
                    h,
                ))
        else:
            victim = min(self.table,
                         key=lambda h: (self.table[h]["last_accessed"], h))
        del self.table[victim]

    def _rrip_victim(self):
        while True:
            candidates = [
                h for h, entry in self.table.items()
                if entry["rrpv"] >= self.RRIP_MAX_RRPV
            ]
            if candidates:
                return min(candidates,
                           key=lambda h: (self.table[h]["last_accessed"], h))
            for entry in self.table.values():
                entry["rrpv"] = min(self.RRIP_MAX_RRPV, entry["rrpv"] + 1)

    def _pdp_victim_key(self, content_hash: int) -> tuple:
        entry = self.table[content_hash]
        protected_until = entry.get("pdp_protected_until", 0)
        expired = protected_until <= self.pdp_epoch
        if expired:
            return (0, entry.get("pdp_last_access_epoch", 0),
                    entry["last_accessed"], content_hash)
        remaining = protected_until - self.pdp_epoch
        reused = entry.get("pdp_reused", False)
        return (1, 1 if reused else 0, -remaining,
                -entry.get("pdp_last_access_epoch", 0), content_hash)

    def _pdp_bucket_distance(self, distance: int) -> int:
        distance = max(1, min(distance, self.pdp_max_distance))
        if self.pdp_bucket_size == 1:
            return distance
        return min(self.pdp_max_distance,
                   ((distance + self.pdp_bucket_size - 1) //
                    self.pdp_bucket_size) * self.pdp_bucket_size)

    def _pdp_recompute(self):
        if not self.pdp_reuse_distance_counts or self.pdp_total_accesses == 0:
            return
        distances = sorted(self.pdp_reuse_distance_counts.items())
        best_distance = self.pdp_protecting_distance
        best_score = -1.0
        cumulative_hits = 0
        cumulative_cost = 0
        idx = 0
        for distance in range(1, self.pdp_max_distance + 1,
                               self.pdp_bucket_size):
            while idx < len(distances) and distances[idx][0] <= distance:
                reuse_distance, count = distances[idx]
                cumulative_hits += count
                cumulative_cost += reuse_distance * count
                idx += 1
            misses = max(0, self.pdp_total_accesses - cumulative_hits)
            denominator = (cumulative_cost +
                           misses * (distance + self.pdp_eviction_distance))
            score = cumulative_hits / denominator if denominator > 0 else 0.0
            if score > best_score:
                best_score = score
                best_distance = distance
        self.pdp_protecting_distance = best_distance


class EvictionPolicyScheduler:
    POLICIES = ("ml", "lru", "rrip", "fifo", "pdp")
    SHADOW_POLICIES = ("lru", "rrip", "fifo", "pdp")

    def __init__(self, config: dict):
        self.enabled = self._as_bool(config.get("enable_scheduler", "0"))
        self.warmup_s = float(config.get("scheduler_warmup", 100))
        self.min_events = int(config.get("scheduler_min_events", 1000))
        self.small_threshold = float(
            config.get("scheduler_small_threshold", 0.10))
        self.large_threshold = float(
            config.get("scheduler_large_threshold", 0.05))
        self.model_size_b = float(config.get("model_size_b", 7))
        self.current_policy = config.get("scheduler_initial_policy", "ml")
        self.start_time = time.time()
        self.finalized = False
        self.capacity = int(config.get("scheduler_capacity", 0))
        self.observe_stride = max(1, int(config.get(
            "scheduler_observe_stride", 4)))
        self.num_observed_accesses = 0
        self.ml_events = deque()
        self.shadow_policies = self._parse_shadow_policies(
            config.get("scheduler_shadow_policies"))
        self.shadow_caches = {
            policy: ShadowPolicyCache(
                policy, self.capacity,
                config if policy == "pdp" else None)
            for policy in self.shadow_policies
        }

    def _as_bool(self, value) -> bool:
        return str(value).lower() in ("1", "true", "yes", "on")

    def _parse_shadow_policies(self, value) -> Tuple[str, ...]:
        if value is None or value == "":
            return self.SHADOW_POLICIES
        policies = tuple(policy.strip() for policy in str(value).split("|")
                         if policy.strip())
        invalid = [policy for policy in policies if policy not in self.POLICIES]
        if invalid:
            raise ValueError(
                "Unknown scheduler shadow policies: "
                f"{invalid}. Supported policies: {self.POLICIES}")
        policies = tuple(policy for policy in policies if policy != "ml")
        if not policies:
            raise ValueError(
                "scheduler_shadow_policies must include at least one "
                "non-ML policy")
        return policies

    def set_capacity(self, capacity: int):
        self.capacity = capacity
        for cache in self.shadow_caches.values():
            cache.set_capacity(capacity)

    def observe(self, content_hash: int, cache_hint: Optional[dict] = None,
                real_hit: Optional[bool] = None) -> bool:
        if not self.enabled:
            return False
        now = time.time()
        if self.finalized:
            return False
        self.num_observed_accesses += 1
        if self.num_observed_accesses % self.observe_stride != 0:
            return False
        if real_hit is not None:
            self.ml_events.append((now, 1 if real_hit else 0))
        for cache in self.shadow_caches.values():
            cache.access(content_hash, now, cache_hint)
        if now - self.start_time >= self.warmup_s:
            return self._finalize_policy(now)
        return False

    def should_predict(self) -> bool:
        if not self.enabled:
            return True
        return (not self.finalized) or self.current_policy == "ml"

    def _ml_hit_rate(self) -> float:
        if not self.ml_events:
            return 0.0
        return sum(hit for _, hit in self.ml_events) / len(self.ml_events)

    def _finalize_policy(self, now: float) -> bool:
        min_events = min([len(self.ml_events)] + [
            cache.num_events() for cache in self.shadow_caches.values()
        ])
        if min_events == 0:
            print("scheduler finalize: no warmup events, keep ml")
            self.current_policy = "ml"
            self.finalized = True
            return True
        if min_events < self.min_events:
            print("scheduler finalize: insufficient warmup events",
                  min_events, "<", self.min_events)

        hit_rates = {
            "ml": self._ml_hit_rate(),
            **{
                policy: cache.hit_rate()
                for policy, cache in self.shadow_caches.items()
            }
        }
        best_shadow_policy = max(self.shadow_policies,
                                 key=lambda p: hit_rates[p])
        best_other = hit_rates[best_shadow_policy]
        threshold = (self.small_threshold if self.model_size_b <= 14 else
                     self.large_threshold)
        if hit_rates["ml"] >= best_other + threshold:
            best_policy = "ml"
        else:
            best_policy = best_shadow_policy

        print("scheduler finalize:",
              self.current_policy, "->", best_policy, hit_rates)
        self.current_policy = best_policy
        self.finalized = True
        return True

class LRUMLEvictor(Evictor):
    RRIP_MAX_RRPV = 3
    RRIP_INSERT_RRPV = RRIP_MAX_RRPV - 1
    RRIP_HIT_RRPV = 0

    def __init__(self, config):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.sorted_dict = SortedDict()
        self.id_to_last_access = {}
        self.id_to_first_access = {}
        self.rrip_values = {}
        self.to_delete_blocks = []
        self.config = self.parse_str_to_dict(config)
        self.stat = CacheStat()
        self.last_refresh_time = time.time()
        self.INSPECT_INTERVAL = 5
        self.scheduler = EvictionPolicyScheduler(self.config)
        # PDP state — used when the scheduler selects PDP post-warmup
        cfg = self.config
        self.pdp_request_epoch = 0
        self.pdp_protecting_distance = int(
            cfg.get("pdp_initial_pd", PDPEvictor.DEFAULT_INITIAL_PD))
        self.pdp_max_distance = int(
            cfg.get("pdp_max_distance", PDPEvictor.DEFAULT_MAX_DISTANCE))
        self.pdp_recompute_interval = int(
            cfg.get("pdp_recompute_interval",
                    PDPEvictor.DEFAULT_RECOMPUTE_INTERVAL))
        self.pdp_bucket_size = int(
            cfg.get("pdp_bucket_size", PDPEvictor.DEFAULT_BUCKET_SIZE))
        self.pdp_eviction_distance = int(
            cfg.get("pdp_eviction_distance",
                    PDPEvictor.DEFAULT_EVICTION_DISTANCE))
        self.pdp_last_seen_epoch: Dict[int, int] = {}
        self.pdp_reuse_distance_counts = defaultdict(int)
        self.pdp_total_accesses = 0
        self.pdp_protected_until_epoch: Dict[int, int] = {}
        self.pdp_reused: Dict[int, bool] = {}
        self.pdp_last_access_epoch: Dict[int, int] = {}
        self.pdp_content_hash_to_block_id: Dict[int, int] = {}

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def parse_str_to_dict(self, s: str) -> dict:
        return parse_evictor_config(s)

    def get_policy(self, cache_hint: dict) -> str:
        if cache_hint is None:
            return 'lru'
        if cache_hint.get('use_rrip'):
            return 'rrip'
        if cache_hint.get('use_lru'):
            return 'lru'
        if cache_hint.get('use_fifo'):
            return 'fifo'
        if cache_hint.get('scheduler_policy'):
            return cache_hint['scheduler_policy']
        if self.scheduler.enabled:
            return self.scheduler.current_policy
        if 'next_timestamp' in cache_hint:
            return 'belady'
        return 'ml'

    def set_capacity(self, capacity: int):
        self.scheduler.set_capacity(capacity)

    def observe_cache_access(self, content_hash: int,
                             cache_hint: Optional[dict] = None,
                             real_hit: Optional[bool] = None):
        if self.scheduler.enabled and not self.scheduler.finalized:
            finalized_now = self.scheduler.observe(content_hash, cache_hint,
                                                   real_hit)
            if finalized_now:
                self._rebind_scheduler_policy_for_existing_blocks()
        elif self.scheduler.finalized and self.scheduler.current_policy == 'pdp':
            self._pdp_observe(content_hash, real_hit)

    def should_predict_cache_hint(self) -> bool:
        return self.scheduler.should_predict()

    def should_observe_cache_accesses(self) -> bool:
        if self.scheduler.enabled and not self.scheduler.finalized:
            return True
        # Keep observing post-warmup so PDP can advance its epoch
        if self.scheduler.finalized and self.scheduler.current_policy == 'pdp':
            return True
        return False

    def should_record_post_warmup_metric(self) -> bool:
        return self.scheduler.enabled and self.scheduler.finalized

    def _bind_scheduler_policy(self, cache_hint: dict) -> dict:
        if not self.scheduler.enabled:
            return cache_hint
        if (cache_hint.get('use_lru') or cache_hint.get('use_rrip') or
                cache_hint.get('use_fifo') or cache_hint.get('scheduler_policy')):
            return cache_hint
        cache_hint = dict(cache_hint)
        cache_hint['scheduler_policy'] = self.scheduler.current_policy
        return cache_hint

    def _rebind_scheduler_policy_for_existing_blocks(self) -> None:
        if not self.scheduler.enabled:
            return
        final_policy = self.scheduler.current_policy
        self.rrip_values.clear()
        if final_policy == 'pdp':
            # Seed protecting_distance from what the shadow PDP learned during warmup
            shadow_pdp = self.scheduler.shadow_caches.get('pdp')
            if shadow_pdp is not None:
                self.pdp_protecting_distance = shadow_pdp.pdp_protecting_distance
            self.pdp_content_hash_to_block_id.clear()
        for block_id, block in self.free_table.items():
            cache_hint = dict(block.cache_hint)
            cache_hint.pop('use_lru', None)
            cache_hint.pop('use_rrip', None)
            cache_hint.pop('use_fifo', None)
            cache_hint['scheduler_policy'] = final_policy
            block.cache_hint = cache_hint
            if final_policy == 'rrip':
                self.rrip_values[block_id] = self.RRIP_INSERT_RRPV
            elif final_policy == 'pdp':
                # Treat existing blocks as reused; protect for a full PD window
                self.pdp_protected_until_epoch[block_id] = (
                    self.pdp_request_epoch + self.pdp_protecting_distance)
                self.pdp_last_access_epoch[block_id] = self.pdp_request_epoch
                self.pdp_reused[block_id] = True
                self.pdp_content_hash_to_block_id[block.content_hash] = block_id
        self.to_delete_blocks.clear()
        self._rebuild_sorted_dict()
        print("scheduler rebound existing blocks to", final_policy,
              "count", len(self.free_table))

    def calc_score(self, block_id, last_accessed, cache_hint):
        policy = self.get_policy(cache_hint)
        if policy == 'rrip':
            rrpv = self.rrip_values.get(block_id, self.RRIP_INSERT_RRPV)
            return -rrpv
        if policy == 'lru':
            return last_accessed
        if policy == 'fifo':
            return self.id_to_first_access[block_id]
        if policy == 'pdp':
            return self._pdp_score(block_id)
        if policy == 'belady':
            return -cache_hint['next_timestamp']
        if 'prob_has_next' in cache_hint:
            prob = probability_of_future_arrival(
                cache_hint['prob_has_next'], cache_hint['exp_scale'], time.time() - last_accessed)
            return prob
        return last_accessed

    def _rebuild_sorted_dict(self):
        new_sorted_dict = SortedDict()
        for block_id, block in self.free_table.items():
            score = self.calc_score(block_id, block.last_accessed,
                                    block.cache_hint)
            block.score = score
            new_sorted_dict[(score, block.last_accessed, block_id)] = (
                block_id, block.content_hash)
        self.sorted_dict = new_sorted_dict

    def _age_rrip_blocks(self):
        for block_id, block in self.free_table.items():
            if self.get_policy(block.cache_hint) != 'rrip':
                continue
            rrpv = self.rrip_values.get(block_id, self.RRIP_INSERT_RRPV)
            self.rrip_values[block_id] = min(self.RRIP_MAX_RRPV, rrpv + 1)
        self._rebuild_sorted_dict()

    def _peek_valid_candidate(self):
        while self.sorted_dict:
            key, (block_id, content_hash) = self.sorted_dict.peekitem(0)
            if (block_id in self.free_table and
                    self.free_table[block_id].score == key[0] and
                    self.free_table[block_id].last_accessed == key[1]):
                return block_id, content_hash
            self.sorted_dict.popitem(0)
        return None

    def _remove_from_sorted_dict(self, block_id: int):
        block = self.free_table[block_id]
        entry = (block.score, block.last_accessed, block_id)
        if entry in self.sorted_dict:
            del self.sorted_dict[entry]
        else:
            raise ValueError("the score is not found in sorted_dict")

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        # PDP uses a direct scan so scores are always evaluated at the current
        # epoch rather than relying on potentially stale sorted_dict entries.
        sample_policy = self.get_policy(
            next(iter(self.free_table.values())).cache_hint)
        if sample_policy == 'pdp':
            block_id = min(self.free_table.keys(),
                           key=lambda bid: self._pdp_victim_key(bid))
            block = self.free_table[block_id]
            content_hash = block.content_hash
            self.stat.append("survival_times",
                             time.time() - block.last_accessed)
            self._remove_from_sorted_dict(block_id)
            if self.pdp_content_hash_to_block_id.get(content_hash) == block_id:
                del self.pdp_content_hash_to_block_id[content_hash]
            self.pdp_protected_until_epoch.pop(block_id, None)
            self.pdp_last_access_epoch.pop(block_id, None)
            self.pdp_reused.pop(block_id, None)
            del self.free_table[block_id]
            self.id_to_first_access.pop(block_id, None)
            return block_id, content_hash

        while True:
            if len(self.to_delete_blocks) > 0:
                (block_id, content_hash, last_accessed) = self.to_delete_blocks.pop()
                if block_id not in self.free_table or self.free_table[block_id].last_accessed != last_accessed:
                    continue
                break
            else:
                candidate = self._peek_valid_candidate()
                if candidate is None:
                    raise ValueError("No usable cache memory left")
                block_id, content_hash = candidate
                if self.get_policy(self.free_table[block_id].cache_hint) == 'rrip':
                    rrpv = self.rrip_values.get(block_id,
                                                self.RRIP_INSERT_RRPV)
                    if rrpv < self.RRIP_MAX_RRPV:
                        self._age_rrip_blocks()
                        continue
                self._remove_from_sorted_dict(block_id)
                break
        if block_id in self.free_table:
            survival_time = time.time() - self.free_table[block_id].last_accessed
            self.stat.append("survival_times", survival_time)
            if self.get_policy(self.free_table[block_id].cache_hint) == 'rrip':
                self.rrip_values.pop(block_id, None)
            del self.free_table[block_id]
            self.id_to_first_access.pop(block_id, None)
            return block_id, content_hash
        else:
            raise ValueError("block is not in the sorted_dict")
    
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        cache_hint = self._bind_scheduler_policy(cache_hint)
        policy = self.get_policy(cache_hint)
        if policy == 'rrip' and block_id not in self.rrip_values:
            self.rrip_values[block_id] = self.RRIP_INSERT_RRPV
        if policy == 'pdp':
            old_block = self.free_table.get(block_id)
            if old_block is not None:
                old_hash = old_block.content_hash
                if self.pdp_content_hash_to_block_id.get(old_hash) == block_id:
                    del self.pdp_content_hash_to_block_id[old_hash]
            reused = content_hash in self.pdp_last_seen_epoch
            self.pdp_protected_until_epoch[block_id] = (
                self.pdp_request_epoch + self.pdp_protecting_distance)
            self.pdp_last_access_epoch[block_id] = self.pdp_last_seen_epoch.get(
                content_hash, self.pdp_request_epoch)
            self.pdp_reused[block_id] = reused
            self.pdp_content_hash_to_block_id[content_hash] = block_id
        if block_id not in self.id_to_first_access:
            self.id_to_first_access[block_id] = last_accessed
        score = self.calc_score(block_id, last_accessed, cache_hint)
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  cache_hint,
                                                  score)
        self.sorted_dict[(score, last_accessed, block_id)] = (block_id, content_hash)
        self.id_to_last_access[cache_hint['id']] = last_accessed
        if time.time() - self.last_refresh_time > self.INSPECT_INTERVAL:
            self._refresh()

    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        if block_id not in self.free_table:
            raise ValueError("Attempting to update block that's not in the evictor")
        cache_hint = self._bind_scheduler_policy(cache_hint)
        self._remove_from_sorted_dict(block_id)
        policy = self.get_policy(cache_hint)
        if policy == 'rrip':
            self.rrip_values[block_id] = self.RRIP_HIT_RRPV
        elif policy == 'pdp':
            self._pdp_protect_block(block_id, reused=True)
        score = self.calc_score(block_id, last_accessed, cache_hint)
        self.free_table[block_id].last_accessed = last_accessed
        self.free_table[block_id].cache_hint = cache_hint
        self.free_table[block_id].score = score
        self.sorted_dict[(score, last_accessed, block_id)] = (block_id, self.free_table[block_id].content_hash)
        self.id_to_last_access[cache_hint['id']] = last_accessed

    # remove is only called by 'hit', the blocks will be added back later after decoding
    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError("Attempting to remove block that's not in the evictor")
        policy = self.get_policy(self.free_table[block_id].cache_hint)
        if policy == 'rrip':
            self.rrip_values[block_id] = self.RRIP_HIT_RRPV
        elif policy == 'pdp':
            content_hash = self.free_table[block_id].content_hash
            if self.pdp_content_hash_to_block_id.get(content_hash) == block_id:
                del self.pdp_content_hash_to_block_id[content_hash]
            self.pdp_protected_until_epoch.pop(block_id, None)
            self.pdp_last_access_epoch.pop(block_id, None)
            self.pdp_reused.pop(block_id, None)
        self._remove_from_sorted_dict(block_id)
        del self.free_table[block_id]

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

    # ---- PDP helpers -------------------------------------------------------

    def _pdp_bucket_distance(self, distance: int) -> int:
        distance = max(1, min(distance, self.pdp_max_distance))
        if self.pdp_bucket_size == 1:
            return distance
        return min(self.pdp_max_distance,
                   ((distance + self.pdp_bucket_size - 1) //
                    self.pdp_bucket_size) * self.pdp_bucket_size)

    def _pdp_protect_block(self, block_id: int, reused: bool = True) -> None:
        if block_id not in self.free_table:
            return
        self.pdp_protected_until_epoch[block_id] = (
            self.pdp_request_epoch + self.pdp_protecting_distance)
        self.pdp_last_access_epoch[block_id] = self.pdp_request_epoch
        if reused:
            self.pdp_reused[block_id] = True

    def _pdp_observe(self, content_hash: int,
                     real_hit: Optional[bool] = None) -> None:
        self.pdp_request_epoch += 1
        self.pdp_total_accesses += 1
        previous_epoch = self.pdp_last_seen_epoch.get(content_hash)
        if previous_epoch is not None:
            dist = self._pdp_bucket_distance(
                self.pdp_request_epoch - previous_epoch)
            self.pdp_reuse_distance_counts[dist] += 1
        self.pdp_last_seen_epoch[content_hash] = self.pdp_request_epoch
        block_id = self.pdp_content_hash_to_block_id.get(content_hash)
        if block_id is not None and block_id in self.free_table:
            reused = bool(real_hit) or previous_epoch is not None
            self._pdp_protect_block(block_id, reused=reused)
        if self.pdp_total_accesses % self.pdp_recompute_interval == 0:
            self._pdp_recompute_protecting_distance()

    def _pdp_recompute_protecting_distance(self) -> None:
        if not self.pdp_reuse_distance_counts or self.pdp_total_accesses == 0:
            return
        distances = sorted(self.pdp_reuse_distance_counts.items())
        best_distance = self.pdp_protecting_distance
        best_score = -1.0
        cumulative_hits = 0
        cumulative_cost = 0
        idx = 0
        for distance in range(1, self.pdp_max_distance + 1,
                               self.pdp_bucket_size):
            while idx < len(distances) and distances[idx][0] <= distance:
                reuse_distance, count = distances[idx]
                cumulative_hits += count
                cumulative_cost += reuse_distance * count
                idx += 1
            misses = max(0, self.pdp_total_accesses - cumulative_hits)
            denominator = (cumulative_cost +
                           misses * (distance + self.pdp_eviction_distance))
            score = cumulative_hits / denominator if denominator > 0 else 0.0
            if score > best_score:
                best_score = score
                best_distance = distance
        self.pdp_protecting_distance = best_distance

    def _pdp_score(self, block_id: int) -> float:
        """Map PDP victim ordering to a single float for sorted_dict."""
        protected_until = self.pdp_protected_until_epoch.get(block_id, 0)
        expired = protected_until <= self.pdp_request_epoch
        if expired:
            epoch = self.pdp_last_access_epoch.get(block_id, 0)
            return -3e15 + epoch  # most negative tier; evict lowest epoch first
        remaining = max(0, protected_until - self.pdp_request_epoch)
        reused = self.pdp_reused.get(block_id, False)
        if not reused:
            return -2e15 - remaining  # inserted: highest remaining PD evicted first
        return -1e15 - remaining  # reused: highest remaining PD evicted first

    def _pdp_victim_key(self, block_id: int) -> tuple:
        """Exact multi-key ordering matching PDPEvictor._victim_key."""
        protected_until = self.pdp_protected_until_epoch.get(block_id, 0)
        expired = protected_until <= self.pdp_request_epoch
        if expired:
            epoch = self.pdp_last_access_epoch.get(block_id, 0)
            last_accessed = self.free_table[block_id].last_accessed
            return (0, epoch, last_accessed, block_id)
        remaining = max(0, protected_until - self.pdp_request_epoch)
        reused = self.pdp_reused.get(block_id, False)
        return (1, 1 if reused else 0, -remaining,
                -self.pdp_last_access_epoch.get(block_id, 0), block_id)

    # ---- end PDP helpers ---------------------------------------------------

    def _refresh(self):
        self._rebuild_sorted_dict()

        self.last_refresh_time = time.time()
        # Create a list of (block_id, survival_time) tuples
        survival_list = [
            (block_id, self.free_table[block_id].cache_hint['id'], self.free_table[block_id].last_accessed)
            for block_id in self.free_table
        ]
        # Sort by survival_time in descending order
        survival_list.sort(key=lambda x: (x[2], x[1]))

        # mark outdated blocks
        to_delete_cnt = 0
        for block_id, _, _ in survival_list:
            block = self.free_table[block_id]
            if self.get_policy(block.cache_hint) in ('rrip', 'pdp'):
                continue
            id = block.cache_hint['id']
            if self.id_to_last_access[id] == block.last_accessed:
                continue
            if self.id_to_last_access[id] != block.last_accessed:
                self.to_delete_blocks.append((block_id, block.content_hash, block.last_accessed))
                to_delete_cnt += 1

class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the Block. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    # CLEANUP_THRESHOLD determines the maximum allowable size of the priority
    # queue relative to the free table size. When this threshold is exceeded,
    # a cleanup operation is triggered to reduce memory usage.
    CLEANUP_THRESHOLD = 50

    def __init__(self):
        self.free_table: Dict[int, BlockMetaData] = {}
        self.priority_queue = []
        self.stat = CacheStat()

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if len(self.free_table) == 0:
            raise ValueError("No usable cache memory left")

        while self.priority_queue:
            # We do not remove outdated entries from the priority queue at the
            # time of updating the last_accessed timestamp. Instead, outdated
            # entries are filtered out here during eviction. Outdated entries
            # would either not in the free table, or have older last accessed
            # time.
            last_accessed, _, block_id, content_hash = heapq.heappop(
                self.priority_queue)
            if (block_id in self.free_table and
                    self.free_table[block_id].last_accessed == last_accessed):
                survival_time = time.time() - self.free_table[block_id].last_accessed
                self.stat.append("survival_times", survival_time)
                self.free_table.pop(block_id)
                return block_id, content_hash

        raise ValueError("No usable cache memory left")

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float, cache_hint: dict):
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed,
                                                  cache_hint)
        heapq.heappush(
            self.priority_queue,
            (last_accessed, -num_hashed_tokens, block_id, content_hash))
        self._cleanup_if_necessary()

    def update(self, block_id: int, last_accessed: float, cache_hint: dict):
        if 'use_fifo' not in cache_hint or cache_hint['use_fifo'] == 0:
            self.free_table[block_id].last_accessed = last_accessed
        self.free_table[block_id].cache_hint = cache_hint

    def _cleanup_if_necessary(self):
        if len(self.priority_queue) > LRUEvictor.CLEANUP_THRESHOLD * len(
                self.free_table):
            self._cleanup()

    def _cleanup(self):
        new_priority_queue: List[Tuple[float, int, int, int]] = []

        for block_id, block in self.free_table.items():
            new_priority_queue.append(
                (block.last_accessed, -block.num_hashed_tokens, block_id,
                 block.content_hash))
        heapq.heapify(new_priority_queue)

        self.priority_queue = new_priority_queue

    def remove(self, block_id: int):
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return len(self.free_table)

def make_evictor(eviction_algorithm: str, config: str) -> Evictor:
    if eviction_algorithm == 'lru':
        return LRUEvictor()
    if eviction_algorithm == 'pdp':
        return PDPEvictor(config)
    else:
        return LRUMLEvictor(config)
