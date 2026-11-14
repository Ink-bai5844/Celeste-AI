"""Request-local exact memoization for raw offline Simulator.step calls.

The cache belongs to one search and one unchanged simulator/map configuration.
Public route checks must wrap this cache; final route admission replays raw
physics. No frame, phase, resource, timer or player-depth field is normalized.
"""
from __future__ import annotations

from collections import OrderedDict

from .physics import Simulator

_MISSING = object()


class ExactStepCache:
    """Bounded LRU of ``(complete SimState, input) -> (after, dash event)``.

    ``on_cache_hit`` preserves the external guard of a skipped step override.
    Misses use the original step, including its guard, exactly once. The hook
    may also be supplied as ``base.step.cache_hit_guard``. A raw Simulator with
    an instance-level step override is cached only when this hook is present.
    Subclasses, constraint wrappers and test simulators are delegated intact.
    This wrapper is deliberately local to a single-threaded search request.
    """

    def __init__(self, base, *, max_entries=32768, on_cache_hit=None):
        if type(max_entries) is not int or not 1 <= max_entries <= 32768:
            raise ValueError("max_entries must be an integer in 1..32768")
        if on_cache_hit is None and type(base) is Simulator:
            on_cache_hit = getattr(base.step, "cache_hit_guard", None)
        if on_cache_hit is not None and not callable(on_cache_hit):
            raise TypeError("on_cache_hit must be callable")
        self.base = base
        self.max_entries = max_entries
        self.on_cache_hit = on_cache_hit
        self.enabled = (type(base) is Simulator
                        and ("step" not in base.__dict__ or on_cache_hit is not None))
        self._entries = OrderedDict()
        self.hits = self.misses = self.evictions = self.bypasses = 0

    def __getattr__(self, name):
        return getattr(self.base, name)

    def step(self, state, mask):
        if not self.enabled:
            self.bypasses += 1
            return self.base.step(state, mask)
        key = (state, mask)
        cached = self._entries.get(key, _MISSING)
        if cached is not _MISSING:
            if self.on_cache_hit is not None:
                self.on_cache_hit()
            self.hits += 1
            self._entries.move_to_end(key)
            after, started_dash = cached
            self.base.last_step_started_dash = started_dash
            return after
        self.misses += 1
        after = self.base.step(state, mask)
        # Failed steps never enter the cache. Preserve the actual event even
        # when a spring/refill hides dash consumption in the returned state.
        self._entries[key] = (after, bool(self.base.last_step_started_dash))
        if len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.evictions += 1
        return after

    def cache_info(self):
        return dict(enabled=self.enabled, hits=self.hits, misses=self.misses,
                    evictions=self.evictions, bypasses=self.bypasses,
                    entries=len(self._entries), max_entries=self.max_entries)

    def clear(self):
        """Release entries without obscuring the request's work counters."""
        self._entries.clear()
