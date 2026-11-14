"""Bounded asynchronous offline planning for an uninterrupted game loop.

The caller keeps sending real game inputs while workers simulate candidates.
No game-thread method waits on a future. Cancellation invalidates the epoch;
already-running workers finish their bounded search and cannot publish stale
states or create an unbounded queue of replacement work.
"""

from __future__ import annotations

from concurrent.futures import CancelledError, ProcessPoolExecutor
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any


def _worker(config: dict) -> dict:
    if config.pop("recovery",False):
        from .recovery import plan_recovery
        return plan_recovery(**config)
    from .room_planning import plan_room
    return plan_room(**config)


def _warm_worker():
    # Import/initialize before the game connects, outside its input loop.
    from .room_planning import plan_room
    return True


def _state_dict(state) -> dict:
    value = state.to_dict() if hasattr(state, "to_dict") else dict(state)
    # Canonicalize tuples and protect the anchor from later caller mutations.
    return json.loads(json.dumps(value, allow_nan=False))


@dataclass
class _Batch:
    generation: int
    anchor: dict
    remaining: int = 0
    navigation_revision: int = 0
    search_seconds: float = 0
    results: list[dict] = field(default_factory=list)


class BackgroundRoomPlanner:
    """One live generation plus bounded stale work in a persistent process pool.

    ``request`` returns false when a current request exists or no worker slot
    is free. ``poll`` returns immediately and yields at most one result for a
    generation. Returned candidates include ``anchor_state`` and ``generation``
    so the live controller can independently reject a now-incompatible anchor.
    """

    def __init__(self, map_path, workers: int = 2, seconds: float = 2,
                 expansions: int = 3000):
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("seconds must be positive and finite")
        if isinstance(expansions, bool) or not isinstance(expansions, int) or expansions < 1:
            raise ValueError("expansions must be a positive integer")
        self.map_path = str(Path(map_path).resolve())
        self.workers = workers
        self.seconds = float(seconds)
        self.expansions = expansions
        self._executor = ProcessPoolExecutor(max_workers=workers)
        self._futures: dict[Any, int] = {}
        self._batch: _Batch | None = None
        self._generation = 0
        self._closed = False

    @property
    def generation(self) -> int:
        return self._generation

    def warmup(self):
        """Call before connecting the game; never from a per-frame callback."""
        futures=[self._executor.submit(_warm_worker) for _ in range(self.workers)]
        for future in futures:
            future.result()

    @property
    def busy(self) -> bool:
        if self._closed:
            return False
        return self._batch is not None or any(not future.done() for future in self._futures)

    def _discard_finished_stale(self):
        current = self._batch.generation if self._batch else None
        for future, generation in list(self._futures.items()):
            if generation != current and future.done():
                # Its result (including any exception) is intentionally ignored.
                del self._futures[future]

    def request(self, state, variant: int = 0, forbidden=None, visited_rooms=None, exploration=None, recovery=False) -> bool:
        if self._closed:
            return False
        self._discard_finished_stale()
        if self._batch is not None:
            return False
        available = self.workers - sum(not future.done() for future in self._futures)
        if available <= 0:
            return False
        anchor = _state_dict(state)
        self._generation += 1
        self._batch = _Batch(self._generation, anchor,
                             navigation_revision=int((exploration or {}).get("navigation_revision",0)),
                             search_seconds=min(20.0,self.seconds*(1+min(9,variant//12))))
        for offset in range(available):
            config = {
                "map_path": self.map_path, "start_state": anchor,
                "seconds": min(20.0,self.seconds*(1+min(9,variant//12))),
                "expansions": self.expansions*(1+min(9,variant//12)),
                "variant": variant + offset, "forbidden": list(forbidden or []),
                "visited_rooms": list(visited_rooms or []),
            }
            if exploration is not None:
                config["exploration"]=exploration
            if recovery:
                config["recovery"]=True
            try:
                future = self._executor.submit(_worker, config)
            except Exception as error:
                self._batch.results.append({"can_execute": False, "reason": "worker_error",
                                            "error": f"{type(error).__name__}: {error}"})
                # A broken pool must produce one inspectable error, not a burst
                # of repeated submissions from the same game update.
                break
            self._futures[future] = self._generation
            self._batch.remaining += 1
        return True

    @staticmethod
    def _score(candidate, anchor):
        final = candidate.get("final_state") or {}
        gained = len(set(final.get("berries", [])) - set(anchor.get("berries", [])))
        transition = bool(final.get("room") and final["room"] != anchor.get("room"))
        distance = math.hypot(float(final.get("x", anchor.get("x", 0)))-float(anchor.get("x", 0)),
                              float(final.get("y", anchor.get("y", 0)))-float(anchor.get("y", 0)))
        if candidate.get("requires_periodic_anchor") or candidate.get("requires_future_anchor"):
            return gained,0,0,int(transition),-len(candidate.get("actions") or []),-distance
        priority=candidate.get("objective_priority",int(transition))
        return gained,priority,float(candidate.get("goal_progress",0)),int(transition),distance,-len(candidate.get("actions") or [])

    def _finish(self, result: dict) -> dict:
        batch = self._batch
        assert batch is not None
        selected = dict(result)
        selected["anchor_state"] = _state_dict(batch.anchor)
        selected["generation"] = batch.generation
        selected["navigation_revision"]=batch.navigation_revision
        selected["search_seconds"]=batch.search_seconds
        self._batch = None
        # cancel() cannot stop a running process. Retain those futures as stale
        # capacity consumers until done rather than queueing replacements.
        for future, generation in list(self._futures.items()):
            if generation == batch.generation:
                future.cancel()
                if future.done():
                    del self._futures[future]
        return selected

    def poll(self) -> dict | None:
        if self._closed:
            return None
        self._discard_finished_stale()
        batch = self._batch
        if batch is None:
            return None
        for future, generation in list(self._futures.items()):
            if generation != batch.generation or not future.done():
                continue
            del self._futures[future]
            batch.remaining -= 1
            try:
                result = future.result()  # Guarded by done(); never blocks.
                if not isinstance(result, dict):
                    raise TypeError("offline worker returned a non-object candidate")
            except CancelledError:
                result = {"can_execute": False, "reason": "worker_cancelled"}
            except Exception as error:
                result = {"can_execute": False, "reason": "worker_error",
                          "error": f"{type(error).__name__}: {error}"}
            batch.results.append(result)
        executable = [r for r in batch.results if r.get("can_execute") and r.get("actions")]
        respawns=[r for r in batch.results if r.get("requires_room_respawn")]
        if respawns:
            return self._finish(respawns[0])
        strong = [r for r in executable if self._score(r,batch.anchor)[0]>0
                  or (r.get("active_berry") and r.get("goal_progress",0)>0
                      and (r.get("goals") or [{}])[0].get("kind")=="berry")
                  or (not r.get("berry_pending") and (r.get("final_state") or {}).get("room",batch.anchor.get("room"))!=batch.anchor.get("room"))]
        if strong:
            return self._finish(max(strong, key=lambda r: self._score(r, batch.anchor)))
        if batch.remaining:
            return None
        if executable:
            return self._finish(max(executable, key=lambda r: self._score(r, batch.anchor)))
        errors = [r for r in batch.results if r.get("reason") == "worker_error"]
        goals=[r["primary_goal"] for r in batch.results if r.get("primary_goal")]
        return self._finish({
            "can_execute": False, "reason": "worker_error" if errors else "offline_search_no_new_route",
            "candidate_reasons": [r.get("reason", r.get("status")) for r in batch.results],
            "errors": [r["error"] for r in errors],
            "primary_goal":max(goals,key=goals.count) if goals else None,
            "pending_goals":sorted(set(goals)),
            "berry_pending":sorted({berry for r in batch.results for berry in r.get("berry_pending",[])}),
        })

    def cancel_epoch(self) -> int:
        """Invalidate all outstanding results immediately, without waiting."""
        self._generation += 1
        self._batch = None
        for future in list(self._futures):
            future.cancel()
            if future.done():
                del self._futures[future]
        return self._generation

    def close(self, wait: bool = False):
        """Stop accepting work. Default shutdown does not wait for workers."""
        if self._closed:
            return
        self.cancel_epoch()
        self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close(wait=False)
