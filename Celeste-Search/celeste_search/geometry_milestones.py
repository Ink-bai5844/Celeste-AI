"""Transient platform milestones for continuous offline route decomposition.

These points are search targets, never evidence of reachability or safe places
to idle while a worker runs. Every approach and continuation still has to pass
the physics simulator. In particular a useful detour may first move away from
the final berry; geometry generation deliberately does not prune that case.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .planner import Goal


_TRANSIENT_KINDS = frozenset({"crumbleBlock", "zipMover", "fallingBlock"})


@dataclass(frozen=True)
class CassetteSupportGoal(Goal):
    """A live beat-block top; an inactive rectangle is never a landing."""
    entity_key: str = ""
    width: float = 0.0

    @property
    def transient(self):
        return True

    @property
    def requires_continuation(self):
        return True

    @property
    def idle_endpoint_allowed(self):
        return False

    def _block(self, state):
        if state.room != self.room or getattr(state, "cassette_clock", None) is None:
            return None
        return next((block for block in state.cassette_blocks if block.id == self.entity_key), None)

    def waypoint(self, state):
        block = self._block(state)
        return (block.x + block.width / 2, block.y) if block is not None else (self.x, self.y)

    def achieved(self, state):
        block = self._block(state)
        return bool(block is not None and block.active and block.collidable and not state.dead
                    and state.on_ground and state.vy >= 0 and abs(state.y-block.y) < 0.1
                    and state.x+4 > block.x and state.x-4 < block.x+block.width)


def cassette_support_goals(room, state):
    """Inventory observed blocks, retaining inactive ones for timed approaches."""
    if state is None or state.room != room.name or getattr(state, "cassette_clock", None) is None:
        return []
    known = {f"{room.name}:{entity.id}" for entity in room.entities if entity.kind == "cassetteBlock"}
    return [CassetteSupportGoal("surface", room.name, block.x+block.width/2, block.y,
                               id="cassette_support:"+block.id, entity_key=block.id, width=block.width)
            for block in state.cassette_blocks if block.id in known and block.active and block.width >= 8]


@dataclass(frozen=True)
class TransientSupportGoal(Goal):
    """Touch the actual moving support top, then continue without an idle stop."""
    entity_key: str = ""
    entity_kind: str = ""
    width: float = 0.0
    origin_x: float = 0.0
    origin_y: float = 0.0

    @property
    def transient(self) -> bool:
        return True

    @property
    def requires_continuation(self) -> bool:
        return True

    @property
    def idle_endpoint_allowed(self) -> bool:
        return False

    def _record(self, state):
        records = getattr(state, "mechanisms", ()) if state.room == self.room else ()
        return next((row for row in records if row[0] == self.entity_key),
                    (self.entity_key, 0, 0, self.origin_x, self.origin_y, 0.0))

    def waypoint(self, state) -> tuple[float, float]:
        """A heuristic can call this instead of chasing the platform's origin."""
        record = self._record(state)
        return round(record[3]) + self.width / 2, float(round(record[4]))

    def achieved(self, state) -> bool:
        if state.dead or state.room != self.room or not state.on_ground or state.vy < 0:
            return False
        if self.entity_key in getattr(state, "broken", ()):
            return False
        record = self._record(state)
        phase = record[1]
        # These are the simulator's collision phases, not guessed native engine
        # coroutine values. A collapsed crumble can become usable again later.
        if (self.entity_kind == "crumbleBlock" and phase == 3
                or self.entity_kind == "fallingBlock" and phase == 4):
            return False
        left, top = round(record[3]), round(record[4])
        return (abs(state.y - top) < 1.1 and state.x + 4 > left
                and state.x - 4 < left + self.width)


def transient_support_goals(room, state=None) -> list[TransientSupportGoal | CassetteSupportGoal]:
    """Return identity-backed current support tops for modeled moving entities.

    Collapsed crumbles stay in the candidate inventory because normal simulation
    can restore them; their achieved() predicate remains false while absent.
    A removed falling block cannot return in this room lifetime and is excluded.
    """
    records = {row[0]: row for row in getattr(state, "mechanisms", ())} if state is not None and state.room == room.name else {}
    broken = getattr(state, "broken", ()) if state is not None and state.room == room.name else ()
    result = []
    for entity in room.entities:
        if entity.kind not in _TRANSIENT_KINDS or entity.width <= 0:
            continue
        key = f"{room.name}:{entity.id}"
        record = records.get(key, (key, 0, 0, entity.x, entity.y, 0.0))
        if key in broken or entity.kind == "fallingBlock" and record[1] == 4:
            continue
        left, top = round(record[3]), round(record[4])
        result.append(TransientSupportGoal(
            "surface", room.name, left + entity.width / 2, float(top), f"support:{key}",
            entity_key=key, entity_kind=entity.kind, width=float(entity.width),
            origin_x=entity.x, origin_y=entity.y))
    result.extend(cassette_support_goals(room, state))
    return sorted(result, key=lambda goal: goal.key)


def continuous_milestones(room, start, goal=None, *, max_distance: float = 280,
                          max_rise: float = 160) -> list[TransientSupportGoal | CassetteSupportGoal]:
    """Nearby transient supports, retaining detours that increase goal distance.

    The caller chooses a bounded number of these targets, searches from the
    real state to one, then continues from the resulting state to its original
    goal. The output itself must never become a live idle/settling endpoint.
    """
    if start.room != room.name:
        return []
    if max_distance <= 0 or max_rise < 0:
        raise ValueError("Milestone distance must be positive and rise nonnegative")
    protected = getattr(goal, "no_dash_until", "") if goal is not None else ""
    result = []
    for support in transient_support_goals(room, start):
        if support.achieved(start):
            continue
        distance = math.hypot(support.x - start.x, support.y - start.y)
        if distance > max_distance or start.y - support.y > max_rise:
            continue
        if protected:
            # dataclasses.replace retains subclass identity and dynamic checks.
            from dataclasses import replace
            support = replace(support, no_dash_until=protected)
        result.append(support)
    return sorted(result, key=lambda support: (
        math.hypot(support.x - start.x, support.y - start.y), support.key))
