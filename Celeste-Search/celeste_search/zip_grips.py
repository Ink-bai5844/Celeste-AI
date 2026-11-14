"""Actual side grips and rides for zip platforms with unsafe top surfaces."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .physics import GRAB, LEFT, RIGHT, UP, _spike_rect, _spike_rides
from .search_progress import remember_segment, search_tracker
from .wall_moves import WallGripGoal


def zip_top_blocked(entity, room):
    """Whether attached upward spikes cover the complete platform top.

    Partial coverage is deliberately not classified as blocked: the ordinary
    boarding search may still find a safe landing on an uncovered part.
    """
    if entity.kind != "zipMover":
        return False
    intervals = []
    for spike in room.entities:
        if spike.kind == "spikesUp" and _spike_rides(spike, entity):
            left, _, right, _ = _spike_rect(spike)
            intervals.append((max(entity.x, left), min(entity.x + entity.width, right)))
    covered = entity.x
    for left, right in sorted(intervals):
        if left > covered:
            return False
        covered = max(covered, right)
    return covered >= entity.x + entity.width


def zip_side_grip_goals(simulator, start, goal):
    """Keep both real faces, preferring the destination-facing side of each zip."""
    data = simulator.map_data if hasattr(simulator, "map_data") else simulator.map
    if start.dead or start.room not in data.rooms:
        return []
    room = data.rooms[start.room]
    candidates = []
    for entity in room.entities:
        if entity.kind != "zipMover" or not entity.nodes or entity.height < 12:
            continue
        identity = f"{room.name}:{entity.id}"
        if identity in start.broken:
            continue
        record = next((row for row in start.mechanisms if row[0] == identity), None)
        left, top = (round(record[3]), round(record[4])) if record else (entity.x, entity.y)
        bottom, right = top + entity.height, left + entity.width
        inset = min(12, entity.height / 2)
        y = max(top + inset, min(start.y - 32, bottom - inset))
        target_x = getattr(goal, "tile_x", goal.x)
        favored = -1 if target_x >= entity.nodes[0][0] + entity.width / 2 else 1
        original = (entity.x, entity.y, entity.x + entity.width, entity.y + entity.height)
        for facing in (favored, -favored):
            x = left - 4 if facing > 0 else right + 4
            distance = math.hypot(x - start.x, y - start.y)
            if (distance > 400 or not room.x + 4 <= x <= room.x + room.width - 4
                    or simulator.solid_at(room.name, x, y, state=start)):
                continue
            grip = WallGripGoal("wall_grip", room.name, x, y,
                    id=f"zip_grip:{identity}:{facing}", rect=original, facing=facing,
                    identity=identity, mechanism_kind="zipMover", grip_offset=y - top)
            # Face preference is stronger than a modest approach detour. Both
            # faces remain available if the preferred side cannot be reached.
            candidates.append((distance + (0 if facing == favored else 100),
                               identity, facing, grip))
    return [item[-1] for item in sorted(candidates, key=lambda row: row[:3])[:12]]


def _contact(state, grip):
    bounds = grip.bounds(state)
    if (bounds is None or state.dead or state.room != grip.room or state.state != 1
            or state.facing != grip.facing or state.stamina <= 0):
        return False
    left, top, right, bottom = bounds
    x = left - 4 if grip.facing > 0 else right + 4
    return abs(state.x - x) < .1 and state.y > top and state.y - 11 < bottom


@dataclass(frozen=True)
class ZipContactGoal(WallGripGoal):
    """Accept the actual first side contact, independent of a guide height.

    Triggering a moving platform is an event. Requiring the player to climb
    to a sampled height first can skip the departure window of that event.
    Held rides and later climbs still require their own simulated inputs.
    """

    def achieved(self, state):
        return _contact(state, self)


def zip_grip_edges(simulator, start, grip, goal, *, deadline,
                   max_frames=140, stats=None, cycle_windows=False, return_climb=False):
    """Yield held-ride states before a later collision or loss of wall contact.

    These are internal search edges, not user-target successes. Every returned
    state keeps the actual stamina, resources, lift and mechanism timers. The
    caller must solve and replay the entire remaining ordered route.

    ``cycle_windows`` additionally samples the first actual return and renewed
    outgoing launch windows, stopping at that next outgoing endpoint. Optional
    ``return_climb`` climbs only during return, then holds below the top edge.
    Both variants retain real stamina and stop when contact/constraints fail.
    The default retains the original short-ride samples and endpoint stop.
    """
    if (grip.mechanism_kind != "zipMover" or not _contact(start, grip)
            or max_frames < 1 or time.monotonic() >= deadline):
        return
    if (stats is not None and stats.get("max_expansions") is not None
            and stats.get("expansions", 0) >= stats["max_expansions"]):
        return
    if stats is not None:
        stats["expansions"] = stats.get("expansions", 0) + 1
    state = start
    actions, states = [], [start]
    preview = search_tracker(start)
    held = GRAB | (RIGHT if grip.facing > 0 else LEFT)
    initial = next((row for row in start.mechanisms if row[0] == grip.identity), None)
    previous_phase = initial[1] if initial else None
    seen_return = previous_phase in (4, 5)
    for tick in range(1, max_frames + 1):
        if time.monotonic() >= deadline:
            return
        record = next((row for row in state.mechanisms if row[0] == grip.identity), None)
        mask = held
        # A body-safe side inset prevents the deliberately bounded climb
        # policy from pulling over an unsafe top. Collision remains decisive.
        if (cycle_windows and return_climb and record and record[1] == 4
                and state.y-round(record[4]) > 16):
            mask |= UP
        successor = simulator.step(state, mask)
        if stats is not None:
            stats["simulated_frames"] = stats.get("simulated_frames", 0) + 1
        state = successor
        actions.append(mask)
        states.append(state)
        # A public hit between ride samples must survive even if the next
        # update loses contact or crushes the player. A room crossing can also
        # satisfy the target on the same tick that the old grip disappears.
        if not state.dead and goal.achieved(state):
            remember_segment(start, state, states)
            if preview is not None:
                preview.report(lambda: [("best", state, states)], 1, tick, remember=True)
            yield list(actions), state, f"zip_grip:{grip.identity}:{grip.facing}:goal"
            return
        if not _contact(state, grip):
            return
        record = next((row for row in state.mechanisms if row[0] == grip.identity), None)
        phase = record[1] if record else None
        seen_return |= phase in (4, 5)
        complete = phase == 3 and (not cycle_windows or seen_return)
        phase_changed = cycle_windows and phase is not None and phase != previous_phase
        outgoing = bool(cycle_windows and record and phase == 2 and record[2] in (8, 16, 24))
        returning = bool(cycle_windows and record and phase == 4 and record[2] in (8, 16, 24, 48, 72, 96))
        returned = cycle_windows and phase == 5 and phase_changed
        endpoint = phase == 3 and (phase_changed or complete)
        if preview is not None:
            preview.report(lambda: [("alternative", state, states)], 1, tick)
        if tick in (8, 16, 24, 32) or complete or phase_changed or outgoing or returning:
            remember_segment(start, state, states)
            suffix = ("return_end" if returned else "ride" if endpoint
                      else f"return{record[2]}" if returning else f"outgoing{record[2]}" if outgoing
                      else f"phase{phase}" if phase_changed else f"launch{tick}")
            variant = "return_climb:" if return_climb else ""
            cycle = "cycle1:" if seen_return and phase in (0, 1, 2, 3) else ""
            label = f"zip_grip:{grip.identity}:{grip.facing}:" + variant + cycle + suffix
            yield list(actions), state, label
        if complete:
            return
        previous_phase = phase
