"""Offline escape planning from a recurring airborne motion such as a spring.

The caller holds neutral input until an absolute future command frame and must
validate the full forecast state, including subpixels, before executing. A
similar-looking later bounce is not interchangeable with that forecast anchor.
"""
from __future__ import annotations

import math
import time

from .physics import SimState
from .planner import Goal, coverage_goals, room_is_supported
from .room_planning import _context, _finish_safely, _safe_anchor, action_fingerprint
from .routes import snapshot
from .strategies import search_goal


def _recovery_supports(room, start):
    """Sample actual exposed floors, including short ledges, away from springs."""
    points = {(goal.x, goal.y) for goal in coverage_goals(room)}
    # Coverage sampling is deliberately sparse; recovery also needs short safe
    # ledges that may lie between its four-tile samples.
    for ty in range(2, room.height // 8):
        run = []
        for tx in range(1, room.width // 8):
            exposed = (room.tile_at(tx, ty) and not room.tile_at(tx, ty - 1)
                       and not room.tile_at(tx, ty - 2))
            if exposed:
                run.append(tx)
            if run and (not exposed or tx == room.width // 8 - 1):
                points.add((room.x + (run[0] + run[-1] + 1) * 4, room.y + ty * 8))
                run = []
    goals = []
    for x, y in points:
        if not 16 <= math.hypot(x - start.x, y - start.y) <= 240:
            continue
        if any(entity.kind == "spring" and abs(x - entity.x) < 20
               and abs(y - entity.y) < 20 for entity in room.entities):
            continue
        if any(entity.kind == "spikesUp" and abs(entity.y - y) < 4
               and entity.x - 5 < x < entity.x + max(8, entity.width) + 5
               for entity in room.entities):
            continue
        goals.append(Goal("surface", room.name, x, y))
    goals.sort(key=lambda goal: (math.hypot(goal.x - start.x, goal.y - start.y), goal.key))
    return goals


def _future_intercept(simulator, observed, *, minimum_frames=180, phase_window=90):
    """Advance actual neutral dynamics; never identify a cycle by rounded pose."""
    state = observed
    springs = [entity for entity in simulator.map.rooms[observed.room].entities
               if entity.kind == "spring"]
    for advanced in range(1, minimum_frames + phase_window + 1):
        state = simulator.step(state, 0)
        if state.dead or state.room != observed.room:
            return None, advanced, "neutral_forecast_died_or_changed_room"
        if advanced < minimum_frames:
            continue
        rising = (state.state == 0 and state.vy < -150 and any(
            abs(state.x - spring.x) < 20 and spring.y - 176 <= state.y <= spring.y + 12
            for spring in springs))
        if rising or _safe_anchor(simulator, state):
            return state, advanced, "future_anchor"
    return None, minimum_frames + phase_window, "no_future_rising_or_safe_phase"


def plan_recovery(map_path: str, start_state: dict, seconds: float = 2,
                  expansions: int = 3000, variant: int = 0,
                  forbidden: list[str] | None = None,
                  visited_rooms: list[str] | None = None,
                  exploration: dict | None = None) -> dict:
    """Find a continuously simulated route from an airborne anchor to safe rest.

    Parameters match the normal background planner, so workers can select this
    function without changing their serialized request format. Repeated neutral
    spring bouncing never counts as a safe endpoint or meaningful recovery.
    Search lasts at most two seconds against an anchor at least three seconds
    ahead; the caller rejects results that arrive after their execution frame.
    """
    if not math.isfinite(seconds) or seconds <= 0 or expansions < 1:
        raise ValueError("seconds and expansions must be positive finite budgets")
    map_data, simulator = _context(map_path)
    observed = start = SimState.from_dict(start_state)
    if start.room not in map_data.rooms:
        raise ValueError(f"Unknown start room {start.room!r}")
    began = time.monotonic()
    deadline = began + min(seconds, 2.0)
    attempts = []
    total_expansions = total_frames = 0
    blacklist = set(forbidden or ())

    def empty(reason):
        return dict(actions=[], expected=[snapshot(start)], final_state=start.to_dict(),
                    anchor_state=start.to_dict(), observation_anchor_state=observed.to_dict(),
                    goals=[], primary_goal=None,
                    status="no_progress", reason=reason, can_execute=False,
                    fingerprint=action_fingerprint([]), expansions=total_expansions,
                    simulated_frames=total_frames, attempts=attempts,
                    requires_periodic_anchor=False, requires_future_anchor=True, runtime_verified=False,
                    wall_seconds=round(time.monotonic() - began, 3))

    if start.dead or start.state != 0:
        return empty("recovery_requires_alive_normal_state")
    if not room_is_supported(simulator, start.room):
        return empty("unsupported_current_room")
    future, neutral_frames, reason = _future_intercept(simulator, observed)
    total_frames += neutral_frames
    if future is None:
        return empty(reason)
    start = future
    goals = _recovery_supports(map_data.rooms[start.room], start)
    if not goals:
        return empty("no_nearby_safe_supports")
    # Worker variants diversify action search; later rounds can prioritize a
    # different nearby ledge without encoding a particular room or direction.
    offset = (variant // 4) % min(3, len(goals))
    goals = goals[offset:] + goals[:offset]
    for index, goal in enumerate(goals[:8]):
        remaining = expansions - total_expansions
        now = time.monotonic()
        if remaining <= 0 or now >= deadline:
            break
        budget_end = deadline if index else now + (deadline - now) * .72
        result = search_goal(simulator, start, goal, deadline=budget_end,
                             max_expansions=remaining, max_frames=360,
                             variant=variant, allow_unsupported=False)
        total_expansions += result.expansions
        total_frames += result.simulated_frames
        attempts.append(dict(goal=goal.key, kind="surface", reached=result.reached, reason=result.reason))
        actions, final, states = _finish_safely(simulator, start, result.actions)
        total_frames += len(actions)
        fingerprint = action_fingerprint(actions)
        if (not actions or fingerprint in blacklist or final.room != start.room
                or not _safe_anchor(simulator, final)
                or math.hypot(final.x - start.x, final.y - start.y) < 16):
            continue
        # A momentary on-ground flag at the spring is insufficient: prove that
        # neutral input leaves this new support stable for a full second.
        resting = simulator.trace(final, [0] * 60)
        total_frames += 60
        if any(not _safe_anchor(simulator, state) or state.room != final.room
               or abs(state.x - final.x) >= 1 or abs(state.y - final.y) >= 1
               for state in resting):
            continue
        if time.monotonic() >= deadline:
            return empty("recovery_search_budget_exceeded")
        return dict(actions=actions, expected=[snapshot(state) for state in states],
                    final_state=final.to_dict(), anchor_state=start.to_dict(),
                    execution_anchor_state=start.to_dict(), observation_anchor_state=observed.to_dict(),
                    execute_at_frame=start.frame, neutral_forecast_frames=neutral_frames,
                    goals=[dict(goal=goal.key, kind="surface", room=goal.room, reached=goal.achieved(final))],
                    primary_goal=goal.key, status="partial", reason="future_intercept_safe_recovery",
                    safe_endpoint=True, can_execute=True, fingerprint=fingerprint,
                    requires_periodic_anchor=False, requires_future_anchor=True, runtime_verified=False,
                    expansions=total_expansions, simulated_frames=total_frames,
                    attempts=attempts, wall_seconds=round(time.monotonic() - began, 3))
    return empty("offline_recovery_no_safe_route")
