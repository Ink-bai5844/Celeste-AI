"""Climb down actual static walls before a continuous jump/catch transfer."""
from __future__ import annotations

from dataclasses import replace
import time

from .physics import DASH, DOWN, GRAB, JUMP, LEFT, RIGHT, UP
from .search_progress import remember_segment
from .wall_transfers import _static_contact


def _launch_heights(origin, target, wall):
    waypoint = getattr(target, "waypoint", None)
    _, guide = waypoint(origin) if waypoint is not None else (target.x, target.y)
    bounds = getattr(target, "bounds", None)
    bounds = bounds(origin) if callable(bounds) else bounds
    if bounds:
        guide = max(bounds[1]+1, min(guide, bounds[3]+9))
    # A little below the contact guide leaves space to jump around a lip.
    # Neighboring heights account for its thickness without fixing room/ticks.
    lowest = wall[3] + (5 if origin.ducking else 10)
    heights = []
    for offset in (0, -8, 8, -16, 16):
        height = min(lowest, guide+4+offset)
        if height >= origin.y+8 and height not in heights:
            heights.append(height)
    return heights


def _contact_on_wall(state, origin, wall):
    height = 6 if state.ducking else 11
    return (not state.dead and state.room == origin.room and state.state == 1
            and state.facing == origin.facing and state.stamina > 0
            and abs(state.x-origin.x) < .1 and state.y > wall[1]
            and state.y-height < wall[3])


def _launch_recipes(origin, target, allow_dash):
    waypoint = getattr(target, "waypoint", None)
    x, _ = waypoint(origin) if waypoint is not None else (target.x, target.y)
    toward = RIGHT if x >= origin.x else LEFT
    away = LEFT if origin.facing > 0 else RIGHT
    directions = tuple(dict.fromkeys((toward, 0, away)))
    face = (RIGHT if target.facing > 0 else LEFT) if target.kind == "wall_grip" else toward
    catch = GRAB | UP | face
    for ticks in (12, 16, 24):
        for direction in directions:
            yield (JUMP | direction,)*ticks + (catch,)*140, f"jump:{direction}:{ticks}"
    if allow_dash and origin.dashes > 0:
        for ticks in (28, 20):
            for direction in directions:
                for aim in (toward, UP | toward):
                    yield ((JUMP | direction,)*ticks + (DASH | aim,)*6 + (catch,)*140,
                           f"jump_dash:{direction}:{ticks}:{aim}")


def wall_descent_edges(simulator, origin, target, *, deadline,
                       max_frames=220, stats=None, max_expansions=None, max_edges=4):
    """Return real ``(actions, state, label)`` descending-wall relay edges.

    Target height selects a few launch opportunities, never a replacement
    state or goal. Descent stops on lost contact; launches use the caller's
    strict target and constraints on every update. All resources stay physical.
    """
    if max_frames <= 0 or max_edges <= 0 or time.monotonic() >= deadline:
        return []
    stats = stats if stats is not None else {}
    limit = min(max_expansions if max_expansions is not None else float("inf"),
                stats.get("max_expansions", float("inf")),
                stats.get("expansions", 0)+128)
    if stats.get("expansions", 0) >= limit:
        return []
    wall = _static_contact(simulator, origin)
    if wall is None:
        return []
    if target.achieved(origin):
        return [([], origin, "wall_descent:already_reached:dash_starts=0")]
    heights = _launch_heights(origin, target, wall)
    if not heights:
        return []
    allow_dash = not getattr(simulator, "no_dash", False)
    seen = {}

    def remember(actions, states, dash_starts, label):
        state = states[-1]
        key = replace(state, frame=0, reason="")
        rank = (len(dash_starts), -state.dashes, -state.stamina,
                len(actions), tuple(-tick for tick in dash_starts))
        if key not in seen or rank < seen[key][0]:
            seen[key] = (rank, list(actions), state, label, list(states))

    # Share this genuine input prefix across launch branches; every stored
    # point was reached by the same uninterrupted physical descent.
    stats["expansions"] = stats.get("expansions", 0)+1
    descent = GRAB | DOWN | (RIGHT if origin.facing > 0 else LEFT)
    state, actions, states, events = origin, [], [origin], []
    launches = {}
    for _ in range(max_frames):
        if time.monotonic() >= deadline:
            break
        state = simulator.step(state, descent)
        started = bool(getattr(simulator, "last_step_started_dash", False))
        stats["simulated_frames"] = stats.get("simulated_frames", 0)+1
        actions.append(descent);states.append(state)
        if started:
            events.append(len(actions))
        if state.dead or not allow_dash and started:
            break
        if target.achieved(state):
            remember(actions, states, events, "descent_goal")
            break
        if not _contact_on_wall(state, origin, wall):
            break
        for height in heights:
            if height not in launches and state.y >= height:
                launches[height] = (list(actions), list(states), list(events))
        if len(launches) == len(heights):
            break
    # All non-dash launches precede relaxed alternatives across every height.
    # This avoids spending a narrow quota on dashes from the first height.
    recipes_seen = set()
    for dashed in ((False, True) if allow_dash else (False,)):
        for height in heights:
            if height not in launches:
                continue
            prefix, prefix_states, prefix_events = launches[height]
            launch = prefix_states[-1]
            for recipe, label in _launch_recipes(launch, target, dashed):
                if ("jump_dash:" in label) != dashed:
                    continue
                if time.monotonic() >= deadline or stats.get("expansions", 0) >= limit:
                    break
                recipe = recipe[:max_frames-len(prefix)]
                identity = tuple(prefix)+tuple(recipe)
                if not recipe or identity in recipes_seen:
                    continue
                recipes_seen.add(identity)
                stats["expansions"] = stats.get("expansions", 0)+1
                state, actions, states, events = launch, list(prefix), list(prefix_states), list(prefix_events)
                for mask in recipe:
                    if time.monotonic() >= deadline:
                        break
                    state = simulator.step(state, mask)
                    started = bool(getattr(simulator, "last_step_started_dash", False))
                    stats["simulated_frames"] = stats.get("simulated_frames", 0)+1
                    actions.append(mask);states.append(state)
                    if started:
                        events.append(len(actions))
                    if state.dead or not allow_dash and (mask & DASH or started):
                        break
                    if target.achieved(state):
                        remember(actions, states, events, f"height:{height:g}:{label}")
                        break
                    if state.room != origin.room:
                        break
    chosen, deferred, bands = [], [], set()
    for row in sorted(seen.values(), key=lambda row:row[0]):
        rank, _, state, _, _ = row
        band = (rank[0], state.dashes, state.state, round(state.y/8),
                round(state.vx/45), round(state.vy/45), int(state.stamina/20))
        if band in bands:
            deferred.append(row)
        else:
            chosen.append(row);bands.add(band)
        if len(chosen) >= max_edges:
            break
    chosen.extend(deferred[:max(0,max_edges-len(chosen))])
    result = []
    for rank, actions, state, label, states in sorted(chosen, key=lambda row:row[0]):
        remember_segment(origin, state, states)
        result.append((actions, state, f"wall_descent:{label}:dash_starts={rank[0]}"))
    return result
