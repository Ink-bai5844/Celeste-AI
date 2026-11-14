"""Ordered user routes and shared frame-by-frame constraint monitoring.

Segment zero starts at the actual player anchor. Each remaining segment starts
at the frame its preceding waypoint is reached, with momentum and timers intact.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping

from .assist_validation import AssistValidationError, validate_tile_target


def normalize_route(map_data, value) -> dict:
    """Normalize a route, or wrap a legacy single tile in a one-point route."""
    if not isinstance(value, Mapping):
        raise AssistValidationError("invalid_route")
    if "waypoints" not in value:
        points, segments = [value], [{}]
    else:
        points = value["waypoints"]
        if not isinstance(points, list) or not 1 <= len(points) <= 32:
            raise AssistValidationError("invalid_waypoint_count")
        segments = value.get("segments", [{} for _ in points])
        if not isinstance(segments, list) or len(segments) != len(points):
            raise AssistValidationError("invalid_segment_count")
    normalized = []
    for segment in segments:
        if not isinstance(segment, Mapping) or set(segment) - {"no_dash", "no_ground"}:
            raise AssistValidationError("invalid_segment_constraints")
        if any(type(segment.get(flag, False)) is not bool for flag in ("no_dash", "no_ground")):
            raise AssistValidationError("invalid_segment_constraints")
        normalized.append({flag: segment.get(flag, False) for flag in ("no_dash", "no_ground")})
    return {"waypoints": [validate_tile_target(map_data, point) for point in points],
            "segments": normalized}


def _value(state, key, default=None):
    if isinstance(state, Mapping):
        return state.get(key, state.get("model_state", {}).get(key, default))
    return getattr(state, key, default)


def _dash_active(state):
    return (_value(state, "state", 0) in (2, 5)
            or _value(state, "dash_frames", 0) > 0
            or _value(state, "dash_buffer", 0) > 0)


def _grounded(state):
    # Native telemetry uses Player.OnGround(1), a geometric support probe. It
    # can briefly be true while rising through a one-way platform; that is not
    # a landing. Match the movement model's grounded phase using vertical speed.
    return bool(_value(state, "on_ground", False)) and _value(state, "vy", 0) >= 0


@dataclass(frozen=True)
class RouteProgress:
    index: int = 0
    left_ground: bool = False
    violation: str | None = None


def _consume(route, progress, state, before=None):
    # Local import keeps the single target predicate public and avoids a cycle.
    from .assist_planning import target_reached
    index, airborne = progress.index, progress.left_ground
    while index < len(route["waypoints"]):
        limits = route["segments"][index]
        if limits["no_dash"] and _dash_active(state):
            return RouteProgress(index, airborne, "dash_forbidden")
        if limits["no_ground"] and not airborne:
            break
        target=route["waypoints"][index]
        if target.get("transition_to") and (before is None or index!=progress.index
                or _value(before,"room")!=target["room"]
                or _value(state,"room")!=target["transition_to"]):
            # Exit tiles represent a directed crossing, not just proximity in
            # the neighbor. One crossing cannot consume two exit waypoints.
            break
        if not target_reached(state,target):
            break
        index += 1
        airborne = not _grounded(state)
    return RouteProgress(index, airborne)


def start_route_progress(route, state) -> RouteProgress:
    if _value(state, "dead", False):
        return RouteProgress(violation="player_died")
    return _consume(route, RouteProgress(left_ground=not _grounded(state)), state)


def advance_route_progress(route, progress, before, after, action=None) -> RouteProgress:
    """Check one consumed action against its incoming segment, then visit points.

    A no-ground segment may take off from its grounded start, but cannot finish
    by walking and cannot touch down after becoming airborne, even at its target.
    An already-running or buffered dash cannot spill into a no-dash segment.
    ``action=None`` checks telemetry-only steps; callers should pass the consumed
    input mask whenever it is available.
    """
    if progress.violation:
        return progress
    if _value(after, "dead", False):
        return RouteProgress(progress.index, progress.left_ground, "player_died")
    if route_complete(route, progress):
        return progress
    if action is not None and (type(action) is not int or not 0 <= action <= 127):
        return RouteProgress(progress.index, progress.left_ground, "invalid_input_mask")
    limits = route["segments"][progress.index]
    if limits["no_dash"] and ((action is not None and action & 32)
                              or _dash_active(before) or _dash_active(after)):
        return RouteProgress(progress.index, progress.left_ground, "dash_forbidden")
    grounded = _grounded(after)
    airborne = progress.left_ground or not _grounded(before)
    if limits["no_ground"] and airborne and grounded:
        return RouteProgress(progress.index, airborne, "landing_forbidden")
    airborne = airborne or not grounded
    return _consume(route, RouteProgress(progress.index, airborne), after, before)


def route_complete(route, progress) -> bool:
    return progress.violation is None and progress.index == len(route["waypoints"])
