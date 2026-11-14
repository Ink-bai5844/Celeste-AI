"""Check idle anchors for asynchronous planning while the game keeps updating.

These checks do not pause or advance the game. The caller keeps sending neutral
input while an offline job runs, then discards its result unless the live state
still matches. Wall-clock time and frame numbers are deliberately not matched.

Player.Update resets jump grace and wall-slide time every grounded normal tick;
positive dash/force/lift timers and pending berry followers do change the next
route. AutoJump's boolean must settle independently of its optional timer.
Source: https://github.com/NoelFB/Celeste/blob/master/Source/Player/Player.cs

OnSafeGround excludes unsafe zip/crumble surfaces. It does not prove that every
entity elsewhere in the room is stationary. Optional live environment snapshots
are checked when available; missing entity-coroutine telemetry remains an explicit
model limitation. Room/death/spawn epochs should also be tracked by the controller.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

from .physics import SimState


_VELOCITY_EPS = 0.01
_SUBPIXEL_EPS = 0.0001
_STAMINA_EPS = 0.05
_ZERO_TIMERS = ("dash_cooldown", "refill_cooldown", "jump_buffer", "dash_buffer",
                "var_jump", "force_frames", "lift_timer", "auto_jump")
_ESSENTIAL = _ZERO_TIMERS + ("auto_jump_active", "prev_input", "facing", "ducking",
                            "rem_x", "rem_y", "max_fall", "lift_vx", "lift_vy")
_ENV_FIELDS = ("mechanisms", "refill_timers", "broken", "fled")
_EPOCH_FIELDS = ("spawn_id", "deaths", "deaths_in_room", "room_epoch")


def _model(observation: Mapping[str, Any]) -> Mapping[str, Any]:
    result = observation.get("model_state", {})
    return result if isinstance(result, Mapping) else {}


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def _integer(value: Any) -> bool:
    return type(value) is int


def _identities(value: Any) -> frozenset[str] | None:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return None
    if any(not isinstance(item, str) or not item for item in value):
        return None
    return frozenset(value)


def _environment_is_settled(model: Mapping[str, Any]) -> bool:
    """Reject known ticking phases; absent snapshots do not invent moving state."""
    for name in ("broken", "fled"):
        if name in model and _identities(model[name]) is None:
            return False
    if "refill_timers" in model:
        timers = model["refill_timers"]
        if not isinstance(timers, (list, tuple)):
            return False
        for record in timers:
            if not isinstance(record, (list, tuple)) or len(record) != 2 or record[1] != 0:
                return False
    if "mechanisms" in model:
        mechanisms = model["mechanisms"]
        if not isinstance(mechanisms, (list, tuple)):
            return False
        for row in mechanisms:
            if not isinstance(row, (list, tuple)) or len(row) != 6:
                return False
            # phase0 is inactive. phase3+timer0+speed0 represents a settled
            # falling block; zip wait and collapsed crumble have ticking timers.
            if (not _integer(row[1]) or not _integer(row[2]) or
                    any(not _finite(row[i]) for i in (3, 4, 5)) or
                    row[1] not in (0, 3) or row[2] != 0 or row[5] != 0):
                return False
    return True


def _idle_player(observation: Mapping[str, Any], *, require_safe_ground: bool) -> bool:
    if not isinstance(observation, Mapping):
        return False
    if observation.get("dead", False) or observation.get("transitioning", False) or observation.get("in_cutscene", False):
        return False
    if type(observation.get("state")) is not int or observation["state"] != 0 or observation.get("on_ground") is not True:
        return False
    if require_safe_ground and observation.get("on_safe_ground") is not True:
        return False
    if observation.get("input_phase", "before_virtual_inputs") != "before_virtual_inputs":
        return False
    if not isinstance(observation.get("room"), str) or not observation["room"]:
        return False
    for name in ("x", "y", "vx", "vy", "stamina"):
        if not _finite(observation.get(name)):
            return False
    if abs(observation["vx"]) > _VELOCITY_EPS or abs(observation["vy"]) > _VELOCITY_EPS:
        return False
    if abs(observation["stamina"] - 110.0) > _STAMINA_EPS:
        return False  # Ground refills stamina on the next normal update.
    if not _integer(observation.get("dashes")) or observation["dashes"] < 0:
        return False
    if _identities(observation.get("berries")) is None or _identities(observation.get("following")) != frozenset():
        return False  # Let pending followers confirm naturally before planning.
    model = _model(observation)
    unavailable = observation.get("model_state_unavailable", ())
    if not isinstance(unavailable, (list, tuple)):
        return False
    if any(name not in model or name in unavailable for name in _ESSENTIAL):
        return False
    if any(not _integer(model[name]) or model[name] != 0 for name in _ZERO_TIMERS):
        return False
    if model["auto_jump_active"] is not False:
        return False
    if not _integer(model["prev_input"]) or not 0 <= model["prev_input"] <= 127 or model["prev_input"] & 112:
        return False
    if model["facing"] not in (-1, 1) or type(model["facing"]) is not int or type(model["ducking"]) is not bool:
        return False
    if any(not _finite(model[name]) for name in ("rem_x", "rem_y", "max_fall", "lift_vx", "lift_vy")):
        return False
    if abs(model["max_fall"] - 160.0) > 0.01:
        return False
    if abs(model["lift_vx"]) > _VELOCITY_EPS or abs(model["lift_vy"]) > _VELOCITY_EPS:
        return False
    return _environment_is_settled(model)


def can_plan_while_idle(observation: Mapping[str, Any]) -> bool:
    """True for a settled, safe-ground normal player under neutral controls.

    This is a launch condition for a background job, not authority to play its
    result later. Call ``anchor_matches`` again when the job finishes.
    """
    return _idle_player(observation, require_safe_ground=True)


def _same_meaningful_state(observation: Mapping[str, Any], anchor: Mapping[str, Any], tolerance: float,
                           *, require_epoch_presence: bool = True) -> bool:
    if not _finite(tolerance) or tolerance < 0:
        return False
    if observation["room"] != anchor["room"]:
        return False
    if math.hypot(observation["x"] - anchor["x"], observation["y"] - anchor["y"]) > tolerance:
        return False
    for name in ("vx", "vy"):
        if abs(observation[name] - anchor[name]) > _VELOCITY_EPS:
            return False
    if abs(observation["stamina"] - anchor["stamina"]) > _STAMINA_EPS:
        return False
    if observation["dashes"] != anchor["dashes"]:
        return False
    for name in ("berries", "following"):
        if _identities(observation[name]) != _identities(anchor[name]):
            return False
    current_model, anchor_model = _model(observation), _model(anchor)
    for name in ("facing", "ducking", "auto_jump_active") + _ZERO_TIMERS:
        if current_model[name] != anchor_model[name]:
            return False
    # Directions have no button-edge semantics. With vx=vy=0 and no jump/dash/
    # grab held, normalize their previous mask to neutral instead of staling an
    # otherwise identical job after the first neutral tick.
    if current_model["prev_input"] & 112 != anchor_model["prev_input"] & 112:
        return False
    # ClimbNoMoveTimer is ignored in normal state and reset by ClimbBegin; it
    # can remain positive after exiting climb early, so it must not block idle.
    for name in ("rem_x", "rem_y"):
        if abs(current_model[name] - anchor_model[name]) > _SUBPIXEL_EPS:
            return False
    if abs(current_model["max_fall"] - anchor_model["max_fall"]) > 0.01:
        return False
    for name in _ENV_FIELDS:
        if name in current_model and name in anchor_model:
            if name in ("broken", "fled"):
                if _identities(current_model[name]) != _identities(anchor_model[name]):
                    return False
            elif tuple(tuple(row) for row in current_model[name]) != tuple(tuple(row) for row in anchor_model[name]):
                return False
    # A future bridge can supply an entity-only fingerprint without making the
    # existing player fingerprint/raw negative timers part of this comparison.
    for name in ("environment_fingerprint",) + _EPOCH_FIELDS:
        if ((name in observation or name in anchor) if require_epoch_presence
                else (name in observation and name in anchor)):
            if observation.get(name) != anchor.get(name):
                return False
    return True


def anchor_matches(observation: Mapping[str, Any], anchor: Mapping[str, Any], tolerance: float = 2) -> bool:
    """Reject stale async results using actual player state, never elapsed frames."""
    return (can_plan_while_idle(observation) and can_plan_while_idle(anchor)
            and _same_meaningful_state(observation, anchor, tolerance))


def candidate_compatible(observation: Mapping[str, Any], initial: SimState | Mapping[str, Any],
                         tolerance: float = 2) -> bool:
    """Check a candidate's saved initial model state before sending its first key.

    ``initial`` can be a SimState, its serialized dictionary, or a bridge anchor
    dictionary. Never pass the candidate's final state. The live observation must
    independently establish safe ground; SimState itself has no safe-ground bit.
    SimState also has no spawn/death epoch: the caller must separately call
    anchor_matches with its original raw observation before accepting a job.
    """
    if isinstance(initial, SimState):
        data = initial.to_dict()
    elif isinstance(initial, Mapping):
        data = dict(initial)
    else:
        return False
    if "model_state" not in data:
        data["model_state"] = dict(data)
    return (can_plan_while_idle(observation) and _idle_player(data, require_safe_ground=False)
            and _same_meaningful_state(observation, data, tolerance, require_epoch_presence=False))
