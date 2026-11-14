"""Validate clicked targets and fresh human-assist handoffs, entirely offline.

The controller must keep physical-input passthrough active during search. Call
``observe_assist_anchor`` on every received STATE and ``admit_assist_candidate``
immediately before taking control. A human input, respawn, or changed request
invalidates the job permanently, even if the player later returns to its pose.

An optional neutral projection predicts natural motion; it never sends that
prefix to the game. Admission requires the matching observed player-update frame.
It does not require landing, and imposes no restriction on a route's room changes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
import math
import struct
from typing import Any, Mapping

from .anchor import can_plan_while_idle
from .maps import MapData, TILE_SIZE
from .physics import DT, SimState, Simulator, _runtime_height
from .runtime_state import RuntimeStateError, state_from_observation
from .assist_targets import expand_exit_target, room_exit_targets
from .cassette import CassetteClock, CassetteBlockState


class AssistValidationError(ValueError):
    """A target or observed anchor cannot be used, with a stable reason code."""

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(reason + (": " + detail if detail else ""))


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_tile_target(map_data: MapData, target: Mapping[str, Any]) -> dict:
    """Return an unchanged world 8px tile origin, or reject an invalid selection.

    Air tiles are valid. Breakable walls, one-way platforms and hazards are left
    to route search. Only a tile fully occupied by permanent static geometry is
    rejected here; the player is never silently redirected to a nearby tile.
    """
    if not isinstance(target, Mapping) or not isinstance(target.get("room"), str):
        raise AssistValidationError("invalid_target", "room and world tile origin x/y are required")
    room = map_data.rooms.get(target["room"])
    if room is None:
        raise AssistValidationError("unknown_target_room")
    if room.dummy:
        raise AssistValidationError("target_room_has_no_player_area")
    size = target.get("tile_size", TILE_SIZE)
    x, y = target.get("x"), target.get("y")
    if type(size) is not int or size != TILE_SIZE:
        raise AssistValidationError("invalid_tile_size")
    if not _number(x) or not _number(y) or (x-room.x) % TILE_SIZE or (y-room.y) % TILE_SIZE:
        raise AssistValidationError("invalid_tile_origin")
    right, bottom = room.x + room.width, room.y + _runtime_height(room)
    if not (room.x <= x < right and room.y <= y < bottom):
        exits = [item for item in room_exit_targets(map_data,room.name)
                 if item["x"] == x and item["y"] == y]
        if "transition_to" in target:
            exits = [item for item in exits if item["transition_to"] == target["transition_to"]]
        if "edge" in target:
            exits = [item for item in exits if item["edge"] == target["edge"]]
        if exits:
            if "exit_region" in target:
                if not isinstance(target["exit_region"], (tuple, list)) or not target["exit_region"]:
                    raise AssistValidationError("invalid_exit_region")
                # Rebuild from permanent portal geometry. Never trust a stored
                # region's membership, destination or expanded arrival boxes.
                return expand_exit_target(map_data, exits[0])
            return exits[0]
        raise AssistValidationError("target_outside_room")
    if "transition_to" in target or "edge" in target or "exit_region" in target:
        raise AssistValidationError("invalid_transition_target")
    tx, ty = int((x-room.x)//TILE_SIZE), int((y-room.y)//TILE_SIZE)
    if room.tile_at(tx, ty):
        raise AssistValidationError("target_is_solid")
    for entity in room.entities:
        if (entity.kind in {"solid", "invisibleBarrier"} and entity.x <= x
                and entity.y <= y and entity.x + entity.width >= min(x+TILE_SIZE, right)
                and entity.y + entity.height >= min(y+TILE_SIZE, bottom)):
            raise AssistValidationError("target_is_solid")
    return {"room": room.name, "x": int(x), "y": int(y), "tile_size": TILE_SIZE}


def target_from_world(map_data: MapData, room: str, x: float, y: float) -> dict:
    """Map a click to its room-local tile; negative world coordinates are valid."""
    selected = map_data.rooms.get(room)
    if selected is None:
        raise AssistValidationError("unknown_target_room")
    if not _number(x) or not _number(y):
        raise AssistValidationError("invalid_click_position")
    if not (selected.x <= x < selected.x+selected.width
            and selected.y <= y < selected.y+_runtime_height(selected)):
        for target in room_exit_targets(map_data,room):
            if target["x"] <= x < target["x"]+TILE_SIZE and target["y"] <= y < target["y"]+TILE_SIZE:
                return target
        raise AssistValidationError("target_outside_room")
    return validate_tile_target(map_data, {
        "room": room, "x": selected.x + math.floor((x-selected.x)/TILE_SIZE)*TILE_SIZE,
        "y": selected.y + math.floor((y-selected.y)/TILE_SIZE)*TILE_SIZE,
    })


def cassette_pickup_required(map_data: MapData, target: Mapping[str, Any], state: SimState | None = None) -> bool:
    """Whether every centre position in this tile hits the unsupported pickup.

    Use the smallest (ducked) player pickup box relative to target_reached's
    centre: x +/-4, y -3..+1. Standing overlaps a superset. Partially overlapping
    adjacent tiles remain searchable; no target is moved or expanded.
    """
    if target.get("transition_to"):
        return False
    if state is not None and state.room==target["room"] and state.cassette_stopped is True:
        return False
    room=map_data.rooms[target["room"]]
    x,y,size=target["x"],target["y"],target.get("tile_size",TILE_SIZE)
    return any(entity.kind=="cassette" and entity.x-12<x
               and x+size<=entity.x+12 and entity.y-9<y
               and y+size<=entity.y+11 for entity in room.entities)


_CORE_FIELDS = {"room", "x", "y", "vx", "vy", "dashes", "stamina", "on_ground",
                "dead", "berries", "following", "state", "frame", "reason"}
_ENVIRONMENT_FIELDS = {"mechanisms", "broken", "refill_timers", "fled",
                       "cassette_clock", "cassette_blocks", "cassette_stopped", "player_update_order",
                       "player_depth", "player_actual_depth"}
# The currently running bridge predates the simulated moving-solid attachment.
# runtime_state requires it only during an active ClimbHop; after that interval
# the next native update clears it, so a stationary anchor needs no guess.
_CONDITIONAL_MODEL_FIELDS = {"climb_hop_solid", "climb_hop_x", "climb_hop_y"}
_REQUIRED_MODEL = ({field.name for field in fields(SimState)} - _CORE_FIELDS
                   - _ENVIRONMENT_FIELDS - _CONDITIONAL_MODEL_FIELDS)
_EPOCH_FIELDS = ("area_id", "area_mode", "spawn_id", "deaths", "deaths_in_room", "room_epoch",
                 "human_input_epoch")
_ENV_KINDS = {"zipMover", "fallingBlock", "crumbleBlock", "dashBlock", "refill", "cassetteBlock"}


def _manual_state(map_data: MapData, observation: Mapping[str, Any]) -> SimState:
    if not isinstance(observation, Mapping):
        raise AssistValidationError("invalid_observation")
    if observation.get("control_mode") != "manual":
        raise AssistValidationError("physical_passthrough_required")
    if observation.get("input_bindings_supported") is not True:
        raise AssistValidationError("unsupported_input_bindings")
    # A different fixed step can change midpoint-to-even Actor pixel rounding.
    # Time-rate assists and unknown timing must not silently reuse this model.
    if any(not _number(observation.get(field)) or abs(observation[field]-DT) > 1e-8
           for field in ("delta_time", "raw_delta_time")):
        raise AssistValidationError("unsupported_time_step")
    mask = observation.get("human_input_mask")
    if type(mask) is not int or not 0 <= mask <= 127:
        raise AssistValidationError("human_input_unavailable")
    if mask:
        raise AssistValidationError("human_input")
    if type(observation.get("human_menu_input")) is not bool:
        raise AssistValidationError("human_input_unavailable")
    if observation["human_menu_input"]:
        raise AssistValidationError("human_input")
    model = observation.get("model_state", {})
    if not isinstance(model, Mapping):
        raise AssistValidationError("missing_model_state")
    unavailable = observation.get("model_state_unavailable", ())
    if not isinstance(unavailable, (tuple, list)) or any(not isinstance(x, str) for x in unavailable):
        raise AssistValidationError("invalid_observation")
    missing = _REQUIRED_MODEL - model.keys() | (_REQUIRED_MODEL & set(unavailable))
    if missing:
        raise AssistValidationError("missing_model_state", ", ".join(sorted(missing)))
    for field in ("spawn_id", "deaths", "frame", "human_input_epoch"):
        if type(observation.get(field)) is not int or observation[field] < 0:
            raise AssistValidationError("missing_runtime_epoch", field)
    try:
        return state_from_observation(map_data, observation)
    except RuntimeStateError as error:
        raise AssistValidationError(getattr(error, "reason", "invalid_runtime_state"), str(error)) from error


def _epoch(observation: Mapping[str, Any]) -> dict:
    return {field: observation[field] for field in _EPOCH_FIELDS if field in observation}


def _environment(observation: Mapping[str, Any]) -> dict:
    """Compare gameplay movers, not decorative berry bobbing or negative timers."""
    snapshot = observation.get("room_entities")
    if not isinstance(snapshot, Mapping):
        return {"snapshot": None}
    records = []
    for record in snapshot.get("entities", ()):
        if record.get("kind") in _ENV_KINDS:
            records.append({key: value for key, value in record.items()
                            if key not in {"retained_observation", "unavailable"}
                            and not (key == "update_order" and record.get("kind") == "cassetteBlock")})
    result = {"entities": sorted(records, key=lambda row: row["id"]),
              "do_not_load": sorted(snapshot.get("do_not_load", ())),
              "unavailable": snapshot.get("unavailable", ())}
    if any(record.get("kind") == "cassetteBlock" for record in records):
        manager = snapshot.get("cassette_manager")
        result.update(cassette_manager={key: value for key, value in manager.items() if key != "update_order"}
                      if isinstance(manager, Mapping) else manager,
                      player_depth=snapshot.get("player_depth"),
                      player_actual_depth=snapshot.get("player_actual_depth"),
                      depth_counters=snapshot.get("depth_counters"))
    return deepcopy(result)


def _same(left: Any, right: Any, *, tolerance: float = 0.0001) -> bool:
    if type(left) is int or type(right) is int:
        return type(left) is type(right) and left == right
    if _number(left) and _number(right):
        return abs(left-right) <= tolerance
    if is_dataclass(left) and is_dataclass(right) and not isinstance(left, type) and type(left) is type(right):
        return all(_same(getattr(left, field.name), getattr(right, field.name),
                         tolerance=1e-10 if field.name in {"actual_depth", "depth_counters"} else tolerance)
                   for field in fields(left)
                   if not (field.name == "update_order" and isinstance(left, (CassetteClock, CassetteBlockState))))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_same(value, right[key],
            tolerance=1e-10 if key in {"actual_depth", "player_actual_depth", "depth_counters"} else tolerance)
            for key, value in left.items())
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(_same(a, b, tolerance=tolerance) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def cassette_comparison_snapshot(snapshot: Mapping[str, Any]) -> dict:
    """Compare physical order, allowing irrelevant gaps from decorative entities.

    Native depth allocations monotonically append within an integer-depth
    bucket. Extra trail/particle allocations can change absolute offsets but
    cannot change the relative order of existing player/manager/blocks or any
    later append. Runtime parsing bounds every relevant bucket below width 1.
    Raw offsets and counters remain in the original snapshot for audit/replay.
    """
    clock = snapshot.get("cassette_clock")
    stopped=snapshot.get("cassette_stopped",False)
    if type(stopped) is not bool:
        raise ValueError("invalid cassette stopped flag")
    if clock is None and not stopped:
        if snapshot.get("cassette_blocks"):
            raise ValueError("cassette blocks have no comparison clock")
        return {}
    blocks = snapshot.get("cassette_blocks")
    if ((clock is not None and not isinstance(clock, Mapping))
            or not isinstance(blocks, (list, tuple)) or stopped and (clock is not None or not blocks)):
        raise ValueError("invalid cassette comparison snapshot")
    ordered = [(snapshot.get("player_actual_depth"), "@player")]
    if clock is not None:
        ordered.append((clock.get("actual_depth"), "@manager"))
    normalized = []
    for block in blocks:
        if not isinstance(block, Mapping) or not isinstance(block.get("id"), str):
            raise ValueError("invalid cassette comparison block")
        if stopped and (block.get("activated") is not False or block.get("collidable") is not False
                        or block.get("active") is not True):
            raise ValueError("stopped cassette comparison contains an active collision block")
        ordered.append((block.get("actual_depth"), block["id"]))
        normalized.append({key: value for key, value in block.items() if key not in {"actual_depth", "update_order"}})
    if any(not _number(depth) for depth, _ in ordered) or len({depth for depth, _ in ordered}) != len(ordered):
        raise ValueError("cassette comparison needs a strict physical update order")
    result={"cassette_clock": {key: value for key, value in clock.items()
                                if key not in {"actual_depth", "depth_counters", "update_order"}} if clock is not None else None,
            "cassette_blocks": sorted(normalized, key=lambda block: block["id"]),
            "player_depth": snapshot["player_depth"],
            "cassette_order": [identity for _, identity in sorted(ordered, reverse=True)]}
    if stopped:
        result["cassette_stopped"]=True
    return result


def _different_fields(actual: SimState, expected: SimState, *, idle: bool) -> list[str]:
    # Native list indices include unrelated scene entities and are captured
    # before the next depth sort. Actual depths and Tracker order are the
    # predictive state; the absolute captured indices are audit metadata only.
    ignored = {"frame", "reason", "player_update_order", "player_depth", "player_actual_depth",
               "cassette_clock", "cassette_blocks"}
    if idle:
        # Grounded NormalUpdate refreshes these, and ClimbBegin resets climb_wait.
        ignored |= {"grace", "wall_slide", "climb_wait"}
    differences = [field.name for field in fields(SimState) if field.name not in ignored
            and not _same(getattr(actual, field.name), getattr(expected, field.name),
                          tolerance=1e-10 if field.name == "player_actual_depth" else 0.0001)]
    if (actual.cassette_clock is not None or expected.cassette_clock is not None
            or actual.cassette_stopped or expected.cassette_stopped):
        def cassette_state(state):
            return cassette_comparison_snapshot({"cassette_clock": state.cassette_clock.to_dict() if state.cassette_clock else None,
                "cassette_blocks": [block.to_dict() for block in state.cassette_blocks],
                "cassette_stopped":state.cassette_stopped,
                "player_depth": state.player_depth, "player_actual_depth": state.player_actual_depth})
        current, predicted = cassette_state(actual), cassette_state(expected)
        differences.extend(key for key in sorted(current.keys() | predicted.keys())
                           if key not in current or key not in predicted or not _same(current[key], predicted[key]))
    return differences


def make_assist_anchor(simulator: Simulator, observation: Mapping[str, Any], *,
                       request_id: str | int, future_frames: int = 0) -> dict:
    """Capture a safe stationary anchor or an explicitly timed neutral forecast.

    A future anchor requires no human keys held now. Its assumption is checked
    on every observation, and its route is admitted only at its exact frame.
    Transition animation timing is outside the player model, so forecasts that
    cross a room boundary are rejected (the subsequently planned route may cross).
    """
    if type(future_frames) is not int or not 0 <= future_frames <= 3600:
        raise AssistValidationError("invalid_projection_frames")
    if type(request_id) not in (str, int) or request_id == "":
        raise AssistValidationError("invalid_request_id")
    start = _manual_state(simulator.map, observation)
    state = start
    if not future_frames:
        if not can_plan_while_idle(observation) or start.prev_input != 0:
            raise AssistValidationError("waiting_for_safe_idle")
        next_state = simulator.step(start, 0)
        if _different_fields(next_state, start, idle=True):
            raise AssistValidationError("environment_or_player_still_changing")
        # A tiny native residual velocity can leave the pixel position fixed
        # while MoveV accumulates its subpixel remainder on every update. A
        # one-step tolerance cannot certify that state indefinitely stationary.
        # Compare native float32 values so decimal telemetry round-tripping
        # alone does not trigger a forecast. The controller may use the existing
        # exact-frame future anchor; admission precision remains unchanged.
        if any(struct.pack("!f", getattr(next_state, name)) != struct.pack("!f", getattr(start, name))
               for name in ("rem_x", "rem_y")):
            raise AssistValidationError("timed_projection_required")
    else:
        for _ in range(future_frames):
            state = simulator.step(state, 0)
            if state.dead:
                raise AssistValidationError("projected_death")
            if state.room != start.room:
                raise AssistValidationError("projection_crosses_transition")
        if state.state not in (0, 1):
            raise AssistValidationError("unsupported_projected_state")
    return {"schema_version": 1, "request_id": request_id,
            "mode": "future" if future_frames else "idle", "room": start.room,
            "source_frame": start.frame, "target_frame": state.frame,
            "state": state.to_dict(), "source_state": start.to_dict(),
            "epoch": _epoch(observation), "environment": _environment(observation),
            "last_observed_frame": start.frame, "invalid_reason": None}


def observe_assist_anchor(map_data: MapData, anchor: dict,
                          observation: Mapping[str, Any], *, request_id: str | int) -> dict:
    """Latch cancellation/staleness while leaving human play fully untouched."""
    if anchor.get("invalid_reason"):
        return {"valid": False, "reason": anchor["invalid_reason"]}
    try:
        if request_id != anchor.get("request_id"):
            raise AssistValidationError("request_changed")
        current = _manual_state(map_data, observation)
        if _epoch(observation) != anchor["epoch"]:
            raise AssistValidationError("runtime_epoch_changed")
        if current.room != anchor["room"]:
            raise AssistValidationError("room_changed")
        if current.frame < anchor["last_observed_frame"]:
            raise AssistValidationError("frame_rewound")
        if anchor["mode"] == "idle":
            if not can_plan_while_idle(observation) or current.prev_input != 0:
                raise AssistValidationError("player_changed")
            if _different_fields(current, SimState.from_dict(anchor["state"]), idle=True):
                raise AssistValidationError("player_or_environment_changed")
            if not _same(_environment(observation), anchor["environment"]):
                raise AssistValidationError("environment_changed")
        elif current.frame > anchor["target_frame"]:
            raise AssistValidationError("future_frame_missed")
        anchor["last_observed_frame"] = current.frame
        return {"valid": True, "reason": "anchor_valid", "state": current.to_dict()}
    except (AssistValidationError, KeyError, TypeError, ValueError) as error:
        reason = error.reason if isinstance(error, AssistValidationError) else "invalid_anchor"
        anchor["invalid_reason"] = reason
        return {"valid": False, "reason": reason}


def admit_assist_candidate(map_data: MapData, anchor: dict,
                            observation: Mapping[str, Any], *, request_id: str | int,
                            candidate_state: SimState | Mapping[str, Any] | None = None) -> dict:
    """Final per-frame gate; accepted state is the fresh actual replay origin.

    Pass the worker's ``anchor_state`` as candidate_state when available. This
    also catches a route accidentally computed from a spawn or another job.
    """
    status = observe_assist_anchor(map_data, anchor, observation, request_id=request_id)
    if not status["valid"]:
        return {"accepted": False, "reason": status["reason"], "waiting": False}
    idle = anchor["mode"] == "idle"
    if not idle and observation["frame"] < anchor["target_frame"]:
        return {"accepted": False, "reason": "future_frame_not_reached", "waiting": True}
    expected = SimState.from_dict(anchor["state"])
    current = SimState.from_dict(status["state"])
    if candidate_state is not None:
        try:
            candidate = candidate_state if isinstance(candidate_state, SimState) else SimState.from_dict(dict(candidate_state))
            if _different_fields(candidate, expected, idle=False):
                return {"accepted": False, "reason": "candidate_origin_changed", "waiting": False}
        except (TypeError, ValueError, KeyError):
            return {"accepted": False, "reason": "invalid_candidate_origin", "waiting": False}
    differences = _different_fields(current, expected, idle=idle)
    if differences:
        anchor["invalid_reason"] = "projected_state_mismatch"
        return {"accepted": False, "reason": "projected_state_mismatch", "waiting": False,
                "different_fields": differences}
    return {"accepted": True, "reason": "fresh_actual_anchor", "waiting": False,
            "state": current.to_dict()}
