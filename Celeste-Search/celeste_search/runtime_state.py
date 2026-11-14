"""Anchor offline planning to an observed live state without moving the player.

Only telemetry is consumed here. There are no sockets, game actions, subprocesses
or map-spawn substitutions. Runtime collection IDs replace predicted IDs.

The bridge's ``model_state`` uses SimState names and integer player-update counts
for timers (converted by the bridge from seconds). Unavailable coroutine state
is not fabricated: in particular a mid-dash observation is not a valid anchor.
"""
from __future__ import annotations

from dataclasses import fields
import math
from typing import Any, Mapping

from .maps import MapData
from .physics import SimState
from .cassette import CassetteClock, CassetteBlockState, validate_blocks
from .falling_runtime import FallingRuntimeError, decode_falling_record


class RuntimeStateError(ValueError):
    """The observation cannot safely anchor the offline model."""


class UnsupportedFallingRuntimeState(RuntimeStateError):
    reason = "unsupported_falling_runtime_state"


_CORE = ("room", "x", "y", "vx", "vy", "dashes", "stamina", "on_ground",
         "state", "berries", "following", "frame")
_CORE_NUMBERS = ("x", "y", "vx", "vy", "stamina")
_MID_AIR_REQUIRED = ("rem_x", "rem_y", "prev_input", "grace", "var_jump",
                     "var_speed", "auto_jump", "auto_jump_active", "dash_cooldown", "refill_cooldown")
_CLIMB_REQUIRED = _MID_AIR_REQUIRED + ("facing", "climb_wait", "force_x", "force_frames")
_BOOL_FIELDS = {"ducking", "dash_started_ground", "auto_jump_active"}
_FLOAT_FIELDS = {"rem_x", "rem_y", "var_speed", "dash_dx", "dash_dy", "dash_before_x",
                 "max_fall", "lift_vx", "lift_vy", "wall_retained", "hop_wait_x_speed",
                 "climb_hop_x", "climb_hop_y"}
_FRAME_FIELDS = {"grace", "jump_buffer", "dash_buffer", "var_jump", "dash_frames",
                 "dash_cooldown", "refill_cooldown", "force_frames", "wall_slide",
                 "climb_wait", "auto_jump", "collect_delay", "lift_timer", "dash_attack", "wall_retention"}
_INTEGER_FIELDS = _FRAME_FIELDS | {"prev_input", "facing", "force_x", "collect_ground", "hop_wait_x"}
_ENVIRONMENT_FIELDS = {"mechanisms", "broken", "refill_timers", "fled"}
_PRIVATE_FIELDS = (_BOOL_FIELDS | _FLOAT_FIELDS | _INTEGER_FIELDS | _ENVIRONMENT_FIELDS
                   | {"follow_order", "climb_hop_solid"})


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise RuntimeStateError(f"{name} must be a finite number")
    return float(value)


def _integer(value: Any, name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeStateError(f"{name} must be an integer, in model frame units")
    if minimum is not None and value < minimum:
        raise RuntimeStateError(f"{name} must be at least {minimum}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise RuntimeStateError(f"{name} must be a boolean")
    return value


def _identities(value: Any, name: str) -> frozenset[str]:
    if not isinstance(value, (list, tuple, set, frozenset)) or any(
            not isinstance(item, str) or not item for item in value):
        raise RuntimeStateError(f"{name} must contain nonempty identity strings")
    return frozenset(value)


def _metadata_list(observation: Mapping[str, Any], name: str) -> list[str]:
    value = observation.get(name, ())
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise RuntimeStateError(f"{name} must be a list of field names")
    return list(value)


def _room_entities(observation: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], frozenset[str]] | None:
    snapshot = observation.get("room_entities")
    if snapshot is None:
        return None
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("entities"), list):
        raise RuntimeStateError("room_entities must contain an entities list")
    records = snapshot["entities"]
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("id"), str):
            raise RuntimeStateError("room_entities record needs a stable entity identity")
        for name in ("x", "y", "percent", "respawn_timer", "collect_timer", "follow_delay_timer", "lift_vx", "lift_vy"):
            if name in record:
                _number(record[name], "room_entities." + name)
        for name in ("present", "collidable", "flying_away", "has_started_falling", "triggered"):
            if name in record:
                _boolean(record[name], "room_entities." + name)
    return records, _identities(snapshot.get("do_not_load", []), "room_entities.do_not_load")


def falling_snapshot_from_observation(map_data: MapData, observation: Mapping[str, Any]) -> list[list]:
    """Read every current-room falling phase without constructing a player anchor.

    Execution-time dash telemetry can omit player coroutine fields. Those fields
    are irrelevant here: only the room and fresh native entity records are used.
    The returned six-field records match the model's sorted mechanism tuples.
    """
    if not isinstance(observation, Mapping):
        raise UnsupportedFallingRuntimeState("falling snapshot needs an observation")
    room_name = observation.get("room")
    if not isinstance(room_name, str) or room_name not in map_data.rooms:
        raise UnsupportedFallingRuntimeState("falling snapshot needs a known current room")
    room = map_data.rooms[room_name]
    expected = {f"{room.name}:{entity.id}": entity for entity in room.entities
                if entity.kind == "fallingBlock"}
    if not expected:
        return []
    try:
        snapshot = _room_entities(observation)
        if snapshot is None:
            raise UnsupportedFallingRuntimeState("fallingBlock room requires a complete native entity snapshot")
        records, _ = snapshot
        result = []
        for identity, entity in sorted(expected.items()):
            matches = [record for record in records if record["id"] == identity]
            if len(matches) != 1 or matches[0].get("room", room.name) != room.name:
                raise UnsupportedFallingRuntimeState(f"{identity} needs exactly one fresh presence/phase record")
            result.append(list(decode_falling_record(matches[0], entity, room)))
        return result
    except (RuntimeStateError, FallingRuntimeError) as error:
        if isinstance(error, UnsupportedFallingRuntimeState):
            raise
        raise UnsupportedFallingRuntimeState(str(error)) from error


def _merge_environment(map_data: MapData, observation: Mapping[str, Any], data: dict[str, Any]) -> None:
    """Prefer measured identities/cooldowns/positions over predicted entities.

    Mover coroutine timing remains an explicitly approximate inference from
    observed position and velocity. No inferred field becomes confirmed loot.
    """
    room = map_data.rooms[data["room"]]
    falling = {f"{room.name}:{entity.id}" for entity in room.entities if entity.kind == "fallingBlock"}
    snapshot = _room_entities(observation)
    if snapshot is None:
        if falling:
            raise UnsupportedFallingRuntimeState("fallingBlock room requires a complete native entity snapshot")
        return
    records, do_not_load = snapshot
    for identity in falling:
        matches = [record for record in records if record["id"] == identity]
        if len(matches) != 1 or matches[0].get("room", room.name) != room.name:
            raise UnsupportedFallingRuntimeState(f"{identity} needs exactly one fresh presence/phase record")
    entities = {f"{room.name}:{entity.id}": entity for entity in room.entities}
    broken, fled = set(data["broken"]), set(data["fled"])
    # A fresh native entity snapshot supersedes speculative broken geometry.
    # An absent snapshot record is *unknown*, not evidence of destruction.
    broken.difference_update(identity for identity, entity in entities.items() if entity.kind == "dashBlock")
    refills = dict(data["refill_timers"])
    mechanisms = {record[0]: record for record in data["mechanisms"]}
    for identity in do_not_load:
        entity = entities.get(identity)
        if entity and entity.kind == "dashBlock":
            broken.add(identity)
    for record in records:
        identity = record["id"]
        entity = entities.get(identity)
        if entity is None or record.get("room", room.name) != room.name:
            continue
        present = record.get("present")
        if entity.kind == "dashBlock":
            if present is False or identity in do_not_load:
                broken.add(identity)
            elif present is True:
                broken.discard(identity)
        elif entity.kind == "refill":
            if present is False and record.get("one_use") is True:
                refills[identity] = 10 ** 9
            elif "respawn_timer" in record:
                frames = max(0, math.ceil(record["respawn_timer"] * 60 - 1e-4))
                if frames:
                    refills[identity] = frames
                else:
                    refills.pop(identity, None)
        elif entity.kind == "strawberry":
            if record.get("flying_away") is True and identity not in data["following"]:
                fled.add(identity)
            elif present is True and record.get("flying_away") is False:
                fled.discard(identity)
        elif entity.kind == "fallingBlock":
            try:
                mechanisms[identity] = decode_falling_record(record, entity, room)
            except FallingRuntimeError as error:
                raise UnsupportedFallingRuntimeState(f"{identity}: {error}") from error
        elif entity.kind in ("zipMover", "crumbleBlock") and "x" in record and "y" in record:
            x, y = float(record["x"]), float(record["y"])
            prior = mechanisms.get(identity)
            phase, timer, speed = (prior[1], prior[2], prior[5]) if prior else (0, 0, 0.0)
            if entity.kind == "zipMover" and entity.nodes:
                # Platform.ExactPosition includes Actor-style movement counters;
                # these fractional coordinates determine the next lift boost.
                x += float(record.get("rem_x",0))
                y += float(record.get("rem_y",0))
                target_x, target_y = entity.nodes[0]
                dx, dy = target_x - entity.x, target_y - entity.y
                distance_squared = dx * dx + dy * dy
                progress = ((x - entity.x) * dx + (y - entity.y) * dy) / max(1, distance_squared)
                progress = min(1.0, max(0.0, progress))
                along = float(record.get("lift_vx", 0)) * dx + float(record.get("lift_vy", 0)) * dy
                if 0 < progress < 1:
                    phase = 4 if along < 0 or (along == 0 and phase == 4) else 2
                    duration = 120 if phase == 4 else 30
                    eased = 1 - progress if phase == 4 else progress
                    timer = round(math.acos(1 - eased) / (math.pi / 2) * duration)
                elif progress == 1:
                    phase, timer = 3, timer if phase == 3 else 30
                elif phase not in (1, 5) or not prior:
                    phase, timer = 0, 0
                native_phase = _zip_coroutine_phase(record)
                if native_phase is not None:
                    phase, timer = native_phase
            elif entity.kind == "crumbleBlock":
                if record.get("collidable") is False:
                    phase, timer = 3, timer if phase == 3 else 120
                elif phase == 3:
                    phase, timer = 0, 0
            mechanisms[identity] = (identity, phase, timer, x, y, speed)
    data.update(broken=frozenset(broken), fled=frozenset(fled),
                refill_timers=tuple(sorted(refills.items())), mechanisms=tuple(sorted(mechanisms.values())))


def _merge_cassette_environment(map_data: MapData, observation: Mapping[str, Any], data: dict[str, Any]) -> None:
    """Require a fresh clock or a complete, explicitly observed stopped room."""
    room = map_data.rooms[data["room"]]
    expected = {f"{room.name}:{entity.id}": entity for entity in room.entities
                if entity.kind == "cassetteBlock"}
    # Even same-room preceding predictions cannot stand in for measured timing.
    data.update(cassette_clock=None, cassette_blocks=(), cassette_stopped=False, player_update_order=-1,
                player_depth=0, player_actual_depth=0.0)
    if not expected:
        return
    snapshot = observation.get("room_entities")
    if not isinstance(snapshot, Mapping):
        raise RuntimeStateError("cassette room requires fresh room_entities telemetry")
    unavailable = snapshot.get("unavailable", ())
    if not isinstance(unavailable, (tuple, list)) or any(not isinstance(name, str) for name in unavailable):
        raise RuntimeStateError("invalid room_entities.unavailable metadata")
    if any(name in {"player_update_order", "player_depth", "player_actual_depth", "depth_counters"}
           or name.startswith("cassette") for name in unavailable):
        raise RuntimeStateError("cassette timing or player update order is unavailable")
    missing = [name for name in ("player_update_order", "player_depth", "player_actual_depth", "depth_counters")
               if name not in snapshot or snapshot[name] is None]
    if missing:
        raise RuntimeStateError("missing cassette timing telemetry: " + ", ".join(missing))
    manager = snapshot.get("cassette_manager")
    if (not isinstance(manager, Mapping) or type(manager.get("present")) is not bool
            or manager.get("retained_observation", False) is not False):
        raise RuntimeStateError("cassette manager presence must be explicitly and freshly observed")
    if not isinstance(manager.get("unavailable"), (list, tuple)) or manager["unavailable"]:
        raise RuntimeStateError("cassette manager has unknown telemetry fields")
    stopped=manager["present"] is False
    if not stopped and manager.get("active") is not True:
        raise RuntimeStateError("cassette manager must be active")
    if stopped:
        # Native A-side LoadLevel creates a manager unless this session's
        # cassette has finished. We certify the measured mechanical state,
        # without inventing collection flags or a beat phase.
        if (_integer(observation.get("area_mode"),"area_mode")!=0
                or _integer(observation.get("state"),"state") not in (0,1,2)
                or observation.get("in_cutscene",False)
                or observation.get("transitioning",False)
                or manager.get("active",False) is not False):
            raise RuntimeStateError("cassette stopped state requires a normal vanilla A-side observation")
        if "session_cassette" in observation and observation["session_cassette"] is not True:
            raise RuntimeStateError("cassette stopped state contradicts the native session flag")
    player_order = _integer(snapshot.get("player_update_order"), "player_update_order", minimum=0)
    player_depth = _integer(snapshot.get("player_depth"), "player_depth")
    player_actual_depth = _number(snapshot.get("player_actual_depth"), "player_actual_depth")
    depth_counters = snapshot["depth_counters"]
    if not isinstance(depth_counters, Mapping):
        raise RuntimeStateError("cassette depth_counters must be a complete native dictionary")
    for key, value in depth_counters.items():
        try:
            valid_key = isinstance(key, str) and str(int(key)) == key
        except ValueError:
            valid_key = False
        if not valid_key or _number(value, "depth counter") < 0:
            raise RuntimeStateError("invalid cassette depth counter")
    records = snapshot.get("entities")
    if not isinstance(records, list):
        raise RuntimeStateError("cassette room requires an entities list")
    blocks = []
    seen = set()
    try:
        clock = None if stopped else CassetteClock.from_dict({**manager, "depth_counters": depth_counters})
        for record in records:
            if record.get("kind") != "cassetteBlock" and record.get("id") not in expected:
                continue
            identity = record["id"]
            if identity in seen or identity not in expected or record.get("room") != room.name:
                raise RuntimeStateError("unknown or duplicate cassette block identity: " + identity)
            if record.get("kind") != "cassetteBlock":
                raise RuntimeStateError("cassette block kind does not match the map: " + identity)
            if (record.get("present") is not True or record.get("active") is not True
                    or record.get("retained_observation") is not False):
                raise RuntimeStateError("cassette block must be present, active and freshly observed: " + identity)
            if not isinstance(record.get("unavailable"), (list, tuple)) or record["unavailable"]:
                raise RuntimeStateError("cassette block has unknown telemetry fields: " + identity)
            if "lift_vx" not in record or "lift_vy" not in record:
                raise RuntimeStateError("cassette block lift speed is unavailable: " + identity)
            block = CassetteBlockState.from_dict({**record,
                "lift_x": record["lift_vx"], "lift_y": record["lift_vy"]})
            entity = expected[identity]
            if (block.width != entity.width or block.height != entity.height
                    or block.index != int(entity.attrs.get("index", 0))):
                raise RuntimeStateError("cassette block geometry or index disagrees with the map: " + identity)
            if clock is not None and block.index >= clock.max_beat:
                raise RuntimeStateError("cassette block color exceeds the manager phase count: " + identity)
            if stopped and (block.activated or block.collidable or block.depth!=8990
                    or block.block_height not in (0,1,2) or block.x!=entity.x
                    or block.y!=entity.y+2-block.block_height
                    or block.rem_y!=0 or block.lift_x!=0 or block.lift_y!=0):
                raise RuntimeStateError("cassette stopped state requires every block to be fully inactive and stationary: " + identity)
            seen.add(identity)
            blocks.append(block)
        if seen != expected.keys():
            raise RuntimeStateError("missing cassette block telemetry: " + ", ".join(sorted(expected.keys()-seen)))
        validate_blocks(clock, tuple(blocks))
        if {block.tracker_order for block in blocks} != set(range(len(blocks))):
            raise RuntimeStateError("incomplete or duplicate cassette Tracker order")
        depths = {player_depth, *(block.depth for block in blocks)}
        if clock is not None:
            depths.add(clock.depth)
        counters = {int(key):value for key,value in depth_counters.items()}
        if not depths.issubset(counters):
            raise RuntimeStateError("cassette depth counters omit an observed entity depth")
        relevant_depths = depths | {-10, -5, 10, 15, 8990, 8995}
        if any(counters.get(depth, 0.0) >= 1 for depth in relevant_depths):
            raise RuntimeStateError("cassette depth allocation crosses integer depth buckets")
        ordering = [(player_depth, player_actual_depth),
                    *((block.depth, block.actual_depth) for block in blocks)]
        if clock is not None:
            ordering.append((clock.depth,clock.actual_depth))
        if len({actual for _, actual in ordering}) != len(ordering):
            raise RuntimeStateError("cassette entity depth ordering is ambiguous")
        for depth, actual in ordering:
            offset = depth-actual
            if not 0 <= offset < 1 or offset > counters[depth]+1e-10:
                raise RuntimeStateError("cassette actual depth is outside its observed allocation bucket")
        orders={block.update_order for block in blocks}
        if clock is not None:
            orders.add(clock.update_order)
        if player_order in orders:
            raise RuntimeStateError("player and cassette entities have duplicate update orders")
    except (ValueError, TypeError, KeyError) as error:
        if isinstance(error, RuntimeStateError):
            raise
        raise RuntimeStateError("invalid cassette telemetry: " + str(error)) from error
    data.update(cassette_clock=clock,cassette_stopped=stopped,
                cassette_blocks=tuple(sorted(blocks, key=lambda block: block.id)),
                player_update_order=player_order, player_depth=player_depth,
                player_actual_depth=player_actual_depth)


def cassette_snapshot_from_observation(map_data: MapData, observation: Mapping[str, Any]) -> dict:
    """Strict cassette execution evidence, including during a native dash.

    This does not construct a player anchor or guess unavailable dash timers.
    Absolute EntityList indices are omitted exactly as in routes.snapshot;
    actual depths, Tracker order and all clock/block fields remain present.
    """
    if not isinstance(observation, Mapping) or not isinstance(observation.get("room"), str):
        raise RuntimeStateError("cassette snapshot requires an observed room")
    room = map_data.rooms.get(observation["room"])
    if room is None:
        raise RuntimeStateError("cassette snapshot room is not in the loaded map")
    if not any(entity.kind == "cassetteBlock" for entity in room.entities):
        return {}
    _room_entities(observation)
    data = {"room": room.name}
    _merge_cassette_environment(map_data, observation, data)
    clock = data["cassette_clock"].to_dict() if data["cassette_clock"] is not None else None
    if clock is not None:
        clock.pop("update_order", None)
    blocks = []
    for block in data["cassette_blocks"]:
        value = block.to_dict()
        value.pop("update_order", None)
        blocks.append(value)
    result={"cassette_clock": clock, "cassette_blocks": blocks,
            "player_depth": data["player_depth"], "player_actual_depth": data["player_actual_depth"]}
    if data["cassette_stopped"]:
        result["cassette_stopped"]=True
    return result


def _zip_coroutine_phase(record: Mapping[str, Any]) -> tuple[int, int] | None:
    """Decode only the locally verified Celeste 1.4 ZipMover iterator layout.

    Local executable IL confirms these compiler states; the bridge exports raw
    values, so other builds/iterator layouts continue using the approximate
    geometric fallback. Coroutine decrements positive waits and returns before
    resuming; movement loops also yield null before their first movement.
    """
    coroutines = record.get("coroutines", ())
    if not isinstance(coroutines, list):
        return None
    for coroutine in coroutines:
        if not isinstance(coroutine, Mapping) or coroutine.get("active") is not True:
            continue
        stack = coroutine.get("stack", ())
        if not isinstance(stack, list):
            continue
        for iterator in stack:
            if not isinstance(iterator, Mapping) or iterator.get("iterator_type") != "Celeste.ZipMover+<Sequence>d__24":
                continue
            fields = iterator.get("fields", {})
            if not isinstance(fields, Mapping):
                continue
            native_state = fields.get("<>1__state")
            if not isinstance(native_state, int) or isinstance(native_state, bool):
                continue
            wait = _number(coroutine.get("wait_timer", 0), "zip coroutine wait_timer")
            waiting = max(0, math.ceil(wait * 60 - 1e-4))
            if native_state in (0, 1):
                return 0, 0
            if native_state == 2:
                return 1, waiting + 1
            if native_state in (3, 5):
                at = _number(fields.get("<at>5__3", 0), "zip iterator progress")
                if not 0 <= at <= 1:
                    return None
                return (2, round(at * 30)) if native_state == 3 else (4, round(at * 120))
            if native_state == 4:
                return 3, waiting + 1
            if native_state == 6:
                return (5, waiting) if waiting else (0, 0)
    return None


def _validate_core(observation: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(observation, Mapping):
        raise RuntimeStateError("Observation must be an object")
    missing = [name for name in _CORE if name not in observation]
    if missing:
        raise RuntimeStateError("Missing observed fields: " + ", ".join(missing))
    room = observation["room"]
    if not isinstance(room, str) or not room:
        raise RuntimeStateError("room must be a nonempty string")
    data = {"room": room}
    for name in _CORE_NUMBERS:
        data[name] = _number(observation[name], name)
    for name in ("dashes", "frame", "state"):
        data[name] = _integer(observation[name], name, minimum=0)
    data["on_ground"] = _boolean(observation["on_ground"], "on_ground")
    data["dead"] = _boolean(observation.get("dead", False), "dead")
    data["berries"] = _identities(observation["berries"], "berries")
    # A confirmed berry cannot simultaneously remain a pending follower.
    data["following"] = _identities(observation["following"], "following") - data["berries"]
    return data


def _validate_private(model: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(model, Mapping):
        raise RuntimeStateError("model_state must be an object")
    data = {}
    for name in _PRIVATE_FIELDS & model.keys():
        value = model[name]
        if name in _BOOL_FIELDS:
            data[name] = _boolean(value, name)
        elif name in _FLOAT_FIELDS:
            data[name] = _number(value, name)
        elif name in _INTEGER_FIELDS:
            data[name] = _integer(value, name, minimum=0 if name in _FRAME_FIELDS else None)
        elif name == "climb_hop_solid":
            if not isinstance(value, str):
                raise RuntimeStateError("climb_hop_solid must be an identity string or empty")
            data[name] = value
        elif name in ("broken", "fled"):
            data[name] = _identities(value, name)
        elif name == "follow_order":
            _identities(value, name)
            if len(value) != len(set(value)):
                raise RuntimeStateError("follow_order must not contain duplicate identities")
            data[name] = tuple(value)
        elif name == "refill_timers":
            if not isinstance(value, (list, tuple)):
                raise RuntimeStateError("refill_timers must be a list of [identity, frames]")
            records = []
            for row in value:
                if not isinstance(row, (list, tuple)) or len(row) != 2 or not isinstance(row[0], str):
                    raise RuntimeStateError("Invalid refill_timers record")
                records.append((row[0], _integer(row[1], "refill_timers.frames", minimum=0)))
            data[name] = tuple(sorted(records))
        elif name == "mechanisms":
            if not isinstance(value, (list, tuple)):
                raise RuntimeStateError("mechanisms must be a list of state records")
            records = []
            for row in value:
                if not isinstance(row, (list, tuple)) or len(row) != 6 or not isinstance(row[0], str):
                    raise RuntimeStateError("Invalid mechanisms record")
                records.append((row[0], _integer(row[1], "mechanisms.phase", minimum=0),
                                _integer(row[2], "mechanisms.timer"),
                                _number(row[3], "mechanisms.x"), _number(row[4], "mechanisms.y"),
                                _number(row[5], "mechanisms.speed")))
            data[name] = tuple(sorted(records))
    if "prev_input" in data and not 0 <= data["prev_input"] <= 127:
        raise RuntimeStateError("prev_input must be a seven-button mask in 0..127")
    if "facing" in data and data["facing"] not in (-1, 1):
        raise RuntimeStateError("facing must be -1 or 1")
    if "force_x" in data and data["force_x"] not in (-1, 0, 1):
        raise RuntimeStateError("force_x must be -1, 0 or 1")
    if "hop_wait_x" in data and data["hop_wait_x"] not in (-1, 0, 1):
        raise RuntimeStateError("hop_wait_x must be -1, 0 or 1")
    return data


def state_readiness(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Describe whether this observation is a supported planning boundary.

    ``ready`` permits construction, not a promise of exact prediction. Legacy
    grounded normal-state observations are accepted with explicit approximation
    warnings. Airborne/climbing anchors require the essential live private
    fields. A dash coroutine, cutscene, transition-in-progress or death is
    rejected; the controller must keep waiting or stop, not invent a state.
    """
    warnings: list[str] = []
    try:
        core = _validate_core(observation)
        model = _validate_private(observation.get("model_state", {}))
        unavailable = _metadata_list(observation, "model_state_unavailable")
        approximate = _metadata_list(observation, "model_state_approximate")
        snapshot = _room_entities(observation)
        if observation.get("input_phase", "before_virtual_inputs") != "before_virtual_inputs":
            return {"ready": False, "reason": "incompatible_input_phase", "warnings": []}
        if core["dead"]:
            return {"ready": False, "reason": "player_dead", "warnings": []}
        if core["state"] == 2:
            return {"ready": False, "reason": "dash_coroutine_unavailable", "warnings": []}
        if core["state"] not in (0, 1):
            return {"ready": False, "reason": "unsupported_player_state", "state": core["state"], "warnings": []}
        if model.get("force_frames", 0) > 0 and model.get("force_x", 0) == 0:
            # The existing bridge does not export Player.climbHopSolid. During
            # this short interval neither proximity nor a prior prediction can
            # establish its identity/last position. Wait for an anchor whose
            # next update no longer depends on that hidden attachment.
            required_hop = {"climb_hop_solid", "climb_hop_x", "climb_hop_y"}
            missing_hop = sorted(required_hop - model.keys() | required_hop.intersection(unavailable))
            if missing_hop:
                return {"ready": False, "reason": "climb_hop_attachment_unavailable",
                        "missing_fields": missing_hop, "warnings": []}
        if observation.get("transitioning", False):
            return {"ready": False, "reason": "room_transition_in_progress", "warnings": []}
        if observation.get("in_cutscene", False):
            return {"ready": False, "reason": "cutscene_in_progress", "warnings": []}
        required = _CLIMB_REQUIRED if core["state"] == 1 else (
            _MID_AIR_REQUIRED if not core["on_ground"] or core["vy"] < 0 else ())
        missing = sorted(name for name in required if name not in model or name in unavailable)
        if missing:
            return {"ready": False, "reason": "missing_airborne_model_state", "missing_fields": missing,
                    "warnings": warnings, "unavailable_fields": unavailable}
        if not observation.get("model_state"):
            warnings.append("Private timers, input edges and subpixels are unavailable; grounded normal-state defaults are approximate.")
        elif unavailable:
            warnings.append("Unavailable model fields: " + ", ".join(unavailable))
        if approximate:
            warnings.append("Approximate observed fields: " + ", ".join(approximate))
        if snapshot is None:
            warnings.append("Live dynamic-entity phases are unavailable unless included in model_state; retained predictions are not measurements.")
        else:
            warnings.append("Entity identities, positions and refill cooldowns are observed; mover coroutine timing is still approximate, inferred from progress or retained from the preceding prediction.")
        return {"ready": True, "reason": "ready", "warnings": warnings,
                "unavailable_fields": unavailable, "approximate_fields": approximate,
                "exact": False}
    except (RuntimeStateError, TypeError) as exc:
        return {"ready": False, "reason": "invalid_observation", "error": str(exc), "warnings": warnings}


def state_from_observation(map_data: MapData, observation: Mapping[str, Any],
                           previous: SimState | None = None) -> SimState:
    """Create an offline search state at the actual observed world position.

    ``previous`` supplies unobserved entity state only for the same room. It is
    never allowed to supply an observed position or confirmed collection. On a
    room change the previous room's platform, refill, break and fled state is
    discarded, unless fresh live fields explicitly supply replacements.
    """
    readiness = state_readiness(observation)
    if not readiness["ready"]:
        detail = readiness.get("error") or ", ".join(readiness.get("missing_fields", []))
        raise RuntimeStateError(readiness["reason"] + (": " + detail if detail else ""))
    core = _validate_core(observation)
    if core["room"] not in map_data.rooms:
        raise RuntimeStateError(f"Observed room {core['room']!r} is not in the loaded map")
    model = _validate_private(observation.get("model_state", {}))
    unavailable = set(observation.get("model_state_unavailable", ()))
    model = {name: value for name, value in model.items() if name not in unavailable}
    same_room = previous is not None and previous.room == core["room"]
    # Never call Simulator.initial_state(): that would replace actual entry
    # position with a spawn and make a route appear reachable by teleporting.
    if same_room:
        data = {field.name: getattr(previous, field.name) for field in fields(SimState)}
    else:
        fresh = SimState(core["room"], core["x"], core["y"])
        data = {field.name: getattr(fresh, field.name) for field in fields(SimState)}
        if core["on_ground"] and core["vy"] >= 0:
            data["grace"] = 6
    data.update(model)
    data.update(core)  # Runtime core always overrides duplicates/predictions.
    # Never inherit an unobserved player attachment from the previous state.
    # An expired forced interval clears it before its next native update.
    if not data["force_frames"] or "climb_hop_solid" not in model:
        data.update(climb_hop_solid="", climb_hop_x=0.0, climb_hop_y=0.0)
    _merge_environment(map_data, observation, data)
    _merge_cassette_environment(map_data, observation, data)
    # The native game can still catch a flying berry during its departure.
    # Its flyingAway flag may remain set, but actual follower/collected IDs win.
    data["fled"] = data["fled"] - core["following"] - core["berries"]
    data["dash_frames"] = 0  # Only normal/climb anchors are accepted.
    actual_following = core["following"]
    supplied_order = model.get("follow_order")
    if supplied_order is None:
        supplied_order = previous.follow_order if same_room else ()
    order = [identity for identity in supplied_order if identity in actual_following]
    order.extend(sorted(actual_following - set(order)))
    data["follow_order"] = tuple(order)
    if not actual_following:
        data["collect_delay"] = data["collect_ground"] = 0
    elif "collect_delay" not in model and (not same_room or actual_following != previous.following):
        # Unknown follower ages are conservatively delayed, never pre-collected.
        data["collect_delay"] = 18
        data["collect_ground"] = 0
    data["reason"] = "runtime_anchor_approximate"
    return SimState(**data)
