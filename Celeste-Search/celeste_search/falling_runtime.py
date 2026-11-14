"""Decode the verified vanilla FallingBlock iterator without guessing a phase.

The raw fields come from Celeste.FallingBlock+<Sequence>d__21. The six-element
mechanism record stores exact position and the same timer convention used by
Simulator: elapsed ticks for initial/contact shake, remaining ticks for waits.
"""
from __future__ import annotations

import math
import struct
from typing import Mapping

_ITERATOR = "Celeste.FallingBlock+<Sequence>d__21"
_DT = struct.unpack("<f", struct.pack("<f", .0166667))[0]


class FallingRuntimeError(ValueError):
    pass


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FallingRuntimeError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise FallingRuntimeError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise FallingRuntimeError(f"{name} must be a finite number")
    return number


def _single(value, name):
    """Restore the native Single from its round-trippable JSON decimal."""
    number = _number(value, name)
    try:
        native = struct.unpack("<f", struct.pack("<f", number))[0]
    except (OverflowError, struct.error) as error:
        raise FallingRuntimeError(f"{name} must fit a finite native Single") from error
    if not math.isfinite(native):
        raise FallingRuntimeError(f"{name} must fit a finite native Single")
    return native


def _boolean(record, name):
    if type(record.get(name)) is not bool:
        raise FallingRuntimeError(f"{name} must be explicitly observed")
    return record[name]


def _remaining(value, maximum, name):
    value = _number(value, name)
    if not -_DT-1e-6 <= value <= maximum+1e-6:
        raise FallingRuntimeError(f"{name} is outside the verified wait range")
    # JSON decimal spelling can land just above an integer tick multiple.
    # Restore the observed C# Single before converting it to update counts.
    native = _single(value, name)
    return math.ceil(max(0., native)/_DT)


def _tiles_below(room, record, entity):
    # Native completion specifically checks SolidTiles, not any Platform.
    x, y = (_single(record[key], key) for key in ("x", "y"))
    left, right = math.floor((x-room.x)/8), math.ceil((x+entity.width-room.x)/8)
    top, bottom = math.floor((y+1-room.y)/8), math.ceil((y+entity.height+1-room.y)/8)
    return any(room.tile_at(tx, ty) for ty in range(top, bottom) for tx in range(left, right))


def decode_falling_record(record, entity, room):
    """Return (identity, phase, timer, exact_x, exact_y, speed) or refuse.

    Explicit absence is fresh evidence even when its last coordinates are retained.
    For present entities, all phase-bearing fields must come from this observation.
    Triggered=true needs a persistent flag that the six-field representation lacks.
    """
    # Behind blocks update before the player, unlike the verified ordinary
    # schedule. Native boss blocks have their own map kind; the attribute check
    # also refuses normalized/custom records before an early idle phase can pass.
    for flag in ("behind", "finalBoss"):
        value = entity.attrs.get(flag, False)
        if value is True or str(value).lower() in ("true", "1"):
            raise FallingRuntimeError(f"{flag}=true fallingBlock variant is not modeled")
    if entity.kind == "finalBossFallingBlock":
        raise FallingRuntimeError("finalBossFallingBlock variant is not modeled")
    identity = f"{room.name}:{entity.id}"
    if record.get("id") != identity or record.get("room", room.name) != room.name:
        raise FallingRuntimeError("fallingBlock identity does not match the current room")
    present = _boolean(record, "present")
    x, y = (_single(record.get(key), key) for key in ("x", "y"))
    if not present:
        return identity, 4, 0, x, y, 0.
    if record.get("retained_observation") is not False:
        raise FallingRuntimeError("present fallingBlock needs a fresh observation")
    if not _boolean(record, "active"):
        raise FallingRuntimeError("inactive fallingBlock cannot advance in the model")
    collidable = _boolean(record, "collidable")
    started = _boolean(record, "has_started_falling")
    if _boolean(record, "triggered"):
        raise FallingRuntimeError("Triggered=true requires an unmodeled persistent trigger flag")
    delay = _number(record.get("fall_delay"), "fall_delay")
    unavailable = record.get("unavailable")
    if not isinstance(unavailable, list) or any(not isinstance(v, str) for v in unavailable):
        raise FallingRuntimeError("fallingBlock field availability must be explicit")
    if set(unavailable)-{"coroutine_phase"}:
        raise FallingRuntimeError("fallingBlock raw fields are unavailable: "+", ".join(unavailable))
    for key, expected in (("width", entity.width), ("height", entity.height)):
        if _number(record.get(key), key) != expected:
            raise FallingRuntimeError("fallingBlock collision size differs from the map")
    # Keep integer-position + subpixel as a double sum. Rounding that sum to
    # Single would erase part of the movement remainder at world coordinates.
    x += _single(record.get("rem_x"), "rem_x")
    y += _single(record.get("rem_y"), "rem_y")
    for key in ("lift_vx", "lift_vy"):
        _number(record.get(key), key)
    coroutines = record.get("coroutines")
    if not isinstance(coroutines, list):
        raise FallingRuntimeError("fallingBlock raw coroutine list is missing")
    if not coroutines:
        # The bridge omits finished empty stacks. Sequence's sole normal,
        # still-collidable termination is its SolidTiles-supported completion.
        if started and collidable and delay <= 0 and _tiles_below(room, record, entity):
            return identity, 3, 0, x, y, 0.
        raise FallingRuntimeError("missing Sequence cannot establish a fallingBlock phase")
    if len(coroutines) != 1 or not isinstance(coroutines[0], Mapping):
        raise FallingRuntimeError("unsupported fallingBlock coroutine schema")
    coroutine = coroutines[0]
    active = _boolean(coroutine, "active")
    finished = _boolean(coroutine, "finished")
    stack = coroutine.get("stack")
    if finished and not active and stack == []:
        if started and collidable and delay <= 0 and _tiles_below(room, record, entity):
            return identity, 3, 0, x, y, 0.
        raise FallingRuntimeError("finished fallingBlock is not supported by SolidTiles")
    if not active or finished or not isinstance(stack, list) or len(stack) != 1:
        raise FallingRuntimeError("fallingBlock requires its active single Sequence iterator")
    iterator = stack[0]
    if not isinstance(iterator, Mapping) or iterator.get("iterator_type") != _ITERATOR:
        raise FallingRuntimeError("unsupported fallingBlock iterator layout")
    fields = iterator.get("fields")
    if not isinstance(fields, Mapping) or type(fields.get("<>1__state")) is not int:
        raise FallingRuntimeError("fallingBlock iterator state is unavailable")
    native = fields["<>1__state"]
    wait = _number(coroutine.get("wait_timer"), "coroutine.wait_timer")
    if native in (6, 7):
        if not started or collidable:
            raise FallingRuntimeError("removed fallingBlock has contradictory collision state")
        _remaining(wait, .2, "coroutine.wait_timer")
        return identity, 4, 0, x, y, 0.
    if not collidable:
        raise FallingRuntimeError("noncollidable fallingBlock lacks a verified removal phase")
    if native in (0, 1):
        if started or delay > 0:
            raise FallingRuntimeError("idle fallingBlock has an unsupported pending delay/state")
        return identity, 0, 0, x, y, 0.
    if native == 2:
        if started:
            raise FallingRuntimeError("FallDelay cannot follow HasStartedFalling")
        return identity, 7, _remaining(delay, 60., "fall_delay"), x, y, 0.
    if not started:
        raise FallingRuntimeError("active falling phase requires HasStartedFalling")
    speed = _single(fields.get("<speed>5__2"), "iterator.speed")
    if not 0 <= speed <= 160:
        raise FallingRuntimeError("unsupported fallingBlock iterator speed")
    if native == 3:
        return identity, 1, 12-_remaining(wait, .2, "coroutine.wait_timer"), x, y, speed
    if native == 4:
        timer = _number(fields.get("<timer>5__4"), "iterator.timer")
        if not 0 < timer <= .400001:
            raise FallingRuntimeError("contact wait timer is outside the vanilla phase")
        return identity, 1, 37-_remaining(timer, .4, "iterator.timer"), x, y, speed
    if native == 8:
        maximum = _number(fields.get("<maxSpeed>5__3"), "iterator.maxSpeed")
        if maximum != 160 or not 0 < speed <= maximum:
            raise FallingRuntimeError("unsupported fallingBlock speed/final-boss variant")
        return identity, 2, 0, x, y, speed
    if native == 5:
        return identity, 5, _remaining(wait, .2, "coroutine.wait_timer"), x, y, speed
    if native == 9:
        return identity, 6, _remaining(wait, .1, "coroutine.wait_timer"), x, y, speed
    raise FallingRuntimeError(f"unsupported fallingBlock iterator state {native}")
