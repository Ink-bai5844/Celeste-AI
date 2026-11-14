"""Pure, explicitly anchored vanilla cassette-block mechanics.

Reference: the local Celeste.exe CassetteBlock, CassetteBlockManager, Platform,
Solid, Actor and Player classes.  This module does not infer a beat from a room
frame, create an unobserved clock, model the Cassette pickup cutscene, or mark a
room supported.  Callers must supply complete telemetry and all other solid /
jump-through rectangles, and apply returned static-mover events to attachments.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import struct
from typing import Mapping
from .state_copy import evolve as replace

Rect = tuple[float, float, float, float]


def _f32(value: float) -> float:
    return struct.unpack("f", struct.pack("f", value))[0]


def _read(data: Mapping, names: tuple[str, ...]) -> dict:
    if not isinstance(data, Mapping) or data.get("unavailable"):
        raise ValueError("cassette telemetry is unavailable")
    missing = [name for name in names if name not in data or data[name] is None]
    if missing:
        raise ValueError("missing cassette telemetry: " + ", ".join(missing))
    return {name: data[name] for name in names}


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid cassette " + name)
    return float(value)


def _integer(value, name: str) -> int:
    number = _number(value, name)
    if number != int(number):
        raise ValueError("invalid cassette integer " + name)
    return int(number)


def _boolean(value, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError("invalid cassette boolean " + name)
    return value


@dataclass(frozen=True)
class CassetteClock:
    current_index: int
    beat_timer: float
    beat_index: int
    tempo_mult: float
    lead_beats: int
    max_beat: int
    is_level_music: bool
    beat_index_offset: int
    active: bool
    music_ready: bool
    update_order: int
    depth: int = 0
    actual_depth: float = 0.0
    depth_counters: tuple[tuple[int, float], ...] = ()

    @classmethod
    def from_dict(cls, data: Mapping) -> "CassetteClock":
        if not isinstance(data, Mapping) or data.get("present") is False:
            raise ValueError("cassette manager is absent")
        values = _read(data, tuple(cls.__dataclass_fields__))
        for name in ("current_index", "beat_index", "lead_beats", "max_beat", "beat_index_offset", "update_order", "depth"):
            values[name] = _integer(values[name], name)
        for name in ("beat_timer", "tempo_mult", "actual_depth"):
            values[name] = _number(values[name], name)
        counters = values["depth_counters"]
        if isinstance(counters, Mapping):
            try:
                counters = [(int(key), value) for key, value in counters.items()]
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid cassette depth counters") from exc
        if not isinstance(counters, (list, tuple)):
            raise ValueError("invalid cassette depth counters")
        values["depth_counters"] = tuple(sorted((_integer(key, "depth key"), _number(value, "depth counter"))
                                                for key, value in counters))
        if len({key for key, _ in values["depth_counters"]}) != len(values["depth_counters"]):
            raise ValueError("duplicate cassette depth counters")
        for name in ("is_level_music", "active", "music_ready"):
            values[name] = _boolean(values[name], name)
        if (values["max_beat"] < 1 or not -1 <= values["current_index"] < values["max_beat"]
                or not 0 <= values["beat_index"] < 256 or values["lead_beats"] < 0
                or values["beat_timer"] < 0 or values["tempo_mult"] <= 0 or values["update_order"] < 0):
            raise ValueError("invalid cassette clock phase")
        return cls(**values)

    def to_dict(self) -> dict:
        result = {"present": True, **asdict(self)}
        result["depth_counters"] = {str(key): value for key, value in self.depth_counters}
        return result


@dataclass(frozen=True)
class CassetteBlockState:
    id: str
    index: int
    tempo: float
    x: float
    y: float
    width: float
    height: float
    activated: bool
    collidable: bool
    active: bool
    block_height: int
    group_leader: bool
    group_ids: tuple[str, ...]
    update_order: int
    rem_y: float
    lift_x: float
    lift_y: float
    depth: int = 0
    actual_depth: float = 0.0
    tracker_order: int = -1

    @classmethod
    def from_dict(cls, data: Mapping) -> "CassetteBlockState":
        values = _read(data, tuple(cls.__dataclass_fields__))
        for name in ("index", "block_height", "update_order", "depth", "tracker_order"):
            values[name] = _integer(values[name], name)
        for name in ("tempo", "x", "y", "width", "height", "rem_y", "lift_x", "lift_y", "actual_depth"):
            values[name] = _number(values[name], name)
        for name in ("activated", "collidable", "active", "group_leader"):
            values[name] = _boolean(values[name], name)
        if not isinstance(values["id"], str) or not values["id"]:
            raise ValueError("invalid cassette block identity")
        group = values["group_ids"]
        if not isinstance(group, (list, tuple)) or any(not isinstance(x, str) or not x for x in group):
            raise ValueError("invalid cassette group")
        values["group_ids"] = tuple(group)
        if (values["id"] not in group or len(set(group)) != len(group)
                or values["width"] <= 0 or values["height"] <= 0
                or values["index"] < 0 or values["update_order"] < 0):
            raise ValueError("invalid cassette block geometry/group")
        return cls(**values)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["group_ids"] = list(self.group_ids)
        return data

    @property
    def rect(self) -> Rect:
        return self.x, self.y, self.x + self.width, self.y + self.height


@dataclass(frozen=True)
class CassetteActor:
    """Normal player collider/fields affected by a block update (feet position)."""
    x: float
    y: float
    ducking: bool = False
    state: int = 0
    facing: int = 1
    climb_trigger_dir: int = 0
    dead: bool = False
    rem_y: float = 0.0
    lift_vx: float = 0.0
    lift_vy: float = 0.0
    lift_timer: int = 0
    allow_pushing: bool = True
    ignore_jumpthrus: bool = False

    @property
    def rect(self) -> Rect:
        return self.x - 4, self.y - (6 if self.ducking else 11), self.x + 4, self.y


@dataclass(frozen=True)
class CassetteEvent:
    kind: str
    identity: str = ""
    amount: float = 0.0


@dataclass(frozen=True)
class CassetteStep:
    clock: CassetteClock
    blocks: tuple[CassetteBlockState, ...]
    actor: CassetteActor
    events: tuple[CassetteEvent, ...]


def advance_clock(clock: CassetteClock, dt: float = 1 / 60, *, music_time: bool = False) -> tuple[CassetteClock, tuple[str, int] | None]:
    """Advance exactly one native manager update, preserving Single rounding.

    Events are ('activate', index) or ('will_toggle', next_index).  Native uses
    one ``if``, not a catch-up loop, even when tempo/delta crosses several beats.
    """
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("invalid cassette delta time")
    if not clock.active:
        return clock, None
    if not music_time and not clock.is_level_music and not clock.music_ready:
        return replace(clock, music_ready=True), None
    amount = _f32(dt) if music_time else _f32(_f32(dt) * clock.tempo_mult)
    timer = _f32(clock.beat_timer + amount)
    beat, current, lead = clock.beat_index, clock.current_index, clock.lead_beats
    event = None
    interval = _f32(1 / 6)
    if timer >= interval:
        timer = _f32(timer - interval)
        beat = (beat + 1) % 256
        if beat % 8 == 0:
            current = (current + 1) % clock.max_beat
            event = ("activate", current)
        elif (beat + 1) % 8 == 0:
            event = ("will_toggle", (current + 1) % clock.max_beat)
        if lead > 0:
            lead -= 1
            if lead == 0:
                beat = 0
    return replace(clock, beat_timer=timer, beat_index=beat, current_index=current, lead_beats=lead), event


def _overlap(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _translated(rect: Rect, dx: float = 0, dy: float = 0) -> Rect:
    return rect[0] + dx, rect[1] + dy, rect[2] + dx, rect[3] + dy


def validate_blocks(clock: CassetteClock | None, blocks: tuple[CassetteBlockState, ...]) -> None:
    """Reject partial groups and ambiguous update ordering; never guess them."""
    identities = {block.id: block for block in blocks}
    orders = [block.update_order for block in blocks]
    if clock is not None:
        orders.append(clock.update_order)
    if len(identities) != len(blocks) or len(set(orders)) != len(orders):
        raise ValueError("duplicate cassette identity or update order")
    for block in blocks:
        if any(identity not in identities for identity in block.group_ids):
            raise ValueError("incomplete cassette group telemetry")
        group = [identities[identity] for identity in block.group_ids]
        if (sum(member.group_leader for member in group) != 1
                or any(member.group_ids != block.group_ids or member.index != block.index for member in group)):
            raise ValueError("inconsistent cassette group telemetry")


def step_cassette(clock: CassetteClock, blocks: tuple[CassetteBlockState, ...], actor: CassetteActor,
                  *, solid_rects: tuple[Rect, ...] = (), jumpthru_rects: tuple[Rect, ...] = (),
                  dt: float = 1 / 60, after_order: int | None = None,
                  before_order: int | None = None,
                  update_schedule: tuple[tuple[int, str], ...] | None = None,
                  music_advance: float | None = None,
                  validate: bool = True) -> CassetteStep:
    """Run an open interval of native entity update order, without player input.

    Split around the observed player order: call with before_order=player_order,
    perform Player.Update, then call with after_order=player_order on the returned
    clock/blocks.  Callers must include other actors in their unsupported gate:
    Theo's earlier BlockedCheck and arbitrary static-mover callbacks are outside
    this player-only module.  Nothing here advances a room transition/camera.
    """
    if validate:
        validate_blocks(clock, blocks)
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("invalid cassette delta time")
    state = {block.id: block for block in blocks}
    events: list[CassetteEvent] = []
    counters = dict(clock.depth_counters)
    depth_changed = False

    def visual_state(identity: str) -> None:
        nonlocal depth_changed
        block = state[identity]
        depth = 8990 if not block.collidable else (10 if actor.rect[1] >= block.rect[3] - 1 else -10)
        if depth != block.depth:
            depth_changed = True
            offset = counters.get(depth, 0.0)
            counters[depth] = offset + 9.999999974752427e-7
            state[identity] = replace(block, depth=depth, actual_depth=depth - offset)
            # The separate visual side entity changes Depth in this same call.
            # It uses a distinct layer but must advance Scene's shared counters.
            side_depth = depth + 5
            counters[side_depth] = counters.get(side_depth, 0.0) + 9.999999974752427e-7

    def collides(candidate: CassetteActor, *, exclude: str = "", force: str = "") -> bool:
        return (any(_overlap(candidate.rect, rect) for rect in solid_rects)
                or any(block.id != exclude and (block.collidable or block.id == force)
                       and _overlap(candidate.rect, block.rect) for block in state.values()))

    def squish(player: CassetteActor, target_y: float) -> CassetteActor:
        # Player.OnSquish first tries crouching, then Actor.TrySquishWiggle's
        # exact positive/negative X/Y iteration order at current and target pos.
        tried_duck = not player.ducking and player.state != 1
        if tried_duck:
            player = replace(player, ducking=True)
            if not collides(player):
                return player
            target = replace(player, y=target_y)
            if not collides(target):
                return target
        for base in (player, replace(player, y=target_y)):
            for dx in range(4):
                for dy in range(6):
                    if dx == dy == 0:
                        continue
                    for sx in (1, -1):
                        for sy in (1, -1):
                            candidate = replace(base, x=base.x + dx * sx, y=base.y + dy * sy)
                            if not collides(candidate):
                                if tried_duck and not collides(replace(candidate, ducking=False)):
                                    candidate = replace(candidate, ducking=False)
                                return candidate
        return replace(player, dead=True)

    def shift(identity: str, amount: int) -> None:
        nonlocal actor
        old = state[identity]
        side = actor.climb_trigger_dir or (actor.facing if actor.state in (1, 6) else 0)
        rider = old.collidable and actor.state != 23 and _overlap(
            _translated(actor.rect, dx=side, dy=0 if side or actor.state == 9 else 1), old.rect)
        remainder = _f32(old.rem_y + amount)
        move = round(remainder)  # System.Math.Round: midpoint to even.
        moved = replace(old, y=_f32(old.y + move), rem_y=_f32(remainder - move),
                        block_height=old.block_height - amount,
                        lift_y=_f32(amount / _f32(dt)) if dt else 0.0)
        state[identity] = moved
        events.append(CassetteEvent("shift", identity, move))
        if not move or not old.collidable or not actor.allow_pushing or actor.dead:
            return
        pushed = _overlap(actor.rect, moved.rect)
        distance = move
        if pushed:
            distance = (move - int(actor.rect[3] - old.rect[1]) if move <= 0
                        else move - int(actor.rect[1] - old.rect[3]))
        if not pushed and not rider:
            return
        target_y = actor.y + distance
        direction = 1 if distance > 0 else -1
        for _ in range(abs(distance)):
            candidate = replace(actor, y=actor.y + direction)
            jumpthru_hit = (direction > 0 and not actor.ignore_jumpthrus
                            and any(not _overlap(actor.rect, rect) and _overlap(candidate.rect, rect)
                                    for rect in jumpthru_rects))
            if collides(candidate, exclude=identity) or jumpthru_hit:
                actor = replace(actor, rem_y=0.0)
                if pushed:
                    actor = squish(actor, target_y)
                    events.append(CassetteEvent("squish" if actor.dead else "squish_escape", identity))
                break
            actor = candidate
        # Actor.LiftSpeed stores nonzero speed for its native 0.16s grace.
        if moved.lift_x or moved.lift_y:
            actor = replace(actor, lift_vx=moved.lift_x, lift_vy=moved.lift_y, lift_timer=10)

    def blocked(identity: str) -> bool:
        nonlocal actor
        block = state[identity]
        if actor.dead or not _overlap(actor.rect, block.rect):
            return False
        # The native group check intentionally ignores other blocks' Collidable.
        if any(other != identity and _overlap(actor.rect, _translated(state[other].rect, dy=4))
               for other in block.group_ids):
            return True
        for amount in range(1, 5):
            candidate = replace(actor, y=actor.y - amount)
            if not collides(candidate, force=identity):
                actor = candidate  # direct Position write does not clear remainder.
                events.append(CassetteEvent("wiggle_up", identity, -amount))
                return False
        return True

    ordered = ([(clock.update_order, "")] if music_advance is not None else
               list(update_schedule) if update_schedule is not None else
               [(clock.update_order, ""), *((block.update_order, block.id) for block in blocks)])
    for order, identity in sorted(ordered):
        if ((after_order is not None and order <= after_order)
                or (before_order is not None and order >= before_order)):
            continue
        if not identity:
            clock, event = advance_clock(clock, dt if music_advance is None else music_advance,
                                          music_time=music_advance is not None)
            if event is None:
                continue
            kind, index = event
            if kind == "activate":
                for key, block in tuple(state.items()):
                    state[key] = replace(block, activated=block.index == index)
            else:
                # Tracker insertion order is independent of EntityList's
                # depth-sorted Update order. Old isolated fixtures may omit it.
                for block in sorted(blocks, key=lambda item: item.tracker_order if item.tracker_order >= 0 else item.update_order):
                    current = state[block.id]
                    if current.index == index or current.activated:
                        shift(current.id, 1 if current.collidable else -1)
                        visual_state(current.id)
            events.append(CassetteEvent(kind, amount=index))
            continue
        block = state[identity]
        if not block.active:
            continue
        # Platform.Update clears LiftSpeed before CassetteBlock.Update.
        if block.lift_x or block.lift_y:
            state[identity] = block = replace(block, lift_x=0.0, lift_y=0.0)
        if block.group_leader and block.activated and not block.collidable:
            if any(blocked(member) for member in block.group_ids):
                events.append(CassetteEvent("blocked", identity))
                visual_state(identity)
                continue
            for member in block.group_ids:
                state[member] = replace(state[member], collidable=True)
                events.append(CassetteEvent("enable", member))
                shift(member, -1)
        elif not block.activated and block.collidable:
            shift(identity, 1)
            state[identity] = replace(state[identity], collidable=False)
            events.append(CassetteEvent("disable", identity))
        visual_state(identity)
    if depth_changed:
        clock = replace(clock, depth_counters=tuple(sorted(counters.items())))
    return CassetteStep(clock, tuple(state[block.id] for block in blocks), actor, tuple(events))
