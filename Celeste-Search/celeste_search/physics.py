"""Deterministic, offline, source-guided Celeste movement model.

This is a planning model, not the original engine and not a conformance claim.
It never imports, launches or calls Celeste. Unsupported mechanics are reported
by ``capability_report`` and must not silently be treated as solved.

Movement constants/ordering were checked against the author-published source:
https://github.com/NoelFB/Celeste/blob/master/Source/Player/Player.cs
Entity semantics were checked against Spikes.cs, Strawberry.cs and Level.cs:
https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/tree/main/Celeste

Coordinates are world pixels; x/y is the player's bottom centre. One step is
one *player update*, excluding engine freeze and camera-transition frames.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import math
import struct
from typing import Iterable

from .maps import Entity, MapData, Room
from .state_copy import evolve as replace
from .cassette import CassetteClock, CassetteBlockState, CassetteActor, step_cassette, validate_blocks

RIGHT, LEFT, UP, DOWN, JUMP, DASH, GRAB = 1, 2, 4, 8, 16, 32, 64
_SINGLE = struct.Struct("<f")


def _f32(value: float) -> float:
    """C# Single rounding at gameplay/Actor accumulation boundaries."""
    return _SINGLE.unpack(_SINGLE.pack(value))[0]


# Instrumentation's RawDeltaTime getter fixes the raw tick to this Single
# constant; normal (TimeRate=1) player updates use the same Engine.DeltaTime.
# Exact 1/60 with Python doubles changes midpoint-to-even pixel decisions.
DT = _f32(0.0166667)
MAX_RUN, RUN_ACCEL, RUN_REDUCE, AIR_MULT = 90.0, 1000.0, 400.0, 0.65
GRAVITY, MAX_FALL, JUMP_SPEED, DASH_SPEED = 900.0, 160.0, -105.0, 240.0
MODEL_VERSION = "offline-physics-v2"

# Decorative entities have no collision or route-state role. Unknown entities
# remain unsupported, even if a route happens to avoid their initial position.
DECORATIVE_KINDS = frozenset({
    "player", "checkpoint", "backgroundTiles", "foregroundTiles", "decal",
    "lamp", "hanginglamp", "hangingLamp", "light", "torch", "bonfire",
    "campfire", "reflections", "bridgeFixed", "birdForsakenCityGem",
    "clouds", "memorial", "memorialTextController", "soundSource", "ambienceParamTrigger",
    "musicTrigger", "musicFadeTrigger", "cameraOffsetTrigger", "cameraTargetTrigger",
    "cameraAdvanceTargetTrigger", "cameraAdvanceTarget", "changeRespawnTrigger",
    "respawnTargetTrigger", "miniTextboxTrigger", "rumbleTrigger", "everest/ambienceVolumeTrigger",
    "blackGem", "interactTrigger", "wavedashMachine", "wire", "fgTiles", "bgTiles",
    "flutterbird", "lightbeam", "pointLight", "bloomPoint", "wallTorch", "cobweb", "clothesline",
})
SUPPORTED_KINDS = frozenset({
    "strawberry", "spikesUp", "spikesDown", "spikesLeft", "spikesRight",
    "jumpThru", "spring", "wallSpringLeft", "wallSpringRight", "refill",
    "invisibleBarrier", "solid", "player",
    "dashBlock", "crumbleBlock", "zipMover", "fallingBlock",
})
APPROXIMATE_KINDS = frozenset({"spinner", "fakeWall", "coverupWall", "crumbleBlock", "zipMover", "fallingBlock"})
# Entity/Trigger subclasses do not block Actor movement. Their scripted effects
# belong to the running engine; telemetry must invalidate playback on a cutscene.
RUNTIME_SCRIPT_KINDS = frozenset({"npc", "eventTrigger", "goldenBerryCollectTrigger"})
MECHANISM_KINDS = frozenset({"dashBlock", "crumbleBlock", "zipMover", "fallingBlock"})
UNSUPPORTED_COLLECTIBLES = frozenset({"goldenBerry", "memorialTextController", "snowberry"})


def _approach(value: float, target: float, amount: float) -> float:
    amount = _f32(amount)
    return min(_f32(value + amount), target) if value < target else max(_f32(value - amount), target)


@lru_cache(maxsize=256)
def _zip_ease(timer, duration):
    """ZipMover's repeated Single approach and Ease.SineIn, not t/duration."""
    at = 0.0
    for _ in range(min(timer,duration)):
        at = _approach(at,1.0,(2.0 if duration == 30 else .5)*DT)
    # Native x86 Ease keeps the multiplication temporary until Math.Cos;
    # only the cosine result and returned easing value are stored as Single.
    return _f32(1.0-_f32(math.cos(_f32(math.pi/2)*at)))


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _truth(value: object) -> bool:
    return value is True or str(value).lower() in ("true", "1")


def spring_player_enabled(entity: Entity) -> bool:
    """Spring's EntityData.Bool(playerCanUse, true), including string values."""
    value = entity.attrs.get("playerCanUse", True)
    if isinstance(value, bool):
        return value
    # Native bool.TryParse accepts true/false; other representations fall back
    # to the constructor's true default rather than Python truthiness.
    return str(value).strip().lower() != "false"


def _runtime_height(room: Room) -> int:
    # LevelData's constructor normalizes the legacy 184px editor room height.
    # Tile bounds still use ceil(height/8), so the 23-row tile grid is retained.
    return 180 if room.height == 184 else room.height


def _spike_rect(entity: Entity, x: float | None = None, y: float | None = None):
    x = entity.x if x is None else x
    y = entity.y if y is None else y
    if entity.kind == "spikesUp":
        return x, y - 3, x + max(8, entity.width), y
    if entity.kind == "spikesDown":
        return x, y, x + max(8, entity.width), y + 3
    if entity.kind == "spikesLeft":
        return x - 3, y, x, y + max(8, entity.height)
    return x, y, x + 3, y + max(8, entity.height)


def _spike_rides(entity: Entity, mover: Entity) -> bool:
    """Spikes.IsRiding: outside now, overlapping after one pixel toward support."""
    shift = {"spikesUp": (0, 1), "spikesDown": (0, -1),
             "spikesLeft": (1, 0), "spikesRight": (-1, 0)}.get(entity.kind)
    if shift is None:
        return False
    support = (mover.x, mover.y, mover.x + mover.width,
               mover.y + (8 if mover.kind == "crumbleBlock" else mover.height))
    return (not _overlap(_spike_rect(entity), support) and
            _overlap(_spike_rect(entity, entity.x + shift[0], entity.y + shift[1]), support))


@dataclass(frozen=True)
class SimState:
    room: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    dashes: int = 1
    stamina: float = 110.0
    on_ground: bool = False
    dead: bool = False
    berries: frozenset[str] = frozenset()
    following: frozenset[str] = frozenset()
    frame: int = 0
    facing: int = 1
    state: int = 0
    prev_input: int = 0
    rem_x: float = 0.0
    rem_y: float = 0.0
    grace: int = 0
    jump_buffer: int = 0
    dash_buffer: int = 0
    var_jump: int = 0
    var_speed: float = JUMP_SPEED
    dash_frames: int = 0
    dash_attack: int = 0
    dash_cooldown: int = 0
    refill_cooldown: int = 0
    dash_dx: float = 0.0
    dash_dy: float = 0.0
    dash_before_x: float = 0.0
    dash_started_ground: bool = False
    force_x: int = 0
    force_frames: int = 0
    wall_slide: int = 72
    wall_retention: int = 0
    wall_retained: float = 0.0
    hop_wait_x: int = 0
    hop_wait_x_speed: float = 0.0
    # ClimbHop follows the contacted moving solid during its forced-input
    # interval, including airborne frames. Position is the last integer solid
    # position observed by Player.Update, before the later entity update.
    climb_hop_solid: str = ""
    climb_hop_x: float = 0.0
    climb_hop_y: float = 0.0
    climb_wait: int = 0
    ducking: bool = False
    # Player.AutoJump is independent of its optional countdown. Active + zero
    # timer is indefinite (dash exit, upward transition, spring or rebound).
    auto_jump: int = 0
    auto_jump_active: bool = False
    max_fall: float = MAX_FALL
    collect_delay: int = 0
    collect_ground: int = 0
    follow_order: tuple[str, ...] = ()
    fled: frozenset[str] = frozenset()
    # Sorted tuples keep state immutable and include refill availability.
    refill_timers: tuple[tuple[str, int], ...] = ()
    broken: frozenset[str] = frozenset()
    # key, phase, timer, world_x, world_y, falling_speed. Kept in every branch.
    mechanisms: tuple[tuple[str, int, int, float, float, float], ...] = ()
    lift_vx: float = 0.0
    lift_vy: float = 0.0
    lift_timer: int = 0
    reason: str = ""
    # Current-room evidence of a removed manager and fully inactive blocks.
    # This is not a session-persistent claim that a cassette was collected.
    cassette_stopped: bool = False
    cassette_clock: CassetteClock | None = None
    cassette_blocks: tuple[CassetteBlockState, ...] = ()
    player_update_order: int = -1
    player_depth: int = 0
    player_actual_depth: float = 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        for key in ("berries", "following", "fled", "broken"):
            data[key] = sorted(data[key])
        data["cassette_clock"] = self.cassette_clock.to_dict() if self.cassette_clock else None
        data["cassette_blocks"] = [block.to_dict() for block in self.cassette_blocks]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SimState":
        data = dict(data)
        for key in ("berries", "following", "fled", "broken"):
            if key in data:
                data[key] = frozenset(data[key])
        if "follow_order" in data:
            data["follow_order"] = tuple(data["follow_order"])
        if "refill_timers" in data:
            data["refill_timers"] = tuple(tuple(x) for x in data["refill_timers"])
        if "mechanisms" in data:
            data["mechanisms"] = tuple(tuple(x) for x in data["mechanisms"])
        if "cassette_stopped" in data and type(data["cassette_stopped"]) is not bool:
            raise ValueError("invalid cassette_stopped boolean")
        if data.get("cassette_clock") is not None:
            data["cassette_clock"] = CassetteClock.from_dict(data["cassette_clock"])
        if "cassette_blocks" in data:
            data["cassette_blocks"] = tuple(CassetteBlockState.from_dict(x) for x in data["cassette_blocks"])
        return cls(**data)


class Simulator:
    """Copyable-state offline simulator; no game process or socket dependency."""

    def __init__(self, map_data: MapData, *, max_dashes: int = 1):
        self.map = map_data
        self.max_dashes = max_dashes
        self.last_step_started_dash = False
        self._tiles = {}
        # LevelLoader merges every room into one SolidTiles grid. Keep the
        # room-local sets for geometry/display; border collision probes must
        # still see static tiles belonging to the neighboring room.
        self._world_tiles = frozenset(
            (room.x + x*8, room.y + y*8)
            for room in map_data.rooms.values()
            for y, row in enumerate(room.solids)
            for x, tile in enumerate(row) if tile not in ("0", " ")
        )
        self._solids = {}
        self._platforms = {}
        self._hazards = {}
        self._interact = {}
        self._neighbors = {}
        self._mechanisms = {}
        self._spike_attachments = {}
        self._cassette_entities = {}
        self._static_rects = {}
        # Per-simulator bounded caches retain exact collision inputs. Static
        # tile spans never include breakable/moving geometry; those rectangles
        # are separately keyed by the complete mechanism and broken state.
        self._tile_span_occupied = lru_cache(maxsize=32768)(self._tile_span_occupied)
        self._cached_mechanism_rects = lru_cache(maxsize=2048)(self._cached_mechanism_rects)
        for room in map_data.rooms.values():
            self._tiles[room.name] = frozenset(
                (x, y) for y, row in enumerate(room.solids)
                for x, tile in enumerate(row) if tile not in ("0", " ")
            )
            self._solids[room.name] = tuple(
                (e.x, e.y, e.x + e.width, e.y + e.height)
                for e in room.entities if e.kind in ("invisibleBarrier", "solid")
            )
            self._platforms[room.name] = tuple(
                (e.x, e.y, e.x + e.width, e.y + 5)
                for e in room.entities if e.kind == "jumpThru"
            )
            self._hazards[room.name] = tuple(
                e for e in room.entities if e.kind.startswith("spikes") or e.kind == "spinner"
            )
            self._interact[room.name] = tuple(
                e for e in room.entities if e.kind in (
                    "strawberry", "spring", "wallSpringLeft", "wallSpringRight", "refill")
            )
            self._neighbors[room.name] = tuple(map_data.neighbors(room.name))
            self._mechanisms[room.name] = tuple(e for e in room.entities if e.kind in MECHANISM_KINDS)
            self._cassette_entities[room.name] = {f"{room.name}:{e.id}": e for e in room.entities if e.kind == "cassetteBlock"}
            spans = []
            for ty, row in enumerate(room.solids):
                first = None
                for tx in range(len(row) + 1):
                    occupied = tx < len(row) and row[tx] not in ("0", " ")
                    if occupied and first is None:
                        first = tx
                    elif not occupied and first is not None:
                        spans.append((room.x + first * 8, room.y + ty * 8, room.x + tx * 8, room.y + (ty + 1) * 8))
                        first = None
            self._static_rects[room.name] = tuple(spans) + self._solids[room.name]
            attachments = {}
            for hazard in self._hazards[room.name]:
                for mover in self._mechanisms[room.name]:
                    if mover.kind in ("zipMover", "fallingBlock", "dashBlock") and _spike_rides(hazard, mover):
                        attachments[(hazard.kind, hazard.id)] = mover
                        break
            self._spike_attachments[room.name] = attachments

    def initial_state(self, room_name: str | None = None, *, spawn_index: int | None = None) -> SimState:
        room = self.map.rooms[room_name] if room_name is not None else self.map.start_room()
        if not room.spawns:
            raise ValueError(f"Room {room.name!r} has no player spawn")
        # Level.DefaultSpawnPoint -> Session.GetSpawnPoint(room bottom-left).
        # Native entity order is not the selection criterion; min preserves ties.
        # Explicit spawn_index is only an offline fixture/debug override.
        if spawn_index is None:
            bottom = room.y + _runtime_height(room)
            x, y = min(room.spawns, key=lambda p: (p[0] - room.x) ** 2 + (p[1] - bottom) ** 2)
        else:
            x, y = room.spawns[spawn_index]
        s = SimState(room.name, float(x), float(y), dashes=self.max_dashes)
        grounded = self.on_ground(s)
        return replace(s, on_ground=grounded, grace=6 if grounded else 0)

    def cassette_ready(self, state: SimState | None, room_name: str) -> bool:
        """Accept a measured local phase, or a completely observed stopped room."""
        if state is None or state.room != room_name:
            return False
        blocks, clock = state.cassette_blocks, state.cassette_clock
        expected = self._cassette_entities[room_name]
        if (not expected or {b.id for b in blocks} != set(expected)
                or state.player_update_order < 0
                or any(not b.active or b.tracker_order < 0 for b in blocks)
                or len({b.tracker_order for b in blocks}) != len(blocks)
                or state.player_update_order in {b.update_order for b in blocks}):
            return False
        if state.cassette_stopped:
            if clock is not None:
                return False
            # Finish() leaves the observed size intact; a later room reload
            # starts at height 2. Never synthesize a beat or normalize geometry.
            for block in blocks:
                entity = expected[block.id]
                shift = 2 - block.block_height
                if (block.activated or block.collidable or block.depth != 8990
                        or block.block_height not in (0, 1, 2)
                        or block.x != entity.x or block.y != entity.y + shift
                        or block.width != entity.width or block.height != entity.height
                        or block.rem_y != 0 or block.lift_x != 0 or block.lift_y != 0):
                    return False
        elif clock is None or not clock.active or not clock.depth_counters:
            return False
        try:
            validate_blocks(clock, blocks)
        except ValueError:
            return False
        depths = [state.player_actual_depth, *(b.actual_depth for b in blocks)]
        if clock is not None:
            depths.append(clock.actual_depth)
        if not all(math.isfinite(value) for value in depths) or len(set(depths)) != len(depths):
            return False
        room = self.map.rooms[room_name]
        # These interactions add freeze time or static-mover callbacks that
        # are not yet integrated with the musical clock. Unknown actors are
        # already rejected by the ordinary room capability gate.
        if any(e.kind in {"dashBlock", "refill", "spring", "wallSpringLeft", "wallSpringRight"}
               or e.kind == "fallingBlock" and _truth(e.attrs.get("behind")) for e in room.entities):
            return False
        if any(_spike_rides(hazard, entity) for hazard in self._hazards[room_name]
               for entity in expected.values()):
            return False
        return True

    def capability_report(self, room_name: str | None = None, *, state: SimState | None = None) -> dict:
        rooms = [self.map.rooms[room_name]] if room_name is not None else list(self.map.rooms.values())
        unsupported = Counter()
        approximate = Counter()
        unsupported_collectibles = Counter()
        unsupported_rooms = {}
        for room in rooms:
            reasons = []
            cassette_supported = self.cassette_ready(state, room.name)
            for e in room.entities:
                if e.kind in {"cassetteBlock", "cassette"} and cassette_supported:
                    scope = "observed_stopped_room" if state.cassette_stopped else "observed_local_phase"
                    approximate["cassette:" + scope] += 1
                elif e.kind in UNSUPPORTED_COLLECTIBLES:
                    unsupported_collectibles[e.kind] += 1
                elif e.kind in APPROXIMATE_KINDS:
                    approximate[e.kind] += 1
                elif e.kind in RUNTIME_SCRIPT_KINDS:
                    approximate[e.kind + ":runtime_script"] += 1
                elif e.kind not in SUPPORTED_KINDS and e.kind not in DECORATIVE_KINDS:
                    unsupported[e.kind] += 1
                    reasons.append(e.kind)
                if e.kind == "strawberry" and (
                    e.nodes or _truth(e.attrs.get("golden")) or _truth(e.attrs.get("moon"))):
                    key = "strawberry:seeds/golden/moon"
                    unsupported_collectibles[key] += 1
                if e.kind == "strawberry" and _truth(e.attrs.get("winged")):
                    approximate["winged strawberry bob/flee timing"] += 1
                if e.kind == "zipMover" and not e.nodes:
                    unsupported["zipMover:missing_target"] += 1
                    reasons.append("zipMover:missing_target")
            if self._spike_attachments[room.name]:
                approximate["platform_attached_spikes"] += len(self._spike_attachments[room.name])
            for mover in self._mechanisms[room.name]:
                if mover.kind == "crumbleBlock" and any(_spike_rides(h, mover) for h in self._hazards[room.name]):
                    unsupported["crumble_attached_spikes"] += 1
                    reasons.append("crumble_attached_spikes")
            for attr in ("windPattern", "underwater", "space"):
                val = room.attrs.get(attr)
                if val not in (None, False, "", "None", "none", 0, "0", "false"):
                    unsupported[f"room:{attr}={val}"] += 1
                    reasons.append(f"room:{attr}")
            if reasons:
                unsupported_rooms[room.name] = sorted(set(reasons))
        return {
            "model": MODEL_VERSION,
            "fully_accurate": False,
            "supported": sorted(SUPPORTED_KINDS),
            "unsupported": dict(sorted(unsupported.items())),
            "unsupported_rooms": unsupported_rooms,
            "unsupported_collectibles": dict(sorted(unsupported_collectibles.items())),
            "approximate_entities": dict(sorted(approximate.items())),
            "planning_supported": not unsupported,
            "limitations": [
                "Source-guided model; not empirically calibrated against the game.",
                "Subpixel/timer order, corner correction, climbing and transition timing are approximations.",
                "Zip/falling/crumble platforms use approximate entity ordering, carry/lift and lifecycle timing. Attached spikes follow zip/falling/dash blocks; crumble-attached spikes remain unsupported.",
                "No scripted cutscenes, water, wind, dream blocks, keys, seeds, golden or moon berries.",
                "NPC/event/golden-collect triggers are non-solid runtime scripts; live control must yield to engine cutscenes and re-anchor afterwards. Cassette interiors require observed local phase/order or a completely observed stopped room; stopped rooms may exit to supported ordinary rooms. Cassette pickup and cross-room clock continuation remain unsupported.",
                "Unknown entities are reported, not simulated. Do not auto-play a route through an unsupported room.",
                "A predicted collectible is not an in-game confirmed collectible.",
            ],
        }

    def solid_at(self, room_name: str, x: float, y: float, *, ducking: bool = False,
                 state: SimState | None = None, dynamic_rects=None) -> bool:
        """Whether the player hitbox at a feet position intersects solid geometry."""
        room = self.map.rooms[room_name]
        h = 6 if ducking else 11
        box = (x - 4, y - h, x + 4, y)
        tx0 = math.floor((box[0] - room.x) / 8)
        tx1 = math.floor((box[2] - room.x - 1e-6) / 8)
        ty0 = math.floor((box[1] - room.y) / 8)
        ty1 = math.floor((box[3] - room.y - 1e-6) / 8)
        if self._tile_span_occupied(room_name, tx0, tx1, ty0, ty1):
            return True
        for rect in self._solids[room_name]:
            if _overlap(box, rect):
                return True
        if dynamic_rects is None:
            dynamic_rects = self._mechanism_rects(room_name, state)
        for _, _, rect in dynamic_rects:
            if _overlap(box, rect):
                return True
        return False

    def _tile_span_occupied(self, room_name, tx0, tx1, ty0, ty1):
        tiles = self._tiles[room_name]
        room = self.map.rooms[room_name]
        width, height = math.ceil(room.width/8), math.ceil(room.height/8)
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                if (tx, ty) in tiles:
                    return True
                if (not (0 <= tx < width and 0 <= ty < height)
                        and (room.x + tx*8, room.y + ty*8) in self._world_tiles):
                    return True
        return False

    def _mechanism_rects(self, room_name: str, state: SimState | None = None):
        rectangles = (self._cached_mechanism_rects(room_name,
                    state.mechanisms if state else (), state.broken if state else frozenset())
                    if self._mechanisms[room_name] else ())
        if state is not None and state.room == room_name and state.cassette_clock is not None:
            entities = self._cassette_entities[room_name]
            rectangles += tuple((block.id, entities[block.id], block.rect) for block in state.cassette_blocks
                                if block.collidable and block.id in entities)
        return rectangles

    def _cassette_segment(self, state, *, schedule=None, before=None, after=None, music_advance=None):
        clock = state.cassette_clock
        if clock is None:
            return state
        actor = CassetteActor(state.x, state.y, ducking=state.ducking, state=state.state,
                    facing=state.facing, dead=state.dead, rem_y=state.rem_y,
                    lift_vx=state.lift_vx, lift_vy=state.lift_vy, lift_timer=state.lift_timer)
        solids = self._static_rects[state.room] + tuple(rect for _, _, rect in
                    self._cached_mechanism_rects(state.room, state.mechanisms, state.broken))
        result = step_cassette(clock, state.cassette_blocks, actor, solid_rects=solids,
                    jumpthru_rects=self._platforms[state.room], dt=DT,
                    before_order=before, after_order=after, update_schedule=schedule,
                    music_advance=music_advance, validate=False)
        actor = result.actor
        return replace(state, cassette_clock=result.clock, cassette_blocks=result.blocks,
                    x=actor.x, y=actor.y, ducking=actor.ducking, dead=actor.dead,
                    rem_y=actor.rem_y, lift_vx=actor.lift_vx, lift_vy=actor.lift_vy,
                    lift_timer=actor.lift_timer, reason="cassette_squish" if actor.dead else state.reason)

    def _cached_mechanism_rects(self, room_name, mechanisms, broken):
        records = {row[0]: row for row in mechanisms}
        result = []
        for e in self._mechanisms[room_name]:
            key = f"{room_name}:{e.id}"
            if key in broken:
                continue
            record = records.get(key, (key, 0, 0, e.x, e.y, 0.0))
            _, phase, _, x, y, _ = record
            if (e.kind == "crumbleBlock" and phase == 3) or (e.kind == "fallingBlock" and phase == 4):
                continue
            height = 8 if e.kind == "crumbleBlock" else e.height
            x, y = round(x), round(y)
            result.append((key, e, (x, y, x + e.width, y + height)))
        return tuple(result)

    def _jumpthru_at(self, room_name: str, x: float, old_y: float, new_y: float) -> bool:
        return any(x + 4 > l and x - 4 < r and old_y <= t and new_y > t
                   for l, t, r, _ in self._platforms[room_name])

    def on_ground(self, s: SimState) -> bool:
        if s.vy < 0:
            return False
        return self.solid_at(s.room, s.x, s.y + 1, ducking=s.ducking, state=s) or self._jumpthru_at(s.room, s.x, s.y, s.y + 1)

    def _move_without_callback(self, s: SimState, amount: float, *, horizontal: bool = False,
                               exact: bool = False, dynamic_rects=None) -> SimState:
        """Actor.Move[HV] without velocity callbacks, optionally excluding a pusher."""
        remainder_name = "rem_x" if horizontal else "rem_y"
        remainder = getattr(s, remainder_name)
        total = amount if exact else _f32(_f32(amount) + remainder)
        pixels = int(amount) if exact else round(total)
        if not exact:
            remainder = _f32(total - pixels)
        direction = _sign(pixels)
        x, y = s.x, s.y
        for _ in range(abs(pixels)):
            nx, ny = (x + direction, y) if horizontal else (x, y + direction)
            if (self.solid_at(s.room, nx, ny, ducking=s.ducking, state=s,
                              dynamic_rects=dynamic_rects)
                    or (not horizontal and direction > 0 and self._jumpthru_at(s.room, x, y, ny))):
                remainder = 0.0
                break
            x, y = nx, ny
        return replace(s, x=x, y=y, **{remainder_name: remainder})

    def _wall(self, room: str, x: float, y: float, direction: int, distance: int = 3, *, dynamic_rects=None) -> bool:
        bounds = self.map.rooms[room]
        if x + direction * distance < bounds.x or x + direction * distance >= bounds.x + bounds.width:
            return False
        return self.solid_at(room, x + direction * distance, y, dynamic_rects=dynamic_rects)

    def _tiles_overlap(self, room_name, rect):
        room = self.map.rooms[room_name]
        if self._tile_span_occupied(room_name,
                math.floor((rect[0] - room.x) / 8), math.floor((rect[2] - room.x - 1e-6) / 8),
                math.floor((rect[1] - room.y) / 8), math.floor((rect[3] - room.y - 1e-6) / 8)):
            return True
        for other in self._solids[room_name]:
            if _overlap(rect, other):
                return True
        return False

    def _ledge_blocked(self, s, x, y, *, ducking=False):
        """Native LedgeBlocker overlap at a proposed player-collider position.

        Up/left/right spikes and crystal spinners block automatic ledge hops
        independently of whether their directional damage check would kill the
        player. Down spikes have no LedgeBlocker. Attached spike positions follow
        their solid, including permanent dash-block removal.
        """
        box = (x-4, y-(6 if ducking else 11), x+4, y)
        records = {row[0]: row for row in s.mechanisms}
        for entity in self._hazards[s.room]:
            if entity.kind == "spikesDown":
                continue
            hx, hy = entity.x, entity.y
            mover = self._spike_attachments[s.room].get((entity.kind, entity.id))
            if mover is not None:
                key = f"{s.room}:{mover.id}"
                if key in s.broken:
                    continue
                record = records.get(key)
                if record is not None:
                    _, phase, _, mover_x, mover_y, _ = record
                    if mover.kind == "fallingBlock" and phase == 4:
                        continue
                    hx += round(mover_x)-mover.x
                    hy += round(mover_y)-mover.y
            if entity.kind.startswith("spikes"):
                if _overlap(box, _spike_rect(entity,hx,hy)):
                    return True
            elif entity.kind == "spinner":
                # CrystalStaticSpinner's ColliderList is Circle(6) union a
                # 16x4 hitbox at (-8,-3), and uses the full player hitbox here.
                cx, cy = max(box[0],min(hx,box[2])), max(box[1],min(hy,box[3]))
                if ((cx-hx)**2+(cy-hy)**2 < 36
                        or _overlap(box,(hx-8,hy-3,hx+8,hy+1))):
                    return True
        return False

    def _advance_mechanisms(self, s: SimState, *, decay_lift: bool = True) -> SimState:
        """Advance only the current room's entities, then carry/push the player.

        Coroutine phases and easing follow source constants. Exact engine entity
        update ordering and squish wiggle are deliberately not claimed.
        """
        entities = self._mechanisms[s.room]
        if not entities:
            if s.lift_timer and decay_lift:
                timer=max(0,s.lift_timer-1)
                return replace(s,lift_timer=timer,lift_vx=s.lift_vx if timer else 0.0,
                               lift_vy=s.lift_vy if timer else 0.0)
            return s
        records = {row[0]: row for row in s.mechanisms}
        x, y = s.x, s.y
        rem_x, rem_y = s.rem_x, s.rem_y
        lift_timer = max(0, s.lift_timer - int(decay_lift))
        lift_x, lift_y = (s.lift_vx,s.lift_vy) if lift_timer else (0.0,0.0)
        room = self.map.rooms[s.room]
        for e in entities:
            key = f"{s.room}:{e.id}"
            if e.kind == "dashBlock" or key in s.broken:
                continue
            _, phase, timer, ex, ey, speed = records.get(key, (key, 0, 0, e.x, e.y, 0.0))
            height = 8 if e.kind == "crumbleBlock" else e.height
            old_x, old_y = round(ex), round(ey)
            old_rect = (old_x, old_y, old_x + e.width, old_y + height)
            on_top = s.vy >= 0 and abs(y - old_y) < 0.1 and x + 4 > old_x and x - 4 < old_x + e.width
            climbing = s.state == 1 and _overlap(
                (x-4+s.facing,y-(6 if s.ducking else 11),x+4+s.facing,y),old_rect)
            rider = on_top or climbing
            nx, ny = ex, ey
            falling_move_requested = False
            falling_lift_y = 0.0
            if e.kind == "crumbleBlock":
                if phase == 0 and rider:
                    phase, timer = (1, 0) if on_top else (2, 0)
                elif phase in (1, 2):
                    timer += 1
                    if (phase == 1 and timer >= 12 and (not on_top or timer >= 36)) or (phase == 2 and timer >= 60):
                        phase, timer = 3, 120
                elif phase == 3:
                    timer -= 1
                    player_box = (x - 4, y - 11, x + 4, y)
                    if timer <= 0 and not _overlap(player_box, old_rect):
                        phase, timer = 0, 0
            elif e.kind == "zipMover" and e.nodes:
                target_x, target_y = e.nodes[0]
                if phase == 0 and rider:
                    # Coroutine's positive-wait tick returns; the next resume
                    # hits the movement loop's yield-null before its first move.
                    phase, timer = 1, 7
                elif phase == 1:
                    timer -= 1
                    if timer <= 0:
                        phase, timer = 2, 0
                elif phase in (2, 4):
                    timer += 1
                    duration = 30 if phase == 2 else 120
                    p = _zip_ease(timer,duration)
                    ax,ay,bx,by = ((e.x,e.y,target_x,target_y) if phase == 2
                                   else (target_x,target_y,e.x,e.y))
                    # XNA Vector2.Lerp stores the component after multiply
                    # plus add, not an extra rounded multiplication temporary.
                    nx = _f32(ax+(bx-ax)*p)
                    ny = _f32(ay+(by-ay)*p)
                    if timer >= duration:
                        phase, timer = (3, 31) if phase == 2 else (5, 30)
                elif phase in (3, 5):
                    timer -= 1
                    if timer <= 0:
                        phase, timer = (4, 0) if phase == 3 else (0, 0)
            elif e.kind == "fallingBlock":
                climb_fall = _truth(e.attrs.get("climbFall", True))
                player_box = (x-4, y-(6 if s.ducking else 11), x+4, y)
                # PlayerWaitCheck deliberately includes ordinary side contact.
                # Releasing Grab or jumping while touching the side must not
                # skip the remaining .4-second wait after the initial shake.
                side_contact = (_overlap(player_box,(old_x-1,old_y,old_x+e.width-1,old_y+height))
                                or _overlap(player_box,(old_x+1,old_y,old_x+e.width+1,old_y+height)))
                top_contact = _overlap(player_box,(old_x,old_y-1,old_x+e.width,old_y+height-1))
                should_trigger = rider if climb_fall else top_contact
                should_wait = should_trigger or climb_fall and side_contact

                def tile_support(rect):
                    return self._tile_span_occupied(s.room,
                        math.floor((rect[0]-room.x)/8),math.floor((rect[2]-room.x-1e-6)/8),
                        math.floor((rect[1]-room.y)/8),math.floor((rect[3]-room.y-1e-6)/8))

                def other_solids():
                    current = replace(s,mechanisms=tuple(sorted(records.values())))
                    return tuple(rect for identity,_,rect in self._mechanism_rects(s.room,current)
                                 if identity!=key)

                def platform_below():
                    probe=(old_x,old_y+1,old_x+e.width,old_y+height+1)
                    return (self._tiles_overlap(s.room,probe)
                            or any(_overlap(probe,rect) for rect in other_solids()+self._platforms[s.room]))

                if phase == 0 and should_trigger:
                    phase, timer = 1, 0
                elif phase == 7:  # Native FallDelay loop yields null after each decrement.
                    if timer>0:
                        timer-=1
                    else:
                        phase,timer=1,0
                elif phase == 1:
                    timer += 1
                    # Trigger update yields .2; twelve positive wait updates
                    # return without resuming. The next resume either starts
                    # falling now, or yields the first .4-second contact wait.
                    if timer>=13 and (not should_wait or timer>=37):
                        phase,timer,speed=2,0,0.0
                elif phase in (5,6):
                    if timer>0:
                        timer-=1
                    else:
                        below=(old_x,old_y+1,old_x+e.width,old_y+height+1)
                        if phase==5 and tile_support(below):
                            phase,timer=3,0
                        elif platform_below():
                            phase,timer=6,6  # yield .1; resume after six positive waits.
                        else:
                            phase,timer=1,0  # Supporting platform left: start a new shake.
                if phase in (3,4):
                    speed = 0.0  # No coroutine remains to carry a future velocity.
                if phase == 2:
                    speed = _approach(speed,160.0,500.0*DT)
                    # Platform.MoveVCollideSolids accumulates a Single counter,
                    # subtracts the rounded request before collision, and keeps
                    # that remainder even if only part of the move is possible.
                    counter=_f32(_f32(ey-old_y)+_f32(speed*DT))
                    pixels=round(counter)
                    falling_move_requested = pixels != 0
                    falling_lift_y = _f32(_f32(speed*DT)/DT)
                    remainder=_f32(counter-pixels)
                    actual_y=old_y
                    for _ in range(max(0,pixels)):
                        before=(old_x,actual_y,old_x+e.width,actual_y+height)
                        probe=(old_x,actual_y+1,old_x+e.width,actual_y+height+1)
                        current=replace(s,mechanisms=tuple(sorted(records.values())))
                        broken=set(s.broken)
                        for identity,entity,rect in self._mechanism_rects(s.room,current):
                            if entity.kind=="dashBlock" and _overlap(probe,rect):
                                broken.add(identity)  # Native thruDashBlocks=true.
                        if broken!=set(s.broken):
                            s=replace(s,broken=frozenset(broken))
                        collision=(self._tiles_overlap(s.room,probe)
                            or any(_overlap(probe,rect) for rect in other_solids())
                            or any(_overlap(probe,rect) and not _overlap(before,rect)
                                   for rect in self._platforms[s.room]))
                        if collision:
                            phase,timer=5,12  # Land yield .2, then test permanent tile support.
                            break
                        actual_y+=1
                    ny=float(actual_y)+remainder
                    below=(old_x,actual_y+1,old_x+e.width,actual_y+height+1)
                    outside_bottom=(actual_y>room.y+room.height-1 and
                        (self._tiles_overlap(s.room,below)
                         or any(_overlap(below,rect) for rect in other_solids())))
                    if phase==2 and (actual_y>room.y+room.height+16 or outside_bottom):
                        phase,timer,speed=4,0,0.0
            records[key] = (key, phase, timer, float(nx), float(ny), speed)
            dx, dy = round(nx) - round(ex), round(ny) - round(ey)
            if dx or dy or falling_move_requested:
                body_height = 6 if s.ducking else 11
                mover_lift_x = _f32(_f32(nx-ex)/DT)
                mover_lift_y = _f32(_f32(ny-ey)/DT)
                if e.kind == "fallingBlock":
                    mover_lift_x, mover_lift_y = 0.0, falling_lift_y
                # Platform.MoveTo performs H before V. Each Solid move takes
                # fresh riders and rechecks overlap after the preceding axis.
                # A horizontal push may entirely clear a diagonal mover; in
                # that case its vertical move must not also push the player.
                for horizontal, amount in ((True, dx), (False, dy)):
                    if not amount and not (not horizontal and falling_move_requested):
                        continue
                    left = old_x if horizontal else round(nx)
                    before_rect = (left, old_y, left + e.width, old_y + height)
                    if s.state == 1:
                        riding_axis = _overlap((x-4+s.facing, y-body_height,
                                                x+4+s.facing, y), before_rect)
                    else:
                        riding_axis = _overlap((x-4, y-body_height+1, x+4, y+1), before_rect)
                    after_rect = ((round(nx), old_y, round(nx)+e.width, old_y+height)
                                  if horizontal else
                                  (round(nx), round(ny), round(nx)+e.width, round(ny)+height))
                    pushed = _overlap((x-4, y-body_height, x+4, y), after_rect)
                    if pushed:
                        if horizontal:
                            x = after_rect[2]+4 if amount > 0 else after_rect[0]-4
                        else:
                            y = after_rect[3]+body_height if amount > 0 else after_rect[1]
                    elif riding_axis:
                        # Native Solid carries a rider with Actor.Move*Exact
                        # and no SquishCallback. Disable only this mover's
                        # collision, clip against other solids/jumpthrus, and
                        # clear only the blocked axis's movement remainder.
                        # Direct overlap above remains a pushed/squish case.
                        rider_state = replace(s, x=x, y=y, rem_x=rem_x, rem_y=rem_y,
                            mechanisms=tuple(sorted(records.values())))
                        obstacles = tuple(row for row in
                            self._mechanism_rects(s.room, rider_state) if row[0] != key)
                        carried = self._move_without_callback(rider_state, amount,
                            horizontal=horizontal, exact=True, dynamic_rects=obstacles)
                        x, y = carried.x, carried.y
                        rem_x, rem_y = carried.rem_x, carried.rem_y
                    else:
                        continue
                    # Platform.Update clears LiftSpeed. During H, MoveV has
                    # not assigned its Y component yet; Actor receives that
                    # actual intermediate vector even if V later moves alone.
                    lift_x, lift_y, lift_timer = mover_lift_x, 0.0 if horizontal else mover_lift_y, 10
                    box = (x-4, y-body_height, x+4, y)
                    if (self._tiles_overlap(s.room, box) or
                            any(block.collidable and _overlap(box, block.rect)
                                for block in s.cassette_blocks)):
                        return replace(s, x=x, y=y, rem_x=rem_x, rem_y=rem_y,
                                       dead=True, reason="squish", mechanisms=tuple(sorted(records.values())))
        return replace(s, x=x, y=y, rem_x=rem_x, rem_y=rem_y,
                       mechanisms=tuple(sorted(records.values())),
                       lift_vx=lift_x, lift_vy=lift_y, lift_timer=lift_timer)

    def _safe_ground(self, s: SimState) -> bool:
        if s.vy < 0:
            return False
        box = (s.x - 4, s.y - 10, s.x + 4, s.y + 1)
        if self._tiles_overlap(s.room, box) or self._jumpthru_at(s.room, s.x, s.y, s.y + 1):
            return True
        records = {row[0]: row for row in s.mechanisms}
        for key, e, rect in self._mechanism_rects(s.room, s):
            if _overlap(box, rect) and (e.kind == "dashBlock" or
                    (e.kind == "fallingBlock" and records.get(key, (0, 0))[1] == 3)):
                return True
        return False

    def roll(self, state: SimState, mask: int, frames: int) -> SimState:
        if frames < 0:
            raise ValueError("frames must be non-negative")
        for _ in range(frames):
            state = self.step(state, mask)
            if state.dead:
                break
        return state

    def trace(self, state: SimState, inputs: Iterable[int]) -> list[SimState]:
        result = [state]
        for mask in inputs:
            state = self.step(state, mask)
            result.append(state)
            if state.dead:
                break
        return result

    def step(self, s: SimState, mask: int) -> SimState:
        # Transient event for this call only; read immediately after stepping.
        # Charge refills or springs can hide a dash in the returned state.
        self.last_step_started_dash = False
        if mask < 0 or mask > 127:
            raise ValueError("input mask must be in 0..127")
        if s.dead:
            return s
        cassette_schedule = None
        cassette_player_order = None
        if s.cassette_clock is not None:
            # Scene.BeforeUpdate sorts once; depth changes during this tick
            # affect the next tick only. Keep this schedule across both halves.
            entries = [(s.cassette_clock.actual_depth, ""), (s.player_actual_depth, "@player")]
            entries += [(block.actual_depth, block.id) for block in s.cassette_blocks]
            ordered = sorted(entries, key=lambda item: -item[0])
            cassette_player_order = next(index for index, (_, identity) in enumerate(ordered) if identity == "@player")
            cassette_schedule = tuple((index, identity) for index, (_, identity) in enumerate(ordered) if identity != "@player")
            s = self._cassette_segment(s, schedule=cassette_schedule, before=cassette_player_order)
            if s.dead:
                return replace(s, frame=s.frame + 1)
        dynamic_rects = self._mechanism_rects(s.room, s)
        def solid(room_name, x, y, *, ducking=False):
            return self.solid_at(room_name, x, y, ducking=ducking, dynamic_rects=dynamic_rects)
        def wall(room_name, x, y, direction, distance=3):
            return self._wall(room_name, x, y, direction, distance, dynamic_rects=dynamic_rects)
        def hop_attachment(x, y, facing):
            box = (x + facing - 4, y - 11, x + facing + 4, y)
            for identity, entity, rect in dynamic_rects:
                if entity.kind in ("zipMover", "fallingBlock", "cassetteBlock") and _overlap(box, rect):
                    return identity, float(rect[0]), float(rect[1])
            return "", 0.0, 0.0
        # Work on a fresh dict, retaining an immutable public state for branching.
        d = vars(s).copy()
        d["frame"] += 1
        d["prev_input"] = mask
        d["reason"] = ""
        room = self.map.rooms[s.room]
        x, y, vx, vy = s.x, s.y, s.vx, s.vy
        mx = int(bool(mask & RIGHT)) - int(bool(mask & LEFT))
        my = int(bool(mask & DOWN)) - int(bool(mask & UP))
        jump_held = bool(mask & JUMP)
        d["jump_buffer"] = 5 if mask & JUMP and not s.prev_input & JUMP else max(0, s.jump_buffer - 1)
        d["dash_buffer"] = 5 if mask & DASH and not s.prev_input & DASH else max(0, s.dash_buffer - 1)
        for key in ("dash_attack", "dash_cooldown", "refill_cooldown", "var_jump"):
            d[key] = max(0, d[key] - 1)
        if s.state == 1:
            d["climb_wait"] = max(0, s.climb_wait - 1)
        ground = self.on_ground(s)
        d["on_ground"] = ground
        d["grace"] = 6 if ground else max(0, s.grace - 1)
        if ground:
            if s.state != 1:
                d["auto_jump_active"] = False
            d["stamina"] = 110.0
            d["wall_slide"] = 72
            if not s.refill_cooldown:
                d["dashes"] = self.max_dashes
        # Source Player.Update only counts down a *positive* timer. A zero
        # timer never expires an active flag. Ground/manual jump/climb can.
        if s.auto_jump > 0:
            if d["auto_jump_active"]:
                d["auto_jump"] = max(0, s.auto_jump - 1)
                if d["auto_jump"] == 0:
                    d["auto_jump_active"] = False
            else:
                d["auto_jump"] = 0
        if s.force_frames:
            mx = s.force_x
            d["force_frames"] -= 1
            if s.climb_hop_solid:
                attached = next((rect for identity, _, rect in dynamic_rects
                                 if identity == s.climb_hop_solid), None)
                if attached is None:
                    d.update(climb_hop_solid="", climb_hop_x=0.0, climb_hop_y=0.0)
                else:
                    dx, dy = attached[0] - s.climb_hop_x, attached[1] - s.climb_hop_y
                    d.update(climb_hop_x=float(attached[0]), climb_hop_y=float(attached[1]))
                    carried = SimState(**(d | {"x": x, "y": y}))
                    if dx:
                        carried = self._move_without_callback(carried, dx, horizontal=True, exact=True)
                    if dy:
                        carried = self._move_without_callback(carried, dy, exact=True)
                    x, y, d["rem_x"], d["rem_y"] = carried.x, carried.y, carried.rem_x, carried.rem_y
        else:
            d.update(climb_hop_solid="", climb_hop_x=0.0, climb_hop_y=0.0)
        if mx and s.state != 1:
            d["facing"] = mx
        facing = d["facing"]
        # Player.Update restores speed retained by OnCollideH once the wall is
        # cleared (for example by jumping above its lip), before NormalUpdate.
        if s.wall_retention:
            if _sign(vx) == -_sign(s.wall_retained):
                d["wall_retention"] = 0
            elif not solid(s.room, x + _sign(s.wall_retained), y, ducking=s.ducking):
                vx, d["wall_retention"] = s.wall_retained, 0
            else:
                d["wall_retention"] -= 1
        if s.hop_wait_x:
            if _sign(vx) == -s.hop_wait_x or vy > 0:
                d["hop_wait_x"] = 0
            elif not solid(s.room, x + s.hop_wait_x, y, ducking=s.ducking):
                vx, d["hop_wait_x"] = s.hop_wait_x_speed, 0
        lift_x = max(-250.0, min(250.0, s.lift_vx)) if s.lift_timer else 0.0
        lift_y = max(-130.0, min(0.0, s.lift_vy)) if s.lift_timer else 0.0
        if s.on_ground and not ground and vy >= 0 and lift_y < 0:
            vy = lift_y
        # Un-duck on released Down belongs to NormalUpdate. DashUpdate must
        # still see a crouched horizontal dash when Jump starts a hyperjump;
        # clearing it here for StDash incorrectly produces a normal superjump.
        if s.state == 0 and d["ducking"]:
            if ground and my != 1 and not solid(s.room, x, y):
                d["ducking"] = False
        elif ground and my == 1 and s.state == 0:
            d["ducking"] = True

        # NormalUpdate returns StClimb BEFORE acceleration, gravity, jump and
        # dash checks. ClimbBegin runs this tick; ClimbUpdate runs next tick.
        entered_climb = False
        last_climb_move = 0  # ClimbBegin resets this; only actual upward climbing boosts.
        if (s.state == 0 and mask & GRAB and d["stamina"] >= 20 and not d["ducking"]
                and vy >= 0 and _sign(vx) != -facing):
            climb_y = None
            if wall(s.room, x, y, facing, 2):
                climb_y = y
            elif my < 1:
                for shift in (1, 2):
                    if not solid(s.room, x, y - shift) and wall(s.room, x, y - shift, facing, 2):
                        climb_y = y - shift
                        break
            if climb_y is not None:
                y = climb_y
                for _ in range(2):
                    if solid(s.room, x + facing, y):
                        break
                    x += facing
                vx, vy = 0.0, vy * 0.2
                d.update(state=1, climb_wait=6, ducking=False, auto_jump_active=False,
                         wall_slide=72, wall_retention=0)
                entered_climb = True

        # DashCoroutine first yields null, then sets Speed and yields .15s.
        # Coroutine.Update returns on *every* positive waitTimer tick, including
        # the tick that crosses zero. At 60Hz that is ten full-speed movements,
        # followed by the end-speed movement on the twelfth total update.
        started_dash = False
        if (not entered_climb and d["dash_buffer"] and d["dashes"] > 0 and not d["dash_cooldown"]
                and not (s.state == 1 and d["jump_buffer"])):
            d.update(state=2, dash_frames=11, dash_attack=18, dash_buffer=0, dashes=d["dashes"] - 1,
                     dash_cooldown=12, refill_cooldown=6, dash_before_x=vx + lift_x,
                     dash_started_ground=ground, dash_dx=0.0, dash_dy=0.0, wall_slide=72)
            vx = vy = 0.0
            # DashBegin can crouch even an airborne downward dash for its
            # initial frozen frame. Falling motion un-ducks later in Update.
            if not ground and d["ducking"] and not solid(s.room,x,y):
                d["ducking"] = False
            elif not d["ducking"] and my == 1:
                d["ducking"] = True
            started_dash = True
            self.last_step_started_dash = True
            if d["cassette_clock"] is not None:
                # Celeste.Freeze(0.05f) advances music immediately, without
                # tempo multiplication; frozen engine frames send no inputs.
                frozen = self._cassette_segment(SimState(**(d | {"x": x, "y": y, "vx": vx, "vy": vy})),
                                                music_advance=_f32(.05))
                d.update(cassette_clock=frozen.cassette_clock, cassette_blocks=frozen.cassette_blocks,
                         rem_y=frozen.rem_y, lift_vx=frozen.lift_vx, lift_vy=frozen.lift_vy,
                         lift_timer=frozen.lift_timer, dead=frozen.dead)
                x, y = frozen.x, frozen.y
                dynamic_rects = self._mechanism_rects(s.room, frozen)
                if frozen.dead:
                    return frozen

        jump_lift_applied = False
        if d["state"] == 2 and not started_dash:
            # StateMachine runs DashUpdate before advancing its coroutine.
            # Thus the first opportunity for a super jump still sees DashDir=0.
            if d["jump_buffer"] and d["dash_dy"] == 0 and d["grace"]:
                # SuperJump adds LiftBoost before applying the crouched jump
                # multipliers. Adding it later gives a rising-platform hyper
                # twice the native vertical boost (and too little horizontal).
                vx, vy = _f32(260.0 * facing + lift_x), _f32(JUMP_SPEED + lift_y)
                jump_lift_applied = True
                if d["ducking"]:
                    vx, vy = _f32(vx * 1.25), _f32(vy * 0.5)
                d.update(state=0, dash_frames=0, dash_attack=0, ducking=False, grace=0,
                         jump_buffer=0, var_jump=12, var_speed=vy, auto_jump_active=False)
            elif d["jump_buffer"] and (wall(s.room, x, y, 1) or wall(s.room, x, y, -1)):
                direction = -1 if wall(s.room, x, y, 1) else 1
                super_wall = d["dash_dx"] == 0 and d["dash_dy"] == -1
                vx, vy = direction * (170.0 if super_wall else 130.0), -160.0 if super_wall else JUMP_SPEED
                d.update(state=0, dash_frames=0, dash_attack=0, grace=0, jump_buffer=0,
                         var_jump=15 if super_wall else 12, var_speed=vy,
                         force_x=direction, force_frames=0 if super_wall else (10 if mx else 0),
                         auto_jump_active=False)
            else:
                if s.dash_frames == 11:
                    # lastAim is raw input, independent of forceMoveX.
                    dx, dy = int(bool(mask & RIGHT)) - int(bool(mask & LEFT)), my
                    if dx == dy == 0:
                        dx = facing
                    length = math.hypot(dx, dy)
                    dx, dy = _f32(dx / length), _f32(dy / length)
                    d["dash_dx"], d["dash_dy"] = dx, dy
                    vx, vy = _f32(dx * DASH_SPEED), _f32(dy * DASH_SPEED)
                    if _sign(s.dash_before_x) == _sign(vx) and abs(s.dash_before_x) > abs(vx):
                        vx = s.dash_before_x
                    if ground and dx and dy > 0:
                        d["dash_dx"], d["dash_dy"] = _sign(dx), 0.0
                        vy, vx = 0.0, _f32(vx * _f32(1.2))
                        d["ducking"] = True
                    if dx:
                        d["facing"] = _sign(dx)
                d["dash_frames"] -= 1
                if d["dash_frames"] <= 0:
                    if d["dash_dy"] <= 0:
                        vx, vy = _f32(d["dash_dx"] * 160.0), _f32(d["dash_dy"] * 160.0)
                    if vy < 0:
                        vy = _f32(vy * 0.75)
                    d.update(state=0, auto_jump_active=True, auto_jump=0)
            # DashEnd also fires listeners if a jump cancels before direction.
            if s.dash_frames == 11:
                d["fled"] = s.fled | frozenset(
                    f"{s.room}:{e.id}" for e in self._interact[s.room]
                    if e.kind == "strawberry" and _truth(e.attrs.get("winged"))
                    and not e.nodes and f"{s.room}:{e.id}" not in s.following | s.berries)
        elif d["state"] == 1 and not started_dash and not entered_climb:
            if d["jump_buffer"]:
                if mx == -facing:
                    vx, vy = -facing * 130.0, JUMP_SPEED
                    d.update(force_x=-facing, force_frames=10)
                else:
                    vx, vy = vx + 40 * mx, JUMP_SPEED
                    if not ground:
                        d["stamina"] -= 27.5
                d.update(state=0, jump_buffer=0, grace=0, var_jump=12, var_speed=vy,
                         dash_attack=0, wall_slide=72, auto_jump_active=False)
            elif not mask & GRAB or d["stamina"] <= 0:
                # ClimbUpdate's let-go / tired exits retain the moving wall's
                # capped LiftBoost, even without a jump or dash input.
                vx, vy = _f32(vx + lift_x), _f32(vy + lift_y)
                d["state"] = 0
            elif not wall(s.room, x, y, facing, 1):
                if vy < 0:
                    vx, vy = facing * 100.0, -120.0
                    d.update(force_x=0, force_frames=12, hop_wait_x=0)
                    identity, hx, hy = hop_attachment(x, y, facing)
                    d.update(climb_hop_solid=identity, climb_hop_x=hx, climb_hop_y=hy)
                d["state"] = 0
            else:
                vx = 0.0
                target = 0.0 if d["climb_wait"] else (-45.0 if my < 0 else 80.0 if my > 0 else 0.0)
                # SlipCheck samples the hand and top corner, not the entire
                # player's body. A ledge below the hands slides at 30px/s even
                # during the initial no-move interval.
                point_x = x + 4 if facing > 0 else x - 5
                def point_solid(px, py):
                    tile=(math.floor((px-room.x)/8),math.floor((py-room.y)/8))
                    # A native point query is not an epsilon rectangle: adding
                    # then subtracting 1e-6 at x2288 can round below the boundary
                    # and produce an empty tile range, inventing a climb lip.
                    return (self._tile_span_occupied(s.room,tile[0],tile[0],tile[1],tile[1])
                            or any(l<=px<r and t<=py<b for l,t,r,b in self._solids[s.room])
                            or any(l<=px<r and t<=py<b for _,_,(l,t,r,b) in dynamic_rects))
                def slip_check(add_y=0):
                    return (not point_solid(point_x, y - 7 + add_y)
                            and not point_solid(point_x, y - 11 + 2 * add_y))
                slipping = slip_check()
                hopped = False
                try_slip = bool(d["climb_wait"] or my == 0)
                if not d["climb_wait"] and my < 0:
                    hop_blocked = (solid(s.room, x, y - 6)
                                   or self._ledge_blocked(s,x+facing*8,y,ducking=d["ducking"]))
                    if solid(s.room, x, y - 1) or (hop_blocked and slip_check(-1)):
                        vy = max(0.0, vy)
                        target, try_slip = 0.0, True
                    elif slipping:
                        # ClimbHop starts as soon as both hand points clear the
                        # lip, while the lower body may still touch the wall.
                        # Horizontal motion waits until the body clears it.
                        touching = solid(s.room, x + facing, y)
                        d.update(state=0, hop_wait_x=facing if touching else 0,
                                 hop_wait_x_speed=facing * 100.0, force_x=0, force_frames=12)
                        identity, hx, hy = hop_attachment(x, y, facing)
                        d.update(climb_hop_solid=identity, climb_hop_x=hx, climb_hop_y=hy)
                        if not touching:
                            vx = facing * 100.0
                        vy = min(vy, -120.0)
                        hopped = True
                if ground and target > 0:
                    target = 0.0
                last_climb_move = _sign(target)
                if try_slip and slipping:
                    target = 30.0
                if ground and target > 0:
                    target = 0.0
                if not hopped:
                    vy = _approach(vy, target, 900.0 * DT)
                    if my != 1 and vy > 0 and not solid(s.room, x + facing, y + 1):
                        vy = 0.0
                    if not d["climb_wait"]:
                        drain = _f32(45.454544) if last_climb_move < 0 else 10.0 if last_climb_move == 0 else 0.0
                        d["stamina"] = _f32(d["stamina"] - _f32(drain*DT))
                        if d["stamina"] <= 0:
                            vx, vy = _f32(vx + lift_x), _f32(vy + lift_y)
                            d["state"] = 0
        elif not started_dash and not entered_climb:
            # Ground / air acceleration is asymmetric when retaining dash speed.
            if d["ducking"] and ground:
                vx = _approach(vx, 0.0, 500.0 * DT)
            else:
                rate = RUN_REDUCE if abs(vx) > MAX_RUN and _sign(vx) == mx else RUN_ACCEL
                vx = _approach(vx, MAX_RUN * mx, rate * (1.0 if ground else AIR_MULT) * DT)
            if not ground:
                d["max_fall"] = _approach(s.max_fall, 240.0 if my > 0 and vy >= MAX_FALL else MAX_FALL, 300.0 * DT)
                max_fall = d["max_fall"]
                if (mx == facing or (not mx and mask & GRAB)) and my != 1 and vy >= 0 and s.wall_slide > 0 and wall(s.room, x, y, facing, 1):
                    max_fall = MAX_FALL + (20.0 - MAX_FALL) * (s.wall_slide / 72)
                    d["wall_slide"] -= 1
                half_gravity = abs(vy) < 40 and (jump_held or d["auto_jump_active"])
                vy = _approach(vy, max_fall, GRAVITY * DT * (0.5 if half_gravity else 1.0))
            if d["var_jump"]:
                if jump_held or d["auto_jump_active"]:
                    vy = min(vy, d["var_speed"])
                else:
                    d["var_jump"] = 0
            if d["jump_buffer"]:
                if d["grace"]:
                    vx += 40.0 * mx
                    vy = JUMP_SPEED
                    d.update(grace=0, jump_buffer=0, var_jump=12, var_speed=vy,
                             dash_attack=0, wall_slide=72, auto_jump_active=False)
                else:
                    wall_dir = 1 if wall(s.room, x, y, 1) else -1 if wall(s.room, x, y, -1) else 0
                    if wall_dir:
                        if facing == wall_dir and mask & GRAB and d["stamina"] > 0:
                            vx += 40.0 * mx
                            vy = JUMP_SPEED
                            d["stamina"] -= 27.5
                        else:
                            super_wall = d["dash_attack"] > 0 and d["dash_dx"] == 0 and d["dash_dy"] == -1
                            vx, vy = -wall_dir * (170.0 if super_wall else 130.0), -160.0 if super_wall else JUMP_SPEED
                            d.update(force_x=-wall_dir, force_frames=0 if super_wall else (10 if mx else 0))
                        d.update(grace=0, jump_buffer=0, var_jump=15 if vy == -160.0 else 12,
                                 var_speed=vy, ducking=False, wall_slide=72, dash_attack=0,
                                 auto_jump_active=False)

        if s.state == 0 and d["state"] != 0:
            # Native NormalEnd clears the old wall speed and pending hop
            # before entering another state. Otherwise a dash that clears a
            # wall lip can incorrectly restore the pre-dash running speed.
            d.update(wall_retention=0, hop_wait_x=0)

        if s.state == 1 and d["state"] != 1:
            # ClimbEnd clears retention; a subsequent wall collision this same
            # tick may retain the new climb-jump speed.
            d["wall_retention"] = 0

        if not jump_lift_applied and s.lift_timer and d["var_jump"] >= 9 and d["jump_buffer"] == 0 and (
                s.jump_buffer or (mask & JUMP and not s.prev_input & JUMP)) and not started_dash:
            vx, vy = vx + lift_x, vy + lift_y
            d["var_speed"] = vy
        # These position assists occur after StateMachine.Update but before the
        # main Actor movement. Jump-through platforms speed an upward crossing
        # by 40px/s; a horizontal dash snaps down at most three pixels.
        player_box = (x - 4, y - (6 if d["ducking"] else 11), x + 4, y)
        if (not ground and vy <= 0 and (d["state"] != 1 or last_climb_move == -1)
                and any(_overlap(player_box, rect) for rect in self._platforms[s.room])
                and not self._ledge_blocked(s,x,y-2,ducking=d["ducking"])):
            assisted = self._move_without_callback(SimState(**(d | {"x": x, "y": y})), -40 * DT)
            x, y, d["rem_y"] = assisted.x, assisted.y, assisted.rem_y
        if not ground and d["dash_attack"] and d["dash_dy"] == 0:
            if solid(s.room, x, y + 3, ducking=d["ducking"]) or self._jumpthru_at(s.room, x, y, y + 3):
                snapped = self._move_without_callback(SimState(**(d | {"x": x, "y": y})), 3, exact=True)
                x, y, d["rem_y"] = snapped.x, snapped.y, snapped.rem_y
        if vy > 0 and d["ducking"] and not ground and not d["grace"] and not solid(s.room,x,y):
            d["ducking"] = False
        def dashed_block(px, py):
            nonlocal dynamic_rects
            if not d["dash_attack"]:
                return False
            if (px != x and _sign(px - x) != _sign(d["dash_dx"])) or (
                    py != y and _sign(py - y) != _sign(d["dash_dy"])):
                return False
            box = (px - 4, py - (6 if d["ducking"] else 11), px + 4, py)
            for key, entity, rect in tuple(dynamic_rects):
                if entity.kind == "dashBlock" and _truth(entity.attrs.get("canDash", True)) and _overlap(box, rect):
                    d["broken"] = d["broken"] | frozenset({key})
                    # Detach this branch's geometry without modifying the
                    # immutable rectangle tuple shared by other search states.
                    dynamic_rects = tuple(row for row in dynamic_rects if row[0] != key)
                    return True
            return False
        # Actor.MoveH/MoveV retain fractional remainders and move pixel by pixel.
        vx, vy = _f32(vx), _f32(vy)
        rx = _f32(s.rem_x + _f32(vx * DT))
        dx = round(rx)
        rx = _f32(rx - dx)
        sx = _sign(dx)
        fresh_rebound = False
        for _ in range(abs(dx)):
            if solid(s.room, x + sx, y, ducking=d["ducking"]):
                # Actor clears the remainder before invoking Player.OnCollideH,
                # even when the callback successfully performs a corner nudge.
                rx = 0.0
                if dashed_block(x + sx, y):
                    vx, vy, rx = -sx * 120.0, -120.0, 0.0
                    d.update(state=0, dash_frames=0, dash_attack=0, var_jump=9, var_speed=vy,
                             auto_jump_active=True, auto_jump=0, force_frames=0, wall_slide=72)
                    fresh_rebound = True
                    break
                corrected = False
                if d["state"] == 2 and ground and not solid(s.room, x + sx, y, ducking=True):
                    d["ducking"] = True
                    corrected = True
                elif d["state"] == 2 and vy == 0:
                    for shift in (1, -1, 2, -2, 3, -3, 4, -4):
                        if not solid(s.room, x + sx, y + shift, ducking=d["ducking"]) and not solid(s.room, x, y + shift, ducking=d["ducking"]):
                            y += shift
                            x += sx
                            corrected = True
                            break
                if not corrected:
                    if not d["wall_retention"]:
                        d["wall_retained"], d["wall_retention"] = vx, 4
                    d["dash_attack"] = 0
                    vx, rx = 0.0, 0.0
                break
            else:
                x += sx
        ry = _f32(d["rem_y"] + _f32(vy * DT))
        dy = round(ry)
        ry = _f32(ry - dy)
        sy = _sign(dy)
        for _ in range(abs(dy)):
            collision = solid(s.room, x, y + sy, ducking=d["ducking"])
            if sy > 0 and self._jumpthru_at(s.room, x, y, y + sy):
                collision = True
            if collision:
                ry = 0.0
                if dashed_block(x, y + sy):
                    vx, vy, ry = 0.0, -120.0, 0.0
                    d.update(state=0, dash_frames=0, dash_attack=0, var_jump=9, var_speed=vy,
                             auto_jump_active=True, auto_jump=0, force_frames=0, wall_slide=72)
                    break
                corrected = False
                if sy > 0 and d["state"] == 2 and not d["dash_started_ground"]:
                    shifts = ((-1, -2, -3, -4) if vx <= 0 else ()) + ((1, 2, 3, 4) if vx >= 0 else ())
                    for shift in shifts:
                        if (not solid(s.room, x + shift, y + 1, ducking=d["ducking"])
                                and not self._jumpthru_at(s.room, x + shift, y, y + 1)):
                            # MoveHExact respects solids along the nudge.
                            direction = _sign(shift)
                            for _ in range(abs(shift)):
                                if solid(s.room, x + direction, y, ducking=d["ducking"]):
                                    rx = 0.0
                                    break
                                x += direction
                            if not solid(s.room, x, y + 1, ducking=d["ducking"]):
                                y += 1
                            corrected = True
                            break
                elif sy < 0:
                    # A vertical dash attack receives a fifth correction pixel
                    # in native OnCollideV; ordinary upward motion gets four.
                    correction = 5 if d["dash_attack"] and abs(vx)<.01 else 4
                    shifts = (tuple(-n for n in range(1,correction+1)) if vx <= .01 else ()) + (tuple(range(1,correction+1)) if vx >= -.01 else ())
                    for shift in shifts:
                        if not solid(s.room, x + shift, y - 1, ducking=d["ducking"]):
                            x += shift
                            y -= 1
                            corrected = True
                            break
                if not corrected:
                    # Three Single decrements from 0.2f leave 0.149999902f,
                    # below native's strict 0.15f ceiling-hit threshold even
                    # though telemetry still rounds that duration to 9 ticks.
                    # A rebound set to exactly 0.15f earlier in this same
                    # update is the exception and must retain its new timer.
                    if sy < 0 and (d["var_jump"] < 9 or
                            (d["var_jump"] == 9 and not fresh_rebound)):
                        d["var_jump"] = 0
                    # Native OnCollideV uses the retained DashDir even after
                    # DashUpdate/its attack window have ended. A later landing
                    # still flattens a downward diagonal and grants its boost.
                    if sy > 0 and d["dash_dx"] and d["dash_dy"] > 0:
                        d["dash_dx"], d["dash_dy"] = _sign(d["dash_dx"]), 0.0
                        vx = _f32(vx * _f32(1.2))
                        d["ducking"] = True
                    vy, ry = 0.0, 0.0
                    d["dash_attack"] = 0
                break
            else:
                y += sy
        d.update(x=x, y=y, vx=vx, vy=vy, rem_x=rx, rem_y=ry)
        result = SimState(**d)
        if self._hazard_hit(result):
            return replace(result, dead=True, reason="hazard", following=frozenset(), follow_order=())
        result = self._entities(result, safe_ground=self._safe_ground(s))
        if result.dead:
            return result
        if result.cassette_clock is not None:
            timer = max(0, result.lift_timer - 1)
            result = replace(result, lift_timer=timer, lift_vx=result.lift_vx if timer else 0.0,
                             lift_vy=result.lift_vy if timer else 0.0)
            result = self._cassette_segment(result, schedule=cassette_schedule, after=cassette_player_order)
            if result.dead:
                return result
        # Map movers update after the player in the native level entity list.
        # A player who lands this update is carried immediately, whereas a jump
        # uses the preceding frame's retained fractional platform velocity.
        result = self._advance_mechanisms(result, decay_lift=result.cassette_clock is None)
        # PlayerCollider checks precede Level.EnforceBounds; contacts at the
        # outgoing edge therefore belong to the outgoing room.
        result = self._bounds(result)
        if s.cassette_clock is not None and result.room != s.room:
            return replace(result, dead=True, reason="unsupported_cassette_transition")
        return replace(result, on_ground=self.on_ground(result))

    def _bounds(self, s: SimState) -> SimState:
        r = self.map.rooms[s.room]
        room_bottom = r.y + _runtime_height(r)
        top, bottom = s.y - (6 if s.ducking else 11), s.y
        center_y = (top + bottom) / 2
        probes = []
        if s.x - 4 < r.x and top >= r.y and bottom < room_bottom:
            probes.append((s.x - 8, center_y, -1, 0))
        if s.x + 4 > r.x + r.width and top >= r.y and bottom < room_bottom:
            probes.append((s.x + 8, center_y, 1, 0))
        if center_y < r.y:
            probes.append((s.x, center_y - 12, 0, -1))
        if bottom > room_bottom and not _truth(r.attrs.get("disableDownTransition", False)):
            if not self.solid_at(s.room, s.x, s.y + 4, ducking=s.ducking, state=s):
                probes.append((s.x, center_y + 12, 0, 1))
        for px, py, dx, dy in probes:
            for n in self._neighbors[s.room]:
                if not n.spawns or not (n.x <= px < n.x + n.width and n.y <= py < n.y + _runtime_height(n)):
                    continue
                x, y, vx, vy = s.x, s.y, s.vx, s.vy
                var_jump, var_speed, state, auto_jump = s.var_jump, s.var_speed, s.state, s.auto_jump
                auto_jump_active = s.auto_jump_active
                if dx > 0:
                    x = max(x, n.x + 4)
                elif dx < 0:
                    # IsInBounds tests < Right-4 and < Bottom-4, so the
                    # furthest integer target on left/up entries is edge-5.
                    x = min(x, n.x + n.width - 5)
                elif dy < 0:
                    y, vx, vy = min(y, n.y + _runtime_height(n) - 5), 0.0, JUMP_SPEED
                    var_jump, var_speed, state, auto_jump = 12, JUMP_SPEED, 0, 0
                    auto_jump_active = True
                else:
                    y, vy, state, var_jump = max(y, n.y + 12), max(0, vy), 0, 0
                    auto_jump_active = False
                return replace(s, room=n.name, x=x, y=y, vx=round(vx), vy=round(vy),
                               rem_x=0.0, rem_y=0.0, dashes=self.max_dashes, stamina=110.0,
                               grace=0, force_frames=0, wall_slide=72, var_jump=var_jump,
                               climb_hop_solid="", climb_hop_x=0.0, climb_hop_y=0.0,
                               var_speed=var_speed, state=state, auto_jump=auto_jump,
                               auto_jump_active=auto_jump_active,
                               dash_cooldown=12 if dy < 0 else s.dash_cooldown,
                               dash_frames=0 if dy else s.dash_frames,
                               # These entities reload on a room revisit. Permanent
                               # broken dash blocks remain in the session set.
                               mechanisms=(), refill_timers=(), lift_timer=0,
                               cassette_stopped=False, cassette_clock=None, cassette_blocks=(),
                               player_update_order=-1, player_depth=0, player_actual_depth=0.0,
                               # Uncollected winged berries are fresh entities
                               # when their room reloads; departure is not a
                               # session-persistent flag like a permanent wall.
                               fled=frozenset(key for key in s.fled if not key.startswith(n.name + ":")),
                               broken=frozenset(key for key in s.broken if not any(
                                   key == f"{s.room}:{e.id}" and e.kind == "dashBlock" and
                                   not _truth(e.attrs.get("permanent", True))
                                   for e in self._mechanisms[s.room])),
                               reason="transition")
        x, y, vx, vy = s.x, s.y, s.vx, s.vy
        if x - 4 < r.x:
            x, vx = r.x + 4, 0.0
        elif x + 4 > r.x + r.width:
            x, vx = r.x + r.width - 4, 0.0
        if top < r.y - 24:
            y, vy = r.y - 24 + (6 if s.ducking else 11), 0.0
        if top > room_bottom + 4:
            return replace(s, x=x, y=y, vx=vx, vy=vy, dead=True, reason="fall", following=frozenset(), follow_order=())
        return replace(s, x=x, y=y, vx=vx, vy=vy)

    def _hazard_hit(self, s: SimState) -> bool:
        h = 6 if s.ducking else 11
        hurt = (s.x - 4, s.y - h, s.x + 4, s.y - 2)
        records = {row[0]: row for row in s.mechanisms}
        for e in self._hazards[s.room]:
            x, y = e.x, e.y
            mover = self._spike_attachments[s.room].get((e.kind, e.id))
            if mover is not None:
                key = f"{s.room}:{mover.id}"
                if key in s.broken:
                    continue  # DashBlock destroys its attached StaticMovers.
                record = records.get(key)
                if record is not None:
                    _, phase, _, mover_x, mover_y, _ = record
                    if mover.kind == "fallingBlock" and phase == 4:
                        continue  # Fallen out of the room: DestroyStaticMovers.
                    x += round(mover_x) - mover.x
                    y += round(mover_y) - mover.y
            if e.kind == "spikesUp":
                rect = _spike_rect(e, x, y)
                # Player.Update temporarily installs its hurtbox for the
                # PlayerCollider callbacks; Spikes.OnCollide reads Bottom from
                # that collider, two pixels above the movement hitbox's feet.
                dangerous = s.vy >= 0 and hurt[3] <= y
            elif e.kind == "spikesDown":
                rect = _spike_rect(e, x, y)
                dangerous = s.vy <= 0
            elif e.kind == "spikesLeft":
                rect = _spike_rect(e, x, y)
                dangerous = s.vx >= 0
            elif e.kind == "spikesRight":
                rect = _spike_rect(e, x, y)
                dangerous = s.vx <= 0
            else:
                # Crystal spinner collision union is approximated conservatively.
                cx = max(hurt[0], min(e.x, hurt[2]))
                cy = max(hurt[1], min(e.y, hurt[3]))
                if (cx - e.x) ** 2 + (cy - e.y) ** 2 < 6 ** 2:
                    return True
                continue
            if dangerous and _overlap(hurt, rect):
                return True
        return False

    def _entities(self, s: SimState, *, safe_ground: bool | None = None) -> SimState:
        # In the vanilla Normal-level lifecycle, an observed stopped manager
        # and inactive blocks imply the pickup actor has already been removed.
        # Live telemetry establishes that invariant; an absent clock alone does not.
        if s.cassette_clock is not None:
            box = (s.x - 4, s.y - (6 if s.ducking else 11), s.x + 4, s.y - 2)
            if any(e.kind == "cassette" and _overlap(box, (e.x - 8, e.y - 8, e.x + 8, e.y + 8))
                   for e in self.map.rooms[s.room].entities):
                return replace(s, dead=True, reason="unsupported_cassette_pickup")
        following, order = set(s.following), list(s.follow_order)
        berries = set(s.berries)
        # Strawberry.Update checks delay before its Follower component updates
        # at the end of the entity tick. A newly picked berry cannot collect in
        # that tick, and a delay reaching zero becomes eligible next tick.
        collection_ready = bool(order) and s.collect_delay == 0
        delay = max(0, s.collect_delay - 1)
        ground_frames = s.collect_ground
        refill_timers = {key: timer - 1 for key, timer in s.refill_timers if timer > 1}
        for e in self._interact[s.room]:
            # Player temporarily installs its hurtbox for ALL PlayerCollider
            # components, including springs, refills and strawberries.
            box = (s.x - 4, s.y - (6 if s.ducking else 11), s.x + 4, s.y - 2)
            key = f"{s.room}:{e.id}"
            if e.kind == "strawberry":
                if key in berries or key in following or key in s.fled:
                    continue
                if e.nodes or _truth(e.attrs.get("golden")) or _truth(e.attrs.get("moon")):
                    continue
                if _overlap(box, (e.x - 7, e.y - 7, e.x + 7, e.y + 7)):
                    following.add(key)
                    order.append(key)
                    if len(order) == 1:
                        # GainFollower starts 0.3s during Player.Update; the
                        # later Follower.Update consumes its first tick now.
                        delay, ground_frames = 17, 0
            elif e.kind == "refill":
                if key not in refill_timers and _overlap(box, (e.x - 8, e.y - 8, e.x + 8, e.y + 8)):
                    maximum = 2 if _truth(e.attrs.get("twoDash")) else self.max_dashes
                    if s.dashes < maximum or s.stamina < 20:
                        s = replace(s, dashes=maximum, stamina=110.0)
                        refill_timers[key] = 10 ** 9 if _truth(e.attrs.get("oneUse")) else 150
            elif e.kind == "spring":
                if not spring_player_enabled(e):
                    continue
                if s.vy >= 0 and _overlap(box, (e.x - 8, e.y - 6, e.x + 8, e.y)):
                    # Spring calls SuperBounce(Spring.Top), not generic Bounce.
                    was_ducking = s.ducking
                    s = self._move_without_callback(replace(s, ducking=False), e.y - 6 - s.y)
                    s = replace(s, ducking=was_ducking, vy=-185.0, vx=0.0, dashes=self.max_dashes,
                                stamina=110.0, var_jump=12, var_speed=-185.0, auto_jump=0,
                                auto_jump_active=True,
                                state=0, dash_frames=0, dash_attack=0, wall_slide=72,
                                grace=0)
            elif e.kind in ("wallSpringLeft", "wallSpringRight"):
                if not spring_player_enabled(e):
                    continue
                direction = 1 if e.kind == "wallSpringLeft" else -1
                rect = (e.x, e.y - 8, e.x + 6, e.y + 8) if direction > 0 else (e.x - 6, e.y - 8, e.x, e.y + 8)
                if not (abs(s.vx) > 240 and _sign(s.vx) == direction) and _overlap(box, rect):
                    # SideBounce uses the normal body for alignment and permits
                    # an outgoing <=240px/s player to touch the spring again.
                    was_ducking = s.ducking
                    s = self._move_without_callback(replace(s, ducking=False), max(-4, min(4, e.y - s.y)))
                    s = self._move_without_callback(s, e.x + direction * 10 - s.x, horizontal=True)
                    s = replace(s, ducking=was_ducking, vx=direction * 240.0, vy=-140.0, dashes=self.max_dashes,
                                stamina=110.0, state=0, dash_frames=0, var_jump=12,
                                var_speed=-140.0, auto_jump=0, auto_jump_active=True,
                                force_x=direction, force_frames=18, dash_attack=0, wall_slide=72, grace=0)
        if order and collection_ready:
            if self._safe_ground(s) if safe_ground is None else safe_ground:
                ground_frames += 1
                # Nine fixed Single ticks total 0.150000274, already greater
                # than native Strawberry's strict 0.15f collect threshold.
                if ground_frames >= 9:
                    collected = order.pop(0)
                    following.remove(collected)
                    berries.add(collected)
                    # Later followers have a longer staggered collection delay.
                    ground_frames = -9 if order else 0
            else:
                ground_frames = min(ground_frames, 0)
        return replace(s, berries=frozenset(berries), following=frozenset(following),
                       follow_order=tuple(order), collect_delay=delay, collect_ground=ground_frames,
                       refill_timers=tuple(sorted(refill_timers.items())))
