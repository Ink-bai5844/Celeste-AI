"""Read the user's installed Celeste maps without loading or modifying the game.

Format references (read 2026-11-12):
https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/BinaryPacker.cs
https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/RunLengthEncoding.cs
https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/LevelData.cs
https://github.com/TheCyndaquilDecompilers/Celeste_Decompiled/blob/main/Celeste/MapData.cs

This is an independent reader, not a redistribution of game code or assets.
Room/entity coordinates are world pixels; tile indices are room-local.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
import math
import struct
from typing import Any

TILE_SIZE = 8
BERRY_KINDS = frozenset({"strawberry", "snowberry", "goldenBerry", "memorialTextController"})
HAZARD_KINDS = frozenset({
    "spikesUp", "spikesDown", "spikesLeft", "spikesRight", "spinner",
    "trackSpinner", "rotateSpinner", "dustStaticSpinner", "dustTrackSpinner",
    "dustRotateSpinner", "seeker", "badelineBoss", "risingLava", "sandwichLava",
    "fireBall", "lightning", "killbox",
})

# Capability reports can pass the simulator's exact set to MapData.to_dict().
# The default is deliberately conservative: all behavior-changing entities are
# listed unless they are known static geometry / collectibles supported below.
DEFAULT_SUPPORTED_KINDS = frozenset({
    "player", "strawberry", "snowberry", "spikesUp", "spikesDown",
    "spikesLeft", "spikesRight", "jumpThru", "refill",
})
DECORATIVE_KINDS = frozenset({
    "light", "pointLight", "bloomPoint", "torch", "wallTorch", "lamp",
    "hanginglamp", "hangingLamp", "birdForsakenCityGem", "memorial",
    "flutterbird", "wire", "cobweb", "clothesline",
})


class MapFormatError(ValueError):
    """A map is truncated, invalid, or exceeds the reader's safety limits."""


@dataclass(frozen=True)
class Entity:
    kind: str
    id: int
    x: float
    y: float
    width: float
    height: float
    attrs: dict[str, Any] = field(default_factory=dict)
    nodes: tuple[tuple[float, float], ...] = ()

    @property
    def is_berry(self) -> bool:
        return self.kind in BERRY_KINDS

    @property
    def berry_category(self) -> str | None:
        if not self.is_berry:
            return None
        if self.kind in {"goldenBerry", "memorialTextController"}:
            return "golden"
        if self.attrs.get("moon", False):
            return "moon"
        if self.nodes:
            return "seeded"
        if self.attrs.get("winged", False):
            return "winged"
        return "ordinary"

    def berry_id(self, room_name: str) -> str:
        """Match EntityID's room/entity identity (unique within a map)."""
        return f"{room_name}:{self.id}"


@dataclass(frozen=True)
class Room:
    name: str
    x: int
    y: int
    width: int
    height: int
    solids: tuple[str, ...]
    entities: tuple[Entity, ...]
    spawns: tuple[tuple[float, float], ...]
    attrs: dict[str, Any] = field(default_factory=dict)

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """World x, y, width, height; right/bottom are exclusive."""
        return self.x, self.y, self.width, self.height

    @property
    def dummy(self) -> bool:
        return not self.spawns

    @property
    def berries(self) -> tuple[Entity, ...]:
        return tuple(e for e in self.entities if e.is_berry)

    @property
    def hazard_entities(self) -> tuple[Entity, ...]:
        return tuple(e for e in self.entities if e.kind in HAZARD_KINDS)

    def contains(self, x: float, y: float) -> bool:
        return self.x <= x < self.x + self.width and self.y <= y < self.y + self.height

    def tile_at(self, tx: int, ty: int) -> bool:
        """Whether a local 8px tile is solid. Outside the room is not a tile."""
        return (
            0 <= ty < len(self.solids)
            and 0 <= tx < len(self.solids[ty])
            and self.solids[ty][tx] not in {"0", " ", "\r", "\n"}
        )


@dataclass(frozen=True)
class MapData:
    package: str
    rooms: dict[str, Room]
    source: Path

    def start_room(self) -> Room:
        # Native MapData.StartLevel uses GetAt(Vector2.Zero), not room sorting.
        for room in self.rooms.values():
            if not room.dummy and room.contains(0, 0):
                return room
        for room in self.rooms.values():
            if not room.dummy:
                return room
        raise MapFormatError(f"{self.package}: map contains no playable room/spawn")

    def neighbors(self, name: str) -> list[Room]:
        """Geometric transition candidates, not a claim of traversability.

        Border obstacles, scripted gates and one-way mechanics must still be
        handled by the simulator. Dummy rooms and disabled downward exits are
        excluded, as in the engine's transition eligibility checks.
        """
        room = self.rooms[name]
        result = []
        for other in self.rooms.values():
            if other.name == name or other.dummy:
                continue
            overlap_y = min(room.y + room.height, other.y + other.height) - max(room.y, other.y)
            overlap_x = min(room.x + room.width, other.x + other.width) - max(room.x, other.x)
            horizontal = overlap_y > 0 and (
                room.x + room.width == other.x or other.x + other.width == room.x
            )
            above = overlap_x > 0 and other.y + other.height == room.y
            below = overlap_x > 0 and room.y + room.height == other.y
            if horizontal or above or (below and not room.attrs.get("disableDownTransition", False)):
                result.append(other)
        return result

    def to_dict(self, supported_kinds: set[str] | frozenset[str] | None = None) -> dict[str, Any]:
        """Compact inventory; parsing the map never marks anything explored."""
        supported = DEFAULT_SUPPORTED_KINDS if supported_kinds is None else supported_kinds
        total_entities: Counter[str] = Counter()
        unsupported: Counter[str] = Counter()
        berry_categories: Counter[str] = Counter()
        rooms = []
        for room in self.rooms.values():
            counts = Counter(e.kind for e in room.entities)
            total_entities.update(counts)
            missing = {kind: n for kind, n in counts.items() if kind not in supported and kind not in DECORATIVE_KINDS}
            unsupported.update(missing)
            berries = []
            for entity in room.berries:
                category = entity.berry_category
                berry_categories[category or "unknown"] += 1
                berries.append({
                    "id": entity.berry_id(room.name), "kind": entity.kind,
                    "x": entity.x, "y": entity.y, "category": category,
                    "winged": bool(entity.attrs.get("winged", False)),
                    "moon": bool(entity.attrs.get("moon", False)),
                    "seed_count": len(entity.nodes),
                })
            rooms.append({
                "name": room.name, "bounds": list(room.bounds), "dummy": room.dummy,
                "spawns": [list(p) for p in room.spawns], "berries": berries,
                "entities": dict(sorted(counts.items())),
                "unsupported_mechanics": dict(sorted(missing.items())),
                "neighbor_candidates": [r.name for r in self.neighbors(room.name)],
            })
        return {
            "package": self.package, "source": str(self.source),
            "room_count": len(self.rooms),
            "playable_room_count": sum(not r.dummy for r in self.rooms.values()),
            "strawberry_count": sum(berry_categories.values()),
            "strawberry_categories": dict(sorted(berry_categories.items())),
            "entity_counts": dict(sorted(total_entities.items())),
            "unsupported_mechanics": dict(sorted(unsupported.items())),
            "coverage_note": "Inventory only; no room or strawberry has been verified in game.",
            "rooms": rooms,
        }


@dataclass
class _Element:
    name: str
    attrs: dict[str, Any]
    children: list[_Element]


class _Reader:
    MAX_STRING = 8 * 1024 * 1024
    MAX_ELEMENTS = 1_000_000

    def __init__(self, data: bytes, source: Path):
        self.data = data
        self.offset = 0
        self.source = source
        self.lookup: list[str] = []
        self.elements = 0

    def fail(self, message: str) -> MapFormatError:
        return MapFormatError(f"{self.source.name}: byte {self.offset}: {message}")

    def read(self, count: int) -> bytes:
        if count < 0 or self.offset + count > len(self.data):
            raise self.fail(f"truncated payload (need {count} bytes)")
        result = self.data[self.offset:self.offset + count]
        self.offset += count
        return result

    def number(self, code: str) -> Any:
        return struct.unpack("<" + code, self.read(struct.calcsize("<" + code)))[0]

    def string(self) -> str:
        size = 0
        for shift in range(0, 35, 7):
            byte = self.number("B")
            if shift == 28 and byte > 7:
                raise self.fail("invalid .NET string length")
            size |= (byte & 0x7F) << shift
            if not byte & 0x80:
                if size > self.MAX_STRING:
                    raise self.fail("string exceeds size limit")
                try:
                    return self.read(size).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise self.fail("invalid UTF-8 string") from exc
        raise self.fail("unterminated string length")

    def lookup_string(self) -> str:
        index = self.number("h")
        if not 0 <= index < len(self.lookup):
            raise self.fail(f"lookup index {index} out of range")
        return self.lookup[index]

    def value(self) -> Any:
        kind = self.number("B")
        if kind == 0:
            return bool(self.number("B"))
        if kind in {1, 2, 3, 4}:
            value = self.number({1: "B", 2: "h", 3: "i", 4: "f"}[kind])
            if isinstance(value, float) and not math.isfinite(value):
                raise self.fail("non-finite map attribute")
            return value
        if kind == 5:
            return self.lookup_string()
        if kind == 6:
            return self.string()
        if kind == 7:
            size = self.number("h")
            if size < 0 or size % 2:
                raise self.fail("invalid RLE byte length")
            encoded = self.read(size)
            if sum(encoded[::2]) > self.MAX_STRING:
                raise self.fail("decoded RLE exceeds size limit")
            return "".join(chr(encoded[i + 1]) * encoded[i] for i in range(0, size, 2))
        raise self.fail(f"unsupported attribute type {kind}")

    def element(self, depth: int = 0) -> _Element:
        self.elements += 1
        if depth > 128 or self.elements > self.MAX_ELEMENTS:
            raise self.fail("element nesting/count limit exceeded")
        name = self.lookup_string()
        attrs = {}
        for _ in range(self.number("B")):
            key = self.lookup_string()
            if key in attrs:
                raise self.fail(f"duplicate attribute {key!r}")
            attrs[key] = self.value()
        count = self.number("h")
        if count < 0:
            raise self.fail("negative child count")
        return _Element(name, attrs, [self.element(depth + 1) for _ in range(count)])


def _coordinate(attrs: dict[str, Any], key: str, default: float = 0) -> float:
    try:
        value = float(attrs.get(key, default))
    except (ValueError, TypeError) as exc:
        raise MapFormatError(f"invalid numeric {key!r} attribute") from exc
    if not math.isfinite(value):
        raise MapFormatError(f"non-finite {key!r} attribute")
    return value


def _room(element: _Element) -> Room:
    attrs = dict(element.attrs)
    raw_name = str(attrs.get("name", ""))
    # Native LevelData strips the editor's 'lvl_' prefix. EntityID and session
    # room names use the stripped form; retain the raw value in attrs.
    name = raw_name[4:] if raw_name.startswith("lvl_") else raw_name
    if not name:
        raise MapFormatError("room is missing its name")
    x, y, width, height = (int(_coordinate(attrs, k)) for k in ("x", "y", "width", "height"))
    if width <= 0 or height <= 0 or width > 131072 or height > 131072:
        raise MapFormatError(f"room {name!r} has invalid dimensions {width}x{height}")
    tile_width, tile_height = math.ceil(width / TILE_SIZE), math.ceil(height / TILE_SIZE)
    if tile_width * tile_height > 16_000_000:
        raise MapFormatError(f"room {name!r} has too many tiles")
    solids_text = ""
    entities, spawns = [], []
    for child in element.children:
        if child.name == "solids":
            solids_text = str(child.attrs.get("innerText", ""))
        elif child.name in {"entities", "triggers"}:
            for raw in child.children:
                entity_attrs = dict(raw.attrs)
                if child.name == "triggers":
                    entity_attrs["_trigger"] = True
                ex = x + _coordinate(entity_attrs, "x")
                ey = y + _coordinate(entity_attrs, "y")
                if raw.name == "player":
                    spawns.append((ex, ey))
                    continue
                nodes = tuple(
                    (x + _coordinate(node.attrs, "x"), y + _coordinate(node.attrs, "y"))
                    for node in raw.children if node.name == "node"
                )
                entities.append(Entity(
                    kind=raw.name, id=int(_coordinate(entity_attrs, "id", -1)),
                    x=ex, y=ey, width=_coordinate(entity_attrs, "width"),
                    height=_coordinate(entity_attrs, "height"), attrs=entity_attrs, nodes=nodes,
                ))
    rows = solids_text.splitlines()
    if len(rows) > tile_height or any(len(row) > tile_width for row in rows):
        raise MapFormatError(f"room {name!r} solids exceed room bounds")
    solids = tuple((rows[i] if i < len(rows) else "").ljust(tile_width, "0") for i in range(tile_height))
    return Room(name, x, y, width, height, solids, tuple(entities), tuple(spawns), attrs)


def load_map(path: str | Path) -> MapData:
    """Read a native .bin map. Raises MapFormatError for malformed content."""
    source = Path(path).resolve()
    if source.stat().st_size > 64 * 1024 * 1024:
        raise MapFormatError(f"{source.name}: file exceeds 64 MiB limit")
    reader = _Reader(source.read_bytes(), source)
    if reader.string() != "CELESTE MAP":
        raise reader.fail("not a CELESTE MAP file")
    package = reader.string()
    count = reader.number("h")
    if count < 0:
        raise reader.fail("negative string lookup count")
    reader.lookup = [reader.string() for _ in range(count)]
    root = reader.element()
    if reader.offset != len(reader.data):
        raise reader.fail("unexpected trailing bytes")
    levels = [child for child in root.children if child.name == "levels"]
    if len(levels) != 1:
        raise MapFormatError(f"{source.name}: expected one levels element")
    rooms = {}
    for raw in levels[0].children:
        room = _room(raw)
        if room.name in rooms:
            raise MapFormatError(f"{source.name}: duplicate room {room.name!r}")
        rooms[room.name] = room
    if not rooms:
        raise MapFormatError(f"{source.name}: empty map")
    return MapData(package, rooms, source)


def discover_maps(directory: str | Path) -> list[Path]:
    """Return sorted .bin files in a Maps directory (non-recursive)."""
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    return sorted(directory.glob("*.bin"), key=lambda p: p.name.casefold())
