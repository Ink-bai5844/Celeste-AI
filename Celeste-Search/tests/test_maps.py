"""Parser/geometry checks use generated fixtures, plus optional owned map files."""

from pathlib import Path
import struct
import tempfile
import unittest

from celeste_search.maps import (
    Entity, MapData, MapFormatError, Room, discover_maps, load_map,
)


def dotnet_string(value):
    data = value.encode("utf-8")
    size = len(data)
    prefix = bytearray()
    while size >= 128:
        prefix.append((size & 127) | 128)
        size >>= 7
    prefix.append(size)
    return bytes(prefix) + data


def fixture():
    lookup = ["Map", "levels", "level", "name", "lvl_1", "x", "y", "width",
              "height", "entities", "player", "strawberry", "id", "winged",
              "node", "solids", "innerText", "float", "long", "short", "text"]

    def element(name, attrs=(), children=()):
        data = struct.pack("<hB", lookup.index(name), len(attrs))
        for key, kind, value in attrs:
            data += struct.pack("<hB", lookup.index(key), kind)
            if kind in {0, 1, 2, 3, 4}:
                data += struct.pack("<" + {0: "B", 1: "B", 2: "h", 3: "i", 4: "f"}[kind], value)
            elif kind == 5:
                data += struct.pack("<h", lookup.index(value))
            elif kind == 6:
                data += dotnet_string(value)
            elif kind == 7:
                data += struct.pack("<h", len(value)) + value
        return data + struct.pack("<h", len(children)) + b"".join(children)

    player = element("player", [("x", 1, 4), ("y", 1, 8)])
    berry = element("strawberry", [("id", 2, 7), ("x", 1, 12), ("y", 1, 4), ("winged", 0, 1)],
                    [element("node", [("x", 1, 8), ("y", 1, 2)])])
    # Two 2-character rows; RLE includes the newline explicitly.
    solids = element("solids", [("innerText", 7, b"\x011\x010\x01\n\x020")])
    room = element("level", [("name", 5, "lvl_1"), ("x", 2, -8), ("y", 2, -8),
                             ("width", 1, 16), ("height", 1, 16),
                             ("float", 4, 1.5), ("long", 3, 100000),
                             ("short", 2, -300), ("text", 6, "雪" * 80)],
                   [element("entities", children=[player, berry]), solids])
    root = element("Map", children=[element("levels", children=[room])])
    return dotnet_string("CELESTE MAP") + dotnet_string("test") + struct.pack("<h", len(lookup)) + b"".join(map(dotnet_string, lookup)) + root


class MapReaderTests(unittest.TestCase):
    def read_fixture(self, data):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.bin"
            path.write_bytes(data)
            return load_map(path)

    def test_all_attribute_types_rle_and_world_coordinates(self):
        result = self.read_fixture(fixture())
        room = result.start_room()
        self.assertEqual(result.package, "test")
        self.assertEqual(room.name, "1")
        self.assertEqual(room.spawns, ((-4, 0),))
        self.assertEqual(room.solids, ("10", "00"))
        self.assertTrue(room.tile_at(0, 0))
        self.assertFalse(room.tile_at(1, 0))
        self.assertFalse(room.tile_at(-1, 0))
        self.assertFalse(room.tile_at(0, 2))
        berry = room.berries[0]
        self.assertEqual((berry.x, berry.y, berry.nodes), (4, -4, ((0, -6),)))
        self.assertEqual(berry.berry_id(room.name), "1:7")
        self.assertEqual(berry.berry_category, "seeded")
        self.assertEqual(room.attrs["float"], 1.5)
        self.assertEqual(room.attrs["long"], 100000)
        self.assertEqual(room.attrs["short"], -300)
        self.assertEqual(room.attrs["text"], "雪" * 80)
        self.assertTrue(result.to_dict()["rooms"][0]["berries"][0]["winged"])

    def test_truncation_and_trailing_bytes_rejected(self):
        data = fixture()
        for size in [0, 1, 10, 50, len(data) - 1]:
            with self.subTest(size=size), self.assertRaises(MapFormatError):
                self.read_fixture(data[:size])
        with self.assertRaisesRegex(MapFormatError, "trailing"):
            self.read_fixture(data + b"\x00")

    def test_invalid_header_lookup_and_rle_rejected(self):
        with self.assertRaisesRegex(MapFormatError, "not a CELESTE MAP"):
            self.read_fixture(dotnet_string("NOT A MAP"))
        prefix = dotnet_string("CELESTE MAP") + dotnet_string("test")
        with self.assertRaisesRegex(MapFormatError, "lookup"):
            self.read_fixture(prefix + struct.pack("<hh", 0, 0))
        data = fixture().replace(b"\x07\x08\x00\x011\x010\x01\n\x020", b"\x07\x07\x00\x011\x010\x01\n\x02", 1)
        self.assertNotEqual(data, fixture())
        with self.assertRaisesRegex(MapFormatError, "RLE"):
            self.read_fixture(data)

    def test_neighbors_exclude_dummy_corner_and_disabled_down(self):
        def room(name, x, y, attrs=None, spawns=((1, 1),)):
            return Room(name, x, y, 16, 16, ("00", "00"), (), spawns, attrs or {})
        rooms = {r.name: r for r in [room("center", 0, 0, {"disableDownTransition": True}),
                 room("right", 16, 0), room("above", 0, -16), room("below", 0, 16),
                 room("corner", 16, 16), room("dummy", -16, 0, spawns=())]}
        data = MapData("test", rooms, Path("test.bin"))
        self.assertEqual({r.name for r in data.neighbors("center")}, {"right", "above"})

    def test_inventory_keeps_unknown_mechanics_visible(self):
        unknown = Entity("futureMechanic", 2, 0, 0, 16, 16)
        room = Room("1", 0, 0, 16, 16, ("00", "00"), (unknown,), ((0, 8),))
        inventory = MapData("test", {"1": room}, Path("test.bin")).to_dict()
        self.assertEqual(inventory["unsupported_mechanics"], {"futureMechanic": 1})
        self.assertEqual(inventory["playable_room_count"], 1)

    def test_native_owned_maps_when_available(self):
        maps = Path(__file__).resolve().parents[2] / "GAME" / "Celeste-1" / "Content" / "Maps"
        if not maps.is_dir():
            self.skipTest("owned game assets not installed")
        paths = discover_maps(maps)
        self.assertGreater(len(paths), 0)
        for path in paths:
            with self.subTest(path=path.name):
                parsed = load_map(path)
                self.assertGreater(len(parsed.rooms), 0)
                self.assertFalse(parsed.start_room().dummy)
        city = load_map(maps / "1-ForsakenCity.bin")
        self.assertEqual(city.start_room().name, "1")
        self.assertEqual(len(city.rooms), 38)
        self.assertEqual(sum(e.kind == "strawberry" for r in city.rooms.values() for e in r.entities), 20)


if __name__ == "__main__":
    unittest.main()
