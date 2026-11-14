"""Portal tiles are geometry controls; no real game is used by these tests."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.assist_targets import room_exit_targets, transition_target_reached
from celeste_search.assist_validation import AssistValidationError, target_from_world, validate_tile_target
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import SimState, Simulator
from celeste_search.assist_routes import normalize_route, start_route_progress, advance_route_progress


def room(name, x, y, *, height=64, solids=None, entities=(), spawns=None, attrs=None):
    return Room(name, x, y, 64, height, solids or ("0"*8,)*((height+7)//8),
                tuple(entities), ((x+32.,y+32.),) if spawns is None else spawns, attrs or {})


def data(*rooms):
    return MapData("portal-fixture", {item.name:item for item in rooms}, Path("portal-fixture.bin"))


class ExitTargetTests(unittest.TestCase):
    def test_all_four_edges_are_selectable_and_resolve_neighbor(self):
        map_data = data(room("a",0,0), room("left",-64,0), room("right",64,0),
                        room("top",0,-64), room("bottom",0,64))
        positions = {"left":(-7,34),"right":(66,34),"top":(34,-7),"bottom":(34,66)}
        for edge, (x,y) in positions.items():
            with self.subTest(edge=edge):
                selected = target_from_world(map_data,"a",x,y)
                self.assertEqual(selected["edge"],edge)
                self.assertEqual(selected["transition_to"],edge)
                self.assertEqual(validate_tile_target(map_data,selected),selected)
                self.assertFalse(transition_target_reached(SimState("a",x,y),selected))
        exits = room_exit_targets(map_data,"a")
        self.assertIsInstance(exits,list)
        self.assertEqual(set(item["edge"] for item in exits),set(positions))

    def test_outside_without_neighbor_dummy_or_diagonal_is_not_an_exit(self):
        for extra in ((), (room("dummy",64,0,spawns=()),), (room("diagonal",64,64),)):
            map_data = data(room("a",0,0),*extra)
            self.assertEqual(room_exit_targets(map_data,"a"),[])
            with self.assertRaises(AssistValidationError):
                target_from_world(map_data,"a",65,33)

    def test_permanent_solid_on_either_side_hides_portal(self):
        for blocked_name in ("a","b"):
            rooms = [room("a",0,0),room("b",64,0)]
            index = 0 if blocked_name == "a" else 1
            rows = tuple("00000001" if index == 0 else "10000000" for _ in range(8))
            rooms[index] = replace(rooms[index],solids=rows)
            self.assertFalse(room_exit_targets(data(*rooms),"a"))

    def test_solid_entity_blocks_but_breakable_wall_remains_candidate(self):
        for kind in ("solid","invisibleBarrier","dashBlock"):
            wall = Entity(kind,1,56,0,8,64)
            exits = room_exit_targets(data(room("a",0,0,entities=(wall,)),room("b",64,0)),"a")
            self.assertEqual(bool(exits),kind == "dashBlock")

    def test_disabled_down_transition_hides_bottom_only(self):
        map_data = data(room("a",0,0,attrs={"disableDownTransition":True}),
                        room("b",0,64),room("c",64,0))
        self.assertEqual({item["edge"] for item in room_exit_targets(map_data,"a")},{"right"})

    def test_negative_coordinates_and_only_one_extension_tile(self):
        map_data = data(room("a",-128,-64),room("b",-64,-64))
        target = target_from_world(map_data,"a",-62.5,-30)
        self.assertEqual((target["x"],target["y"]),(-64,-32))
        with self.assertRaises(AssistValidationError):
            target_from_world(map_data,"a",-55,-30)
        with self.assertRaises(AssistValidationError):
            target_from_world(map_data,"a",-63,-65)

    def test_legacy_height_184_bottom_has_grid_extension_after_partial_tile(self):
        map_data = data(room("a",0,0,height=184),room("b",0,184))
        self.assertEqual(target_from_world(map_data,"a",34,179)["y"],176)
        for y in (180,181,183.9):
            with self.assertRaises(AssistValidationError):
                target_from_world(map_data,"a",34,y)
        target = target_from_world(map_data,"a",34,185)
        self.assertEqual(target["y"],184)
        self.assertEqual(target["transition_to"],"b")
        entered = Simulator(map_data)._bounds(SimState("a",34,181))
        self.assertEqual(entered.room,"b")
        self.assertTrue(transition_target_reached(entered,target))

    def test_upward_native_reposition_can_skip_visual_tile(self):
        map_data = data(room("a",0,0),room("b",0,-184,height=184))
        target = target_from_world(map_data,"a",34,-7)
        entered = Simulator(map_data)._bounds(SimState("a",34,5))
        self.assertEqual(entered.room,"b")
        self.assertLess(entered.y-5.5,target["y"])
        self.assertTrue(transition_target_reached(entered,target))
        self.assertFalse(transition_target_reached(replace(entered,x=42),target))
        self.assertFalse(transition_target_reached(replace(entered,dead=True),target))
        self.assertFalse(transition_target_reached(replace(entered,y=-80),target))

    def test_forged_metadata_is_rejected_or_replaced_by_geometry(self):
        map_data = data(room("a",0,0),room("b",64,0))
        target = target_from_world(map_data,"a",65,34)
        normalized = validate_tile_target(map_data,{**target,"arrival_bounds":[-999,-999,999,999]})
        self.assertEqual(normalized,target)
        for changes in ({"transition_to":"missing"},{"edge":"left"},{"x":56}):
            with self.assertRaises(AssistValidationError):
                validate_tile_target(map_data,{**target,**changes})
        # Cached values are isolated from caller edits.
        exits = room_exit_targets(map_data,"a")
        exits[0]["arrival_bounds"][0] = -999
        self.assertNotEqual(room_exit_targets(map_data,"a")[0]["arrival_bounds"][0],-999)

    def test_route_exit_requires_actual_directed_crossing_not_neighbor_presence(self):
        map_data = data(room("a",0,0),room("b",64,0),room("c",64,-64))
        target = target_from_world(map_data,"a",65,34)
        route = normalize_route(map_data,{"waypoints":[target]})
        source = SimState("a",61,38)
        arrival = Simulator(map_data)._bounds(source)
        self.assertTrue(transition_target_reached(arrival,target))
        initial = start_route_progress(route,arrival)
        self.assertEqual(initial.index,0)
        walking = advance_route_progress(route,initial,arrival,replace(arrival,x=arrival.x+1),1)
        self.assertEqual(walking.index,0)
        wrong_room = advance_route_progress(route,initial,replace(source,room="c"),arrival,1)
        self.assertEqual(wrong_room.index,0)
        progress = start_route_progress(route,source)
        crossed = advance_route_progress(route,progress,source,arrival,1)
        self.assertEqual(crossed.index,1)

    def test_reported_city_room_top_opening_has_three_clickable_exit_tiles(self):
        from celeste_search.maps import load_map
        source=Path(__file__).resolve().parents[2]/"GAME/Celeste-2/Content/Maps/1-ForsakenCity.bin"
        if not source.exists():
            self.skipTest("Local owned game map is not installed")
        map_data=load_map(source)
        selected=map_data.rooms["5"]
        exits=[item for item in room_exit_targets(map_data,"5") if item["edge"]=="top"]
        self.assertEqual({((item["x"]-selected.x)//8,item["y"]-selected.y,item["transition_to"])
                          for item in exits},{(35,-8,"6"),(36,-8,"6"),(37,-8,"6")})
        for item in exits:
            self.assertEqual(target_from_world(map_data,"5",item["x"]+4,item["y"]+4),item)

    def test_one_crossing_cannot_consume_same_exit_twice(self):
        map_data = data(room("a",0,0),room("b",64,0))
        target = target_from_world(map_data,"a",65,34)
        route = normalize_route(map_data,{"waypoints":[target,target]})
        source = SimState("a",61,38)
        arrival = Simulator(map_data)._bounds(source)
        progress = start_route_progress(route,source)
        crossed = advance_route_progress(route,progress,source,arrival,1)
        self.assertEqual(crossed.index,1)
        waiting = advance_route_progress(route,crossed,arrival,arrival,0)
        self.assertEqual(waiting.index,1)


if __name__ == "__main__":
    unittest.main()
