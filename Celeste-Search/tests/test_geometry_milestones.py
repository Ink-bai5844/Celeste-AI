"""Generated geometry targets do not manufacture reachability or idle safety."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.geometry_milestones import continuous_milestones, transient_support_goals
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import SimState, Simulator
from celeste_search.planner import Goal


def room_with(*entities):
    return Room("arbitrary", 0, 0, 320, 240, ("0" * 40,) * 30,
                tuple(entities), ((32.0, 200.0),), {})


class GeometryMilestoneTests(unittest.TestCase):
    def test_required_crumble_detour_is_preserved_even_farther_from_berry(self):
        room = room_with(Entity("crumbleBlock", 1, 120, 160, 24, 0),
                         Entity("crumbleBlock", 2, 168, 136, 24, 0))
        start = SimState(room.name, 80, 184, on_ground=True)
        berry = Goal("berry", room.name, 48, 104, "arbitrary:7", "arbitrary:7")
        goals = continuous_milestones(room, start, berry)
        self.assertEqual({goal.entity_key for goal in goals}, {"arbitrary:1", "arbitrary:2"})
        self.assertTrue(all(goal.x > start.x > berry.x for goal in goals))
        self.assertTrue(all(goal.no_dash_until == berry.id for goal in goals))
        self.assertTrue(all(goal.transient and goal.requires_continuation and not goal.idle_endpoint_allowed
                            for goal in goals))
        self.assertTrue(all(not goal.achieved(start) for goal in goals))

    def test_moved_zip_top_uses_actual_mechanism_position_not_map_origin(self):
        room = room_with(Entity("zipMover", 4, 40, 160, 24, 16, nodes=((184, 80),)))
        state = SimState(room.name, 116, 112, on_ground=True,
                         mechanisms=(("arbitrary:4", 2, 15, 104.0, 112.0, 0.0),))
        goal = transient_support_goals(room, state)[0]
        self.assertEqual((goal.x, goal.y), (116, 112))
        self.assertTrue(goal.achieved(state))
        moved = replace(state, x=156, y=96,
                        mechanisms=(("arbitrary:4", 2, 24, 144.0, 96.0, 0.0),))
        self.assertEqual(goal.waypoint(moved), (156, 96))
        self.assertTrue(goal.achieved(moved))
        self.assertFalse(goal.achieved(replace(moved, x=116, y=112)))
        self.assertFalse(goal.achieved(replace(moved, on_ground=False)))

    def test_collapsed_crumble_is_not_achieved_until_normal_model_restore(self):
        room = room_with(Entity("crumbleBlock", 2, 80, 160, 24, 0))
        state = SimState(room.name, 92, 160, on_ground=True,
                         mechanisms=(("arbitrary:2", 3, 40, 80.0, 160.0, 0.0),))
        goal = transient_support_goals(room, state)[0]
        self.assertFalse(goal.achieved(state))
        restored = replace(state, mechanisms=(("arbitrary:2", 0, 0, 80.0, 160.0, 0.0),))
        self.assertTrue(goal.achieved(restored))
        self.assertFalse(goal.idle_endpoint_allowed)

    def test_removed_falling_block_and_unrelated_entities_are_excluded(self):
        room = room_with(Entity("fallingBlock", 1, 64, 80, 24, 16),
                         Entity("fallingBlock", 2, 96, 80, 24, 16),
                         Entity("jumpThru", 3, 128, 80, 24, 0),
                         Entity("strawberry", 4, 160, 80, 0, 0))
        state = SimState(room.name, 24, 160,
                         mechanisms=(("arbitrary:1", 4, 0, 64.0, 260.0, 160.0),
                                     ("arbitrary:2", 2, 0, 96.0, 112.5, 60.0)))
        goals = transient_support_goals(room, state)
        self.assertEqual([goal.entity_key for goal in goals], ["arbitrary:2"])
        self.assertEqual(goals[0].y, 112)  # Same collider rounding as Simulator.

    def test_transient_top_is_not_safe_to_wait_for_background_work(self):
        room = room_with(Entity("crumbleBlock", 2, 80, 160, 24, 0))
        sim = Simulator(MapData("fixture", {room.name: room}, Path("fixture.bin")))
        state = SimState(room.name, 92, 160, on_ground=True)
        goal = transient_support_goals(room, state)[0]
        self.assertTrue(goal.achieved(state))
        self.assertFalse(sim._safe_ground(state))
        self.assertEqual(continuous_milestones(room, state), [])


if __name__ == "__main__":
    unittest.main()
