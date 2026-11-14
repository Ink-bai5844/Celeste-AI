"""A measured stopped cassette room is stationary and may exit normally."""
from dataclasses import replace
import time
import unittest

from celeste_search.cassette import validate_blocks
from celeste_search.maps import Entity
from celeste_search.physics import RIGHT, SimState, Simulator
from celeste_search.planner import Goal, room_is_supported, search_leg, state_key
from celeste_search.routes import snapshot
from test_cassette_physics import fixture


def stopped_fixture(*extra, height=1):
    data, simulator, state = fixture(*extra)
    block = state.cassette_blocks[0]
    block = replace(block, activated=False, collidable=False, block_height=height,
                    y=block.y + 2 - height,
                    depth=8990, actual_depth=8990.)
    return data, simulator, replace(state, cassette_stopped=True, cassette_clock=None,
                                    cassette_blocks=(block,))


class CassetteStoppedPhysicsTests(unittest.TestCase):
    def test_complete_stopped_state_round_trips_and_has_distinct_search_identity(self):
        _, _, state = stopped_fixture()
        self.assertEqual(SimState.from_dict(state.to_dict()), state)
        encoded = snapshot(state)
        self.assertTrue(encoded["cassette_stopped"])
        self.assertIsNone(encoded["cassette_clock"])
        self.assertEqual(encoded["cassette_blocks"][0]["block_height"], 1)
        self.assertNotEqual(state_key(state), state_key(replace(state, cassette_stopped=False)))
        with self.assertRaises(ValueError):
            SimState.from_dict({**state.to_dict(), "cassette_stopped": 1})

    def test_all_observed_finish_and_reload_heights_remain_stationary_and_nonsolid(self):
        for height in (0, 1, 2):
            with self.subTest(block_height=height):
                _, simulator, state = stopped_fixture(height=height)
                self.assertTrue(room_is_supported(simulator, "a", state))
                self.assertFalse(simulator.on_ground(replace(state, x=40, y=state.cassette_blocks[0].y)))
                tail = simulator.trace(state, [0] * 120)
                self.assertEqual(len(tail), 121)
                self.assertTrue(all(not item.dead and item.cassette_stopped and
                                    item.cassette_clock is None and
                                    item.cassette_blocks == state.cassette_blocks for item in tail))

    def test_absence_or_partial_inactive_evidence_never_implies_a_stopped_room(self):
        _, simulator, state = stopped_fixture()
        block = state.cassette_blocks[0]
        for invalid in (replace(state, cassette_stopped=False),
                        replace(state, cassette_blocks=()),
                        replace(state, cassette_clock=fixture()[2].cassette_clock),
                        *(replace(state, cassette_blocks=(replace(block, **change),))
                          for change in ({"activated": True}, {"collidable": True},
                                         {"active": False}, {"depth": 10}, {"y": 40.},
                                         {"lift_y": 60.}, {"tracker_order": -1}))):
            with self.subTest(state=invalid):
                self.assertFalse(room_is_supported(simulator, "a", invalid))
        with self.assertRaises(ValueError):
            validate_blocks(None, (replace(block, group_ids=("missing",)),))

    def test_stopped_pickup_actor_is_absent_while_active_pickup_remains_guarded(self):
        _, simulator, stopped = stopped_fixture(Entity("cassette", 2, 80., 24., 0., 0.))
        self.assertFalse(simulator.step(replace(stopped, x=80., y=29., on_ground=False), 0).dead)
        _, active_simulator, active = fixture(Entity("cassette", 2, 80., 24., 0., 0.))
        self.assertEqual(active_simulator.step(replace(active, x=80., y=29., on_ground=False), 0).reason,
                         "unsupported_cassette_pickup")

    def test_exit_clears_local_evidence_and_reentry_requires_fresh_observation(self):
        data, _, state = stopped_fixture()
        other = replace(data.rooms["a"], name="b", x=128, spawns=((144., 64.),), entities=())
        simulator = Simulator(replace(data, rooms={**data.rooms, "b": other}))
        start = replace(state, x=124., vx=90.)
        moved = simulator.step(start, RIGHT)
        self.assertEqual(moved.room, "b")
        self.assertFalse(moved.dead)
        self.assertFalse(moved.cassette_stopped)
        self.assertIsNone(moved.cassette_clock)
        self.assertEqual(moved.cassette_blocks, ())
        self.assertEqual((moved.player_update_order, moved.player_depth, moved.player_actual_depth),
                         (-1, 0, 0.))
        self.assertNotIn("cassette_stopped", snapshot(moved))
        self.assertTrue(room_is_supported(simulator, "b", moved))
        self.assertFalse(room_is_supported(simulator, "a", moved))
        result = search_leg(simulator, start, Goal("target", "b", 144., 64.),
                            deadline=time.monotonic() + 1, max_expansions=100, max_frames=60)
        self.assertTrue(result.reached, result.reason)
        self.assertEqual(simulator.trace(start, result.actions)[-1], result.state)
        self.assertFalse(result.state.cassette_stopped)


if __name__ == "__main__":
    unittest.main()
