"""Directionless normal wall jumps are real, resource-preserving transfers."""
from dataclasses import replace
from pathlib import Path
import time
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, GRAB, JUMP, LEFT, RIGHT, UP, SimState, Simulator
from celeste_search.wall_moves import WallGripGoal
from celeste_search.wall_transfers import static_wall_transfer_edges


def fixture(*, mirror=False, offset=(0, 0), step=False, stamina=25):
    ox, oy = offset
    def x(value):
        return ox + (256-value if mirror else value)
    def rect(left, top, width, height):
        return (x(left+width) if mirror else x(left), oy+top, width, height)
    blocks = [(144, 96, 16, 128), (128, 16, 32, 80)] if step else [(128, 16, 16, 208)]
    entities = tuple(Entity("solid", i, *rect(*block)) for i, block in enumerate(blocks))
    room = Room("translated_wall", ox, oy, 256, 240, ("0"*32,)*30, entities, ())
    sim = Simulator(MapData("wall-regrip", {room.name: room}, Path("fixture.bin")))
    direction = LEFT if mirror else RIGHT
    state = SimState(room.name, x(140 if step else 124), oy+168,
                     state=1, facing=-1 if mirror else 1, stamina=stamina, dashes=0,
                     prev_input=JUMP | GRAB | UP | direction)
    left, top, width, height = rect(128, 16, 32 if step else 16, 80 if step else 208)
    goal = WallGripGoal("wall_grip", room.name, x(124), oy+64,
                       rect=(left, top, left+width, top+height), facing=state.facing)
    return sim, state, goal


class WallRegripTests(unittest.TestCase):
    def edges(self, sim, state, goal, **kwargs):
        return static_wall_transfer_edges(sim, state, goal,
            deadline=time.monotonic()+1, max_expansions=3, **kwargs)

    def test_same_wall_low_stamina_mirror_and_negative_translation(self):
        for mirror, offset in ((False, (0, 0)), (True, (-768, -480))):
            with self.subTest(mirror=mirror):
                sim, start, goal = fixture(mirror=mirror, offset=offset)
                edges = self.edges(sim, start, goal, max_frames=180)
                self.assertTrue(edges)
                actions, expected, label = edges[0]
                self.assertIn("release_regrip:0", label)
                self.assertEqual(actions[:2], [0, JUMP])
                state = start
                for mask in actions:
                    self.assertFalse(mask & DASH)
                    state = sim.step(state, mask)
                    self.assertFalse(state.dead)
                    self.assertFalse(sim.last_step_started_dash)
                self.assertEqual(state, expected)
                self.assertTrue(goal.achieved(state))
                self.assertEqual(state.stamina, start.stamina)
                self.assertEqual(state.dashes, 0)
                self.assertGreater(start.y-state.y, 90)

    def test_neighboring_step_uses_real_catches_without_resource_reset(self):
        for mirror in (False, True):
            sim, start, goal = fixture(mirror=mirror, offset=(-512, -256), step=True)
            edges = self.edges(sim, start, goal, max_frames=220)
            self.assertTrue(edges)
            actions, expected, _ = edges[0]
            state = start
            for mask in actions:
                state = sim.step(state, mask)
                self.assertFalse(state.dead)
            self.assertEqual(state, expected)
            self.assertTrue(goal.achieved(state))
            self.assertNotEqual(state.x, start.x)
            self.assertEqual(state.stamina, start.stamina)

    def test_first_same_wall_catch_is_not_a_distant_target_success(self):
        sim, start, goal = fixture()
        self.assertEqual(self.edges(sim, start, goal, max_frames=30), [])

    def test_below_native_regrab_stamina_cannot_be_replenished(self):
        sim, start, goal = fixture(stamina=19)
        self.assertEqual(self.edges(sim, start, goal, max_frames=180), [])

    def test_no_dash_wrapper_and_buffered_dash_rejection(self):
        sim, start, goal = fixture()
        class Wrapped:
            no_dash = True
            def __getattr__(self, name):
                return getattr(sim, name)
            def step(self, state, mask):
                if mask & DASH:
                    raise AssertionError("No-dash family emitted Dash")
                return sim.step(state, mask)
        self.assertTrue(self.edges(Wrapped(), start, goal, max_frames=180))
        buffered = replace(start, state=1, dashes=1, dash_buffer=3,
                           prev_input=GRAB | RIGHT)
        self.assertEqual(self.edges(Wrapped(), buffered, goal, max_frames=180), [])


if __name__ == "__main__":
    unittest.main()
