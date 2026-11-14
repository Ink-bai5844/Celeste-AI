"""Local timing refinement keeps strict goals, old branches and finite budgets."""
from dataclasses import dataclass, replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_targets import room_exit_targets
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, DOWN, GRAB, JUMP, LEFT, RIGHT, UP, SimState, Simulator
from celeste_search.planner import Goal
from celeste_search.wall_transfers import _RegripRecipe, static_wall_transfer_edges
from celeste_search.wall_underpasses import wall_underpass_edges


def exit_fixture(mirror=False, offset=(0, 0)):
    ox, oy = offset
    def x(value):
        return ox+(128-value if mirror else value)
    wall_x = x(48) if mirror else x(32)
    room = Room("lower", ox, oy, 128, 160, ("0"*16,)*20,
                (Entity("solid", 1, wall_x, oy, 16, 160),), ((x(52), oy+150),))
    upper = Room("upper", ox, oy-160, 128, 160, ("0"*16,)*20,
                 (Entity("jumpThru", 2, ox+48, oy-8, 32, 0),), ((x(52), oy-8),))
    data = MapData("short-regrip", {room.name: room, upper.name: upper}, Path("fixture.bin"))
    sim = Simulator(data)
    start = SimState(room.name, x(52), oy+77, state=1, facing=1 if mirror else -1,
                     stamina=40, dashes=0, prev_input=GRAB | UP | (RIGHT if mirror else LEFT))
    tile_x = ox+(72 if mirror else 48)
    target = next(t for t in room_exit_targets(data, room.name) if t["edge"] == "top" and t["x"] == tile_x)
    return sim, start, replace(_selected_goal(target)[0], crossing_only=True)


@dataclass(frozen=True)
class Contact(Goal):
    facing: int = 1

    def achieved(self, state):
        return not state.dead and state.x == 20


class TimingSimulator:
    """Contract fixture: only release durations 19..21 can actually catch."""
    no_dash = True
    last_step_started_dash = False

    def __init__(self, reject_neighbor=False):
        self.attempted = []
        self.reject_neighbor = reject_neighbor

    def step(self, state, mask):
        self.last_step_started_dash = False
        changed = dict(frame=state.frame+1, prev_input=mask)
        if mask & DOWN:
            return replace(state, **changed, rem_y=state.rem_y+.01)
        duration = round(state.rem_y*100)
        self.attempted.append(duration)
        dead = duration not in (19, 20, 21) or self.reject_neighbor and duration == 19
        return replace(state, **changed, x=20., dead=dead)


class WallTimingVariantTests(unittest.TestCase):
    def test_short_regrip_reaches_exact_exit_lane_under_mirror_and_translation(self):
        for mirror, offset in ((False, (0, 0)), (True, (-512, -320))):
            with self.subTest(mirror=mirror):
                sim, start, goal = exit_fixture(mirror, offset)
                # The original full-jump variants cross at the wrong x; a
                # later in-room move must not count as the selected exit.
                self.assertEqual(static_wall_transfer_edges(sim, start, goal,
                    deadline=time.monotonic()+1, max_expansions=3), [])
                stats = {}
                edges = static_wall_transfer_edges(sim, start, goal,
                    deadline=time.monotonic()+1, max_expansions=6, stats=stats)
                self.assertTrue(edges)
                self.assertLessEqual(stats["expansions"], 6)
                for actions, expected, label in edges:
                    self.assertIn(":held8:", label)
                    state, runs, held = start, [], 0
                    for index, mask in enumerate(actions):
                        held = held+1 if mask & JUMP else held
                        if not mask & JUMP and held:
                            runs.append(held); held = 0
                        self.assertFalse(mask & DASH)
                        state = sim.step(state, mask)
                        self.assertFalse(state.dead)
                        self.assertFalse(sim.last_step_started_dash)
                        if index < len(actions)-1:
                            self.assertEqual(state.room, start.room)
                    self.assertEqual(runs[-1], 8)
                    self.assertTrue(any(length > 8 for length in runs[:-1]))
                    self.assertTrue(goal.achieved(state))
                    self.assertEqual(state, expected)
                    self.assertEqual(state.x, start.x)

    def test_jump_hold_counts_initial_directionless_jump_and_returns_to_grab(self):
        for held in (8, 4):
            state = SimState("a", 0, 0, state=0)
            recipe = _RegripRecipe(1, 0, jump_hold=held)
            inputs = recipe.inputs(None, lambda: state)
            masks = [next(inputs) for _ in range(held+4)]
            self.assertEqual(masks[0], 0)
            self.assertEqual(masks[1], JUMP)
            self.assertEqual(sum(bool(mask & JUMP) for mask in masks), held)
            self.assertEqual(masks[held+1:], [GRAB | UP | RIGHT]*3)

    def refine(self, *, max_edges=4, max_expansions=96, max_frames=180, reject_neighbor=False):
        sim = TimingSimulator(reject_neighbor)
        start = SimState("a", 0., 8., state=1, facing=1, dashes=0, stamina=70.)
        target = Contact("wall_grip", "a", 20., 8.)
        recipes = [((DOWN | RIGHT,)*wait+(GRAB | UP | RIGHT,)*140,
                    f"drop_catch:{wait}:{DOWN | RIGHT}:{GRAB | UP | RIGHT}") for wait in (32, 20)]
        stats = {}
        with patch("celeste_search.wall_underpasses._static_contact", return_value=(4, 0, 12, 80)), \
                patch("celeste_search.wall_underpasses._recipes", return_value=iter(recipes)):
            edges = wall_underpass_edges(sim, start, target, deadline=time.monotonic()+1,
                max_expansions=max_expansions, max_frames=max_frames, max_edges=max_edges, stats=stats)
        return sim, start, target, edges, stats

    def test_refines_only_actual_success_and_retains_original_and_new_family(self):
        sim, start, target, edges, stats = self.refine(max_edges=2)
        self.assertEqual(sim.attempted, [32, 20, 19, 21])
        self.assertEqual(stats["expansions"], 4)
        self.assertEqual(len(edges), 2)
        self.assertEqual({label.split(":", 1)[0] for _, _, label in edges},
                         {"wall_underpass", "wall_underpass_refined"})
        self.assertEqual({round(state.rem_y*100) for _, state, _ in edges}, {19, 20})
        for actions, expected, _ in edges:
            state = start
            for mask in actions:
                self.assertFalse(mask & DASH)
                state = sim.step(state, mask)
                self.assertFalse(state.dead)
            self.assertTrue(target.achieved(state))
            self.assertEqual(state, expected)
            self.assertEqual(state.stamina, start.stamina)
            self.assertEqual(state.dashes, 0)

    def test_one_slot_preserves_original_and_refinement_obeys_caller_constraints(self):
        _, _, _, edges, _ = self.refine(max_edges=1)
        self.assertEqual(len(edges), 1)
        self.assertTrue(edges[0][2].startswith("wall_underpass:"))
        _, _, _, edges, _ = self.refine(reject_neighbor=True)
        self.assertNotIn(19, {round(state.rem_y*100) for _, state, _ in edges})
        self.assertTrue(any(label.startswith("wall_underpass_refined:") for _, _, label in edges))

    def test_refinement_never_exceeds_budget_or_invents_truncated_contact(self):
        sim, _, _, edges, stats = self.refine(max_expansions=2)
        self.assertEqual(stats["expansions"], 2)
        self.assertEqual(sim.attempted, [32, 20])
        self.assertTrue(all(label.startswith("wall_underpass:") for _, _, label in edges))
        sim, _, _, edges, stats = self.refine(max_frames=10)
        self.assertEqual(edges, [])
        self.assertEqual(sim.attempted, [])
        self.assertEqual(stats, {"expansions": 1, "simulated_frames": 10})


if __name__ == "__main__":
    unittest.main()
