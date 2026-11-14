"""Contact diversity and paid-jump reserves guide search without authorizing it."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import math
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import RouteProgress, route_complete
from celeste_search.maps import Entity, MapData, Room
from celeste_search.mechanism_planning import _Node, _trim_frontier, search_mechanism_routes
from celeste_search.physics import GRAB, JUMP, LEFT, RIGHT, SimState, Simulator
from celeste_search.planner import Goal, SearchResult
from celeste_search.search_resources import grip_resource_band


class GripResourceBandTests(unittest.TestCase):
    def test_paid_jump_threshold_separates_68_and_78_and_both_sides_of_75(self):
        self.assertEqual(grip_resource_band(68), (3, 1))
        self.assertEqual(grip_resource_band(78), (3, 2))
        below, above = math.nextafter(75., -math.inf), math.nextafter(75., math.inf)
        self.assertEqual(grip_resource_band(below), (3, 1))
        self.assertEqual(grip_resource_band(75.), (3, 2))
        self.assertEqual(grip_resource_band(above), (3, 2))
        self.assertEqual([grip_resource_band(value)[1] for value in (19., 20., 47.5, 102.5)],
                         [0, 0, 1, 3])

    def test_real_two_climb_jumps_leave_13_or_23_and_only_latter_can_regrab(self):
        room = Room("reserve", 0, 0, 160, 288, ("0"*20,)*36,
                    (Entity("solid", 1, 64, 0, 16, 272),), ())
        sim = Simulator(MapData("reserve", {"reserve": room}, Path("reserve.bin")))
        caught = sim.step(SimState("reserve", 59, 240, vx=90, facing=1), GRAB | RIGHT)
        self.assertEqual(caught.state, 1)
        for stamina, remaining, regrab in ((68., 13., False), (78., 23., True)):
            with self.subTest(stamina=stamina):
                state = replace(caught, stamina=stamina)
                for index in range(2):
                    state = sim.step(state, GRAB | RIGHT | JUMP)
                    self.assertEqual(state.stamina, stamina-(index+1)*27.5)
                    for _ in range(8):
                        state = sim.step(state, GRAB | RIGHT)
                    if index == 0:
                        self.assertEqual(state.state, 1)
                self.assertEqual(state.stamina, remaining)
                self.assertEqual(state.state == 1, regrab)
                self.assertFalse(state.dead)
        # Zero reserve is not a ban on jumping: a caught player can perform
        # a free outward wall jump, and geometry still decides its result.
        tired = replace(caught, stamina=10.)
        outward = sim.step(tired, GRAB | LEFT | JUMP)
        self.assertEqual(grip_resource_band(tired.stamina)[1], 0)
        self.assertEqual(outward.stamina, tired.stamina)
        self.assertEqual(outward.state, 0)
        self.assertLess(outward.vx, 0)
        self.assertLess(outward.vy, 0)

    def test_zero_paid_jump_reserve_is_not_a_planner_reachability_rejection(self):
        class FreeExitSimulator:
            map_data = None
            _tiles = {}
            last_step_started_dash = False

            def step(self, state, action):
                x = {(0., RIGHT): 20., (20., LEFT): 84.}.get((state.x, action))
                return replace(state, x=state.x if x is None else x, frame=state.frame+1,
                               prev_input=action, state=1 if state.x == 0 else 0,
                               dead=x is None)

        sim = FreeExitSimulator()
        start = SimState("a", 0., 8., state=1, stamina=10., on_ground=False)
        private = Goal("wall_grip", "a", 20., 8., id="free_contact")
        route = dict(waypoints=[dict(room="a", x=80, y=0, tile_size=8)],
                     segments=[dict(no_dash=True, no_ground=False)])
        public = _selected_goal(route["waypoints"][0])[0]

        def edge(wrapped, origin, target, **kwargs):
            return [([RIGHT], wrapped.step(origin, RIGHT), "free_contact")] if origin.x == 0 else []

        def suffix(wrapped, origin, target, **kwargs):
            if origin.x != 20:
                return SearchResult([], origin, False, 1, 0, "no_direct")
            after = wrapped.step(origin, LEFT)
            return SearchResult([LEFT], after, target.achieved(after), 1, 1, "free_wall_exit")

        with ExitStack() as stack:
            stack.enter_context(patch("celeste_search.mechanism_routes.relay_candidates",
                side_effect=lambda s, state, *a, **k: [private] if state.x == 0 else []))
            stack.enter_context(patch("celeste_search.mechanism_routes.relay_priority", return_value=0.))
            stack.enter_context(patch("celeste_search.mechanism_planning.search_leg", side_effect=suffix))
            stack.enter_context(patch("celeste_search.mechanism_moves.mechanism_transfer_edges", side_effect=edge))
            for path in ("wall_transfers.static_wall_transfer_edges", "wall_underpasses.wall_underpass_edges",
                         "wall_descents.wall_descent_edges", "wall_moves.wall_approach_edges",
                         "zip_grips.zip_grip_edges"):
                stack.enter_context(patch("celeste_search."+path, return_value=[]))
            result = search_mechanism_routes(sim, start, route, [public],
                deadline=time.monotonic()+1, max_expansions=20, max_frames=8, use_detours=False)
        self.assertTrue(result[2])
        self.assertEqual(result[0], [RIGHT, LEFT])
        self.assertTrue(route_complete(route, result[5]))
        self.assertEqual(result[1], sim.step(sim.step(start, RIGHT), LEFT))
        self.assertEqual(result[1].stamina, 10.)


class MechanismFrontierDiversityTests(unittest.TestCase):
    def node(self, *, contact="zip-a", timer=0, stamina=68., facing=1, phase=0):
        state = SimState("a", 24., 64., state=1, stamina=stamina, facing=facing,
                         mechanisms=((contact, 2, timer, 28., 48., 0.),))
        return _Node(state, [], RouteProgress(phase), contact_key=contact,
                     contact_identity=contact)

    def test_forty_cheap_timing_variants_cannot_erase_another_actual_contact(self):
        nodes = [self.node(timer=index*8) for index in range(40)]
        nodes.append(self.node(contact="zip-b"))
        frontier = [(float(index), index) for index in range(40)] + [(1000., 40)]
        frontier.reverse()
        original_frontier, original_nodes = list(frontier), deepcopy(nodes)
        selected = _trim_frontier(frontier, nodes)
        self.assertEqual(len(selected), 32)
        self.assertIn((0., 0), selected)
        self.assertIn((1000., 40), selected)
        self.assertEqual(len(selected), len(set(selected)))
        self.assertTrue(set(selected) <= set(frontier))
        self.assertEqual(frontier, original_frontier)
        self.assertEqual(nodes, original_nodes)

    def test_capacity_facing_and_public_phase_survive_duplicate_low_cost_states(self):
        nodes = [self.node() for _ in range(20)]
        nodes.extend((self.node(stamina=78.), self.node(facing=-1),
                      self.node(phase=1), self.node(timer=8)))
        frontier = [(float(index), index) for index in range(20)] + [
            (50., 20), (60., 21), (70., 22), (80., 23)]
        selected = _trim_frontier(frontier, nodes, width=5)
        self.assertEqual(set(selected), {(0., 0), (50., 20), (60., 21), (70., 22), (80., 23)})
        # Scores and node references retain their original meaning. Selection
        # may round-robin groups; it must not rewrite the heap's cost values.
        self.assertTrue(all(entry == frontier[entry[1]] for entry in selected))

    def test_positive_widths_bound_output_without_duplicate_or_fabricated_entries(self):
        nodes = [self.node(contact=f"zip-{index%4}", timer=(index//4)*8)
                 for index in range(50)]
        frontier = [(float(index), index) for index in range(len(nodes))]
        for width in (1, 2, 5, 16, 32, 64):
            with self.subTest(width=width):
                selected = _trim_frontier(frontier, nodes, width=width)
                self.assertLessEqual(len(selected), width)
                self.assertEqual(len(selected), min(width, len(frontier)))
                self.assertEqual(len(selected), len(set(selected)))
                self.assertTrue(set(selected) <= set(frontier))
        self.assertEqual(_trim_frontier([], nodes), [])


if __name__ == "__main__":
    unittest.main()
