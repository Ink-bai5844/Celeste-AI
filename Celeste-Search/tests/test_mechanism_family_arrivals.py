"""A bounded arrival beam admits later macro families and replays their inputs."""
from contextlib import ExitStack
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import advance_route_progress, route_complete, start_route_progress
from celeste_search.mechanism_planning import search_mechanism_routes
from celeste_search.physics import DASH, DOWN, GRAB, JUMP, LEFT, RIGHT, UP, SimState
from celeste_search.planner import Goal, SearchResult


@dataclass(frozen=True)
class ContactGoal(Goal):
    identity: str = "a:zip"
    mechanism_kind: str = "zipMover"

    def achieved(self, state):
        return not state.dead and state.room == self.room and state.x == 20 and state.state == 1


class FamilySimulator:
    map_data = None
    _tiles = {}  # Enable the static-wall macro family in the integration path.
    last_step_started_dash = False

    def __init__(self, winning_action=JUMP, fake_contact=False):
        self.winning_action = winning_action
        self.fake_contact = fake_contact

    def step(self, state, action):
        self.last_step_started_dash = bool(action & DASH)
        changed = dict(frame=state.frame+1, prev_input=action, stamina=state.stamina-2)
        if state.x == 0 and action in (LEFT, RIGHT, UP, DOWN):
            timer = (LEFT, RIGHT, UP, DOWN).index(action)*8
            return replace(state, **changed, x=20., state=1,
                           mechanisms=(("a:zip", 1, timer, 12., 0., 0.),))
        if state.x == 0 and action == self.winning_action:
            return replace(state, **changed, x=4. if self.fake_contact else 20., state=1,
                           dashes=0 if action & DASH else state.dashes,
                           mechanisms=(("a:zip", 2, 9, 12., 0., 0.),))
        if state.x == 20 and state.mechanisms[0][1] == 2 and action == GRAB:
            return replace(state, **changed, x=84., state=0)
        return replace(state, **changed, dead=True, reason="fixture_no_continuation")


class MechanismFamilyArrivalTests(unittest.TestCase):
    def solve(self, simulator, *, no_dash=False):
        start = SimState("a", 0., 8., state=1, frame=91, stamina=83.,
                         dashes=1, rem_x=.125, rem_y=-.25,
                         mechanisms=(("a:zip", 0, 0, 12., 0., 0.),))
        route = dict(waypoints=[dict(room="a", x=80, y=0, tile_size=8)],
                     segments=[dict(no_dash=no_dash, no_ground=False)])
        original = deepcopy(route)
        public = _selected_goal(route["waypoints"][0])[0]
        private = ContactGoal("wall_grip", "a", 20., 8., id="private_zip")
        offered, rides = [], []

        def first_family(sim, origin, target, **kwargs):
            if origin.x != 0 or target.key != private.key:
                return []
            edges = []
            for action in (LEFT, RIGHT, UP, DOWN):
                state = sim.step(origin, action)
                edges.append(([action], state, f"mechanism_transfer:choice:{action}"))
            offered.append(("first", sim.no_dash, len(edges)))
            return edges

        def later_family(sim, origin, target, **kwargs):
            if origin.x != 0 or target.key != private.key:
                return []
            actual = sim.step(origin, simulator.winning_action)
            offered.append(("later", sim.no_dash, 0 if actual.dead else 1))
            if actual.dead:
                return []
            # Deliberately lie only in the negative test. The planner must
            # replay actions and reject x=4 instead of trusting claimed x=20.
            claimed = replace(actual, x=20.) if simulator.fake_contact else actual
            return [([simulator.winning_action], claimed, "static_wall_transfer:only_continuation")]

        def ride(sim, origin, portal, target, **kwargs):
            rides.append(origin)
            return []

        def search(sim, origin, target, **kwargs):
            if origin.x == 20 and origin.mechanisms[0][1] == 2:
                state = sim.step(origin, GRAB)
                return SearchResult([GRAB], state, target.achieved(state), 1, 1, "fixture")
            return SearchResult([], origin, False, 1, 0, "no_continuation")

        with ExitStack() as stack:
            stack.enter_context(patch("celeste_search.mechanism_routes.relay_candidates",
                                      side_effect=lambda sim, state, *a, **k: [private] if state.x == 0 else []))
            stack.enter_context(patch("celeste_search.mechanism_routes.relay_priority", return_value=0.))
            stack.enter_context(patch("celeste_search.mechanism_moves.mechanism_transfer_edges",
                                      side_effect=first_family))
            stack.enter_context(patch("celeste_search.wall_transfers.static_wall_transfer_edges",
                                      side_effect=later_family))
            stack.enter_context(patch("celeste_search.wall_underpasses.wall_underpass_edges", return_value=[]))
            stack.enter_context(patch("celeste_search.wall_descents.wall_descent_edges", return_value=[]))
            stack.enter_context(patch("celeste_search.wall_moves.wall_approach_edges", return_value=[]))
            stack.enter_context(patch("celeste_search.zip_grips.zip_grip_edges", side_effect=ride))
            stack.enter_context(patch("celeste_search.mechanism_planning.search_leg", side_effect=search))
            answer = search_mechanism_routes(simulator, start, route, [public],
                deadline=time.monotonic()+2, max_expansions=60, max_frames=8)
        self.assertEqual(route, original)
        return start, route, answer, offered, rides

    def assert_complete_replay(self, simulator, start, route, answer):
        self.assertTrue(answer[2])
        state, progress = start, start_route_progress(route, start)
        for action in answer[0]:
            before, state = state, simulator.step(state, action)
            progress = advance_route_progress(route, progress, before, state, action)
            self.assertFalse(state.dead)
            self.assertIsNone(progress.violation)
        self.assertTrue(route_complete(route, progress))
        self.assertEqual(state, answer[1])
        self.assertEqual(progress, answer[5])

    def test_later_family_reaches_ride_and_suffix_after_first_family_offers_four_distinct_arrivals(self):
        simulator = FamilySimulator()
        # One conservative pass makes family fairness necessary: the later
        # generator cannot be rescued accidentally by a second relaxed pass.
        start, route, answer, offered, rides = self.solve(simulator, no_dash=True)
        self.assertIn(("first", True, 4), offered)
        self.assertIn(("later", True, 1), offered)
        self.assertTrue(all(conservative for _, conservative, _ in offered))
        self.assertEqual(len(set(rides)), 4)
        self.assertTrue(all(count == 2 for count in Counter(rides).values()))
        self.assertEqual(answer[0], [JUMP, GRAB])
        actual = simulator.step(start, JUMP)
        self.assertIn(actual, rides)
        self.assertEqual((actual.frame, actual.stamina, actual.rem_x, actual.rem_y),
                         (92, 81., .125, -.25))
        self.assert_complete_replay(simulator, start, route, answer)

    def test_interleaving_cannot_admit_a_fabricated_later_family_contact(self):
        simulator = FamilySimulator(fake_contact=True)
        start, route, answer, offered, _ = self.solve(simulator)
        self.assertIn(("later", True, 1), offered)
        self.assertEqual(simulator.step(start, JUMP).x, 4.)
        self.assertFalse(answer[2])
        self.assertFalse(route_complete(route, answer[5]))
        self.assertNotEqual(answer[1].x, 84.)

    def test_later_dash_family_requires_the_relaxed_pass_and_cannot_escape_no_dash_segment(self):
        permitted = FamilySimulator(winning_action=DASH)
        start, route, answer, offered, _ = self.solve(permitted)
        self.assertIn(("later", True, 0), offered)
        self.assertIn(("later", False, 1), offered)
        self.assertEqual(answer[0], [DASH, GRAB])
        self.assert_complete_replay(permitted, start, route, answer)

        forbidden = FamilySimulator(winning_action=DASH)
        _, route, answer, offered, _ = self.solve(forbidden, no_dash=True)
        self.assertTrue(all(conservative for _, conservative, _ in offered))
        self.assertFalse(answer[2])
        self.assertFalse(route_complete(route, answer[5]))
        self.assertTrue(all(not action & DASH for action in answer[0]))


if __name__ == "__main__":
    unittest.main()
