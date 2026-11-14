"""Keep a fifth launch family and distinct lift resources through the arrival beam."""
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
from celeste_search.physics import DOWN, GRAB, JUMP, LEFT, RIGHT, UP, SimState
from celeste_search.planner import Goal, SearchResult


@dataclass(frozen=True)
class ContactGoal(Goal):
    identity: str = "a:zip"
    mechanism_kind: str = "zipMover"

    def achieved(self, state):
        return not state.dead and state.x == 20 and state.state == 1


class ArrivalSimulator:
    map_data = None
    _tiles = {}
    last_step_started_dash = False

    def __init__(self, resource=None):
        self.resource = resource

    def can_continue(self, state):
        if self.resource is None:
            return state.mechanisms[0][1] == 2
        return getattr(state, self.resource) == {"lift_vx": 130., "lift_vy": -130.,
                                                "lift_timer": 6}[self.resource]

    def step(self, state, action):
        self.last_step_started_dash = False
        changed = dict(frame=state.frame+1, prev_input=action, stamina=state.stamina-1)
        if state.x == 0 and action in (LEFT, RIGHT, UP, DOWN, JUMP):
            phase = 2 if action == JUMP and self.resource is None else 1
            timer = 0 if action == JUMP else (LEFT, RIGHT, UP, DOWN).index(action)*8
            resource = {}
            if self.resource:
                resource[self.resource] = ({"lift_vx": 130., "lift_vy": -130., "lift_timer": 6}
                                           [self.resource] if action == JUMP else 0)
            return replace(state, **changed, **resource, x=20., state=1,
                           mechanisms=(("a:zip", phase, timer, 12., 0., 0.),))
        if state.x == 20 and self.can_continue(state) and action == GRAB:
            return replace(state, **changed, x=84., state=0)
        return replace(state, **changed, dead=True, reason="fixture_missing_launch_resource")


class MechanismArrivalResourceTests(unittest.TestCase):
    def solve(self, simulator, families):
        start = SimState("a", 0., 8., state=1, stamina=100., on_ground=False,
                         mechanisms=(("a:zip", 0, 0, 12., 0., 0.),))
        private = ContactGoal("wall_grip", "a", 20., 8., id="private")
        route = dict(waypoints=[dict(room="a", x=80, y=0, tile_size=8)],
                     segments=[dict(no_dash=True, no_ground=False)])
        original = deepcopy(route)
        public = _selected_goal(route["waypoints"][0])[0]
        calls, rides = [], []

        def emit_family(index, actions):
            def emit(wrapped, origin, target, **kwargs):
                if origin.x != 0 or target.key != private.key:
                    return []
                calls.append((index, wrapped.no_dash))
                return [([action], wrapped.step(origin, action), f"family{index}:choice{j}")
                        for j, action in enumerate(actions)]
            return emit

        def search(wrapped, origin, target, **kwargs):
            if origin.x == 20 and simulator.can_continue(origin):
                after = wrapped.step(origin, GRAB)
                return SearchResult([GRAB], after, target.achieved(after), 1, 1, "fixture")
            return SearchResult([], origin, False, 1, 0, "no_continuation")

        def ride(wrapped, origin, portal, target, **kwargs):
            rides.append(origin)
            return []

        paths = (
            "celeste_search.mechanism_moves.mechanism_transfer_edges",
            "celeste_search.wall_transfers.static_wall_transfer_edges",
            "celeste_search.wall_underpasses.wall_underpass_edges",
            "celeste_search.wall_descents.wall_descent_edges",
            "celeste_search.wall_moves.wall_approach_edges",
        )
        with ExitStack() as stack:
            stack.enter_context(patch("celeste_search.mechanism_routes.relay_candidates",
                                      side_effect=lambda sim, state, *a, **k: [private] if state.x == 0 else []))
            stack.enter_context(patch("celeste_search.mechanism_routes.relay_priority", return_value=0.))
            stack.enter_context(patch("celeste_search.mechanism_planning.search_leg", side_effect=search))
            stack.enter_context(patch("celeste_search.zip_grips.zip_grip_edges", side_effect=ride))
            for index, (path, actions) in enumerate(zip(paths, families)):
                stack.enter_context(patch(path, side_effect=emit_family(index, actions)))
            answer = search_mechanism_routes(simulator, start, route, [public],
                deadline=time.monotonic()+1, max_expansions=60, max_frames=8)
        self.assertEqual(route, original)
        self.assertTrue(all(no_dash for _, no_dash in calls))
        self.assertTrue(answer[2])
        self.assertEqual(answer[0], [JUMP, GRAB])
        state, progress = start, start_route_progress(route, start)
        for action in answer[0]:
            before, state = state, simulator.step(state, action)
            progress = advance_route_progress(route, progress, before, state, action)
            self.assertFalse(state.dead)
            self.assertIsNone(progress.violation)
        self.assertTrue(route_complete(route, progress))
        self.assertEqual(state, answer[1])
        self.assertEqual(progress, answer[5])
        # Each physically distinct contact gets the held and return-climb
        # variants once; that must not duplicate or erase its resource band.
        self.assertTrue(all(count == 2 for count in Counter(rides).values()))
        return start, calls, rides

    def test_fifth_family_survives_four_other_distinct_valid_arrivals(self):
        sim = ArrivalSimulator()
        start, calls, rides = self.solve(sim, [[LEFT], [RIGHT], [UP], [DOWN], [JUMP]])
        self.assertEqual({index for index, _ in calls}, set(range(5)))
        self.assertEqual(len(set(rides)), 5)
        self.assertIn(sim.step(start, JUMP), rides)

    def test_lift_velocity_and_expiry_keep_distinct_arrivals_with_identical_geometry(self):
        for resource in ("lift_vx", "lift_vy", "lift_timer"):
            with self.subTest(resource=resource):
                sim = ArrivalSimulator(resource)
                start, _, rides = self.solve(sim, [[LEFT], [JUMP], [], [], []])
                unboosted, boosted = sim.step(start, LEFT), sim.step(start, JUMP)
                for field in ("x", "y", "vx", "vy", "frame", "state", "facing",
                              "dashes", "stamina", "mechanisms"):
                    self.assertEqual(getattr(unboosted, field), getattr(boosted, field))
                self.assertNotEqual(getattr(unboosted, resource), getattr(boosted, resource))
                self.assertIn(unboosted, rides)
                self.assertIn(boosted, rides)
                self.assertEqual(len(set(rides)), 2)

    def test_paid_jump_capacity_survives_same_twenty_point_stamina_arrival_band(self):
        class StaminaSimulator(ArrivalSimulator):
            def can_continue(self, state):
                return state.stamina >= 75

            def step(self, state, action):
                self.last_step_started_dash = False
                if state.x == 0 and action in (LEFT, JUMP):
                    return replace(state, x=20., state=1, frame=state.frame+1,
                                   prev_input=action, stamina=78. if action == JUMP else 68.,
                                   mechanisms=(("a:zip", 1, 0, 12., 0., 0.),))
                return super().step(state, action)

        sim = StaminaSimulator()
        start, _, rides = self.solve(sim, [[LEFT, JUMP], [], [], [], []])
        low, high = sim.step(start, LEFT), sim.step(start, JUMP)
        self.assertEqual(int(low.stamina/20), int(high.stamina/20))
        self.assertEqual((low.x, low.y, low.state, low.mechanisms),
                         (high.x, high.y, high.state, high.mechanisms))
        self.assertEqual(len(set(rides)), 2)
        self.assertIn(low, rides)
        self.assertIn(high, rides)


if __name__ == "__main__":
    unittest.main()
