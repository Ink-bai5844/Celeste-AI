"""Private macro output ranking must not depend on wall-clock slice speed."""
from contextlib import ExitStack
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.assist_routes import normalize_route, route_complete
from celeste_search.corridor_planning import search_corridor_routes
from celeste_search.macro_continuation import MACRO_EXPANSION
from celeste_search.physics import Simulator, RIGHT
from celeste_search.planner import Goal, SearchResult
from test_corridor_atomic_macro_quantum import fixture, corridor


class StableMacroBatchTests(unittest.TestCase):
    def exercise(self, *, slow=True, public=False, exhausted=False, timeout=False):
        data = fixture()
        sim = Simulator(data)
        start = sim.initial_state()
        clock, events, suffixes = [100.], [], []
        route = normalize_route(data, dict(room="a", x=40 if public else 280,
                                          y=136 if public else 0))
        goal = _selected_goal(route["waypoints"][0])[0]
        portal = Goal("refill", "a", 44, 144, id="ranked")
        candidates = [corridor("ranked", portal)]
        if not public:
            candidates.append(corridor("neighbor", Goal("corridor", "a", 180, 8, id="neighbor")))

        def search(wrapped, origin, target, **kwargs):
            events.append(("search", target.id, origin.x))
            if origin != start:
                suffixes.append(origin)
            # Keep the neighbor's graph cursor pending through multiple turns.
            reason = "time_budget" if target.id == "neighbor" else "frontier_exhausted"
            return SearchResult([], origin, False, 1, 0, reason)

        def macro(wrapped, origin, target, **kwargs):
            handle = kwargs["continuation"]
            stats = kwargs["stats"]
            events.append(("macro", "resume", handle.expansions))
            def work(cursor):
                for number, count in enumerate((22, 24) if public else (14, 20)):
                    yield MACRO_EXPANSION
                    state, actions = origin, [RIGHT] * count
                    for mask in actions:
                        state = wrapped.step(state, mask)
                        stats["simulated_frames"] = stats.get("simulated_frames", 0) + 1
                    self.assertFalse(state.dead)
                    self.assertTrue(target.achieved(state))
                    # The later bounded recipe wins the family's final rank.
                    cursor.results = [(actions, state, "preferred" if number else "provisional")]
                    events.append(("macro", "candidate", number))
                    clock[0] += .2 if slow else .000001
                    yield None
                events.append(("macro", "done", cursor.expansions))
                return cursor.results
            return handle.advance((id(wrapped), origin, target), work,
                deadline=kwargs["deadline"], stats=stats,
                max_expansions=kwargs["max_expansions"], recipe_limit=2)

        with ExitStack() as stack:
            stack.enter_context(patch("celeste_search.corridor_planning.time.monotonic", side_effect=lambda:clock[0]))
            stack.enter_context(patch("celeste_search.corridor_planning.generate_corridor_routes", return_value=candidates))
            stack.enter_context(patch("celeste_search.corridor_planning.search_leg", side_effect=search))
            stack.enter_context(patch("celeste_search.refill_moves.refill_approach_edges", side_effect=macro))
            result = search_corridor_routes(sim, start, route, [goal],
                deadline=100.2 if timeout else 105., max_expansions=2 if exhausted else 300,
                max_frames=200)
        replay = start
        for mask in result[0]:
            replay = sim.step(replay, mask)
            self.assertFalse(replay.dead)
        self.assertEqual(replay, result[1])
        return result, route, events, suffixes, sim, start

    def test_slow_slices_do_not_send_a_provisional_private_arrival_into_deep_suffix(self):
        slow = self.exercise(slow=True)
        fast = self.exercise(slow=False)
        self.assertTrue(slow[3])
        self.assertTrue(fast[3])
        preferred = slow[5]
        for _ in range(20):
            preferred = slow[4].step(preferred, RIGHT)
        self.assertEqual(slow[3][0], preferred)
        self.assertEqual(fast[3][0], preferred)
        self.assertEqual(slow[3][0], fast[3][0])

    def test_an_unfinished_private_family_still_yields_to_neighboring_corridors(self):
        _, _, events, _, _, _ = self.exercise(slow=True)
        partial = events.index(("macro", "candidate", 0))
        done = events.index(("macro", "done", 2))
        self.assertTrue(any(event[:2] == ("search", "neighbor")
                            for event in events[partial + 1:done]), events)

    def test_actual_public_hit_is_delivered_before_family_finishes(self):
        result, route, events, _, _, _ = self.exercise(public=True)
        self.assertTrue(result[2])
        self.assertTrue(route_complete(route, result[5]))
        self.assertNotIn(("macro", "candidate", 1), events)
        self.assertNotIn(("macro", "done", 2), events)

    def test_last_paid_public_result_survives_both_expansion_and_deadline_end(self):
        for budget in (dict(exhausted=True), dict(timeout=True)):
            with self.subTest(budget=budget):
                result, route, events, _, _, _ = self.exercise(public=True, **budget)
                self.assertTrue(result[2])
                self.assertTrue(route_complete(route, result[5]))
                self.assertEqual(result[3], 2)
                self.assertNotIn(("macro", "candidate", 1), events)


if __name__ == "__main__":
    unittest.main()
