"""Previously reached ground remains a real, stateful continuation frontier."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal, plan_to_route, target_reached
from celeste_search.assist_routes import (
    advance_route_progress, normalize_route, route_complete, start_route_progress,
)
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, JUMP, LEFT, RIGHT, SimState, Simulator
from celeste_search.regional_planning import search_regional_routes


def platform_chain(*, turning=False, clock=False):
    positions = (128, 64, 0, 64, 128, 192) if turning else (0, 64, 128, 192, 256)
    first_y = 192 if turning else 176
    width = 288 if turning else 352
    platforms = tuple(Entity("jumpThru", index + 1, x, first_y - 24 * index, 48, 8)
                      for index, x in enumerate(positions))
    extras = ()
    if clock:
        extras = (Entity("zipMover", 91, width - 40, 16, 16, 16, {}, ((width - 24., 16.),)),
                  Entity("refill", 92, width - 16, 208, 0, 0))
    room = Room("relay-room", 0, 0, width, 224, ("0" * (width // 8),) * 28,
                platforms + extras, ((positions[0] + 16., float(first_y)),))
    data = MapData("ground-frontier", {room.name: room}, Path("ground-frontier.bin"))
    simulator = Simulator(data)
    state = simulator.initial_state(room.name)
    if clock:
        state = replace(state, frame=73, stamina=53., rem_x=.2, rem_y=.1,
                        refill_timers=(("relay-room:92", 17),),
                        mechanisms=(("relay-room:91", 3, 7, width - 24., 16., 0.),))
    target = dict(room=room.name, x=positions[-1] + 32, y=first_y - 24 * (len(positions) - 1) - 8)
    return data, simulator, state, target, positions


def known_no_dash_trace(data, start, positions):
    """An independent physical witness: walk to each edge, then jump normally."""
    simulator = Simulator(data)
    state = start
    actions, landings = [], []

    def step(mask):
        nonlocal state
        state = simulator.step(state, mask)
        actions.append(mask)
        if state.dead:
            raise AssertionError("Physical witness died")

    for index in range(len(positions) - 1):
        direction = RIGHT if positions[index + 1] > positions[index] else LEFT
        launch = positions[index] + (44 if direction == RIGHT else 4)
        for _ in range(80):
            if (state.x >= launch if direction == RIGHT else state.x <= launch):
                break
            step(direction)
        airborne = False
        for tick in range(70):
            step(direction | (JUMP if tick < 16 else 0))
            airborne |= not state.on_ground
            if airborne and state.on_ground:
                landings.append(state)
                break
        if len(landings) != index + 1:
            raise AssertionError("Physical witness did not land on the next platform")
    for _ in range(40):
        if positions[-1] + 32 <= state.x < positions[-1] + 40:
            break
        step(RIGHT)
    return actions, state, landings


class GroundFrontierIntegrationTests(unittest.TestCase):
    def assert_replay(self, data, start, route, actions, final, phase=None):
        simulator = Simulator(data)
        state = start
        progress = start_route_progress(route, start)
        landings = []
        for mask in actions:
            before = state
            state = simulator.step(state, mask)
            progress = advance_route_progress(route, progress, before, state, mask)
            self.assertFalse(state.dead)
            self.assertIsNone(progress.violation)
            if route["segments"][0].get("no_dash"):
                self.assertFalse(mask & DASH)
                self.assertFalse(getattr(simulator, "last_step_started_dash", False))
                self.assertNotIn(state.state, (2, 5))
            if not before.on_ground and state.on_ground:
                landings.append(state)
        self.assertEqual(state, final, "Every timer, remainder and mechanism record must survive stitching")
        self.assertEqual(state.frame, start.frame + len(actions))
        if phase is not None:
            self.assertEqual(progress, phase)
        return progress, landings

    def run_regional(self, data, simulator, start, target, *, limits=None, expansions=3500, seconds=12):
        route = normalize_route(data, dict(waypoints=[target], segments=[limits or dict(no_dash=True)]))
        goals = [_selected_goal(point)[0] for point in route["waypoints"]]
        result = search_regional_routes(simulator, start, route, goals,
            deadline=time.monotonic() + seconds, max_expansions=expansions, max_frames=700)
        self.assertLessEqual(result[3], expansions)
        progress, landings = self.assert_replay(data, start, route, result[0], result[1], result[5])
        self.assertEqual(route_complete(route, progress), result[2])
        return result, route, landings

    def test_real_turning_chain_requires_retaining_ground_away_from_public_goal(self):
        data, simulator, start, target, positions = platform_chain(turning=True)
        witness, witnessed_final, witnessed_landings = known_no_dash_trace(data, start, positions)
        self.assertTrue(target_reached(witnessed_final, target))
        self.assertEqual([state.y for state in witnessed_landings], [168., 144., 120., 96., 72.])
        self.assertFalse(any(mask & DASH for mask in witness))
        result, route, landings = self.run_regional(data, simulator, start, target)
        self.assertTrue(result[2], result[6])
        self.assertTrue(target_reached(result[1], route["waypoints"][0]))
        self.assertGreater(result[6]["ground_frontier"]["expanded"], 0)
        self.assertGreaterEqual(len({state.y for state in landings}), 4)
        self.assertLess(min(state.x for state in landings), 48,
                        "The route must retain the far-left relay before coming back right")
        self.assertGreater(result[1].x, start.x)

    def test_public_planner_replays_multiple_platforms_from_exact_original_state(self):
        data, simulator, start, target, _ = platform_chain(clock=True)
        original = start.to_dict()
        route = dict(waypoints=[target], segments=[dict(no_dash=True)])
        with patch("celeste_search.assist_planning._context", return_value=(data, simulator)):
            result = plan_to_route(str(data.source), original, route,
                strategy="ordered_regional", seconds=12, expansions=3500, max_attempts=1)
        self.assertTrue(result["can_execute"], result)
        self.assertEqual(SimState.from_dict(result["anchor_state"]), start)
        normalized = normalize_route(data, route)
        progress, landings = self.assert_replay(data, start, normalized, result["actions"],
                                               SimState.from_dict(result["final_state"]))
        self.assertTrue(route_complete(normalized, progress))
        self.assertGreaterEqual(len({state.y for state in landings}), 3)
        self.assertEqual(start.to_dict(), original)
        self.assertNotEqual(SimState.from_dict(result["final_state"]).mechanisms, start.mechanisms)
        self.assertEqual(SimState.from_dict(result["final_state"]).refill_timers, ())

    def test_no_ground_and_no_dash_remain_strict_across_discovered_relay_platforms(self):
        data, simulator, start, target, _ = platform_chain()
        result, route, landings = self.run_regional(data, simulator, start, target,
            limits=dict(no_ground=True, no_dash=True), expansions=500, seconds=4)
        self.assertFalse(result[2], "Relay landings cannot bypass a public no-ground segment")
        self.assertEqual(result[5].index, 0)
        self.assertFalse(landings)
        self.assertFalse(route_complete(route, result[5]))

    def test_buffered_dash_is_not_hidden_by_a_no_dash_ground_prefix(self):
        data, simulator, start, target, _ = platform_chain()
        start = replace(start, dash_buffer=3, dash_cooldown=0, prev_input=0)
        result, _, _ = self.run_regional(data, simulator, start, target,
            limits=dict(no_dash=True), expansions=250, seconds=3)
        self.assertFalse(result[2])
        self.assertEqual(result[0], [])
        self.assertEqual(result[1], start)


if __name__ == "__main__":
    unittest.main()
