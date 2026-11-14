"""Search behavior against the offline physics model, never the game process."""

from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest

from celeste_search.cli import _target
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, JUMP, RIGHT, Simulator
from celeste_search.planner import Goal, Heuristic, coverage_goals, plan_campaign, search_leg, secret_goals, state_key
from celeste_search.routes import SCHEMA, decode_actions, encode_actions, load_route, snapshot, trace_route, write_json


def synthetic_map(*, berries=(), gap=False, width=16):
    floor = "111110000" + "1" * (width - 9) if gap else "1" * width
    rows = tuple("0" * width for _ in range(8)) + (floor,) * 2
    room = Room("a", 0, 0, width * 8, 80, rows, tuple(berries), ((16.0, 64.0),))
    return MapData("fixture", {"a": room}, Path("fixture.bin"))


def leg(simulator, goal, *, expansions=2000, frames=240):
    return search_leg(simulator, simulator.initial_state(), goal,
                      deadline=time.monotonic()+4, max_expansions=expansions, max_frames=frames)


class PlannerBehaviorTests(unittest.TestCase):
    def test_search_crosses_gap_and_result_survives_replay(self):
        data = synthetic_map(gap=True)
        simulator = Simulator(data)
        goal = Goal("explore", "a", 108, 64)
        result = leg(simulator, goal)
        self.assertTrue(result.reached, result.reason)
        self.assertTrue(any(mask & (JUMP | DASH) for mask in result.actions))
        replay, _, _, _ = trace_route(simulator, "a", result.actions)
        self.assertEqual(replay, result.state)
        self.assertTrue(goal.achieved(replay))

    def test_same_physical_anchor_resumes_frontier_across_frame_counters(self):
        simulator=Simulator(synthetic_map(width=64))
        start=simulator.initial_state()
        goal=Goal("surface","a",300,64)
        first=search_leg(simulator,start,goal,deadline=time.monotonic()+2,
                         max_expansions=1,max_frames=300)
        later=replace(start,frame=100)
        second=search_leg(simulator,later,goal,deadline=time.monotonic()+2,
                          max_expansions=1,max_frames=300)
        self.assertGreater(second.state.x,first.state.x)
        replay=later
        for mask in second.actions:
            replay=simulator.step(replay,mask)
        self.assertEqual(second.state,replay)

    def test_surface_goal_requires_support_not_flying_nearby(self):
        state=Simulator(synthetic_map()).initial_state()
        goal=Goal("surface","a",state.x,state.y)
        self.assertTrue(goal.achieved(state))
        self.assertFalse(goal.achieved(replace(state,on_ground=False)))

    def test_coverage_includes_short_ledge_between_old_grid_samples(self):
        data=synthetic_map()
        rows=list(data.rooms["a"].solids)
        rows[4]="0010000000000000"
        room=replace(data.rooms["a"],solids=tuple(rows))
        self.assertIn((20,32),[(g.x,g.y) for g in coverage_goals(room)])

    def test_only_dash_blocks_are_break_goals_visual_walls_are_regions(self):
        data=synthetic_map(berries=[Entity("dashBlock",1,32,32,8,32),
             Entity("fakeWall",2,80,32,24,32),Entity("coverupWall",3,112,32,16,32)])
        goals=secret_goals(data.rooms["a"])
        self.assertEqual([g.kind for g in goals],["break","hidden","hidden"])
        state=Simulator(data).initial_state()
        self.assertFalse(goals[0].achieved(replace(state,x=36,y=53)))
        self.assertTrue(goals[0].achieved(replace(state,broken=frozenset({"a:1"}))))

    def test_carried_berry_returns_to_spike_free_support(self):
        data=synthetic_map(berries=[Entity("spikesUp",7,48,64,48,0)])
        state=replace(Simulator(data).initial_state(),x=72,y=40,
                      following=frozenset({"a:5"}))
        x,y=Heuristic(data,Goal("berry","a",72,35,"a:5")).waypoint(state)
        self.assertFalse(43<x<101 and y==64)

    def test_search_collects_aerial_berry_and_returns_to_safe_ground(self):
        data = synthetic_map(berries=[Entity("strawberry", 5, 64, 35, 0, 0)])
        simulator = Simulator(data)
        result = leg(simulator, Goal("berry", "a", 64, 35, "a:5"))
        self.assertTrue(result.reached, result.reason)
        self.assertIn("a:5", result.state.berries)
        self.assertNotIn("a:5", result.state.following)
        replay, _, _, _ = trace_route(simulator, "a", result.actions)
        self.assertEqual(replay, result.state)

    def test_room_goal_requires_physical_transition(self):
        data = synthetic_map()
        other = replace(data.rooms["a"], name="b", x=128, spawns=((144.0, 64.0),))
        data = replace(data, rooms={"a": data.rooms["a"], "b": other})
        simulator = Simulator(data)
        result = leg(simulator, Goal("room", "b", 144, 64))
        self.assertTrue(result.reached, result.reason)
        replay, _, rooms, _ = trace_route(simulator, "a", result.actions)
        self.assertEqual(replay.room, "b")
        self.assertEqual(rooms, {"a", "b"})

    def test_serialized_route_keeps_each_frame_and_offline_result(self):
        simulator = Simulator(synthetic_map(gap=True))
        result = leg(simulator, Goal("explore", "a", 108, 64))
        self.assertTrue(result.reached)
        artifact = {"schema": SCHEMA, "frames": len(result.actions),
                    "actions": encode_actions(result.actions), "runtime_verified": False,
                    "predicted": {"state": snapshot(result.state)}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "route.json"
            write_json(path, artifact)
            restored = load_route(path)
        actions = decode_actions(restored["actions"])
        self.assertEqual(actions, result.actions)
        state, _, _, _ = trace_route(simulator, "a", actions)
        self.assertEqual(snapshot(state), restored["predicted"]["state"])
        self.assertFalse(restored["runtime_verified"])

    def test_unsupported_mechanism_is_explicitly_skipped(self):
        data = synthetic_map(berries=[Entity("unmodeledFixtureMechanism", 1, 80, 32, 16, 16)])
        result = plan_campaign(Simulator(data), data, "a", seconds=1,
                               target=Goal("explore", "a", 108, 64))
        self.assertEqual(result["actions"], [])
        self.assertEqual(result["legs"], [])
        self.assertTrue(result["skipped"])
        self.assertEqual(result["skipped"][0]["reason"], "unsupported_mechanics")

    def test_resume_extends_existing_sequence_and_replays_identically(self):
        data = synthetic_map(width=32)
        prefix = [RIGHT] * 24
        simulator = Simulator(data)
        result = plan_campaign(simulator, data, "a", seconds=2,
                               target=Goal("explore", "a", 160, 64), prefix=prefix)
        self.assertEqual(result["actions"][:len(prefix)], prefix)
        self.assertGreater(len(result["actions"]), len(prefix))
        self.assertTrue(result["legs"][0]["reached"])
        replay, _, _, _ = trace_route(simulator, "a", result["actions"])
        self.assertEqual(replay, result["state"])

    def test_resume_rejects_a_prefix_that_dies(self):
        data = synthetic_map(berries=[Entity("spikesUp", 1, 8, 64, 24, 0)])
        with self.assertRaisesRegex(ValueError, "survive"):
            plan_campaign(Simulator(data), data, "a", seconds=1, prefix=[0])

    def test_reverse_bfs_respects_directed_room_adjacency(self):
        base = synthetic_map().rooms["a"]
        upper = replace(base, name="upper", y=-80, spawns=((16.0, -16.0),),
                        attrs={"disableDownTransition": True})
        lower = replace(base, name="lower")
        data = MapData("fixture", {"upper": upper, "lower": lower}, Path("fixture.bin"))
        upward = Heuristic(data, Goal("room", "upper", 16, -16))
        self.assertEqual(upward.parents.get("lower"), "upper")
        downward = Heuristic(data, Goal("room", "lower", 16, 64))
        self.assertNotIn("upper", downward.parents)

    def test_search_key_retains_collection_and_mechanism_progress(self):
        state = Simulator(synthetic_map()).initial_state()
        carrying = replace(state, following=frozenset({"a:5"}), follow_order=("a:5",), collect_delay=18)
        self.assertNotEqual(state_key(carrying), state_key(replace(carrying, collect_delay=2)))
        self.assertNotEqual(state_key(carrying), state_key(replace(carrying, collect_ground=8)))
        self.assertNotEqual(state_key(state), state_key(replace(state, broken=frozenset({"a:2"}))))
        moving = replace(state, mechanisms=(("a:3", 1, 5, 40.0, 40.0, 0.0),))
        later = replace(state, mechanisms=(("a:3", 1, 10, 48.0, 40.0, 0.0),))
        self.assertNotEqual(state_key(moving), state_key(later))

    def test_budget_failure_remains_pending_and_is_never_labeled_reached(self):
        data = synthetic_map(width=64)
        goal = Goal("explore", "a", 480, 64)
        result = plan_campaign(Simulator(data), data, "a", seconds=1,
                               expansions=1, target=goal)
        self.assertTrue(result["legs"])
        self.assertFalse(result["legs"][0]["reached"])
        self.assertIn(goal.key, result["pending_goals"])
        self.assertEqual(result["scope"], "offline_prediction_only")

    def test_target_rejects_non_finite_point_and_room_without_spawn(self):
        data = synthetic_map()
        dummy = replace(data.rooms["a"], name="dummy", spawns=())
        data = replace(data, rooms={"a": data.rooms["a"], "dummy": dummy})
        for text in ("point:nan:0", "point:0:inf", "room:dummy"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                _target(text, data, "a")


if __name__ == "__main__":
    unittest.main()
