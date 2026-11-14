"""Wall relay stages require contact; every launch edge replays physically."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.cassette import CassetteBlockState
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, GRAB, JUMP, LEFT, RIGHT, UP, SimState, Simulator
from celeste_search.planner import Goal
from celeste_search.wall_moves import (WallGripGoal, wall_approach_macros,
    wall_approach_edges, wall_grip_goals, wall_launch_edges)


def fixture(*, facing=1, moving=False):
    wall_x = 80 if facing > 0 else 64
    ledge_x = 16 if facing > 0 else 112
    entities = (Entity("fallingBlock" if moving else "solid", 1, wall_x, 32, 16, 64),
                Entity("solid", 2, ledge_x, 64, 32, 8))
    room = Room("a", 0, 0, 160, 144, ("0"*20,)*16+("1"*20,)*2, entities, ())
    sim = Simulator(MapData("wall-fixture", {"a": room}, Path("fixture.bin")))
    state = SimState("a", wall_x-4 if facing > 0 else wall_x+20, 96,
                     state=1, facing=facing, dashes=0, stamina=110)
    if moving:
        state = replace(state, mechanisms=(("a:1", 1, 0, float(wall_x), 32., 0.),))
    return sim, state


class WallMoveTests(unittest.TestCase):
    def test_approach_edges_preserve_a_no_dash_arrival_and_replay_exactly(self):
        sim, _ = fixture()
        start = SimState("a", 40., 128., on_ground=True, grace=6)
        goal = WallGripGoal("wall_grip", "a", 76., 92.,
                           rect=(80.,32.,96.,96.), facing=1, grip_offset=60.)
        stats={}
        edges=wall_approach_edges(sim,start,goal,deadline=time.monotonic()+1,
                                 max_frames=140,stats=stats,max_edges=2)
        self.assertTrue(edges)
        self.assertTrue(all(not any(mask & DASH for mask in actions)
                            for actions, _, _ in edges))
        for actions, final, _ in edges:
            state=start
            for mask in actions:
                state=sim.step(state,mask)
            self.assertEqual(state,final)
            self.assertTrue(goal.achieved(state))
            self.assertEqual(final.dashes,1)
        self.assertGreater(stats["simulated_frames"],0)

    def test_approach_deadline_and_expansion_limits_do_not_run_actions(self):
        sim,start=fixture()
        goal=WallGripGoal("wall_grip","a",76.,44.,rect=(80.,32.,96.,96.))
        with patch.object(sim,"step",wraps=sim.step) as step:
            self.assertEqual(wall_approach_edges(sim,start,goal,
                deadline=time.monotonic()-1),[])
            self.assertEqual(wall_approach_edges(sim,start,goal,
                deadline=time.monotonic()+1,stats={"expansions":3},max_expansions=3),[])
            step.assert_not_called()

    def test_airborne_wall_underpass_reaches_opposite_wall_with_real_clearance(self):
        room=Room("a",0,0,160,144,("0"*20,)*16+("1"*20,)*2,
            (Entity("solid",1,80,32,16,64),Entity("solid",2,0,0,8,128)),())
        sim=Simulator(MapData("underpass",{"a":room},Path("fixture.bin")))
        start=SimState("a",100.,84.,state=1,facing=-1,dashes=1,stamina=110)
        goal=WallGripGoal("wall_grip","a",12.,60.,rect=(0.,0.,8.,128.),
                          facing=-1,grip_offset=60.)
        edges=wall_approach_edges(sim,start,goal,deadline=time.monotonic()+1,
                                 max_frames=180,max_edges=1)
        self.assertTrue(edges)
        actions,final,label=edges[0]
        self.assertTrue(label.startswith("wall_underpass:"))
        state=start;clearance=False
        for mask in actions:
            state=sim.step(state,mask)
            self.assertFalse(state.dead)
            if 76 < state.x < 100:
                self.assertGreaterEqual(state.y,107.)  # Eleven-pixel body below bottom96.
                clearance=True
        self.assertTrue(clearance)
        self.assertEqual(state,final)
        self.assertTrue(goal.achieved(final))

    def test_thin_static_lip_retains_legal_contact_below_its_bottom(self):
        room = Room("a", 0, 0, 160, 144, ("0"*20,)*16+("1"*20,)*2,
                    (Entity("solid", 1, 80, 32, 48, 8),), ())
        sim = Simulator(MapData("thin-wall", {"a": room}, Path("fixture.bin")))
        start = SimState("a", 40, 128, on_ground=True)
        goal = next(goal for goal in wall_grip_goals(sim, start, Goal("target", "a", 140, 32))
                    if goal.rect == (80, 32, 128, 40) and goal.facing == 1)
        contact = replace(start, x=76., y=48., state=1, facing=1, on_ground=False)
        self.assertTrue(goal.achieved(contact))
        self.assertFalse(goal.achieved(replace(contact, y=51.)))
        self.assertEqual(goal.waypoint(start), (76., 48.))

    def test_cassette_grip_tracks_live_shift_and_disappearing_collision(self):
        block = CassetteBlockState("a:1", 0, 1., 80., 34., 16., 16.,
                                   True, True, True, 2, True, ("a:1",), 0, 0., 0., 0.)
        target = WallGripGoal("wall_grip", "a", 76., 42., "wall:a:1:1",
                             rect=(80., 32., 96., 48.), facing=1,
                             identity="a:1", mechanism_kind="cassetteBlock", grip_offset=10.)
        state = SimState("a", 76., 42., state=1, facing=1, cassette_blocks=(block,))
        self.assertEqual(target.bounds(state), (80., 34., 96., 50.))
        self.assertTrue(target.achieved(state))
        shifted = replace(state, x=84., cassette_blocks=(replace(block, x=88., y=32.),))
        self.assertEqual(target.waypoint(shifted), (84., 42.))
        self.assertTrue(target.achieved(shifted))
        self.assertFalse(target.achieved(replace(shifted, x=76.)))
        for missing in ((), (replace(block, collidable=False),), (replace(block, active=False),)):
            absent = replace(state, cassette_blocks=missing)
            self.assertIsNone(target.bounds(absent))
            self.assertFalse(target.achieved(absent))

    def test_relative_approach_macros_reach_a_real_upper_wall_grip(self):
        for facing in (1, -1):
            with self.subTest(facing=facing):
                sim, _ = fixture(facing=facing, moving=True)
                start = SimState("a", 40 if facing > 0 else 120, 128,
                                 on_ground=True, grace=6, prev_input=JUMP)
                target = next(goal for goal in wall_grip_goals(sim, start, Goal("target", "a", 140, 32))
                              if goal.identity == "a:1" and goal.facing == facing)
                macros = list(wall_approach_macros(start, target))
                self.assertEqual(len(macros), 10)
                reached = []
                vertical_reached = []
                for actions, length in macros:
                    self.assertEqual(length, len(actions))
                    self.assertEqual(actions[0], 0)  # Actual jump-button release.
                    self.assertLessEqual(length, 90)
                    state = start
                    for mask in actions:
                        state = sim.step(state, mask)
                        if state.dead or target.achieved(state):
                            break
                    if target.achieved(state):
                        reached.append(state)
                        if DASH | UP in actions:
                            vertical_reached.append(state)
                self.assertTrue(reached)
                self.assertTrue(vertical_reached)
                self.assertTrue(all(state.state == 1 and state.y <= target.y+4 for state in reached))
                airborne = replace(start, y=120, on_ground=False)
                short = list(wall_approach_macros(airborne, target))
                self.assertEqual([duration for _, duration in short], [6, 12])
                toward = RIGHT if facing > 0 else LEFT
                self.assertTrue(all(set(actions) == {GRAB | UP | toward} for actions, _ in short))
                self.assertEqual(list(wall_approach_macros(start, Goal("target", "a", 140, 32))), [])

    def test_goals_prioritize_movers_and_require_grip_at_requested_height(self):
        sim, grip = fixture(moving=True)
        start = SimState("a", 40, 128)
        goals = wall_grip_goals(sim, start, Goal("target", "a", 140, 32))
        self.assertTrue(goals)
        target = goals[0]
        self.assertIsInstance(target, WallGripGoal)
        self.assertEqual(target.identity, "a:1")
        self.assertEqual(target.facing, 1)
        self.assertLessEqual(len(goals), 16)
        ready = replace(grip, y=target.y)
        self.assertTrue(target.achieved(ready))
        for wrong in (replace(ready, state=0), replace(ready, dead=True),
                      replace(ready, facing=-1), replace(ready, x=ready.x-1),
                      replace(ready, y=target.y+5)):
            self.assertFalse(target.achieved(wrong))
        # A falling face cannot quietly lower the required grip altitude.
        lowered = replace(ready, y=target.y+12,
                          mechanisms=(("a:1", 2, 8, 80., 44., 80.),))
        self.assertFalse(target.achieved(lowered))
        self.assertEqual(target.waypoint(lowered)[1], target.y)
        shifted = replace(ready, x=ready.x+8,
                          mechanisms=(("a:1", 2, 8, 88., 40., 80.),))
        self.assertEqual(target.waypoint(shifted)[0], ready.x+8)
        self.assertTrue(target.achieved(shifted))
        self.assertFalse(target.achieved(replace(shifted, x=ready.x)))
        self.assertFalse(target.achieved(replace(ready, broken=frozenset({"a:1"}))))
        absent = replace(start, mechanisms=(("a:1", 4, 0, 80., 200., 160.),))
        self.assertFalse(any(goal.identity == "a:1" for goal in wall_grip_goals(sim, absent, target)))

    def test_multiple_launches_replay_restore_resources_and_keep_preview_knots(self):
        for facing in (1, -1):
            with self.subTest(facing=facing):
                sim, start = fixture(facing=facing)
                before = start.to_dict()
                with patch("celeste_search.wall_moves.remember_segment") as remember:
                    edges = wall_launch_edges(sim, start)
                self.assertGreaterEqual(len(edges), 3)
                self.assertEqual(remember.call_count, len(edges))
                upper = [(actions, end) for actions, end, _ in edges if end.y == 64]
                self.assertGreaterEqual(len({end.x for _, end in upper}), 2)
                for (actions, end, label), call in zip(edges, remember.call_args_list):
                    replay = sim.trace(start, actions)
                    self.assertEqual(replay[-1], end, label)
                    self.assertTrue(all(not state.dead and state.room == start.room for state in replay))
                    self.assertTrue(end.on_ground)
                    self.assertGreaterEqual(end.vy, 0)
                    self.assertEqual((end.dashes, end.stamina), (1, 110))
                    self.assertEqual(end.frame-start.frame, len(actions))
                    self.assertEqual(actions[-1], 0)
                    self.assertIn(JUMP | (LEFT if facing > 0 else RIGHT), actions)
                    self.assertEqual(call.args, (start, end, replay))
                self.assertEqual(start.to_dict(), before)

    def test_moving_mechanism_state_is_carried_through_every_edge(self):
        sim, start = fixture(moving=True)
        edges = wall_launch_edges(sim, start)
        self.assertTrue(edges)
        for actions, end, _ in edges:
            self.assertEqual(end, sim.trace(start, actions)[-1])
            self.assertTrue(any(row[0] == "a:1" for row in end.mechanisms))
        self.assertTrue(any(end.mechanisms != start.mechanisms for _, end, _ in edges))

    def test_invalid_starts_budget_and_room_changes_never_return_an_edge(self):
        sim, start = fixture()
        for invalid in (replace(start, state=0), replace(start, dead=True), replace(start, stamina=0)):
            self.assertEqual(wall_launch_edges(sim, invalid), [])
        self.assertEqual(wall_launch_edges(sim, start, max_frames=1), [])
        edges = wall_launch_edges(sim, start, max_frames=40)
        self.assertTrue(edges)
        self.assertTrue(all(len(actions) <= 40 for actions, _, _ in edges))
        original = sim.step
        def transition(state, mask):
            moved = original(state, mask)
            return replace(moved, room="b") if mask & JUMP else moved
        with patch.object(sim, "step", side_effect=transition), patch("celeste_search.wall_moves.remember_segment") as remember:
            self.assertEqual(wall_launch_edges(sim, start), [])
            remember.assert_not_called()


if __name__ == "__main__":
    unittest.main()
