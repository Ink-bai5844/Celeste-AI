"""Spiked-top zips remain usable through exact, bounded physical side rides."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import Mock, patch

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import Simulator, SimState
from celeste_search.planner import Goal
from celeste_search.zip_grips import zip_top_blocked, zip_side_grip_goals, zip_grip_edges


def fixture(mirror=False, ox=0, oy=0, hazard=False):
    width = 320
    x = width - 80 - 24 if mirror else 80
    destination = width - 216 - 24 if mirror else 216
    entities = [Entity("zipMover", 1, ox + x, oy + 64, 24, 40, {},
                       ((ox + destination, oy + 64),)),
                Entity("spikesUp", 2, ox + x, oy + 64, 24, 0),
                Entity("zipMover", 3, ox + 8, oy + 144, 16, 16, {},
                       ((ox + 8, oy + 112),))]
    if hazard:
        entities.append(Entity("spikesLeft", 4, ox + 200, oy + 80, 0, 32))
    room = Room("a", ox, oy, width, 192, ("0" * 40,) * 24,
                tuple(entities), ((ox + 52., oy + 152.),))
    data = MapData("zip-grips", {"a": room}, Path("zip-grips.bin"))
    simulator = Simulator(data)
    approach = SimState("a", ox + (268 if mirror else 52), oy + 152,
                        on_ground=True,
                        mechanisms=(("a:1", 0, 0, ox + x, oy + 64., 0.),))
    goal = Goal("point", "a", ox + (64 if mirror else 256), oy + 96)
    grips = zip_side_grip_goals(simulator, approach, goal)
    grip = next(candidate for candidate in grips if candidate.identity == "a:1")
    start = SimState("a", grip.x, oy + 94, state=1, facing=grip.facing,
                     dashes=0, frame=123, stamina=110, climb_wait=5,
                     prev_input=64 | (1 if grip.facing > 0 else 2),
                     mechanisms=(("a:1", 1, 7, ox + x, oy + 64., 0.),
                                 ("a:3", 3, 7, ox + 8., oy + 112., 0.)))
    return simulator, start, grip, goal, grips


class ZipGripTests(unittest.TestCase):
    def test_destination_side_preference_and_mirrored_full_state_rides(self):
        for mirror, ox, oy in ((False, 0, 0), (True, -480, -240)):
            with self.subTest(mirror=mirror, offset=(ox, oy)):
                sim, start, grip, goal, grips = fixture(mirror, ox, oy)
                self.assertEqual([g.facing for g in grips if g.identity == "a:1"],
                                 [1, -1] if mirror else [-1, 1])
                before = start.to_dict()
                edges = list(zip_grip_edges(sim, start, grip, goal,
                                             deadline=time.monotonic() + 1))
                self.assertEqual([len(actions) for actions, _, _ in edges], [8, 16, 24, 32, 37])
                for actions, arrival, _ in edges:
                    state = start
                    for action in actions:
                        self.assertEqual(action, 64 | (1 if grip.facing > 0 else 2))
                        state = sim.step(state, action)
                        self.assertFalse(state.dead)
                    self.assertEqual(state, arrival)
                    self.assertEqual(state.frame, start.frame + len(actions))
                    self.assertEqual(state.dashes, 0)
                    self.assertLess(state.stamina, start.stamina)
                    self.assertNotEqual(state.mechanisms[1], start.mechanisms[1])
                self.assertEqual(edges[-1][1].mechanisms[0][1], 3)
                self.assertEqual(start.to_dict(), before)

    def test_only_complete_attached_upward_spike_coverage_blocks_boarding(self):
        sim, _, _, _, _ = fixture()
        room = sim.map.rooms["a"]
        mover, spike = room.entities[:2]
        self.assertTrue(zip_top_blocked(mover, room))
        for replacement in (replace(spike, width=16), replace(spike, y=spike.y - 8),
                            replace(spike, kind="spikesDown")):
            self.assertFalse(zip_top_blocked(mover, replace(room, entities=(mover, replacement))))
        left = replace(spike, width=8)
        right = replace(spike, id=9, x=spike.x + 8, width=16)
        self.assertTrue(zip_top_blocked(mover, replace(room, entities=(mover, left, right))))
        gap = replace(right, x=right.x + 1)
        self.assertFalse(zip_top_blocked(mover, replace(room, entities=(mover, left, gap))))

    def test_moving_contact_does_not_use_stale_world_height_during_downward_ride(self):
        sim, start, grip, goal, _ = fixture()
        room = sim.map.rooms["a"]
        mover = replace(room.entities[0], nodes=((80., 96.),))
        room = replace(room, entities=(mover,) + room.entities[1:])
        sim = Simulator(MapData("vertical-grip", {"a": room}, Path("vertical-grip.bin")))
        edges = list(zip_grip_edges(sim, start, grip, goal, deadline=time.monotonic() + 1))
        self.assertTrue(edges[-1][2].endswith(":ride"))
        end = edges[-1][1]
        self.assertEqual(end.y, start.y + 32)
        self.assertFalse(grip.achieved(end))  # Original approach-height goal is stale.
        self.assertEqual(end.state, 1)

    def test_earlier_launches_remain_when_later_ride_hits_real_spikes(self):
        sim, start, grip, goal, _ = fixture(hazard=True)
        edges = list(zip_grip_edges(sim, start, grip, goal, deadline=time.monotonic() + 1))
        self.assertTrue(edges)
        self.assertFalse(any(label.endswith(":ride") for _, _, label in edges))
        state = start
        for _ in range(50):
            state = sim.step(state, 66)
            if state.dead:
                break
        self.assertTrue(state.dead)
        self.assertTrue(all(not arrival.dead for _, arrival, _ in edges))

    def test_budgets_and_preview_keep_local_counts_and_complete_physical_prefixes(self):
        sim, start, grip, goal, _ = fixture()
        stats = {"expansions": 7, "simulated_frames": 100, "max_expansions": 8}
        tracker = Mock()
        remembered = []
        with patch("celeste_search.zip_grips.search_tracker", return_value=tracker), \
                patch("celeste_search.zip_grips.remember_segment",
                      side_effect=lambda first, last, states: remembered.append((first, last, list(states)))):
            edges = list(zip_grip_edges(sim, start, grip, goal, max_frames=20,
                                        deadline=time.monotonic() + 1, stats=stats))
        self.assertEqual([len(a) for a, _, _ in edges], [8, 16])
        self.assertEqual(stats["expansions"], 8)
        self.assertEqual(stats["simulated_frames"], 120)
        self.assertEqual(max(call.args[1] for call in tracker.report.call_args_list), 1)
        self.assertEqual(max(call.args[2] for call in tracker.report.call_args_list), 20)
        for (actions, end, _), (first, last, states) in zip(edges, remembered):
            self.assertEqual((first, last), (start, end))
            self.assertEqual(len(states), len(actions) + 1)
            for before, action, after in zip(states, actions, states[1:]):
                self.assertEqual(sim.step(before, action), after)
        with patch.object(sim, "step", wraps=sim.step) as step:
            for options in ({"deadline": time.monotonic() - 1},
                            {"deadline": time.monotonic() + 1, "max_frames": 0},
                            {"deadline": time.monotonic() + 1, "stats": stats}):
                self.assertEqual(list(zip_grip_edges(sim, start, grip, goal, **options)), [])
            step.assert_not_called()

    def test_lost_grip_death_and_other_room_are_rejected(self):
        for changed in ({"state": 0}, {"dead": True}, {"room": "other"}):
            with self.subTest(changed=changed):
                sim, start, grip, goal, _ = fixture()
                with patch.object(sim, "step", return_value=replace(start, **changed)):
                    self.assertEqual(list(zip_grip_edges(sim, start, grip, goal,
                                         deadline=time.monotonic() + 1)), [])

    def test_public_goal_between_samples_is_returned_at_its_first_real_hit(self):
        sim, start, grip, _, _ = fixture()
        class NarrowGoal(Goal):
            def achieved(self, state):
                return not state.dead and state.room==self.room and state.x==self.x
        goal=NarrowGoal("point",start.room,start.x+2,start.y)
        state=start;first=None
        for tick in range(1,40):
            state=sim.step(state,66)
            if goal.achieved(state):
                first=tick
                break
        self.assertIsNotNone(first)
        self.assertNotIn(first,(8,16,24,32))
        stats={}
        edges=list(zip_grip_edges(sim,start,grip,goal,
                                  deadline=time.monotonic()+1,stats=stats))
        self.assertEqual(len(edges[-1][0]),first)
        self.assertTrue(edges[-1][2].endswith(":goal"))
        self.assertTrue(goal.achieved(edges[-1][1]))
        self.assertEqual(stats["simulated_frames"],first)
        self.assertEqual(stats["expansions"],1)

    def test_live_target_hit_is_kept_when_same_update_leaves_the_grip(self):
        sim,start,grip,_,_=fixture()
        goal=Goal("room","neighbor",0,0)
        crossing=replace(start,room="neighbor",state=0,frame=start.frame+1)
        with patch.object(sim,"step",return_value=crossing):
            edges=list(zip_grip_edges(sim,start,grip,goal,deadline=time.monotonic()+1))
        self.assertEqual(len(edges),1)
        self.assertEqual(edges[0][1],crossing)
        self.assertEqual(len(edges[0][0]),1)
        with patch.object(sim,"step",return_value=replace(crossing,dead=True)):
            self.assertEqual(list(zip_grip_edges(sim,start,grip,goal,
                             deadline=time.monotonic()+1)),[])


if __name__ == "__main__":
    unittest.main()
