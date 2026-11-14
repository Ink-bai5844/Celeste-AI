"""Capability proposals remain real contact goals across mover phases."""
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch
import unittest

from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import SimState,Simulator
from celeste_search.planner import Goal
from celeste_search.mechanism_routes import (
    relay_candidates,relay_candidate_details,relay_priority,_inventory)


def fixture(name="relay",offset=(0,0),mirror=False):
    ox,oy=offset
    entities=[Entity("spring",1,48,256,0,0),
              Entity("zipMover",2,80,176,16,24,{},((152,176),)),
              Entity("spikesUp",3,80,176,16,0),
              Entity("zipMover",4,288,192,24,32,{},((288,112),)),
              Entity("spikesUp",5,288,192,24,0),
              Entity("jumpThru",6,264,64,48,8),
              Entity("jumpThru",7,192,56,24,8),
              Entity("refill",8,208,160,0,0)]
    def px(x): return ox+(384-x if mirror else x)
    entities=tuple(replace(e,x=px(e.x+e.width) if mirror else px(e.x),y=oy+e.y,
        nodes=tuple((px(x+e.width) if mirror else px(x),oy+y) for x,y in e.nodes))
        for e in entities)
    room=Room(name,ox,oy,384,288,("0"*48,)*36,entities,())
    sim=Simulator(MapData("relay",{name:room},Path("relay.bin")))
    rows=tuple((f"{name}:{e.id}",0,0,e.x,e.y,0.) for e in entities if e.kind=="zipMover")
    state=SimState(name,px(40),oy+264,on_ground=True,mechanisms=rows)
    goal=Goal("target",name,px(128),oy+32)
    return sim,state,goal


class MechanismRouteTests(unittest.TestCase):
    def test_grounded_near_zip_gets_first_probe_without_losing_far_or_opposite_faces(self):
        # The second zip has a better optimistic ascent to the public target,
        # but the first zip's near face is the local entry from this ground.
        # This is contact ordering only: no future mover endpoint is a state.
        for name, offset, mirror in (("entry", (0, 0), False),
                                     ("translated", (-704, -1200), False),
                                     ("mirrored", (-800, -904), True)):
            with self.subTest(room=name):
                ox, oy = offset
                def px(x):
                    return ox + (384-x if mirror else x)
                entities = (
                    Entity("zipMover", 1, 80, 192, 16, 24, {}, ((80, 128),)),
                    Entity("spikesUp", 2, 80, 192, 16, 0),
                    Entity("zipMover", 3, 160, 192, 16, 24, {}, ((160, 32),)),
                    Entity("spikesUp", 4, 160, 192, 16, 0),
                )
                entities = tuple(replace(e,
                    x=px(e.x+e.width) if mirror else px(e.x), y=oy+e.y,
                    nodes=tuple((px(x+e.width) if mirror else px(x), oy+y)
                                for x, y in e.nodes)) for e in entities)
                room = Room(name, ox, oy, 384, 288, ("0"*48,)*36, entities, ())
                simulator = Simulator(MapData("local-entry", {name: room}, Path("fixture.bin")))
                mechanisms = tuple((f"{name}:{e.id}", 0, 0, e.x, e.y, 0.)
                                   for e in entities if e.kind == "zipMover")
                state = SimState(name, px(40), oy+264, on_ground=True,
                                 mechanisms=mechanisms)
                goal = Goal("target", name, px(352), oy+8)
                before = state.to_dict()
                with patch.object(simulator, "step", side_effect=AssertionError("ranking must not simulate")):
                    single = relay_candidate_details(simulator, state, goal, limit=1)
                    all_rows = relay_candidate_details(simulator, state, goal, limit=32)
                closest = single[0]["goal"]
                self.assertEqual((closest.identity, closest.facing),
                                 (f"{name}:1", -1 if mirror else 1))
                self.assertEqual(all_rows[0]["goal"], closest)
                self.assertTrue(any(row["identity"] == f"{name}:3"
                                    and row["score"] < all_rows[0]["score"]
                                    for row in all_rows))
                self.assertEqual({(row["identity"], row["goal"].facing) for row in all_rows},
                                 {(f"{name}:{identity}", facing)
                                  for identity in (1, 3) for facing in (-1, 1)})
                self.assertTrue(all(not row["goal"].achieved(state) for row in all_rows))
                self.assertFalse(goal.achieved(state))
                self.assertEqual(state.to_dict(), before)

    def test_spiked_zips_are_side_contacts_and_real_spring_is_retained(self):
        sim,state,goal=fixture()
        rows=relay_candidate_details(sim,state,goal,limit=16)
        self.assertTrue(any(row["family"]=="spring" for row in rows))
        zips=[row["goal"] for row in rows if row["family"]=="zip"]
        self.assertEqual({g.identity for g in zips},{"relay:2","relay:4"})
        self.assertTrue(all(g.kind=="wall_grip" for g in zips))
        self.assertTrue(all(not g.achieved(state) for g in zips))
        self.assertLessEqual(len(_inventory(sim,state,goal)),32)

    def test_translation_rename_and_horizontal_mirror_keep_capability_scores(self):
        inventories=[]
        for name,offset,mirror in (("relay",(0,0),False),("other",(-700,-1200),False),
                                   ("mirror",(-800,-900),True)):
            sim,state,goal=fixture(name,offset,mirror)
            rows=relay_candidate_details(sim,state,goal,limit=32)
            inventories.append(sorted((row["family"],row["identity"].split(":")[-1],
                                       round(row["score"],5)) for row in rows))
        self.assertEqual(inventories[0],inventories[1])
        self.assertEqual(inventories[0],inventories[2])

    def test_proposals_never_step_or_mutate_actual_mechanism_states(self):
        sim,state,goal=fixture()
        before=state.to_dict()
        with patch.object(sim,"step",side_effect=AssertionError("ranking must not simulate")):
            candidates=relay_candidates(sim,state,goal)
            self.assertTrue(candidates)
            self.assertGreater(relay_priority(sim,state,goal),0)
        self.assertEqual(state.to_dict(),before)

    def test_new_actual_mover_position_is_contact_position_not_future_endpoint(self):
        sim,state,goal=fixture()
        old=[g for g in relay_candidates(sim,state,goal,limit=32) if getattr(g,"identity","")=="relay:2"]
        moved=replace(state,mechanisms=(("relay:2",2,18,116.,176.,0.),state.mechanisms[1]))
        new=[g for g in relay_candidates(sim,moved,goal,limit=32) if getattr(g,"identity","")=="relay:2"]
        self.assertEqual({g.x for g in old},{76.,100.})
        self.assertEqual({g.x for g in new},{112.,136.})
        self.assertTrue(all(g.bounds(moved)[0]==116 for g in new))
        self.assertNotEqual(relay_priority(sim,state,goal),relay_priority(sim,moved,goal))

    def test_lift_and_resources_change_actual_arrival_priority_without_altering_it(self):
        sim,state,goal=fixture()
        start=replace(state,x=172,y=188,on_ground=False,state=1,dashes=0)
        lifted=replace(start,lift_vx=176,lift_vy=-130)
        self.assertLess(relay_priority(sim,lifted,goal),relay_priority(sim,start,goal))
        self.assertEqual(lifted.dashes,0)
        self.assertEqual(lifted.mechanisms,start.mechanisms)

    def test_contact_history_limits_and_invalid_states(self):
        sim,state,goal=fixture()
        first=relay_candidates(sim,state,goal,limit=3)
        self.assertEqual(len(first),3)
        next_goals=relay_candidates(sim,state,goal,visited=[first[0].key],limit=3)
        self.assertNotIn(first[0],next_goals)
        self.assertEqual(relay_candidates(sim,state,goal,limit=0),[])
        self.assertEqual(relay_candidates(sim,replace(state,dead=True),goal),[])
        self.assertEqual(relay_candidates(sim,replace(state,room="missing"),goal),[])

    def test_safe_top_boarding_is_kept_but_spiked_top_is_not(self):
        sim,state,goal=fixture()
        room=sim.map.rooms[state.room]
        room=replace(room,entities=tuple(e for e in room.entities if e.id!=3))
        sim=Simulator(replace(sim.map,rooms={room.name:room}))
        candidates=relay_candidates(sim,state,goal,limit=32)
        self.assertTrue(any(g.key=="relay_board:relay:2" for g in candidates))
        self.assertFalse(any(g.key=="relay_board:relay:4" for g in candidates))

    def test_real_horizontal_ride_keeps_distinct_forward_mover_before_return_contacts(self):
        for name,offset,mirror in (("relay",(0,0),False),("other",(-700,-900),True)):
            sim,state,goal=fixture(name,offset,mirror)
            ox,oy=offset
            x=384-152-16 if mirror else 152
            state=replace(state,x=ox+x-4 if mirror else ox+x+20,y=oy+188,
                on_ground=False,state=1,facing=1 if mirror else -1,
                lift_vx=-176 if mirror else 176,
                mechanisms=((f"{name}:2",3,31,ox+x,oy+176,0.),state.mechanisms[1]))
            candidates=relay_candidates(sim,state,goal,limit=3)
            self.assertEqual(candidates[0].identity,f"{name}:4")
            self.assertLessEqual(len(candidates),3)

    def test_real_upward_ride_prefers_near_recovery_without_deleting_high_support(self):
        sim,state,goal=fixture()
        state=replace(state,x=284,y=124,on_ground=False,state=1,facing=1,
            lift_vy=-200,mechanisms=(state.mechanisms[0],("relay:4",3,31,288.,112.,0.)))
        candidates=relay_candidates(sim,state,goal,limit=16)
        self.assertEqual((candidates[0].kind,candidates[0].x,candidates[0].y),("surface",288,64))
        self.assertTrue(any(g.kind=="surface" and g.y==56 for g in candidates))


if __name__=="__main__":
    unittest.main()
