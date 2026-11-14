"""A real mover grip keeps a short backward wall catch among onward choices."""
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_routes import normalize_route, start_route_progress
from celeste_search.corridor_planning import _ConstrainedSimulator
from celeste_search.maps import Entity, MapData, Room
from celeste_search.mechanism_moves import mechanism_transfer_edges
from celeste_search.mechanism_routes import held_zip_contacts, relay_candidate_details
from celeste_search.physics import GRAB, LEFT, RIGHT, SimState, Simulator
from celeste_search.planner import Goal
from celeste_search.zip_grips import zip_grip_edges


def fixture(mirror=False, offset=(0,0)):
    ox,oy=offset
    def px(x):return ox+(384-x if mirror else x)
    def left(x,width):return px(x+width) if mirror else px(x)
    entities=(Entity("zipMover",1,left(80,16),oy+48,16,64,{},((left(80,16),oy+136),)),
              Entity("spikesUp",2,left(80,16),oy+48,16,0),
              Entity("zipMover",3,left(8,16),oy+96,16,32,{},((left(8,16),oy+32),)))
    tiles=tuple("".join("1" if ty==38 or 6<=ty<25 and
                         (47-tx if mirror else tx) in (5,16) else "0"
                         for tx in range(48)) for ty in range(40))
    room=Room("retreat",ox,oy,384,320,tiles,entities,((px(200),oy+304),))
    sim=Simulator(MapData("retreat",{room.name:room},Path("fixture.bin")))
    held=GRAB|(RIGHT if mirror else LEFT)
    state=SimState(room.name,px(100),oy+108,state=1,facing=1 if mirror else -1,
        dashes=1,stamina=110,frame=71,prev_input=held,
        mechanisms=(("retreat:1",1,7,entities[0].x,entities[0].y,0.),
                    ("retreat:3",0,0,entities[2].x,entities[2].y,0.)))
    public=Goal("target",room.name,px(16),oy+16)
    return sim,state,public,px


class ZipTemporaryWallTests(unittest.TestCase):
    def test_nearest_backward_wall_keeps_onward_wall_mover_and_recovery_slots(self):
        for mirror,offset in ((False,(0,0)),(True,(-720,-904))):
            with self.subTest(mirror=mirror):
                sim,state,public,px=fixture(mirror,offset)
                before=state.to_dict()
                with patch.object(sim,"step",side_effect=AssertionError("guidance only")):
                    selected=relay_candidate_details(sim,state,public,limit=6)
                self.assertLessEqual(len(selected),6)
                self.assertEqual(selected[0]["family"],"wall")
                self.assertEqual(selected[0]["goal"].x,px(124))
                self.assertLess((selected[0]["goal"].x-state.x)*(public.x-state.x),0)
                self.assertTrue(any(row["family"]=="wall" and row["goal"].x==px(52)
                                    for row in selected))
                self.assertTrue(any(row["identity"]=="retreat:3" for row in selected))
                self.assertTrue(any(row["family"]=="support" for row in selected))
                self.assertEqual(state.to_dict(),before)

    def test_actual_return_can_take_selected_backward_wall_without_dash(self):
        sim,start,public,px=fixture()
        route=normalize_route(sim.map,dict(waypoints=[dict(room=start.room,x=16,y=16)],
            segments=[dict(no_dash=True,no_ground=False)]))
        wrapped=_ConstrainedSimulator(sim,route,no_dash=True,phase=start_route_progress(route,start))
        grip=held_zip_contacts(sim,start)[0]
        returned=next(row for row in zip_grip_edges(wrapped,start,grip,public,
            cycle_windows=True,return_climb=True,max_frames=280,deadline=time.monotonic()+1)
            if row[2].endswith(":return_end"))
        selected=relay_candidate_details(sim,returned[1],public,limit=6)
        retreat=next(row["goal"] for row in selected if row["family"]=="wall"
                     and row["goal"].x==px(124))
        edges=mechanism_transfer_edges(wrapped,returned[1],retreat,
            deadline=time.monotonic()+1,max_frames=140,max_expansions=96,max_edges=3)
        self.assertTrue(edges)
        for actions,final,_ in edges:
            state=start
            for mask in returned[0]+actions:
                state=wrapped.step(state,mask)
                self.assertFalse(state.dead)
            self.assertEqual(state,final)
            self.assertTrue(retreat.achieved(final))
            self.assertEqual(final.dashes,start.dashes)
            self.assertLess(final.stamina,start.stamina)
            self.assertFalse(public.achieved(final))

    def test_held_contact_uses_actual_pose_even_when_sampled_high_guide_is_blocked(self):
        sim,state,_,_=fixture()
        with patch.object(sim,"solid_at",return_value=True):
            contacts=held_zip_contacts(sim,state)
        self.assertEqual(len(contacts),1)
        self.assertEqual(contacts[0].identity,"retreat:1")
        self.assertTrue(contacts[0].achieved(state))


if __name__=="__main__":unittest.main()
