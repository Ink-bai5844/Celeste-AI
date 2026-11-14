"""Toward climb-jumps are distinct from jumping away from a rising zip."""
from dataclasses import replace
from pathlib import Path
import time
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.mechanism_moves import mechanism_transfer_edges, _transfer_recipes
from celeste_search.physics import DASH, GRAB, LEFT, RIGHT, UP, SimState, Simulator
from celeste_search.wall_moves import WallGripGoal


def fixture(mirror=False,offset=(0,0),floor=False):
    ox,oy=offset
    def x(value):return ox+(224-value if mirror else value)
    def left(value,width):return x(value+width) if mirror else x(value)
    zx=left(64,16)
    entities=(Entity("zipMover",1,zx,oy+120,16,40,{},((zx,oy+56),)),
              Entity("spikesUp",2,zx,oy+120,16,0),
              Entity("solid",3,left(96,40),oy+112,40,72),
              Entity("spikesUp",4,left(96,40),oy+112,40,0),
              Entity("spikesDown",5,left(48,64),oy+16,64,0))
    if floor:
        entities += (Entity("solid",6,left(0,72),oy+168,72,24),)
    room=Room("toward",ox,oy,224,192,("0"*28,)*24,entities,())
    sim=Simulator(MapData("moving-wall-toward",{room.name:room},Path("fixture.bin")))
    origin=SimState(room.name,x(60),oy+140,vy=-45,dashes=1,stamina=107.727272,
        state=1,facing=-1 if mirror else 1,prev_input=GRAB|UP|(LEFT if mirror else RIGHT),
        rem_x=-.361169457 if mirror else .361169457,rem_y=.249958754,
        mechanisms=(("toward:1",2,1,float(zx),oy+119.91229248046875,0.),))
    rect=(left(96,40),oy+112,left(96,40)+40,oy+184)
    target=WallGripGoal("wall_grip",room.name,x(92),oy+128,rect=rect,
                        facing=-1 if mirror else 1,grip_offset=16)
    return sim,origin,target


class MechanismTowardLaunchTests(unittest.TestCase):
    def test_moving_spiked_wall_can_launch_toward_another_real_face(self):
        for mirror,offset in ((False,(0,0)),(True,(-640,-800))):
            with self.subTest(mirror=mirror):
                sim,origin,target=fixture(mirror,offset)
                stats={}
                edges=mechanism_transfer_edges(sim,origin,target,
                    deadline=time.monotonic()+1,max_expansions=192,max_edges=4,stats=stats)
                self.assertTrue(edges)
                no_dash=[edge for edge in edges if not any(mask&DASH for mask in edge[0])]
                self.assertTrue(no_dash)
                self.assertTrue(any("toward_jump:" in label for _,_,label in no_dash))
                for actions,expected,label in no_dash:
                    state=origin
                    for mask in actions:
                        state=sim.step(state,mask)
                        self.assertFalse(state.dead)
                        self.assertFalse(sim.last_step_started_dash)
                    self.assertEqual(state,expected)
                    self.assertTrue(target.achieved(state))
                    self.assertEqual(state.dashes,1)
                    self.assertTrue(label.endswith("dash_starts=0"))
                self.assertLessEqual(stats["expansions"],192)

    def test_old_away_neutral_and_dash_order_survive_the_existing_quota(self):
        _,origin,target=fixture()
        rows=list(_transfer_recipes(origin,target,True))
        labels=[label for _,label in rows]
        old_jumps=[f"jump:{direction}:{ticks}:{after}" for direction in (LEFT,0)
                    for ticks in (12,20,8,28) for after in (GRAB|UP|RIGHT,GRAB|RIGHT,0)]
        self.assertEqual(labels[:24],old_jumps)
        old_dashes=[f"jump_dash:{direction}:{delay}:{aim}:{after}"
                    for delay in (20,28,12,36,8,16,24,4) for direction in (LEFT,0)
                    for aim in (UP|RIGHT,RIGHT,UP) for after in (GRAB|UP|RIGHT,GRAB|RIGHT)]
        self.assertEqual([label for label in labels if label.startswith("jump_dash:")],old_dashes)
        self.assertTrue(all(label in labels[:192] for label in old_dashes))
        self.assertTrue(any(label.startswith("toward_jump_dash:") for label in labels[:192]))
        # A lower target adds a fourth old dash direction; every old recipe
        # must still fit before the newly appended toward-dash family.
        left,top,right,bottom = target.rect
        lower = replace(target,y=origin.y+40,rect=(left,top+100,right,bottom+100))
        lower_labels = [label for _,label in _transfer_recipes(origin,lower,True)]
        old_lower = [label for label in lower_labels if label.startswith("jump_dash:")]
        self.assertEqual(len(old_lower),128)
        self.assertTrue(all(label in lower_labels[:192] for label in old_lower))

    def test_releasing_a_triggered_mover_can_recover_on_real_ground_then_jump(self):
        for mirror,offset in ((False,(0,0)),(True,(-640,-800))):
            with self.subTest(mirror=mirror):
                sim,origin,target=fixture(mirror,offset,floor=True)
                class NoDash:
                    no_dash=True
                    def __getattr__(self,name):return getattr(sim,name)
                stats={}
                edges=mechanism_transfer_edges(NoDash(),origin,target,
                    deadline=time.monotonic()+1,max_expansions=192,max_edges=4,stats=stats)
                recovered=[edge for edge in edges if "release_jump:" in edge[2]]
                self.assertTrue(recovered)
                self.assertGreater(recovered[0][1].stamina,90)
                for actions,expected,label in recovered:
                    state=origin
                    touched_ground=False
                    for mask in actions:
                        self.assertFalse(mask&DASH)
                        state=sim.step(state,mask)
                        touched_ground |= state.on_ground
                        self.assertFalse(state.dead)
                    self.assertTrue(touched_ground)
                    self.assertEqual(state,expected)
                    self.assertTrue(target.achieved(state))
                    self.assertEqual(state.dashes,1)
                    self.assertNotEqual(state.mechanisms,origin.mechanisms)
                self.assertLessEqual(stats["expansions"],59)

    def test_no_dash_recipes_and_external_rejection_stay_authoritative(self):
        sim,origin,target=fixture()
        rows=list(_transfer_recipes(origin,target,False))
        self.assertFalse(any(mask&DASH for actions,_ in rows for mask in actions))
        self.assertTrue(any(":32:" in label or ":40:" in label for _,label in rows))
        class Limited:
            no_dash=True
            def __getattr__(self,name):return getattr(sim,name)
            def step(self,state,mask):
                return replace(sim.step(state,mask),dead=True,reason="public_no_ground")
        stats={}
        self.assertEqual(mechanism_transfer_edges(Limited(),origin,target,
            deadline=time.monotonic()+1,max_expansions=32,stats=stats),[])
        self.assertEqual(stats,{"expansions":32,"simulated_frames":32})


if __name__=="__main__":
    unittest.main()
