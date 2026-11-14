"""Native NormalEnd/DashBegin discard pre-dash wall speed at a cleared lip."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import DASH,GRAB,LEFT,RIGHT,SimState,Simulator


def fixture(direction):
    wall_x=80 if direction<0 else 104
    room=Room('a',0,0,224,200,(),(Entity('solid',1,wall_x,32,16,64),),())
    sim=Simulator(MapData('dash-wall-lip',{'a':room},Path('fixture.bin')))
    state=SimState('a',100.,106.,vx=0.,vy=141.11128,on_ground=False,
        facing=direction,wall_retention=4,wall_retained=90.*direction,
        wall_slide=55,rem_y=.389170885,prev_input=LEFT if direction<0 else RIGHT)
    return sim,state


class DashWallRetentionTests(unittest.TestCase):
    def test_dash_clearing_wall_lip_keeps_dash_speed_in_both_directions(self):
        for direction,button in ((-1,LEFT),(1,RIGHT)):
            with self.subTest(direction=direction):
                sim,state=fixture(direction)
                state=sim.step(state,button|DASH)
                self.assertTrue(sim.last_step_started_dash)
                self.assertEqual(state.state,2)
                self.assertEqual(state.wall_retention,0)
                self.assertEqual(state.wall_slide,72)
                state=sim.step(state,button|DASH)
                self.assertEqual(state.y,107.)
                self.assertEqual(state.x,100.+direction)
                self.assertEqual(state.vx,240.*direction)
                state=sim.step(state,button|DASH)
                self.assertEqual(state.x,100.+5*direction)
                self.assertEqual(state.vx,240.*direction)
                self.assertEqual(state.wall_retention,0)

    def test_normal_end_clears_pending_hop_on_climb_and_dash(self):
        for mask in (LEFT|DASH,LEFT|GRAB):
            with self.subTest(mask=mask):
                sim,state=fixture(-1)
                state=replace(state,y=80.,vy=0.,hop_wait_x=-1,hop_wait_x_speed=-100.)
                after=sim.step(state,mask)
                self.assertIn(after.state,(1,2))
                self.assertEqual(after.hop_wait_x,0)
                self.assertEqual(after.wall_retention,0)

    def test_normal_movement_still_restores_retained_speed_after_wall_clearance(self):
        sim,state=fixture(-1)
        state=replace(state,y=108.,wall_retained=-130.,vy=0.)
        after=sim.step(state,LEFT)
        self.assertEqual(after.state,0)
        self.assertLess(after.vx,-90.)
        self.assertEqual(after.wall_retention,0)


if __name__=='__main__':unittest.main()
