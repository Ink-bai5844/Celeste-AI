"""Native rider carries stop at obstructions without a squish callback."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import GRAB, LEFT, SimState, Simulator, _f32, _zip_ease


def vertical(*, upward=False, obstacle="tile", hazard=False, mirror=False):
    grid = [list("0" * 16) for _ in range(16)]
    start_y, target_y = (48, 16) if upward else (16, 48)
    px = lambda x: 128-x if mirror else x
    rx = lambda x,w: 128-x-w if mirror else x
    mover = Entity("zipMover", 1, rx(40,16), start_y, 16,32,
                   nodes=((rx(40,16),target_y),))
    top = 32 if upward else 48
    entities = [mover]
    if obstacle == "tile":
        for x in range(rx(56,16)//8, (rx(56,16)+16)//8):
            grid[top//8][x] = "1"
    elif obstacle == "jumpThru":
        entities.append(Entity(obstacle,2,rx(56,16),top+8 if upward else top,16,8))
    elif obstacle:
        entities.append(Entity(obstacle,2,rx(56,16),top,16,8))
    if hazard:
        entities.append(Entity("spikesUp",3,rx(56,16),48,16,0))
    room = Room("carry",0,0,128,128,tuple("".join(row) for row in grid),tuple(entities),())
    sim = Simulator(MapData("rider-carry",{"carry":room},Path("synthetic")))
    y = _f32(start_y+(target_y-start_y)*_zip_ease(15,30))
    state = SimState("carry",px(60),52 if upward else 47,state=1,
        facing=1 if mirror else -1,stamina=73,dashes=0,frame=91,
        rem_x=.25,rem_y=-.125,
        mechanisms=(("carry:1",2,15,float(mover.x),y,0.),))
    return sim,state


class MoverRiderClippingTests(unittest.TestCase):
    def test_side_rider_vertical_carry_clips_both_directions_and_faces(self):
        for upward in (False,True):
            for mirror in (False,True):
                with self.subTest(upward=upward,mirror=mirror):
                    sim,state = vertical(upward=upward,mirror=mirror)
                    self.assertFalse(sim.solid_at(state.room,state.x,state.y,state=state))
                    after = sim._advance_mechanisms(state)
                    self.assertFalse(after.dead)
                    self.assertEqual(after.y,51 if upward else 48)
                    self.assertEqual(after.x,state.x)
                    self.assertEqual((after.rem_x,after.rem_y),(.25,0.))
                    self.assertEqual((after.dashes,after.stamina,after.frame),(0,73,91))
                    self.assertEqual(after.mechanisms[0][1:3],(2,16))
                    self.assertEqual(after.lift_timer,10)
                    self.assertEqual(after.lift_vy>0,not upward)

    def test_standing_horizontal_rider_clips_and_clears_only_horizontal_remainder(self):
        grid = [list("0"*16) for _ in range(16)]
        grid[0][8] = "1"  # x64, above the moving platform rather than in its path.
        mover = Entity("zipMover",1,40,16,16,32,nodes=((72,16),))
        room = Room("carry",0,0,128,128,tuple("".join(row) for row in grid),(mover,),())
        sim = Simulator(MapData("standing-carry",{"carry":room},Path("synthetic")))
        x = _f32(40+32*_zip_ease(15,30))
        state = SimState("carry",59,16,on_ground=True,rem_x=.25,rem_y=-.125,
                         mechanisms=(("carry:1",2,15,x,16.,0.),))
        after = sim._advance_mechanisms(state)
        self.assertFalse(after.dead)
        self.assertEqual((after.x,after.y),(60,16))
        self.assertEqual((after.rem_x,after.rem_y),(0.,-.125))
        self.assertEqual(after.mechanisms[0][1:3],(2,16))
        self.assertGreater(after.lift_vx,0)

    def test_other_static_and_dynamic_solids_block_the_rider(self):
        for kind in ("solid","dashBlock","crumbleBlock"):
            with self.subTest(kind=kind):
                sim,state = vertical(obstacle=kind)
                after = sim._advance_mechanisms(state)
                self.assertFalse(after.dead)
                self.assertEqual(after.y,48)
                self.assertEqual(after.rem_y,0.)
                self.assertEqual(after.broken,state.broken)

    def test_destroyed_dynamic_obstacle_does_not_block_or_reset_remainder(self):
        sim,state = vertical(obstacle="dashBlock")
        state = replace(state,broken=frozenset({"carry:2"}))
        after = sim._advance_mechanisms(state)
        self.assertFalse(after.dead)
        self.assertEqual(after.y,49)
        self.assertEqual((after.rem_x,after.rem_y),(state.rem_x,state.rem_y))
        self.assertEqual(after.broken,state.broken)

    def test_jumpthru_stops_downward_carry_but_allows_upward_carry(self):
        for upward in (False,True):
            sim,state = vertical(upward=upward,obstacle="jumpThru")
            after = sim._advance_mechanisms(state)
            self.assertFalse(after.dead)
            self.assertEqual(after.y,50 if upward else 48)
            self.assertEqual(after.rem_y,state.rem_y if upward else 0.)

    def test_direct_overlap_push_still_squishes_against_a_tile(self):
        grid = [list("0"*16) for _ in range(16)]
        grid[2][9] = "1"  # Wall x72, player initially ends exactly at its face.
        mover = Entity("zipMover",1,0,16,16,32,nodes=((96,16),))
        room = Room("carry",0,0,128,128,tuple("".join(row) for row in grid),(mover,),())
        sim = Simulator(MapData("pushed-collision",{"carry":room},Path("synthetic")))
        x = _f32(96*_zip_ease(20,30))
        state = SimState("carry",round(x)+20,30,stamina=73,dashes=0,
                         mechanisms=(("carry:1",2,20,x,16.,0.),))
        self.assertFalse(sim.solid_at(state.room,state.x,state.y,state=state))
        after = sim._advance_mechanisms(state)
        self.assertTrue(after.dead)
        self.assertEqual(after.reason,"squish")
        self.assertEqual((after.dashes,after.stamina),(0,73))

    def test_carry_clipping_does_not_remove_actual_spike_death(self):
        sim,state = vertical(hazard=True)
        # Native hazards update before this room's mover. The carry itself
        # stops safely; the next ordinary hazard check still kills the player.
        after = sim.step(state,GRAB|LEFT)
        self.assertFalse(after.dead)
        self.assertEqual(after.y,48)
        self.assertTrue(sim._hazard_hit(after))
        after = sim.step(after,GRAB|LEFT)
        self.assertTrue(after.dead)
        self.assertEqual(after.reason,"hazard")


if __name__ == "__main__":
    unittest.main()
