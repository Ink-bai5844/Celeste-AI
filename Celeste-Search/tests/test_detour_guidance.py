"""A geometric detour potential ranks branches without changing acceptance."""
from pathlib import Path
import time
import unittest
from unittest.mock import patch

from celeste_search.detour_guidance import StaticDetourGuide
from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import SimState,Simulator
from celeste_search.planner import Goal


def fixture(mirror=False,offset=(0,0),wall=True,sealed=False,moving=False):
    ox,oy=offset
    def x(v):return ox+(224-v if mirror else v)
    left=x(112) if mirror else x(80)
    entities=(() if not wall else (Entity("zipMover" if moving else "solid",1,left,oy,
                   32,192 if sealed else 120,{},((left+32,oy),) if moving else ()),))
    if moving:
        entities+=(Entity("spikesUp",2,left,oy,32,0),)
    room=Room("detour",ox,oy,224,192,("0"*28,)*24,entities,())
    sim=Simulator(MapData("detour",{room.name:room},Path("fixture.bin")))
    upper=SimState(room.name,x(76),oy+60)
    lower=SimState(room.name,x(76),oy+116)
    goal=Goal("point",room.name,x(180),oy+44)
    return sim,upper,lower,goal


class DetourGuidanceTests(unittest.TestCase):
    def test_wall_detour_rewards_lower_route_in_mirror_and_translation(self):
        scores=[]
        for mirror,offset in ((False,(0,0)),(True,(-640,-800))):
            sim,upper,lower,goal=fixture(mirror,offset)
            with patch.object(sim,"step",side_effect=AssertionError("Heuristic must not step physics")):
                guide=StaticDetourGuide(sim,deadline=time.monotonic()+1)
                self.assertTrue(guide.prepare(upper,goal))
                a,b=guide.score(upper,goal),guide.score(lower,goal)
                self.assertGreater(a,b)
                self.assertEqual(a,0)
                self.assertLess(b,0)
                # Known progress remains rewarded after the terrain is behind
                # the player and the exit is directly visible.
                visible=SimState(upper.room,goal.x,offset[1]+100)
                self.assertLess(guide.score(visible,goal),b)
                embedded=SimState(upper.room,offset[0]+(128 if mirror else 96),offset[1]+60)
                self.assertEqual(guide.score(embedded,goal),0)
                self.assertEqual(guide.score(upper,goal),a)
                self.assertEqual(guide.cache_info()["builds"],1)
                scores.append((a,b))
                self.assertTrue(guide.prepare(lower,goal))
                self.assertEqual(guide.score(lower,goal),0)
        self.assertEqual(scores[0],scores[1])

    def test_open_straight_route_and_ignored_mover_have_zero_penalty(self):
        for moving in (False,True):
            sim,start,_,goal=fixture(wall=moving,moving=moving)
            guide=StaticDetourGuide(sim,deadline=time.monotonic()+1)
            self.assertFalse(guide.prepare(start,goal))
            self.assertEqual(guide.score(start,goal),0)
            self.assertEqual(guide.score(SimState(start.room,goal.x,goal.y+16),goal),0)

    def test_sealed_or_too_narrow_projection_falls_back_without_rejection(self):
        sim,start,_,goal=fixture(sealed=True)
        guide=StaticDetourGuide(sim,deadline=time.monotonic()+1)
        self.assertEqual(guide.score(start,goal),0)
        # Inside fixed terrain has no valid body-clear connector.
        embedded=SimState(start.room,96,60)
        self.assertEqual(guide.score(embedded,goal),0)
        self.assertFalse(start.dead)

    def test_expired_or_partial_build_is_zero_and_does_not_repeat(self):
        sim,start,_,goal=fixture()
        expired=StaticDetourGuide(sim,deadline=time.monotonic()-1)
        self.assertEqual(expired.score(start,goal),0)
        self.assertEqual(expired.cache_info()["builds"],0)
        disabled=StaticDetourGuide(sim,deadline=time.monotonic()+1,build_seconds=0)
        self.assertEqual(disabled.score(start,goal),0)
        self.assertEqual(disabled.score(start,goal),0)
        self.assertEqual(disabled.cache_info()["builds"],1)
        self.assertEqual(disabled.cache_info()["failures"],1)

    def test_mock_and_other_room_are_harmless_fallbacks(self):
        sim,start,_,goal=fixture()
        class MockSimulator:
            map_data=None
        guide=StaticDetourGuide(MockSimulator(),deadline=time.monotonic()+1)
        self.assertEqual(guide.score(start,goal),0)
        guide=StaticDetourGuide(sim,deadline=time.monotonic()+1)
        self.assertEqual(guide.score(start,Goal("point","other",180,44)),0)
        self.assertEqual(guide.cache_info()["builds"],0)


if __name__=="__main__":unittest.main()
