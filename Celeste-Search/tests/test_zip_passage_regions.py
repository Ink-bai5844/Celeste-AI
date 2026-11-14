"""A zip passage accepts real oriented traversal, never a proximity shortcut."""
from dataclasses import replace
from pathlib import Path
import time
import unittest

from celeste_search.corridor_routes import (ZipMoverPassageGoal,MechanismClearanceGoal,
    MechanismPassageGoal,DashBlockBreakGoal,DashBlockPassageGoal,_current_grid,_passage_goal,
    generate_corridor_routes,clear_corridor_cache)
from celeste_search.maps import Entity,MapData,Room
from celeste_search.physics import Simulator,SimState,GRAB,LEFT
from celeste_search.planner import Goal


def fixture(offset=(0,0)):
    ox,oy=offset
    e=Entity('zipMover',1,ox+96,oy+64,16,48,{},((ox+32,oy+64),))
    r=Room('region',ox,oy,256,192,('0'*32,)*22+('1'*32,)*2,(e,),((ox+144,oy+88),))
    sim=Simulator(MapData('zip-region',{'region':r},Path('region.bin')))
    s=SimState('region',ox+144,oy+88,mechanisms=(('region:1',0,0,ox+96.,oy+64.,0.),))
    return sim,s


def geometry_goal(sim,state,before,after):
    ox,oy=sim.map.rooms['region'].x,sim.map.rooms['region'].y
    clearance=MechanismClearanceGoal('mechanism_clearance','region',*before,
        identity='region:1',rect=(ox+96,oy+64,ox+112,oy+112),
        clearance_box=(ox+92,oy+60,ox+116,oy+124),mechanism_kind='zipMover')
    grid=_current_grid(sim,sim.map.rooms['region'],state,4,time.monotonic()+1,frozenset({'region:1'}))
    return _passage_goal(grid,before,after,clearance.rect,clearance)


class ZipPassageRegionTests(unittest.TestCase):
    def test_horizontal_mirror_preserves_real_side_of_midline(self):
        sim,state=fixture();g=geometry_goal(sim,state,(144,88),(72,88))
        flip=lambda r:(256-r[2],r[1],256-r[0],r[3])
        mirrored=replace(g,x=256-g.x,bounds=flip(g.bounds),
            occupied_rect=flip(g.occupied_rect),footprints=tuple(flip(r) for r in g.footprints),
            crossing_sign=-g.crossing_sign,clearance=replace(g.clearance,
                x=256-g.clearance.x,rect=flip(g.clearance.rect),
                clearance_box=flip(g.clearance.clearance_box)))
        for x,expected in ((100,True),(110,False)):
            original=replace(state,x=x,y=88,mechanisms=(('region:1',3,10,32.,64.,0.),))
            reflected=replace(original,x=256-x,mechanisms=(('region:1',3,10,208.,64.,0.),))
            self.assertEqual(g.achieved(original),expected)
            self.assertEqual(mirrored.achieved(reflected),expected)

    def test_four_crossing_directions_and_translation_derive_from_path(self):
        for offset in ((0,0),(-704,-1200)):
            ox,oy=offset;sim,state=fixture(offset)
            for before,after,axis,sign,passed,near in (
                ((144,88),(72,88),0,-1,(100,88),(110,88)),
                ((72,88),(144,88),0,1,(108,88),(98,88)),
                ((104,32),(104,160),1,1,(104,96),(104,72)),
                ((104,160),(104,32),1,-1,(104,80),(104,108))):
                with self.subTest(offset=offset,axis=axis,sign=sign):
                    g=geometry_goal(sim,state,(ox+before[0],oy+before[1]),(ox+after[0],oy+after[1]))
                    self.assertIsInstance(g,ZipMoverPassageGoal)
                    self.assertEqual((g.crossing_axis,g.crossing_sign),(axis,sign))
                    moved=replace(state,mechanisms=(('region:1',3,10,ox+32.,oy+64.,0.),))
                    hit=replace(moved,x=ox+passed[0],y=oy+passed[1])
                    self.assertTrue(g.achieved(hit))
                    self.assertFalse(g.achieved(replace(moved,x=ox+near[0],y=oy+near[1])))
                    self.assertFalse(g.achieved(replace(hit,dead=True)))
                    self.assertFalse(g.achieved(replace(hit,room='wrong')))
                    self.assertFalse(g.achieved(replace(hit,mechanisms=state.mechanisms)))
                    self.assertFalse(g.achieved(replace(hit,mechanisms=(('region:1',4,10,ox+96.,oy+64.,0.),))))

    def test_discrete_footprints_do_not_fill_empty_space(self):
        sim,state=fixture();g=geometry_goal(sim,state,(144,88),(72,88))
        g=replace(g,occupied_rect=(96,64,112,112),
            footprints=((96,64,112,112),(128,64,144,112)),crossing_sign=1)
        moved=replace(state,mechanisms=(('region:1',3,10,32.,64.,0.),),x=120,y=88)
        self.assertFalse(g.achieved(moved),'Gap between distinct old footprints is not passage')
        self.assertTrue(g.achieved(replace(moved,x=132)))

    def test_old_attached_hazard_footprint_can_supply_the_crossing_region(self):
        sim,state=fixture();g=geometry_goal(sim,state,(144,88),(72,88))
        g=replace(g,footprints=(g.occupied_rect,(96,61,112,64)),
            clearance=replace(g.clearance,attached_hazards=((96,61,112,64),)))
        crossed=replace(state,x=100,y=64,mechanisms=(('region:1',3,10,32.,64.,0.),))
        self.assertTrue(g.achieved(crossed))
        self.assertFalse(replace(g,footprints=(g.occupied_rect,)).achieved(crossed))

    def test_real_carry_can_finish_passage_at_clearance_without_returning_to_guide(self):
        sim,state=fixture()
        origin=replace(state,x=116,y=100,state=1,facing=-1,prev_input=GRAB,
                       frame=221,rem_y=.125,stamina=108)
        clearance=MechanismClearanceGoal('mechanism_clearance','region',116,100,
            identity='region:1',rect=(96,64,112,112),clearance_box=(92,82,118,104),
            facing=-1,mechanism_kind='zipMover')
        g=ZipMoverPassageGoal('corridor','region',104,88,identity='region:1',
            bounds=(102,86,106,90),occupied_rect=clearance.rect,footprints=(clearance.rect,),
            crossing_axis=0,crossing_sign=-1,clearance=clearance)
        old=MechanismPassageGoal('corridor','region',104,88,identity='region:1',bounds=g.bounds)
        actions=[];states=[origin];s=origin
        self.assertFalse(g.achieved(origin))
        for _ in range(80):
            s=sim.step(s,GRAB|LEFT);actions.append(GRAB|LEFT);states.append(s)
            self.assertFalse(s.dead)
            if clearance.achieved(s):break
        self.assertTrue(clearance.achieved(s));self.assertTrue(g.achieved(s))
        self.assertFalse(old.achieved(s));self.assertEqual(s.dashes,origin.dashes)
        replay=origin
        for i,a in enumerate(actions):
            replay=sim.step(replay,a);self.assertEqual(replay,states[i+1])
        self.assertEqual(origin.mechanisms,state.mechanisms)
        # The returning platform occupies this area again. A phase marker
        # alone cannot make the same region count as currently open.
        self.assertFalse(g.achieved(replace(s,mechanisms=(('region:1',4,100,96.,64.,0.),))))

    def test_non_zip_passage_keeps_original_narrow_predicate(self):
        sim,state=fixture();grid=_current_grid(sim,sim.map.rooms['region'],state,4,time.monotonic()+1,frozenset({'region:1'}))
        for kind in ('fallingBlock','dashBlock'):
            if kind=='dashBlock':
                c=DashBlockBreakGoal('dash_block_break','region',116,100,identity='region:1',rect=(96,64,112,112),clearance_box=(90,82,118,104))
            else:c=MechanismClearanceGoal('mechanism_clearance','region',116,100,identity='region:1',rect=(96,64,112,112),clearance_box=(90,82,118,104))
            g=_passage_goal(grid,(144,88),(72,88),c.rect,c)
            self.assertIs(type(g),DashBlockPassageGoal if kind=='dashBlock' else MechanismPassageGoal)

    def test_generation_keeps_final_clearance_reference_and_explicit_preview(self):
        clear_corridor_cache();sim,state=fixture()
        page=generate_corridor_routes(sim,state,Goal('target','region',72,88),deadline=time.monotonic()+1,max_routes=6)
        conditional=[r for r in page if r.conditional_on]
        self.assertTrue(conditional)
        for route in conditional:
            g=next(p for p in route.portals if isinstance(p,ZipMoverPassageGoal))
            c=next(p for p in route.portals if isinstance(p,MechanismClearanceGoal))
            self.assertEqual(g.clearance,c)
            self.assertLess(route.portals.index(c),route.portals.index(g))
            p=next(p for p in route.to_dict()['portals'] if p['id']==g.id)
            self.assertEqual(p['condition'],'actual_open_footprint_crossing')
            self.assertEqual(p['crossing_sign'],g.crossing_sign)
            self.assertFalse(route.to_dict()['physically_verified'])


if __name__=='__main__':unittest.main()
