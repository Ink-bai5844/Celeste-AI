"""Conditional geometry never turns an unbroken dash block into real air."""
from dataclasses import replace
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import celeste_search.corridor_routes as corridors
from celeste_search.corridor_routes import (
    DashBlockApproachGoal, DashBlockBreakGoal, DashBlockPassageGoal,
    FallingBlockTriggerGoal, ZipMoverTriggerGoal, MechanismClearanceGoal, _current_grid,
    clear_corridor_cache, generate_corridor_routes,
)
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import SimState, Simulator
from celeste_search.planner import Goal


def fixture(entities=(), *, width=160, height=160, offset=(0,0)):
    ox,oy=offset
    entities=tuple(replace(entity,x=entity.x+ox,y=entity.y+oy) for entity in entities)
    room=Room('a',ox,oy,width,height,('0'*(width//8),)*(height//8),entities,((24+ox,80+oy),))
    return Simulator(MapData('dash-geometry',{'a':room},Path('dash-geometry.bin')))


def block(identifier=1,x=72,y=0,width=24,height=160,**attrs):
    return Entity('dashBlock',identifier,x,y,width,height,dict(canDash=True,permanent=True)|attrs)


class DashBlockCorridorTests(unittest.TestCase):
    def setUp(self):clear_corridor_cache()

    def routes(self,sim,start,goal):
        return generate_corridor_routes(sim,start,goal,max_routes=6,deadline=time.monotonic()+2)

    def test_all_cardinal_approaches_include_ordered_real_break_and_passage(self):
        cases=[(block(),(24,80),(136,80),(1,0)),
               (block(),(136,80),(24,80),(-1,0)),
               (block(x=0,y=48,width=160,height=64),(80,24),(80,144),(0,1)),
               (block(x=0,y=48,width=160,height=64),(80,144),(80,24),(0,-1))]
        for entity,origin,destination,direction in cases:
            with self.subTest(direction=direction):
                clear_corridor_cache()
                sim=fixture((entity,));start=SimState('a',*origin);anchor=start.to_dict()
                routes=self.routes(sim,start,Goal('target','a',*destination))
                self.assertTrue(routes)
                occupied=_current_grid(sim,sim.map.rooms['a'],start,4,time.monotonic()+1)
                opened=_current_grid(sim,sim.map.rooms['a'],start,4,time.monotonic()+1,frozenset({'a:1'}))
                for route in routes:
                    self.assertEqual(route.conditional_on,('a:1',))
                    stages=[p for p in route.portals if getattr(p,'identity','')=='a:1']
                    self.assertEqual([type(p) for p in stages],
                        [DashBlockApproachGoal,DashBlockBreakGoal,DashBlockPassageGoal])
                    approach,breaking,passage=stages
                    self.assertEqual((breaking.dash_dx,breaking.dash_dy),direction)
                    self.assertFalse(breaking.achieved(start))
                    hit=replace(start,broken=frozenset({'a:1'}))
                    self.assertTrue(breaking.achieved(hit))
                    self.assertFalse(passage.achieved(hit))
                    self.assertFalse(passage.achieved(replace(start,x=passage.x,y=passage.y)))
                    self.assertTrue(passage.achieved(replace(hit,x=passage.x,y=passage.y)))
                    self.assertFalse(breaking.achieved(replace(hit,dead=True)))
                    self.assertTrue(occupied.clear((approach.x,approach.y)))
                    self.assertTrue(occupied.path(occupied.nearest(origin),occupied.nearest((approach.x,approach.y))))
                    self.assertTrue(all(opened.segment_clear(a,b) for a,b in zip(route.points,route.points[1:])))
                    data=route.to_dict()
                    self.assertFalse(data['physically_verified'])
                    self.assertEqual(data['conditions'][0]['kind'],'dash_break')
                    self.assertEqual(data['conditions'][0]['rect'],breaking.rect)
                    self.assertEqual(data['conditions'][0]['identity'],'a:1')
                self.assertEqual(start.to_dict(),anchor,'Geometric omission must not mutate the real anchor')

    def test_conditional_candidate_publishes_before_optional_variants_spend_slice(self):
        sim=fixture((block(),));start=SimState('a',24,80);goal=Goal('target','a',136,80)
        clock=[0.];original=corridors._ClearanceGrid.path
        def expire_at_optional(grid,first,last,penalties=None,forbidden=frozenset(),**options):
            if penalties:
                clock[0]=2.
                return []
            return original(grid,first,last,penalties,forbidden,**options)
        with patch('celeste_search.corridor_routes.time.monotonic',side_effect=lambda:clock[0]), \
                patch.object(corridors._ClearanceGrid,'path',expire_at_optional):
            page=generate_corridor_routes(sim,start,goal,deadline=1.)
        self.assertTrue(page)
        self.assertTrue(all(route.conditional_on==('a:1',) for route in page))
        self.assertFalse(page.complete,'Optional unfinished discovery must not be declared exhausted')

    def test_condition_timeout_retries_same_scene_without_publishing_unvalidated_geometry(self):
        sim=fixture((block(),));start=SimState('a',24,80);goal=Goal('target','a',136,80)
        clock=[0.];calls=[];original=corridors._conditional_route
        def pause_first_condition(*args,**kwargs):
            calls.append(args[1].id)
            if len(calls)==1:
                clock[0]=2.
                return None
            return original(*args,**kwargs)
        with patch('celeste_search.corridor_routes.time.monotonic',side_effect=lambda:clock[0]), \
                patch('celeste_search.corridor_routes._conditional_route',side_effect=pause_first_condition):
            first=generate_corridor_routes(sim,start,goal,deadline=1.)
            self.assertEqual(first,[])
            self.assertFalse(first.complete)
            handle=first.continuation
            clear_corridor_cache()
            second=generate_corridor_routes(sim,start,goal,deadline=3.,continuation=handle)
        self.assertTrue(second)
        self.assertIs(second.continuation,handle)
        self.assertEqual(calls[0],calls[1])
        self.assertTrue(all(route.conditional_on==('a:1',) for route in second))

    def test_cannot_dash_gate_remains_occupied_and_broken_gate_needs_no_condition(self):
        for disabled in (False,'false',0,'0'):
            with self.subTest(canDash=disabled):
                sim=fixture((block(canDash=disabled),));start=SimState('a',24,80)
                self.assertEqual(self.routes(sim,start,Goal('target','a',136,80)),[])
        sim=fixture((block(),));start=SimState('a',24,80,broken=frozenset({'a:1'}))
        routes=self.routes(sim,start,Goal('target','a',136,80))
        self.assertTrue(routes)
        self.assertTrue(all(not route.conditional_on and not route.to_dict()['conditions'] for route in routes))

    def test_attached_spikes_are_removed_only_with_their_broken_owner(self):
        sim=fixture((block(y=48,height=48),Entity('spikesUp',2,72,48,24,0),
                     Entity('spikesUp',3,16,80,16,0)))
        start=SimState('a',24,112)
        occupied=_current_grid(sim,sim.map.rooms['a'],start,4,time.monotonic()+1)
        self.assertIn((72,45,96,48),occupied.hazards)
        gone=_current_grid(sim,sim.map.rooms['a'],replace(start,broken=frozenset({'a:1'})),4,time.monotonic()+1)
        self.assertNotIn((72,45,96,48),gone.hazards)
        self.assertIn((16,77,32,80),gone.hazards,'An unrelated spike must remain dangerous')
        self.assertEqual(start.broken,frozenset())

    def test_fully_spiked_impact_face_is_not_offered_as_a_dash_opening(self):
        sim=fixture((block(),Entity('spikesLeft',2,72,0,0,160)))
        self.assertEqual(self.routes(sim,SimState('a',24,80),Goal('target','a',136,80)),[])

    def test_two_dash_gates_and_mixed_falling_gate_keep_each_condition(self):
        for second in (block(2,x=104,width=16),
                       Entity('fallingBlock',2,104,0,16,160,dict(climbFall=True))):
            with self.subTest(second=second.kind):
                sim=fixture((block(1,x=48,width=16),second));start=SimState('a',24,80)
                routes=self.routes(sim,start,Goal('target','a',144,80))
                self.assertTrue(routes)
                for route in routes:
                    self.assertEqual(route.conditional_on,('a:1','a:2'))
                    gates=[p for p in route.portals if isinstance(p,(DashBlockBreakGoal,MechanismClearanceGoal))]
                    self.assertEqual([p.identity for p in gates],['a:1','a:2'])
                    self.assertIsInstance(gates[0],DashBlockBreakGoal)
                    if second.kind=='fallingBlock':
                        self.assertIsInstance(gates[1],MechanismClearanceGoal)
                        self.assertTrue(any(isinstance(p,FallingBlockTriggerGoal) and p.identity=='a:2'
                                            for p in route.portals))
                    else:self.assertIsInstance(gates[1],DashBlockBreakGoal)

    def test_other_moving_solids_keep_their_observed_occupancy(self):
        mover=Entity('zipMover',2,112,0,16,160,{},((112,200),))
        sim=fixture((block(1,x=48,width=16),mover));start=SimState('a',24,80)
        goal=Goal('target','a',144,80)
        routes=self.routes(sim,start,goal)
        self.assertTrue(routes)
        dash_open=_current_grid(sim,sim.map.rooms['a'],start,4,time.monotonic()+1,
                                frozenset({'a:1'}))
        self.assertFalse(dash_open.path(dash_open.nearest((start.x,start.y)),
            dash_open.nearest((goal.x,goal.y),from_point=False)),
            "Opening the dash block alone cannot erase the occupied zip")
        for route in routes:
            self.assertEqual(route.conditional_on,('a:1','a:2'))
            self.assertTrue(any(isinstance(p,DashBlockBreakGoal) and p.identity=='a:1'
                                for p in route.portals))
            self.assertTrue(any(isinstance(p,ZipMoverTriggerGoal) and p.identity=='a:2'
                                for p in route.portals))
            clearance=next(p for p in route.portals
                           if isinstance(p,MechanismClearanceGoal) and p.identity=='a:2')
            self.assertEqual(clearance.mechanism_kind,'zipMover')
            self.assertFalse(clearance.achieved(start))
            self.assertFalse(route.to_dict()['physically_verified'])
        moved=replace(start,mechanisms=(('a:2',1,10,112.,200.,0.),))
        routes=self.routes(sim,moved,goal)
        self.assertTrue(routes)
        self.assertTrue(all(route.conditional_on==('a:1',) for route in routes))
        self.assertEqual(start.mechanisms,())

    def test_translation_and_no_dash_resource_do_not_fabricate_reachability(self):
        for offset in ((0,0),(-192,-256)):
            ox,oy=offset;sim=fixture((block(),),offset=offset)
            start=SimState('a',24+ox,80+oy,dashes=0)
            routes=self.routes(sim,start,Goal('target','a',136+ox,80+oy))
            self.assertTrue(routes,'Geometry may propose a conditional route, never supply its dash')
            for route in routes:
                breaking=next(p for p in route.portals if isinstance(p,DashBlockBreakGoal))
                self.assertFalse(breaking.achieved(start))
                self.assertEqual(breaking.rect,(72+ox,oy,96+ox,160+oy))
                self.assertFalse(route.to_dict()['physically_verified'])
            self.assertEqual(start.dashes,0)


if __name__=='__main__':unittest.main()
