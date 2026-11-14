"""Optional wall choices compare real arrivals before an expensive suffix.

The small deterministic simulator isolates scheduling from trajectory search.
It exposes the same transient dash event as Simulator, including a buffered
dash whose input press occurred before this edge.
"""
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

from celeste_search.assist_planning import _selected_goal
from celeste_search.corridor_planning import search_corridor_routes
from celeste_search.corridor_routes import (CorridorRoute, CorridorPortalGoal,
                                            MechanismPassageGoal)
from celeste_search.physics import SimState, LEFT, RIGHT, UP, DASH
from celeste_search.planner import SearchResult
from celeste_search.wall_moves import WallGripGoal


class ChoiceSimulator:
    map_data = None
    last_step_started_dash = False

    def __init__(self, *, low_finishes=False, high_dash=False, buffered_high=False):
        self.low_finishes=low_finishes
        self.high_dash=high_dash
        self.buffered_high=buffered_high

    def step(self, before, mask):
        self.last_step_started_dash=False
        fields=dict(frame=before.frame+1,prev_input=mask)
        if before.x==0 and mask==LEFT:
            return replace(before,x=20.,y=80.,state=1,on_ground=True,vy=0.,
                           dashes=1,stamina=110.,dash_buffer=0,**fields)
        if before.x==0 and mask==(RIGHT|DASH if self.high_dash else RIGHT):
            self.last_step_started_dash=self.high_dash or self.buffered_high
            # An actual same-frame refill may restore the consumed dash.
            return replace(before,x=44.,y=40.,state=1,on_ground=False,vy=0.,
                           dashes=1,stamina=110.,dash_buffer=0,**fields)
        if mask==UP and (before.x==44 or before.x==20 and self.low_finishes):
            return replace(before,x=84.,y=8.,state=0,on_ground=False,**fields)
        return replace(before,dead=True,reason='fixture_no_edge',**fields)


class CorridorWallChoiceTests(unittest.TestCase):
    def setUp(self):
        self.start=SimState('a',0.,100.,facing=1,on_ground=False)
        self.low=WallGripGoal('wall_grip','a',20.,80.,id='low',
                             rect=(24,0,32,96),facing=1)
        self.high=WallGripGoal('wall_grip','a',44.,40.,id='high',
                              rect=(48,0,56,96),facing=1)
        self.route=dict(waypoints=[dict(room='a',x=80,y=0,tile_size=8)],
                        segments=[dict(no_dash=False,no_ground=False)])
        self.goal=_selected_goal(self.route['waypoints'][0])[0]

    def solve(self, simulator, *, first=None, deny_first=False, budget=24):
        first=first or self.low
        corridor=CorridorRoute('choices','a',(first,self.high),
                              ((0.,100.),(20.,80.),(44.,40.),(84.,8.)),(),120.)
        probes=[];searches=[]

        def macros(wrapped, origin, goal, *, stats, **kwargs):
            probes.append((origin.x,goal.id,wrapped.no_dash))
            stats['expansions']+=1
            if origin.x!=0:
                return []
            mask=LEFT if goal.id==first.id else RIGHT|DASH if simulator.high_dash else RIGHT
            after=wrapped.step(origin,mask)
            stats['simulated_frames']+=1
            if after.dead or not goal.achieved(after):
                return []
            return [([mask],after,'fixture_macro')]

        def search(wrapped, origin, goal, **kwargs):
            searches.append((origin.x,goal.id,kwargs['max_expansions']))
            actions=[]
            if origin.x==0 and goal.id==first.id and first.kind!='wall_grip' and not deny_first:
                actions=[LEFT]
            elif origin.x==44 or origin.x==20 and simulator.low_finishes:
                actions=[UP]
            state=origin
            for action in actions:
                state=wrapped.step(state,action)
            reached=bool(actions and not state.dead and goal.achieved(state))
            # A bad low-wall suffix really uses the entire quota granted to it.
            expanded=1 if reached else kwargs['max_expansions']
            return SearchResult(actions,state,reached,expanded,len(actions),'fixture')

        with patch('celeste_search.corridor_planning.generate_corridor_routes',return_value=[corridor]), \
             patch('celeste_search.wall_moves.wall_approach_edges',side_effect=macros), \
             patch('celeste_search.corridor_planning.search_leg',side_effect=search):
            result=search_corridor_routes(simulator,self.start,self.route,[self.goal],
                deadline=time.monotonic()+2,max_expansions=budget,max_frames=10)
        return result,probes,searches

    def test_equal_cost_higher_wall_is_compared_before_low_suffix_spends_budget(self):
        result,probes,searches=self.solve(ChoiceSimulator())
        self.assertTrue(result[2])
        self.assertEqual(result[0],[RIGHT,UP])
        self.assertTrue(any(x==0 and name=='low' for x,name,_ in probes))
        self.assertTrue(any(x==0 and name=='high' for x,name,_ in probes))
        self.assertFalse(any(x==20 for x,_,_ in searches))
        self.assertLess(result[3],24)
        choices=[row for row in result[6]['corridor_attempts'] if row['reason']=='wall_alternative']
        self.assertEqual(choices[0]['portal_index'],1)

    def test_extra_dash_high_wall_cannot_outrank_zero_dash_low_wall(self):
        result,_,_=self.solve(ChoiceSimulator(low_finishes=True,high_dash=True))
        self.assertTrue(result[2])
        self.assertEqual(result[0],[LEFT,UP])

    def test_buffered_actual_dash_is_charged_even_without_a_new_dash_press(self):
        self.start=replace(self.start,dash_buffer=1)
        result,_,_=self.solve(ChoiceSimulator(low_finishes=True,buffered_high=True))
        self.assertTrue(result[2])
        self.assertEqual(result[0],[LEFT,UP])

    def test_support_and_resource_stops_do_not_enable_wall_bypass(self):
        for kind,role in [('surface','support'),('refill','refill')]:
            with self.subTest(kind=kind):
                first=CorridorPortalGoal(kind,'a',20.,80.,id=kind,
                    bounds=(19,79,21,81),role=role)
                result,probes,_=self.solve(ChoiceSimulator(low_finishes=True),first=first)
                self.assertTrue(result[2])
                self.assertEqual(result[0][0],LEFT)
                self.assertFalse(any(x==0 and name=='high' for x,name,_ in probes))
                self.assertFalse(any(row['reason']=='wall_alternative'
                                     for row in result[6]['corridor_attempts']))

    def test_identity_backed_wall_is_not_sorted_as_an_optional_static_wall(self):
        first=replace(self.low,identity='a:7',mechanism_kind='fallingBlock')
        result,probes,_=self.solve(ChoiceSimulator(low_finishes=True),first=first)
        self.assertTrue(result[2]);self.assertEqual(result[0],[LEFT,UP])
        self.assertFalse(any(x==0 and name=='high' for x,name,_ in probes))
        self.assertFalse(any(row['reason']=='wall_alternative'
                             for row in result[6]['corridor_attempts']))

    def test_required_mechanism_passage_cannot_be_skipped_for_high_wall(self):
        first=MechanismPassageGoal('corridor','a',20.,80.,id='mechanism_passage:a:7',
                                  bounds=(19,79,21,81),identity='a:7')
        result,probes,_=self.solve(ChoiceSimulator(low_finishes=True),first=first,deny_first=True)
        self.assertFalse(result[2]);self.assertEqual(probes,[])


if __name__=='__main__':
    unittest.main()
