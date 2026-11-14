"""Exact cached stepping preserves full state, events and cancellation guards."""
from dataclasses import replace
from pathlib import Path
import unittest

from celeste_search.cassette import CassetteClock
from celeste_search.maps import Entity, MapData, Room
from celeste_search.physics import DASH, GRAB, JUMP, RIGHT, SimState, Simulator
from celeste_search.step_cache import ExactStepCache


def simulator(entities=()):
    room=Room("cache",0,0,160,80,("0"*20,)*6+("1"*20,)*4,
              tuple(entities),((16.,48.),),{})
    return Simulator(MapData("step-cache",{room.name:room},Path("fixture.bin")))


class ExactStepCacheTests(unittest.TestCase):
    def test_raw_and_cached_trajectories_match_every_field_and_event(self):
        entities=(Entity("zipMover",1,80,24,16,24,{},((104,24),)),
                  Entity("refill",2,32,42,0,0))
        raw=simulator(entities)
        base=simulator(entities)
        cache=ExactStepCache(base)
        start=replace(raw.initial_state(),mechanisms=(("cache:1",2,8,82.,24.,0.),))
        recipes=([RIGHT]*8+[JUMP|RIGHT]*8+[RIGHT|DASH]*10+[GRAB|RIGHT]*12,
                 [RIGHT]*8+[JUMP|RIGHT]*8+[RIGHT]*20,
                 [RIGHT]*8+[JUMP|RIGHT]*8+[RIGHT|DASH]*10+[GRAB|RIGHT]*12)
        for actions in recipes:
            a=b=start
            for mask in actions:
                a=raw.step(a,mask)
                b=cache.step(b,mask)
                self.assertEqual(a,b)
                self.assertEqual(cache.last_step_started_dash,raw.last_step_started_dash)
        self.assertGreater(cache.hits,40)
        self.assertEqual(cache.hits+cache.misses,sum(map(len,recipes)))
        self.assertEqual(start.frame,0)
        self.assertIs(cache.map,base.map)

    def test_hit_restores_hidden_dash_event_and_false_on_next_hit(self):
        for entities in ((),(Entity("refill",1,16,42,0,0),),
                         (Entity("spring",1,16,48,0,0),)):
            with self.subTest(entities=entities):
                base=simulator(entities);cache=ExactStepCache(base);start=base.initial_state()
                dashed=cache.step(start,DASH)
                self.assertTrue(cache.last_step_started_dash)
                dead=replace(start,dead=True)
                self.assertEqual(cache.step(dead,0),dead)
                self.assertFalse(cache.last_step_started_dash)
                self.assertIs(cache.step(start,DASH),dashed)
                self.assertTrue(cache.last_step_started_dash)
                self.assertTrue(base.last_step_started_dash)
                cache.step(dead,0)
                self.assertFalse(cache.last_step_started_dash)
                self.assertEqual(cache.hits,2)
        base=simulator();cache=ExactStepCache(base)
        falling=SimState("cache",16,47,vy=60,dashes=0)
        landed=cache.step(falling,DASH)
        dashed=cache.step(landed,0)
        self.assertTrue(cache.last_step_started_dash)
        cache.step(falling,DASH)
        self.assertFalse(cache.last_step_started_dash)
        self.assertEqual(cache.step(landed,0),dashed)
        self.assertTrue(cache.last_step_started_dash)

    def test_full_frame_phase_timer_depth_clock_and_input_remain_in_key(self):
        base=simulator((Entity("zipMover",1,80,24,16,24,{},((104,24),)),))
        cache=ExactStepCache(base)
        start=replace(base.initial_state(),mechanisms=(("cache:1",2,8,82.,24.,0.),))
        clock=CassetteClock(0,.1,0,1.,0,4,False,0,True,True,3)
        variants=(start,replace(start,frame=1),replace(start,prev_input=RIGHT),
            replace(start,mechanisms=(("cache:1",4,8,82.,24.,0.),)),
            replace(start,mechanisms=(("cache:1",2,9,82.,24.,0.),)),
            replace(start,player_update_order=2),replace(start,player_depth=-1),
            replace(start,player_actual_depth=.00001),replace(start,cassette_clock=clock),
            replace(start,cassette_clock=replace(clock,beat_timer=.12)),
            replace(start,cassette_clock=replace(clock,depth_counters=((10,.00001),))))
        for state in variants:
            self.assertEqual(cache.step(state,0),base.step(state,0))
        cache.step(start,RIGHT)
        self.assertEqual(cache.hits,0)
        self.assertEqual(cache.misses,len(variants)+1)
        for state in variants:cache.step(state,0)
        self.assertEqual(cache.hits,len(variants))

    def test_lru_is_bounded_and_hit_refreshes_recency(self):
        base=simulator();cache=ExactStepCache(base,max_entries=2)
        start=base.initial_state();a,b,c=(replace(start,frame=i) for i in range(3))
        cache.step(a,0);cache.step(b,0);cache.step(a,0);cache.step(c,0)
        self.assertEqual(cache.cache_info()["entries"],2)
        self.assertEqual(cache.evictions,1)
        cache.step(a,0)
        self.assertEqual(cache.hits,2)
        cache.step(b,0)
        self.assertEqual(cache.misses,4)
        self.assertEqual(cache.evictions,2)
        cache.clear()
        self.assertEqual(cache.cache_info()["entries"],0)
        for bad in (0,32769,True,1.5):
            with self.assertRaises(ValueError):ExactStepCache(base,max_entries=bad)

    def test_instance_override_requires_guard_and_checks_every_logical_step_once(self):
        base=simulator();original=base.step;calls=[];guards=[];cancelled=[False]
        def guard():
            guards.append(1)
            if cancelled[0]:raise RuntimeError("cancelled")
        def checked(state,mask):
            guard();calls.append(1)
            return original(state,mask)
        base.step=checked
        bypass=ExactStepCache(base);start=base.initial_state()
        bypass.step(start,0);bypass.step(start,0)
        self.assertFalse(bypass.enabled)
        self.assertEqual((len(calls),len(guards),bypass.bypasses),(2,2,2))
        checked.cache_hit_guard=guard
        cache=ExactStepCache(base)
        cache.step(start,0);cache.step(start,0)
        self.assertTrue(cache.enabled)
        self.assertEqual((len(calls),len(guards)),(3,4))
        self.assertEqual((cache.misses,cache.hits),(1,1))
        cancelled[0]=True
        with self.assertRaisesRegex(RuntimeError,"cancelled"):cache.step(start,0)
        self.assertEqual((len(calls),len(guards)),(3,5))

    def test_mock_constraint_wrapper_and_other_requests_cannot_share_cache(self):
        base=simulator();calls=[]
        class Restricted:
            def __getattr__(self,name):return getattr(base,name)
            def step(self,state,mask):
                calls.append(1)
                return replace(base.step(state,mask),dead=True,reason="route_limit")
        cache=ExactStepCache(Restricted());start=base.initial_state()
        for _ in range(2):self.assertTrue(cache.step(start,0).dead)
        self.assertEqual(len(calls),2)
        self.assertFalse(cache.enabled)
        first,second=ExactStepCache(base),ExactStepCache(simulator())
        first.step(start,0);second.step(start,0)
        self.assertEqual((first.misses,second.misses),(1,1))
        base.last_step_started_dash=True
        with self.assertRaises(ValueError):first.step(start,128)
        self.assertFalse(base.last_step_started_dash)
        self.assertEqual(first.cache_info()["entries"],1)


if __name__=="__main__":
    unittest.main()
