"""Live controller tests: fake background futures and telemetry, no real game."""

from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest

from celeste_search.live import LiveController, LiveOptions
from celeste_search.maps import MapData, Room
from celeste_search.physics import RIGHT, SimState, Simulator
from celeste_search.room_planning import action_fingerprint
from celeste_search.routes import snapshot


def raw_observation(state, frame, *, spawn=1, deaths=0, **changes):
    model=state.to_dict()
    model["auto_jump_active"]=False
    data=snapshot(state)
    data.update(frame=frame,stamina=state.stamina,state=state.state,on_safe_ground=state.on_ground,
                input_phase="before_virtual_inputs",spawn_id=spawn,deaths=deaths,
                model_state=model,model_state_unavailable=[],model_state_approximate=[])
    data.update(changes)
    return data


class FakeBackground:
    """Matches generation invalidation without any blocking result interface."""
    def __init__(self,map_data):
        self.sim=Simulator(map_data)
        self.generation=0
        self.pending=False
        self.requests=[]
        self.ready=[]
        self.cancel_count=0
        self.poll_count=0

    def request(self,state,variant,forbidden,visited_rooms,exploration=None,recovery=False):
        if self.pending:
            return False
        self.generation+=1
        self.pending=True
        self.requests.append({"state":state,"variant":variant,"forbidden":list(forbidden),
                              "visited":list(visited_rooms),"generation":self.generation})
        return True

    def poll(self):
        self.poll_count+=1
        while self.ready:
            generation,result=self.ready.pop(0)
            if generation==self.generation:
                self.pending=False
                return result
        return None

    def cancel_epoch(self):
        self.generation+=1
        self.cancel_count+=1
        self.pending=False

    def publish(self,result,*,generation=None):
        self.ready.append((self.generation if generation is None else generation,result))

    def candidate(self,masks=None):
        masks=list(masks or [RIGHT]*16)
        start=self.requests[-1]["state"]
        states=self.sim.trace(start,masks)
        return {"can_execute":True,"actions":masks,"expected":[snapshot(s) for s in states],
                "final_state":states[-1].to_dict(),"fingerprint":action_fingerprint(masks),
                "anchor_state":start.to_dict(),"generation":self.generation,"status":"partial"},states


class LiveTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        source=Path(temporary.name)/"fixture.bin"
        source.write_bytes(b"live controller fixture map identity")
        rows=("0"*40,)*8+("1"*40,)*2
        room=Room("a",0,0,320,80,rows,(),((24.0,64.0),))
        other=replace(room,name="b",x=320,spawns=((344.0,64.0),))
        self.map=MapData("fixture",{"a":room,"b":other},source)
        self.sim=Simulator(self.map)
        self.start=self.sim.initial_state("a")

    def controller(self,**options):
        background=FakeBackground(self.map)
        return LiveController(self.map,background,LiveOptions(**options)),background

    def prime(self,controller,background):
        for frame in range(4):
            self.assertEqual(controller.observe(raw_observation(self.start,frame)),0)
        self.assertEqual(len(background.requests),1)
        self.assertIsNotNone(controller.pending_anchor)

    def activate(self,controller,background):
        self.prime(controller,background)
        candidate,states=background.candidate()
        background.publish(candidate)
        self.assertEqual(controller.observe(raw_observation(self.start,4)),RIGHT)
        return candidate,states

    def test_game_updates_continue_neutrally_while_background_work_is_pending(self):
        controller,background=self.controller()
        self.prime(controller,background)
        for frame in range(4,90):
            self.assertEqual(controller.observe(raw_observation(self.start,frame)),0)
        self.assertEqual(controller.frame,89)
        self.assertEqual(len(background.requests),1)
        self.assertGreater(background.poll_count,80)
        self.assertFalse(controller.finished)
        self.assertEqual(controller.journal()["chapter_restarts"],0)

    def test_ready_candidate_executes_only_after_actual_anchor_check(self):
        controller,background=self.controller()
        candidate,states=self.activate(controller,background)
        self.assertIs(controller.plan,candidate)
        self.assertEqual(controller.index,1)
        self.assertEqual(controller.observe(raw_observation(states[1],5)),RIGHT)
        self.assertEqual(len(background.requests),1)
        self.assertEqual(controller.confirmed,set())
        event=next(e for e in controller.events if e["event"]=="execute_selected_route")
        self.assertEqual(event["game_frames_during_search"],1)

    def test_moved_anchor_discards_ready_route_without_resetting_or_executing(self):
        controller,background=self.controller()
        self.prime(controller,background)
        candidate,_=background.candidate()
        moved=replace(self.start,x=self.start.x+20)
        self.assertEqual(controller.observe(raw_observation(moved,4)),0)
        background.publish(candidate)
        self.assertEqual(controller.observe(raw_observation(moved,5)),0)
        self.assertIsNone(controller.plan)
        self.assertTrue(any(e["event"]=="stale_plan_discarded" for e in controller.events))
        self.assertEqual(controller.room_retries,0)

    def test_natural_respawn_invalidates_old_epoch_and_never_replays_chapter(self):
        controller,background=self.controller()
        self.prime(controller,background)
        candidate,_=background.candidate()
        old_generation=background.generation
        cancellations=background.cancel_count
        self.assertEqual(controller.observe(raw_observation(self.start,4,spawn=2,deaths=1)),0)
        self.assertGreater(background.cancel_count,cancellations)
        background.publish(candidate,generation=old_generation)
        for frame in range(5,20):
            self.assertEqual(controller.observe(raw_observation(self.start,frame,spawn=2,deaths=1)),0)
        self.assertIsNone(controller.plan)
        self.assertEqual(len(background.requests),2)
        self.assertEqual(controller.deaths,1)
        self.assertFalse(controller.finished)
        self.assertEqual(controller.journal()["rollback_method"],"native_current_room_respawn")
        self.assertEqual(controller.journal()["chapter_restarts"],0)

    def test_natural_death_during_execution_releases_keys_and_replans(self):
        controller,background=self.controller()
        candidate,_=self.activate(controller,background)
        answer=controller.observe(raw_observation(self.start,5,spawn=2,deaths=1))
        self.assertEqual(answer,0)
        self.assertIsNone(controller.plan)
        self.assertEqual(controller.failures[-1]["reason"],"natural_death")
        self.assertIn(candidate["fingerprint"],set().union(*controller.blacklist.values()))
        self.assertFalse(controller.finished)
        self.assertEqual(controller.failures_since_spawn,0)

    def test_prediction_divergence_releases_inputs_instead_of_restarting(self):
        controller,background=self.controller()
        candidate,_=self.activate(controller,background)
        answer=controller.observe(raw_observation(replace(self.start,x=150),5))
        self.assertEqual(answer,0)
        self.assertIsNone(controller.plan)
        self.assertEqual(controller.failures[-1]["reason"],"prediction_diverged")
        self.assertIn(candidate["fingerprint"],set().union(*controller.blacklist.values()))
        self.assertEqual(controller.room_retries,0)
        self.assertFalse(controller.finished)

    def test_repeated_no_route_requests_only_native_current_room_retry(self):
        controller,background=self.controller(retries=2,max_room_retries=3)
        self.prime(controller,background)
        background.publish({"can_execute":False,"reason":"budget"})
        self.assertEqual(controller.observe(raw_observation(self.start,4)),0)
        self.assertEqual(controller.room_retries,0)
        for frame in range(5,19):
            self.assertEqual(controller.observe(raw_observation(self.start,frame)),0)
        self.assertEqual(len(background.requests),2)
        background.publish({"can_execute":False,"reason":"budget"})
        self.assertEqual(controller.observe(raw_observation(self.start,19)),254)
        self.assertEqual(controller.room_retries,1)
        self.assertFalse(controller.finished)
        self.assertEqual(controller.observe(raw_observation(self.start,20,spawn=2,deaths=1)),0)
        retry=next(e for e in controller.events if e["event"]=="retry_current_room")
        self.assertFalse(retry["chapter_restart"])

    def test_search_budget_does_not_kill_player_already_at_unchanged_checkpoint(self):
        controller,background=self.controller(retries=1)
        self.prime(controller,background)
        background.publish({"can_execute":False,"reason":"budget"})
        answer=controller.observe(raw_observation(self.start,4,respawn_x=self.start.x,respawn_y=self.start.y,
                                                  room_entities={"entities":[]}))
        self.assertEqual(answer,0)
        self.assertEqual(controller.room_retries,0)
        self.assertTrue(any(e["event"]=="continue_offline_search" for e in controller.events))

    def test_fled_winged_candidate_requests_only_current_room_respawn(self):
        controller,background=self.controller()
        self.prime(controller,background)
        background.publish({"can_execute":False,"requires_room_respawn":True,
                            "reason":"winged_berry_fled_current_room_respawn_needed",
                            "pending_goals":["a:7"]})
        self.assertEqual(controller.observe(raw_observation(self.start,4)),254)
        self.assertEqual(controller.room_retries,1)
        self.assertFalse(controller.finished)
        self.assertEqual(controller.events[-1]["pending_goals"],["a:7"])

    def test_future_recovery_waits_for_exact_frame_and_private_phase(self):
        controller,background=self.controller()
        self.prime(controller,background)
        controller.pending_kind="periodic"
        candidate,states=background.candidate()
        candidate.update(execute_at_frame=8,execution_anchor_state=self.start.to_dict(),
                         requires_periodic_anchor=True)
        background.publish(candidate)
        for frame in range(4,8):
            self.assertEqual(controller.observe(raw_observation(self.start,frame)),0)
        self.assertEqual(controller.observe(raw_observation(self.start,8)),RIGHT)

    def test_future_recovery_rejects_subpixel_mismatch_without_reset(self):
        controller,background=self.controller()
        self.prime(controller,background)
        controller.pending_kind="periodic"
        candidate,states=background.candidate()
        candidate.update(execute_at_frame=4,execution_anchor_state=self.start.to_dict(),
                         requires_periodic_anchor=True)
        background.publish(candidate)
        self.assertEqual(controller.observe(raw_observation(replace(self.start,rem_y=.4),4)),0)
        self.assertIsNone(controller.plan)
        self.assertEqual(controller.room_retries,0)

    def test_time_budget_detaches_in_place_without_chapter_reset(self):
        controller,background=self.controller(max_seconds=1)
        controller.started=time.monotonic()-2
        self.assertEqual(controller.observe(raw_observation(self.start,0)),253)
        self.assertTrue(controller.finished)
        self.assertEqual(controller.reason,"time_budget")
        self.assertEqual(background.requests,[])

    def test_reattach_adopts_current_counter_without_replaying(self):
        controller,background=self.controller()
        controller.adopt_next_frame=True
        self.assertEqual(controller.observe(raw_observation(self.start,16000)),0)
        self.assertEqual(controller.frame,16000)
        self.assertFalse(controller.finished)
        self.assertEqual(background.requests,[])

    def test_default_room_retry_count_does_not_end_continuous_play(self):
        controller,background=self.controller(retries=1)
        controller.room_retries=100
        self.prime(controller,background)
        background.publish({"can_execute":False,"reason":"budget"})
        self.assertEqual(controller.observe(raw_observation(self.start,4)),254)
        self.assertEqual(controller.room_retries,101)
        self.assertFalse(controller.finished)

    def test_only_actual_observed_berries_are_confirmed(self):
        controller,background=self.controller()
        self.prime(controller,background)
        candidate,states=background.candidate()
        candidate["final_state"]["berries"]=["a:999"]
        background.publish(candidate)
        controller.observe(raw_observation(self.start,4))
        for index in range(1,len(candidate["actions"])+1):
            answer=controller.observe(raw_observation(states[index],4+index,berries=[]))
        self.assertEqual(answer,0)
        self.assertEqual(controller.failures[-1]["reason"],"predicted_berry_not_confirmed")
        self.assertEqual(controller.confirmed,set())
        controller.observe(raw_observation(replace(self.start,x=60),21,berries=["a:7"]))
        self.assertEqual(controller.journal()["confirmed_berries"],["a:7"])

    def test_room_change_cancels_pending_anchor_and_records_actual_room(self):
        controller,background=self.controller()
        self.prime(controller,background)
        candidate,_=background.candidate()
        old_generation=background.generation
        actual=replace(self.start,room="b",x=355)
        self.assertEqual(controller.observe(raw_observation(actual,4)),0)
        background.publish(candidate,generation=old_generation)
        self.assertEqual(controller.observe(raw_observation(actual,5)),0)
        self.assertEqual(controller.visited,{"a","b"})
        self.assertEqual(controller.room,"b")
        self.assertIsNone(controller.plan)
        self.assertIsNone(controller.pending_anchor)


if __name__=="__main__":
    unittest.main()
