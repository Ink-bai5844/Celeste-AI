"""Independent confirmation/pass-through audit; no GUI host or game is used."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import time
import unittest

from celeste_search.assist import AssistController,PASSIVE,SharedAssistState
from celeste_search.assist_gui import AssistWindow,RouteDraft
from celeste_search.maps import MapData,Room
from celeste_search.physics import DT,Simulator
from celeste_search.routes import snapshot


def fixture():
    rows=("0"*16,)*8+("1"*16,)*2
    room=Room("a",0,0,128,80,rows,(),((16.,64.),))
    return MapData("workflow-fixture",{"a":room},Path("workflow-fixture.bin"))


def observed(state,**changes):
    result=state.to_dict()
    result.update(model_state=state.to_dict(),model_state_unavailable=[],
                  on_safe_ground=state.on_ground,input_phase="before_virtual_inputs",
                  control_mode="manual",human_input_mask=0,human_menu_input=False,
                  input_bindings_supported=True,delta_time=DT,raw_delta_time=DT,
                  human_input_epoch=2,spawn_id=1,deaths=0,area_id=1,area_mode=0,
                  room_entities={"entities":[],"do_not_load":[],"unavailable":[]})
    result.update(changes)
    return result


class PlannerStub:
    def __init__(self):
        self.requests=[]
        self.ready=None

    def cancel(self):
        self.ready=None

    def request(self,path,state,target,attempt,forbidden):
        self.requests.append((state,dict(target),attempt,list(forbidden)))
        return True

    def poll(self):
        result=self.ready
        self.ready=None
        return result


class ConfirmedTargetWorkflowAudit(unittest.TestCase):
    def setUp(self):
        self.data=fixture()
        self.state=replace(Simulator(self.data).initial_state("a"),x=40,frame=100)
        self.planner=PlannerStub()
        self.controller=AssistController(self.data,self.planner)

    def test_startup_and_cancelled_old_target_never_trigger_automatic_search(self):
        for frame in range(100,200):
            obs=observed(replace(self.state,frame=frame),human_input_mask=frame%3)
            self.assertEqual(self.controller.observe(obs),PASSIVE)
        self.assertEqual(self.planner.requests,[])
        self.controller.confirm({"room":"a","x":96,"y":56,"tile_size":8})
        self.controller.cancel()
        for frame in range(200,250):
            self.assertEqual(self.controller.observe(observed(replace(self.state,frame=frame))),PASSIVE)
        self.assertEqual(self.planner.requests,[])

    def test_click_only_selects_and_separate_confirmation_queues_exact_route(self):
        window=object.__new__(AssistWindow)
        window.map=self.data
        window.room=self.data.rooms["a"]
        window.scale=2
        window.offset=(20,20)
        window.shared=SharedAssistState()
        window.shared.update(connected=True)
        window.selection=SimpleNamespace(set=Mock())
        window.status=SimpleNamespace(set=Mock())
        window.canvas=SimpleNamespace(canvasx=lambda x:x,canvasy=lambda y:y)
        window.draw_dynamic=Mock()
        window.draft=RouteDraft()
        window.refresh_editor=Mock()
        window.select_tile(SimpleNamespace(x=20+96*2,y=20+56*2))
        self.assertTrue(window.shared.commands.empty())
        self.assertEqual(window.draft.waypoints,[{"room":"a","x":96,"y":56,"tile_size":8}])
        window.confirm()
        self.assertEqual(window.shared.commands.get_nowait(),{"kind":"route","route":window.draft.payload()})
        self.assertTrue(window.shared.commands.empty())

    def test_confirmed_search_uses_observed_nonspawn_state_and_keeps_inputs_passive(self):
        target={"room":"a","x":96,"y":56,"tile_size":8}
        self.controller.confirm(target)
        for frame in range(100,104):
            self.assertEqual(self.controller.observe(observed(replace(self.state,frame=frame))),PASSIVE)
        self.assertEqual(len(self.planner.requests),1)
        requested,target_used,_,_=self.planner.requests[0]
        self.assertEqual(requested.x,40)
        self.assertEqual(requested.y,64)
        self.assertEqual(target_used,target)
        self.assertEqual(self.controller.mode,"planning")

    def test_moving_after_confirmation_invalidates_old_origin_before_execution(self):
        target={"room":"a","x":96,"y":56,"tile_size":8}
        self.controller.confirm(target)
        for frame in range(100,103):
            self.controller.observe(observed(replace(self.state,frame=frame)))
        self.planner.ready={"can_execute":True,"reached":True,"actions":[1],
                            "anchor_state":self.state.to_dict()}
        moved=replace(self.state,x=56,frame=103)
        action=self.controller.observe(observed(moved,human_input_mask=1,human_input_epoch=3))
        self.assertEqual(action,PASSIVE)
        self.assertIsNone(self.controller.plan)
        self.assertEqual(self.controller.mode,"manual")
        self.assertTrue(any(e["event"]=="anchor_invalidated" for e in self.controller.events))

    def test_search_timeout_does_not_interrupt_an_already_admitted_route(self):
        self.controller.confirm({"room":"a","x":96,"y":56,"tile_size":8})
        self.controller.started=time.monotonic()-100
        self.controller.executing_started=time.monotonic()
        self.controller.mode="executing"
        self.controller.execution_human_epoch=2
        self.controller.execution_spawn=1
        self.controller.plan={"actions":[1,1],"expected":[snapshot(self.state)]*3}
        self.controller.index=0
        self.assertEqual(self.controller.observe(observed(self.state)),1)
        self.assertEqual(self.controller.mode,"executing")


if __name__=="__main__":
    unittest.main()
