from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from celeste_search.physics import SimState,Simulator
from celeste_search.maps import Room,MapData
from celeste_search.assist_planning import plan_to_route
from celeste_search.search_progress import (progress_reporter,search_tracker,remember_segment,_ACTIVE,
                                           publish_corridor_progress)


class SearchProgressTests(unittest.TestCase):
    def setUp(self):
        self.start=SimState("a",16,96,on_ground=True)
        self.route={"waypoints":[{"room":"a","x":40,"y":88},
                                  {"room":"a","x":80,"y":88}],
                    "segments":[{"no_dash":False,"no_ground":False}]*2}

    def test_throttles_before_constructing_paths_and_never_publishes_controls(self):
        packets=[];clock=[0.];builds=[]
        def build():
            builds.append(1)
            yield "best",self.start,[self.start]
        with patch("celeste_search.search_progress.time.monotonic",side_effect=lambda:clock[0]), \
             progress_reporter(packets.append,self.start.to_dict(),self.route):
            tracker=search_tracker(self.start)
            tracker.report(build,2,20)
            clock[0]=.2;tracker.report(build,4,40)
            clock[0]=.4;tracker.report(build,7,70)
        self.assertEqual(len(packets),2)
        self.assertEqual(len(builds),2)
        self.assertEqual(packets[-1]["expansions"],7)
        self.assertEqual(packets[-1]["simulated_frames"],70)
        self.assertNotIn("actions",str(packets))

    def test_decimation_preserves_actual_anchor_and_both_sides_of_room_crossing(self):
        states=[replace(self.start,x=16+i,room="a" if i<151 else "b") for i in range(400)]
        packets=[]
        with progress_reporter(packets.append,self.start,self.route):
            search_tracker(self.start).report(lambda:[("best",states[-1],states)],1,400)
        path=packets[0]["paths"][0]
        self.assertFalse(path["local_leg"])
        self.assertLessEqual(len(path["points"]),120)
        actual={(state.room,state.x,state.y) for state in states}
        published={(point["room"],point["x"],point["y"]) for point in path["points"]}
        self.assertTrue(published<=actual)
        for index in (0,150,151,399):
            state=states[index]
            self.assertIn((state.room,state.x,state.y),published)

    def test_completed_internal_legs_and_real_settling_compose_from_job_anchor(self):
        packets=[];clock=[0.]
        midpoint=replace(self.start,x=40,frame=10)
        settled=replace(midpoint,frame=12,prev_input=0,rem_x=.2)
        end=replace(settled,x=80,frame=20)
        with patch("celeste_search.search_progress.time.monotonic",side_effect=lambda:clock[0]), \
             progress_reporter(packets.append,self.start,self.route):
            search_tracker(self.start).report(lambda:[("best",midpoint,[self.start,midpoint])],1,10,remember=True)
            remember_segment(midpoint,settled,[midpoint,settled])
            clock[0]=1
            search_tracker(settled).report(lambda:[("best",end,[settled,end])],2,8)
        path=packets[-1]["paths"][0]
        self.assertFalse(path["local_leg"])
        self.assertEqual(path["points"][0],dict(room="a",x=16,y=96))
        self.assertEqual(path["points"][-1],dict(room="a",x=80,y=96))
        self.assertEqual(path["waypoints_reached"],2)

    def test_short_remembered_prefix_defers_route_scan_until_it_is_displayed(self):
        from celeste_search.search_progress import _route_marks
        packets=[]
        midpoint=replace(self.start,x=40,frame=10)
        end=replace(midpoint,x=80,frame=20)
        with progress_reporter(packets.append,self.start,self.route), \
             patch("celeste_search.search_progress._route_marks",wraps=_route_marks) as marks:
            remember_segment(self.start,midpoint,[self.start,midpoint])
            marks.assert_not_called()
            search_tracker(midpoint).report(lambda:[("best",end,[midpoint,end])],1,10)
            marks.assert_called_once()
        path=packets[-1]["paths"][0]
        self.assertFalse(path["local_leg"])
        self.assertEqual(path["waypoints_reached"],2)
        self.assertEqual(path["points"][0],dict(room="a",x=16,y=96))

    def test_throttled_remember_does_not_construct_unused_alternative(self):
        packets=[];clock=[0.];built=[]
        midpoint=replace(self.start,x=40,frame=10)
        end=replace(midpoint,x=80,frame=20)
        def build():
            built.append("best")
            yield "best",midpoint,[self.start,midpoint]
            built.append("unused")
            yield "alternative",end,[self.start,midpoint,end]
        with patch("celeste_search.search_progress.time.monotonic",side_effect=lambda:clock[0]), \
             progress_reporter(packets.append,self.start,self.route):
            tracker=search_tracker(self.start)
            tracker.report(lambda:[("best",self.start,[self.start])],0,0)
            clock[0]=.1
            tracker.report(build,1,10,remember=True)
            self.assertEqual(len(packets),1)
            self.assertEqual(built,["best"])
            clock[0]=.5
            search_tracker(midpoint).report(lambda:[("best",end,[midpoint,end])],1,10)
        self.assertEqual(packets[-1]["paths"][0]["waypoints_reached"],2)
        self.assertFalse(packets[-1]["paths"][0]["local_leg"])

    def test_unknown_internal_anchor_is_explicitly_local_not_falsely_connected(self):
        packets=[];local=replace(self.start,x=100)
        with progress_reporter(packets.append,self.start,self.route):
            search_tracker(local).report(lambda:[("best",local,[local])],1,1)
        path=packets[0]["paths"][0]
        self.assertTrue(path["local_leg"])
        self.assertEqual(path["points"][0]["x"],100)
        self.assertEqual(path["waypoints_reached"],0)

    def test_public_waypoint_count_does_not_count_internal_resource_knots(self):
        packets=[]
        states=[replace(self.start,x=x) for x in (16,24,32,40,48,64,80)]
        with progress_reporter(packets.append,self.start,self.route):
            search_tracker(self.start).report(lambda:[("best",states[-1],states)],1,7)
        self.assertEqual(packets[0]["waypoints_reached"],2)

    def test_callback_failure_and_context_cleanup_do_not_escape_into_solver(self):
        def failure(packet):
            raise RuntimeError("queue unavailable")
        with progress_reporter(failure,self.start,self.route):
            search_tracker(self.start).report(lambda:[("best",self.start,[self.start])],1,1)
        self.assertIsNone(search_tracker(self.start))

    def test_no_ground_preview_does_not_count_a_grounded_start_as_a_waypoint(self):
        packets=[]
        route={"waypoints":[{"room":"a","x":16,"y":88}],
               "segments":[{"no_ground":True}]}
        with progress_reporter(packets.append,self.start,route):
            search_tracker(self.start).report(lambda:[("best",self.start,[self.start])],1,1)
        self.assertEqual(packets[0]["waypoints_reached"],0)

    def test_prefix_registry_is_bounded(self):
        with progress_reporter(lambda packet:None,self.start,self.route):
            for index in range(180):
                end=replace(self.start,x=100+index)
                remember_segment(self.start,end,[self.start,end])
            reporter=_ACTIVE.get()
            self.assertLessEqual(len(reporter.prefixes),128)
            self.assertLessEqual(reporter.prefix_points,20000)

    def test_geometry_is_separate_and_cannot_claim_public_waypoints_or_controls(self):
        packets=[]
        corridors=[dict(id="c1",status="pending",reason="",actions=[1,2],waypoints_reached=2,
                        points=[dict(room="a",x=16,y=96),dict(room="a",x=80,y=96)])]
        with progress_reporter(packets.append,self.start,self.route):
            publish_corridor_progress(corridors)
        packet=packets[0]
        self.assertEqual(packet["paths"],[])
        self.assertEqual(packet["waypoints_reached"],0)
        self.assertEqual(packet["topology"]["corridors"][0]["status"],"pending")
        self.assertNotIn("actions",str(packet))
        self.assertNotIn("waypoints_reached",packet["topology"]["corridors"][0])
        corridors[0]["points"][0]["x"]=999
        self.assertEqual(packet["topology"]["corridors"][0]["points"][0]["x"],16)

    def test_physical_packets_keep_corridor_status_and_new_attempt_clears_old_paths(self):
        packets=[]
        corridors=[dict(id=f"c{i}",status="trying" if i==1 else "pending",reason="",
                        points=[dict(room="a",x=16,y=96),dict(room="a",x=80,y=96)]) for i in (1,2)]
        with progress_reporter(packets.append,self.start,self.route):
            publish_corridor_progress(corridors,"c1")
            search_tracker(self.start).report(lambda:[("best",self.start,[self.start])],3,20)
            self.assertEqual(packets[-1]["topology"]["active_id"],"c1")
            self.assertTrue(packets[-1]["paths"])
            corridors[0].update(status="failed",reason="portal_unreachable")
            publish_corridor_progress(corridors,"c1")
            self.assertTrue(packets[-1]["paths"])
            corridors[1]["status"]="trying"
            publish_corridor_progress(corridors,"c2")
            self.assertEqual(packets[-1]["paths"],[])
            self.assertEqual(packets[-1]["topology"]["corridors"][0]["reason"],"portal_unreachable")

    def test_corridor_payload_is_bounded_and_invalid_geometry_is_not_connected(self):
        packets=[]
        points=[dict(room="a",x=x,y=96) for x in range(300)]
        corridors=[dict(id=f"c{i}",status="pending",reason="x"*500,points=points) for i in range(30)]
        with progress_reporter(packets.append,self.start,self.route):
            publish_corridor_progress(corridors,"unknown")
            packet=packets[-1]["topology"]
            self.assertEqual(len(packet["corridors"]),24)
            self.assertIsNone(packet["active_id"])
            for item in packet["corridors"]:
                self.assertEqual(len(item["points"]),60)
                self.assertEqual((item["points"][0]["x"],item["points"][-1]["x"]),(0,299))
                self.assertEqual(len(item["reason"]),160)
            bad=dict(id="bad",status="trying",points=[points[0],dict(room="a",x=float("nan"),y=0),points[-1]])
            publish_corridor_progress([bad],"bad")
            self.assertEqual(packets[-1]["topology"]["corridors"],[])

    def test_corridor_callback_failure_and_absent_context_cannot_affect_search(self):
        corridor=dict(id="c",status="verified",points=[dict(room="a",x=16,y=96),dict(room="a",x=40,y=96)])
        publish_corridor_progress([corridor],"c")
        with progress_reporter(lambda packet:(_ for _ in ()).throw(RuntimeError("closed")),self.start,self.route):
            publish_corridor_progress([corridor],"c")
        self.assertIsNone(_ACTIVE.get())

    def test_explicit_empty_corridors_clear_old_geometry_and_physical_preview(self):
        packets=[]
        corridor=dict(id="c",status="trying",points=[dict(room="a",x=16,y=96),dict(room="a",x=40,y=96)])
        with progress_reporter(packets.append,self.start,self.route):
            publish_corridor_progress([corridor],"c")
            search_tracker(self.start).report(lambda:[("best",self.start,[self.start])],1,10)
            self.assertTrue(packets[-1]["paths"])
            publish_corridor_progress([])
        self.assertEqual(packets[-1]["topology"],dict(active_id=None,corridors=[]))
        self.assertEqual(packets[-1]["paths"],[])
        self.assertEqual(packets[-1]["waypoints_reached"],0)

    def test_real_ordered_and_single_leg_solvers_keep_identical_controls_with_previews(self):
        room=Room("a",0,0,128,112,("0"*16,)*12+("1"*16,)*2,(),((16.,96.),))
        data=MapData("preview_fixture",{"a":room},Path("preview_fixture.bin"))
        route={"waypoints":[{"room":"a","x":80,"y":88}]}
        for strategy in ("ordered_greedy","direct_greedy"):
            results=[];packets=[]
            for enabled in (False,True):
                simulator=Simulator(data);start=simulator.initial_state("a")
                with patch("celeste_search.assist_planning._context",return_value=(data,simulator)), \
                     progress_reporter(packets.append if enabled else None,start,route):
                    results.append(plan_to_route("preview_fixture.bin",start.to_dict(),route,
                                   strategy=strategy,seconds=1,expansions=1000,max_attempts=1))
            self.assertTrue(all(result["can_execute"] for result in results))
            self.assertEqual(results[0]["actions"],results[1]["actions"])
            self.assertEqual(results[0]["final_state"],results[1]["final_state"])
            self.assertTrue(packets)
            self.assertEqual(packets[0]["paths"][0]["points"][0],dict(room="a",x=16,y=96))


if __name__=="__main__":
    unittest.main()
