"""Trace auditing is read-only and never attributes later human play to a route."""
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


source=Path(__file__).resolve().parents[2]/"tools/extract_assist_live_proof.py"
spec=importlib.util.spec_from_file_location("extract_assist_live_proof",source)
proof=importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)


def state(frame,*,request=1,epoch=9,dead=False,time=None):
    return dict(kind="state",trace_session="test",request_id=request,
        time=frame/60 if time is None else time,mode="success",action=128,
        _line=frame+1,_byte=frame*100,observation=dict(frame=frame,room="a",x=1,y=2,
        dead=dead,deaths=3+int(dead),spawn_id=4,human_input_mask=0,
        human_menu_input=False,human_input_epoch=epoch,control_mode="manual"))


class TraceIndexTests(unittest.TestCase):
    def setUp(self):
        temporary=TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.trace=Path(temporary.name)/"trace.jsonl"
        self.index_path=Path(temporary.name)/"index.json"

    def append(self,*records):
        with self.trace.open("ab") as stream:
            for record in records:
                stream.write((json.dumps(record)+"\n").encode())

    def test_append_scans_only_new_complete_lines(self):
        self.append(dict(trace_session="a",kind="hello"))
        first,_=proof.index_trace(self.trace,self.index_path)
        with self.trace.open("ab") as stream:
            stream.write(b'{"trace_session":"b"')
        partial,bytes_scanned=proof.index_trace(self.trace,self.index_path)
        self.assertEqual(bytes_scanned,0)
        self.assertEqual(partial["indexed_bytes"],first["indexed_bytes"])
        with self.trace.open("ab") as stream:
            stream.write(b',"kind":"hello"}\n')
        completed,bytes_scanned=proof.index_trace(self.trace,self.index_path)
        self.assertEqual(completed["line_count"],2)
        self.assertEqual(completed["sessions"]["b"]["first_byte"],first["indexed_bytes"])
        self.assertEqual(bytes_scanned,self.trace.stat().st_size-first["indexed_bytes"])
        self.assertEqual(proof.index_trace(self.trace,self.index_path)[1],0)

    def test_same_size_rewrite_invalidates_cached_session_spans(self):
        self.append(dict(trace_session="a",kind="hello"))
        proof.index_trace(self.trace,self.index_path)
        self.trace.write_bytes(self.trace.read_bytes().replace(b'"a"',b'"b"'))
        index,scanned=proof.index_trace(self.trace,self.index_path)
        self.assertEqual(set(index["sessions"]),{"b"})
        self.assertEqual(scanned,self.trace.stat().st_size)

    def test_session_request_filter_keeps_tail_boundary_then_stops(self):
        self.append(dict(trace_session="old",kind="hello"),
            dict(trace_session="test",kind="hello"),state(0,request=9),
            dict(trace_session="test",kind="plan",request_id=1),
            state(60),state(61),state(62,request=2),state(241),state(242),
            dict(trace_session="new",kind="hello"))
        index,_=proof.index_trace(self.trace,self.index_path)
        records,decoded,_=proof.read_request_records(self.trace,index,"test",1,2)
        self.assertEqual(decoded,7)
        self.assertEqual([r["observation"]["frame"] for r in records if r["kind"]=="state"],[60,61,62,241])
        self.assertTrue(all(r["trace_session"]=="test" for r in records))

    def test_failure_window_does_not_decode_all_later_idle_states(self):
        failed=state(60)
        failed["mode"]="failed"
        self.append(dict(trace_session="test",kind="hello"),failed,state(61),state(241),state(242))
        index,_=proof.index_trace(self.trace,self.index_path)
        records,decoded,_=proof.read_request_records(self.trace,index,"test",1,2)
        self.assertEqual(decoded,4)
        self.assertEqual(records[-1]["observation"]["frame"],241)


class NeutralTailTests(unittest.TestCase):
    def test_full_contiguous_neutral_window_is_verified(self):
        result=proof.post_success_tail(state(0),[state(i) for i in range(1,122)],2)
        self.assertTrue(result["full_neutral_tail_verified"])
        self.assertEqual(result["neutral_observations"],120)

    def test_changed_human_epoch_stops_attribution_even_when_mask_is_zero(self):
        result=proof.post_success_tail(state(0),[state(1),state(2,epoch=10),state(3,dead=True)],2)
        self.assertTrue(result["interrupted_by_human"])
        self.assertTrue(result["safe_before_human"])
        self.assertFalse(result["full_neutral_tail_verified"])
        self.assertEqual(result["neutral_observations"],1)

    def test_new_request_stops_attribution(self):
        result=proof.post_success_tail(state(0),[state(1),state(2,request=2),state(3,dead=True)],2)
        self.assertTrue(result["interrupted_by_new_request"])
        self.assertEqual(result["next_request_id"],2)
        self.assertTrue(result["safe_before_human"])
        self.assertEqual(result["neutral_observations"],1)

    def test_dead_or_missing_frames_never_verify_full_tail(self):
        for records in ([state(i,dead=(i==3)) for i in range(1,121)],
                        [state(i) for i in range(1,121) if i!=3],[]):
            with self.subTest(size=len(records)):
                self.assertFalse(proof.post_success_tail(state(0),records,2)["full_neutral_tail_verified"])


class RequestConnectionTests(unittest.TestCase):
    def test_prior_reconnects_do_not_invalidate_a_later_completed_route(self):
        from celeste_search.maps import MapData, Room
        from celeste_search.physics import Simulator
        from celeste_search.routes import snapshot
        with TemporaryDirectory() as directory:
            root=Path(directory)
            map_path=root/"fixture.bin"
            map_path.write_bytes(b"owned map fixture supplied through load_map")
            room=Room("a",0,0,64,80,("0"*8,)*8+("1"*8,)*2,(),((16.,64.),))
            data=MapData("fixture",{"a":room},map_path)
            simulator=Simulator(data)
            start=simulator.initial_state()
            end=simulator.step(start,0)
            route=dict(waypoints=[dict(room="a",x=16,y=56,tile_size=8)],
                       segments=[dict(no_dash=False,no_ground=False)])
            def observed(value):
                return dict(**value.to_dict(),model_state=value.to_dict(),
                    input_phase="before_virtual_inputs",deaths=0,spawn_id=1,
                    human_input_mask=0,human_menu_input=False,human_input_epoch=9,
                    control_mode="manual")
            records=[dict(kind="connection_error",request_id=0,error="old reset",time=-2),
                dict(kind="disconnected",time=-1.9),
                dict(kind="hello",map_path=str(map_path),time=-1),
                dict(kind="event",event=dict(event="route_confirmed",request_id=1),time=-.1),
                dict(kind="plan",request_id=1,time=0,candidate=dict(actions=[0],route=route,
                    expected=[snapshot(start),snapshot(end)],waypoint_indices=[0])),
                dict(kind="state",request_id=1,time=0,mode="executing",action=0,
                    route_index=1,waypoint_index=1,observation=observed(start)),
                dict(kind="state",request_id=1,time=1/60,mode="success",action=128,
                    route_index=1,waypoint_index=1,observation=observed(end))]
            for tick in range(1,122):
                end=simulator.step(end,0)
                records.append(dict(kind="state",request_id=1,time=(tick+1)/60,
                    mode="success",action=128,route_index=1,waypoint_index=1,observation=observed(end)))
            trace=root/"trace.jsonl"
            trace.write_text("".join(json.dumps(dict(trace_session="test",**row))+"\n" for row in records))
            with patch("celeste_search.maps.load_map",return_value=data):
                result=proof.audit(trace,"test",1,map_path=map_path)
            self.assertTrue(result["runtime_route_success_verified"],result["checks"])
            self.assertTrue(result["safe_handoff_verified"])
            self.assertEqual(result["session_prior_connection_error_counts"],
                             {"connection_error":1,"disconnected":1})
            self.assertEqual(result["connection_errors_through_success"],[])

    def test_current_request_and_untagged_tail_disconnects_remain_failures(self):
        success=state(10)
        plan=dict(kind="plan",request_id=1,_line=5,time=0)
        errors=[dict(kind="connection_error",request_id=1,_line=6,time=.05,error="current reset"),
                dict(kind="disconnected",_line=12,time=.2)]
        result=proof.request_connection_audit([plan,*errors,success],1,plan,success,2)
        self.assertEqual([row["source_line"] for row in result["connection_errors_through_success"]],[6])
        self.assertEqual([row["source_line"] for row in result["connection_errors_post_success_tail"]],[12])

    def test_new_request_confirmation_ends_tail_error_attribution(self):
        success=state(10)
        plan=dict(kind="plan",request_id=1,_line=5,time=0)
        confirmation=dict(kind="event",_line=12,time=.2,
                          event=dict(event="route_confirmed",request_id=2))
        error=dict(kind="disconnected",_line=13,time=.21)
        result=proof.request_connection_audit([plan,success,confirmation,error],1,plan,success,2)
        self.assertEqual(result["connection_errors_post_success_tail"],[])


if __name__=="__main__":
    unittest.main()
