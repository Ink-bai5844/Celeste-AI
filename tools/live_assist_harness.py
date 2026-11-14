"""Developer harness for explicitly submitted live assist routes.

Uses the product GUI/controller/service. It never chooses a target, pauses a
game, injects OS input, or starts/restarts Celeste. Write command.json atomically
with a fresh id and the latest request/room/spawn context from assist-session.json.
An acknowledgement of ``queued`` is only admission to the controller queue;
actual completion remains in controller status and trace.jsonl.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from queue import Empty, Full, Queue
import sys
from threading import Event, Lock, Thread
import time
import tkinter as tk

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"Celeste-Search"))

from celeste_search.assist import SharedAssistState,start_assist_service,default_assist_workers
from celeste_search.assist_gui import AssistWindow,RouteDraft
from celeste_search.assist_routes import normalize_route
from celeste_search.game_platform import default_game_dir,normalize_game_dir


def write_json(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")
    temporary.replace(path)


class TraceWriter:
    """Full active traces, sampled passive states, and explicit loss statistics."""
    def __init__(self,path,max_pending=4096,clock=time.monotonic,passive_trace_seconds=0):
        self.path=Path(path)
        self.queue=Queue(maxsize=max_pending)
        self.stop=Event()
        self.lock=Lock()
        self.dropped=0
        self.sampled_out=0
        self.written=0
        self.error=None
        self.session_id=str(time.time_ns())
        self.clock=clock
        self.last_state_key=None
        self.full_rate_until=float("-inf")
        self.passive_full_until=(clock()+passive_trace_seconds if passive_trace_seconds>0 else float("-inf"))
        self.last_passive_sample=float("-inf")
        self.thread=Thread(target=self._run,daemon=True,name="assist-live-trace")
        self.thread.start()

    def enqueue(self,record):
        if self.error or self.stop.is_set():
            with self.lock:
                self.dropped+=1
            return False
        # Deliberate sampling is distinct from queue/disk loss. Keep all active
        # observations, every context/terminal transition, and a two-second tail
        # after execution for post-release drift/death/settling diagnostics.
        if record.get("kind")=="state":
            now=self.clock()
            mode=record.get("mode")
            observation=record.get("observation") or {}
            key=(mode,record.get("request_id"),observation.get("room"),
                 observation.get("spawn_id"),observation.get("dead"))
            with self.lock:
                changed=key!=self.last_state_key
                if mode=="executing" or (self.last_state_key is not None
                                          and self.last_state_key[0]=="executing"):
                    self.full_rate_until=now+2
                self.last_state_key=key
                if mode in {"manual","success","failed","disconnected"}:
                    retain=(changed or now<=self.full_rate_until or now<=self.passive_full_until
                            or now-self.last_passive_sample>=1)
                    if not retain:
                        self.sampled_out+=1
                        return True
                    self.last_passive_sample=now
        value=dict(record)
        value.setdefault("trace_session",self.session_id)
        value.setdefault("wall_time_ns",time.time_ns())
        value.setdefault("monotonic_ns",time.monotonic_ns())
        try:
            self.queue.put_nowait(value)
            return True
        except Full:
            with self.lock:
                self.dropped+=1
            return False

    def stats(self):
        with self.lock:
            return dict(session_id=self.session_id,written=self.written,dropped=self.dropped,sampled_out=self.sampled_out,error=self.error,
                        pending=self.queue.qsize())

    def _run(self):
        try:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            with self.path.open("a",encoding="utf-8") as stream:
                last_flush=time.monotonic()
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        value=self.queue.get(timeout=.1)
                    except Empty:
                        value=None
                    if value is not None:
                        stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+"\n")
                        with self.lock:
                            self.written+=1
                    if time.monotonic()-last_flush>=.5:
                        stream.flush()
                        last_flush=time.monotonic()
                stream.write(json.dumps(dict(kind="trace_closed",**self.stats()))+"\n")
                stream.flush()
        except Exception as error:
            with self.lock:
                self.error=f"{type(error).__name__}: {error}"

    def close(self):
        self.stop.set()
        self.thread.join(timeout=5)
        return self.stats()


class CommandInbox:
    """One operator command per new file value, with frozen route and context."""
    def __init__(self,path,ack_path,shared,on_route=None,on_trace=None):
        self.path,self.ack_path=Path(path),Path(ack_path)
        self.shared=shared
        self.on_route=on_route or (lambda route:None)
        self.on_trace=on_trace or (lambda record:None)
        self.last_bytes=None
        self.seen_ids=set()
        if self.path.exists():
            self.last_bytes=self.path.read_bytes()
            try:
                identifier=json.loads(self.last_bytes).get("id")
            except (ValueError,AttributeError):
                identifier=None
            if isinstance(identifier,str):
                self.seen_ids.add(identifier)
            self._ack(identifier,"rejected","stale_startup_command")

    def _ack(self,identifier,status,reason,**extra):
        value=dict(id=identifier,status=status,reason=reason,wall_time_ns=time.time_ns(),**extra)
        write_json(self.ack_path,value)
        self.on_trace(dict(kind="operator_command_ack",**value))
        return value

    def poll(self):
        try:
            if self.path.stat().st_size>1024*1024:
                return self._ack(None,"rejected","command_too_large")
            raw=self.path.read_bytes()
        except FileNotFoundError:
            return None
        if raw==self.last_bytes:
            return None
        self.last_bytes=raw
        identifier=None
        try:
            command=json.loads(raw)
            if not isinstance(command,dict):
                raise ValueError("invalid_command_object")
            identifier=command.get("id")
            if not isinstance(identifier,str) or not 1<=len(identifier)<=128:
                raise ValueError("invalid_command_id")
            if identifier in self.seen_ids:
                raise ValueError("duplicate_command_id")
            self.seen_ids.add(identifier)
            kind=command.get("kind")
            if kind not in {"route","cancel"}:
                raise ValueError("unsupported_command")
            current=self.shared.snapshot()
            observation=current.get("observation") or {}
            if not current.get("connected") or not current.get("map_data"):
                raise ValueError("game_not_connected")
            expected_request=command.get("expected_request_id")
            expected_spawn=command.get("expected_spawn_id")
            expected_room=command.get("expected_room")
            if type(expected_request) is not int or expected_request!=current.get("request_id"):
                raise ValueError("request_changed")
            if not isinstance(expected_room,str) or expected_room!=observation.get("room"):
                raise ValueError("room_changed")
            if type(expected_spawn) is not int or expected_spawn!=observation.get("spawn_id"):
                raise ValueError("spawn_changed")
            queued=dict(kind=kind,command_id=identifier,
                        guard=dict(request_id=expected_request,room=expected_room,spawn_id=expected_spawn))
            if kind=="route":
                queued["route"]=deepcopy(normalize_route(current["map_data"],command.get("route")))
                self.on_route(deepcopy(queued["route"]))
            self.shared.commands.put(queued)
            return self._ack(identifier,"queued","explicit_operator_command",
                             expected_request_id=expected_request,
                             submitted_request_id=expected_request+(kind=="route"))
        except (ValueError,TypeError,KeyError) as error:
            return self._ack(identifier,"rejected",str(error))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir",type=Path,default=default_game_dir())
    parser.add_argument("--port",type=int,default=25102)
    parser.add_argument("--workers",type=int,default=default_assist_workers())
    parser.add_argument("--plan-seconds",type=float,default=8)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"artifacts"/"live-assist")
    parser.add_argument("--restore-draft",type=Path,
                        help="Restore route_request from a saved session as an unsubmitted GUI draft")
    parser.add_argument("--passive-trace-seconds",type=float,default=0,
                        help="Record every passive state for a bounded initial audit interval")
    args=parser.parse_args(argv)
    args.game_dir=normalize_game_dir(args.game_dir)
    args.output_dir=args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    args.output=args.output_dir/"assist-session.json"
    trace=TraceWriter(args.output_dir/"trace.jsonl",passive_trace_seconds=args.passive_trace_seconds)
    shared=SharedAssistState()
    service=start_assist_service(shared,args.game_dir,args.port,workers=args.workers,
                                 plan_seconds=args.plan_seconds,on_trace=trace.enqueue)
    root=tk.Tk()
    window=AssistWindow(root,shared,args)

    def reflect_route(route):
        window.draft=RouteDraft(deepcopy(route["waypoints"]),deepcopy(route["segments"]),0)
        selected=route["waypoints"][0]["room"]
        if window.map and selected in window.map.rooms:
            window.room_choice.set(selected)
            window.room=window.map.rooms[selected]
        window.refresh_editor()
        window.draw_map(fit=True)

    # A game can connect after the window is created. Preserve the draft until
    # that chapter's map arrives; restoring it never submits a route.
    pending_draft=(json.loads(args.restore_draft.read_text(encoding="utf-8"))
                   if args.restore_draft else None)

    inbox=CommandInbox(args.output_dir/"command.json",args.output_dir/"ack.json",shared,
                       on_route=reflect_route,on_trace=trace.enqueue)
    last_stats=0

    def poll():
        nonlocal last_stats,pending_draft
        if shared.stop.is_set():
            return
        if (pending_draft and window.map and window.last_map_path
                and shared.snapshot().get("connected")
                and Path(window.last_map_path).resolve()==window.map.source.resolve()
                and Path(pending_draft.get("map_path","")).resolve()==window.map.source.resolve()):
            route=pending_draft.get("route_request")
            pending_draft=None
            if route:
                reflect_route(normalize_route(window.map,route))
        try:
            inbox.poll()
        except OSError as error:
            trace.enqueue(dict(kind="inbox_io_error",error=str(error)))
        if time.monotonic()-last_stats>=1:
            last_stats=time.monotonic()
            try:
                write_json(args.output_dir/"trace-status.json",trace.stats())
            except OSError:
                pass
        root.after(100,poll)

    root.after(100,poll)
    trace.enqueue(dict(kind="harness_started",game_dir=str(args.game_dir),port=args.port))
    try:
        root.mainloop()
    finally:
        shared.stop.set()
        service.join(timeout=2)
        write_json(args.output_dir/"trace-status.json",trace.close())
    return 0


if __name__=="__main__":
    raise SystemExit(main())
