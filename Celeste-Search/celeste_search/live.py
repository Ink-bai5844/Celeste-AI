"""Nonblocking room controller: background planning, normal game respawns.

observe() never waits for a worker. While candidates are being computed it
answers the game immediately with neutral input. No chapter reset or input
prefix replay exists in this controller.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import select
import socket
import struct
import subprocess
import time

from .anchor import can_plan_while_idle, anchor_matches
from .background import BackgroundRoomPlanner
from .maps import load_map
from .playback import configure_game, install_bridge, read_message, recv_exact, session_xml, restore_configuration
from .routes import map_hash, write_json
from .runtime_state import state_from_observation
from .periodic import can_seek_spring_recovery, periodic_anchor_matches, projected_anchor_matches
from .exploration import Exploration
from .telemetry import Telemetry, compact_frame
from datetime import datetime


@dataclass
class LiveOptions:
    tolerance:float=12
    retries:int=6
    max_room_retries:int=0
    workers:int=2
    max_seconds:float=0


class LiveController:
    def __init__(self,map_data,background,options=None,*,on_event=None,resume=None,record=None):
        self.map=map_data
        self.map_sha256=map_hash(map_data.source)
        self.physics_sha256=map_hash(Path(__file__).with_name("physics.py"))
        self.background=background
        self.options=options or LiveOptions()
        self.on_event=on_event or (lambda e:None)
        self.record=record or (lambda kind,data:None)
        self.exploration=Exploration((resume or {}).get("exploration"))
        self.started=time.monotonic()
        self.frame=-1
        self.room=None
        self.spawn_id=None
        self.deaths=0
        self.plan=None
        self.index=0
        self.pending_anchor=None
        self.pending_spawn=None
        self.pending_started_frame=None
        self.last_observation=None
        self.visited=set()
        self.confirmed=set()
        self.events=[]
        self.failures=[]
        self.blacklist={}
        self.attempts={}
        self.failures_since_spawn=0
        self.room_retries=0
        self.plans=0
        self.finished=False
        self.reason="running"
        self.phase="waiting_for_stable_position"
        self.idle_frames=0
        self.plan_key=None
        self.search_key=None
        self.last_request_frame=-100
        self.executions=0
        self.execution_id=None
        self.adopt_next_frame=False
        self.room_searches={}
        self.pending_kind="idle"
        self.waiting_candidate=None
        self.waiting_since=None
        self.unstable_frames=0
        if resume:
            if resume.get("map_sha256")!=self.map_sha256:
                raise ValueError("Saved route cache belongs to a different map")
            # Resume search knowledge only. Never restart/replay previous chapters.
            if resume.get("physics_sha256")==self.physics_sha256:
                self.blacklist={k:set(v) for k,v in resume.get("rejected_routes",{}).items()}
                self.attempts=dict(resume.get("attempts",{}))
            else:
                self.exploration.failed_goals.clear()
                self.exploration.failed_regions.clear()
            self.visited.update(resume.get("visited_rooms",[]))

    def event(self,name,**data):
        item={"event":name,"frame":self.frame,"room":self.room,
              "elapsed":round(time.monotonic()-self.started,2),**data}
        self.events.append(item)
        if len(self.events)>2000:
            self.events=self.events[-1500:]
        self.record("event",item)
        self.on_event(item)

    def _key(self,obs):
        value=(obs["room"],round(obs["x"]),round(obs["y"]),obs["dashes"],tuple(sorted(obs["berries"])))
        return sha256(repr(value).encode()).hexdigest()[:20]

    def _cancel(self):
        self.background.cancel_epoch()
        self.pending_anchor=None
        self.pending_spawn=None
        self.waiting_candidate=None
        self.waiting_since=None

    def _reject(self,reason,obs,*,expected=None):
        key=(self.plan_key if self.plan else self.search_key) or self._key(obs)
        if self.plan:
            self.blacklist.setdefault(key,set()).add(self.plan["fingerprint"])
        self.attempts[key]=self.attempts.get(key,0)+1
        self.failures_since_spawn+=1
        goal=(self.plan or {}).get("primary_goal")
        if not goal and (self.plan or {}).get("goals"):
            goal=self.plan["goals"][0]
        self.exploration.reject(obs,reason,goal)
        self.failures.append({"reason":reason,"frame":self.frame,"room":obs["room"],
                              "observed":dict(obs),
                              "expected":expected,"candidate":self.plan.get("fingerprint") if self.plan else None})
        self.failures=self.failures[-200:]
        self.event("route_rejected",reason=reason,action="release_inputs_and_search_in_background")
        self.plan=None
        self.index=0
        self._cancel()
        self.phase="waiting_for_stable_position"

    def observe(self,obs):
        """Return immediately: keys0..127, native roomretry254, detach253."""
        if self.finished:
            return 253
        if self.adopt_next_frame:
            self.frame=obs["frame"]-1
            self.adopt_next_frame=False
        if obs.get("input_phase")!="before_virtual_inputs" or obs["frame"]!=self.frame+1:
            self.finished=True
            self.reason="incompatible_bridge_or_frame_sequence"
            return 253
        self.frame=obs["frame"]
        self.last_observation=obs
        previous_walls=set(self.exploration.opened_walls)
        self.exploration.observe(obs)
        opened=self.exploration.opened_walls-previous_walls
        if opened:
            self.event("walls_opened",walls=sorted(opened),runtime_verified=True)
        if self.options.max_seconds and time.monotonic()-self.started>=self.options.max_seconds:
            self.finished=True
            self.reason="time_budget"
            return 253
        new_berries=set(obs["berries"])-self.confirmed
        self.confirmed.update(obs["berries"])
        if new_berries:
            self.room_searches[obs["room"]]=0
            self.event("berry_confirmed",berries=sorted(new_berries))
        changed_room=obs["room"]!=self.room
        changed_spawn=self.spawn_id is not None and obs.get("spawn_id")!=self.spawn_id
        if changed_spawn:
            self.unstable_frames=0
            if self.plan:
                self._reject("natural_death",obs)
            else:
                self._cancel()
            self.failures_since_spawn=0
            self.idle_frames=0
            self.event("natural_respawn",respawn_room=obs["room"],
                       x=obs.get("respawn_x",obs["x"]),y=obs.get("respawn_y",obs["y"]))
        if changed_room:
            self.unstable_frames=0
            continuation=(self.plan and self.plan.get("allow_room_continuation")
                          and self.index<len(self.plan["actions"])
                          and self.plan["expected"][self.index]["room"]==obs["room"])
            self.room=obs["room"]
            self.visited.add(self.room)
            if not continuation:
                self.plan=None
                self.index=0
            self._cancel()
            self.failures_since_spawn=0
            self.idle_frames=0
            self.event("room_entered",game_paused=False,planned_continuation=bool(continuation))
        self.spawn_id=obs.get("spawn_id")
        self.deaths=obs.get("deaths",self.deaths)
        if can_plan_while_idle(obs):
            self.exploration.mark_settled(self.room)
        if self.plan:
            expected=self.plan["expected"][min(self.index,len(self.plan["expected"])-1)]
            distance=math.inf if expected["room"]!=obs["room"] else math.hypot(obs["x"]-expected["x"],obs["y"]-expected["y"])
            if distance>self.options.tolerance:
                self._reject("prediction_diverged",obs,expected=expected)
                return 0
            if self.index<len(self.plan["actions"]):
                mask=self.plan["actions"][self.index]
                self.index+=1
                return mask
            if not set(self.plan["final_state"]["berries"])<=set(obs["berries"]):
                self._reject("predicted_berry_not_confirmed",obs)
                return 0
            self.event("segment_completed",confirmed_berries=obs["berries"])
            self.plan=None
            self.index=0
            self.idle_frames=0
        candidate=self.waiting_candidate or self.background.poll()
        if candidate is not None:
            expected_spawn=self.pending_spawn
            anchor=self.pending_anchor
            periodic=self.pending_kind=="periodic"
            same_epoch=(expected_spawn==self.spawn_id and anchor is not None and anchor["room"]==obs["room"])
            matching=(periodic_anchor_matches(obs,anchor) if periodic and same_epoch
                      else anchor_matches(obs,anchor) if same_epoch else False)
            if candidate.get("navigation_revision",self.exploration.navigation_revision)!=self.exploration.navigation_revision:
                same_epoch=False
            execute_at=candidate.get("execute_at_frame")
            if periodic and same_epoch and execute_at is not None and candidate.get("can_execute"):
                if self.frame<execute_at:
                    self.waiting_candidate=candidate
                    self.phase="waiting_for_predicted_spring_phase"
                    return 0
                matching=(self.frame==execute_at and projected_anchor_matches(
                    obs,candidate["execution_anchor_state"],anchor))
            if periodic and same_epoch and candidate.get("can_execute") and not matching:
                if execute_at is not None:
                    self.waiting_since=self.frame-360  # Missed interception; search afresh.
                elif self.waiting_since is None:
                    self.waiting_since=self.frame
                if self.frame-self.waiting_since<360:
                    self.waiting_candidate=candidate
                    self.phase="waiting_for_spring_phase"
                    return 0
            self.waiting_candidate=None
            self.waiting_since=None
            self.pending_anchor=None
            self.pending_spawn=None
            if not same_epoch or (candidate.get("can_execute") and not matching):
                self.event("stale_plan_discarded",game_paused=False)
                self.phase="waiting_for_stable_position"
            elif candidate.get("requires_room_respawn"):
                self.room_retries+=1
                self._cancel()
                self.event("retry_current_room",method="native_death_at_current_respawn",
                           reason=candidate.get("reason"),pending_goals=candidate.get("pending_goals",[]),
                           chapter_restart=False)
                return 254
            elif not candidate.get("can_execute") or not candidate.get("actions"):
                self.attempts[self.search_key]=self.attempts.get(self.search_key,0)+1
                self.failures_since_spawn+=1
                goal=candidate.get("primary_goal")
                if goal:
                    self.exploration.failed_goals[goal]+=1
                    if (candidate.get("search_seconds",0)>=19.9 and self.exploration.failed_goals[goal]>=6
                            and self.exploration.failed_goals[goal]%6==0):
                        self.exploration.defer_goal(goal)
                        self.failures_since_spawn=0
                        self.event("goal_deferred",goal=goal,still_pending=True,reason="broadened_offline_search_exhausted")
                self.event("no_offline_route",reason=candidate.get("reason"),
                           candidate_reasons=candidate.get("candidate_reasons",[]))
                if candidate.get("errors"):
                    self.event("worker_error",details=candidate["errors"])
                self.phase="waiting_for_stable_position"
            else:
                self.plan=candidate
                if candidate.get("active_berry"):
                    self.exploration.set_berry(candidate["active_berry"])
                destination=candidate.get("destination_room")
                if destination and destination!=self.room:
                    self.exploration.set_destination(destination)
                self.executions+=1
                self.execution_id=self.executions
                self.record("plan",dict(candidate,run_plan_id=self.executions,actual_anchor=anchor))
                self.plan_key=self.search_key
                self.index=1
                self.phase="executing"
                self.event("execute_selected_route",frames=len(candidate["actions"]),
                           game_frames_during_search=self.frame-(self.pending_started_frame or 0))
                return candidate["actions"][0]
        stable=can_plan_while_idle(obs)
        self.idle_frames=self.idle_frames+1 if stable else 0
        self.unstable_frames=0 if stable else self.unstable_frames+1
        if (self.unstable_frames>=120 and self.pending_anchor is None
                and self.frame-self.last_request_frame>=30 and can_seek_spring_recovery(obs,self.map)):
            state=state_from_observation(self.map,obs)
            key=self._key(obs)
            variant=self.attempts.get(key,0)*self.options.workers
            if self.background.request(state,variant,sorted(self.blacklist.get(key,set())),sorted(self.visited),
                                       exploration=self.exploration.snapshot(),recovery=True):
                self.search_key=key
                self.pending_anchor=dict(obs)
                self.pending_spawn=self.spawn_id
                self.pending_kind="periodic"
                self.pending_started_frame=self.frame
                self.last_request_frame=self.frame
                self.plans+=1
                self.phase="searching_spring_recovery"
                self.event("background_spring_recovery",game_paused=False)
            return 0
        # Wait for residual acceleration/buffers to settle naturally. Never send
        # a reset merely because an asynchronous result is stale.
        if self.idle_frames<4:
            return 0
        if self.failures_since_spawn>=self.options.retries and self.pending_anchor is None:
            # At a safe anchor with unchanged mechanisms, search exhaustion
            # does not justify killing the player and losing a valid waypoint.
            environment=obs.get("room_entities",{})
            environment_known=isinstance(environment,dict) and "entities" in environment
            entities=environment.get("entities",[]) if environment_known else []
            environment_changed=any(
                e.get("flying_away")
                or (e.get("kind") in {"zipMover","fallingBlock","crumbleBlock"}
                    and (e.get("present") is False or e.get("x")!=e.get("map_x") or e.get("y")!=e.get("map_y") or e.get("collidable") is False))
                for e in entities)
            if environment_known and not environment_changed:
                self.failures_since_spawn=0
                self.event("continue_offline_search",reason="unchanged_safe_anchor",chapter_restart=False)
            else:
                return self._retry_checkpoint()
        if self.pending_anchor is None and self.frame-self.last_request_frame>=15:
            state=state_from_observation(self.map,obs)
            key=self._key(obs)
            # Coordinate changes must not reset the budget forever when a room
            # only yields intermediate waypoints. Broaden the offline search.
            stagnation=self.room_searches.get(self.room,0)//3
            variant=(self.attempts.get(key,0)+stagnation)*self.options.workers
            if self.background.request(state,variant,sorted(self.blacklist.get(key,set())),sorted(self.visited),
                                       exploration=self.exploration.snapshot()):
                self.search_key=key
                self.pending_anchor=dict(obs)
                self.pending_kind="idle"
                self.pending_spawn=self.spawn_id
                self.pending_started_frame=self.frame
                self.last_request_frame=self.frame
                self.plans+=1
                self.room_searches[self.room]=self.room_searches.get(self.room,0)+1
                self.phase="searching_in_background"
                self.event("background_search",variant=variant,game_paused=False)
        return 0

    def _retry_checkpoint(self):
        if self.options.max_room_retries and self.room_retries>=self.options.max_room_retries:
            self.finished=True
            self.reason="room_retry_budget_exhausted"
            return 253
        self.room_retries+=1
        self.failures_since_spawn=0
        self._cancel()
        self.event("retry_current_room",method="native_death_at_current_respawn",chapter_restart=False)
        return 254

    def terminal(self,obs):
        self.confirmed.update(obs.get("berries",[]))
        self.last_observation=obs
        self.finished=True
        self.phase="manual_control"
        if obs.get("reason")=="chapter_complete":
            self.reason="chapter_complete"
        elif self.reason=="running":
            self.reason=obs.get("reason","detached")
        self.event("detached",reason=self.reason,game_left_running=True)

    def journal(self):
        return {"schema":"celeste-live-search-v1","map":str(self.map.source.resolve()),
                "map_sha256":self.map_sha256,"physics_sha256":self.physics_sha256,"phase":self.phase,"reason":self.reason,
                "frame":self.frame,"plans":self.plans,"natural_deaths":self.deaths,
                "room_retries":self.room_retries,"visited_rooms":sorted(self.visited),
                "confirmed_berries":sorted(self.confirmed),"attempts":self.attempts,
                "rejected_routes":{k:sorted(v) for k,v in self.blacklist.items()},
                "exploration":self.exploration.snapshot(),
                "events":self.events,"failures":self.failures,"last_observation":self.last_observation,
                "game_paused_for_search":False,"chapter_restarts":0,
                "rollback_method":"native_current_room_respawn"}


def accept_handshake(conn,data,source,*,allow_new_session=True):
    """Attach to a live checkpoint or initialize only a fresh game session."""
    tag=recv_exact(conn,8)
    room=data.start_room()
    header={"map":{"path":str(source)},"start_room":room.name,
            "predicted":{"state":{"x":room.x+room.width,"y":room.y}}}
    xml=session_xml(header,0).replace(b"<EndOnDeath>true</EndOnDeath>",b"<EndOnDeath>false</EndOnDeath>")
    if tag==b"LOADINFO":
        if not allow_new_session:
            raise ValueError("Reconnect must preserve the running session")
        conn.sendall(xml)
        return False
    if tag==b"LIVEINFO":
        size=struct.unpack("<I",recv_exact(conn,4))[0]
        if not 0<size<=1024*1024:
            raise ValueError("Invalid LIVEINFO size")
        hello=json.loads(recv_exact(conn,size))
        import xml.etree.ElementTree as ET
        settings=ET.fromstring(xml)
        if (int(hello["area_id"]),int(hello["area_mode"]))!=(int(settings.findtext("AreaKey")),int(settings.findtext("AreaMode"))):
            raise ValueError("Running chapter differs from --map; refusing to reload it")
        conn.sendall(b"\x00")
        return True
    raise ValueError(f"Expected LOADINFO or LIVEINFO, got {tag!r}")


def run_live(args):
    game_dir=args.game_dir.resolve()
    source=args.map.resolve()
    data=load_map(source)
    if map_hash(game_dir/"Content"/"Maps"/source.name)!=map_hash(source):
        raise ValueError("Offline and selected game maps differ")
    if not 1024<=args.port<=65535:
        raise ValueError("Port must be in 1024..65535")
    resume=json.loads(args.resume.read_text(encoding="utf-8")) if args.resume else None
    if resume and resume.get("schema")!="celeste-live-search-v1":
        raise ValueError("Old chapter-replay checkpoints are retired; run without --resume")
    background=BackgroundRoomPlanner(source,args.workers,args.plan_seconds,args.expansions)
    background.warmup()
    output=args.output.resolve()
    trace_dir=output.parent/(output.stem+"-traces")/datetime.now().strftime("%Y%m%d-%H%M%S")
    telemetry=Telemetry(trace_dir)
    controller=LiveController(data,background,LiveOptions(args.tolerance,args.retries,args.max_rollbacks,args.workers,args.max_seconds),
                              resume=resume,record=telemetry.emit,
                              on_event=lambda e:print(json.dumps(e,ensure_ascii=False),flush=True))
    def journal():
        return dict(controller.journal(),trace_directory=str(trace_dir),
                    telemetry_dropped=telemetry.dropped,telemetry_error=telemetry.error)
    configuration=None
    process=None
    conn=None
    writer=ThreadPoolExecutor(max_workers=1)
    pending_write=None
    try:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET,getattr(socket,"SO_EXCLUSIVEADDRUSE",socket.SO_REUSEADDR),1)
            listener.bind(("127.0.0.1",args.port))
            listener.listen(1)
            listener.settimeout(args.connect_timeout)
            if args.launch:
                if args.install_bridge:
                    install_bridge(game_dir)
                configuration=configure_game(game_dir,args.port)
                startup=subprocess.STARTUPINFO() if hasattr(subprocess,"STARTUPINFO") else None
                if startup:
                    startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW
                    startup.wShowWindow=1
                process=subprocess.Popen([str(game_dir/"Celeste.exe")],cwd=game_dir,startupinfo=startup)
            print(f"Live controller on 127.0.0.1:{args.port}; game keeps running during search",flush=True)
            connection_deadline=time.monotonic()+args.connect_timeout
            while True:
                conn,_=listener.accept()
                conn.settimeout(30)
                conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                try:
                    controller.adopt_next_frame=accept_handshake(conn,data,source)
                    break
                except (ConnectionError,socket.timeout,OSError) as error:
                    conn.close()
                    conn=None
                    controller.event("incomplete_connection",message=str(error))
                    if time.monotonic()>=connection_deadline:
                        raise
            if controller.adopt_next_frame:
                controller.event("reattached_in_place",chapter_restart=False)
            while True:
                # Story scenes may have no controllable player update for a
                # while; silence is not a reason to drop the game connection.
                if not select.select([conn],[],[],1)[0]:
                    continue
                try:
                    kind,obs=read_message(conn)
                except OSError as error:
                    conn.close()
                    conn=None
                    controller._cancel()
                    controller.plan=None
                    controller.index=0
                    controller.phase="waiting_for_game_connection"
                    controller.event("connection_lost",message=str(error),chapter_restart=False)
                    connection_deadline=time.monotonic()+args.connect_timeout
                    while True:
                        conn,_=listener.accept()
                        conn.settimeout(30)
                        conn.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                        try:
                            accept_handshake(conn,data,source,allow_new_session=False)
                            controller.adopt_next_frame=True
                            controller.event("reattached_in_place",chapter_restart=False)
                            break
                        except (ConnectionError,socket.timeout,OSError):
                            conn.close()
                            conn=None
                            if time.monotonic()>=connection_deadline:
                                raise
                    continue
                if kind=="done":
                    conn.sendall(b"\x00")
                    controller.terminal(obs)
                    break
                mask=controller.observe(obs)
                conn.sendall(bytes([mask]))
                plan=controller.plan
                index=max(0,controller.index-1) if plan else None
                expected=plan["expected"][index] if plan and index<len(plan["expected"]) else None
                telemetry.emit("frame",compact_frame(obs,mask,controller.execution_id if plan else None,index,expected))
                if obs["frame"]%120==0 and (pending_write is None or pending_write.done()):
                    pending_write=writer.submit(write_json,output,journal())
    except KeyboardInterrupt:
        controller.reason="user_stopped"
        if conn:
            try:
                conn.sendall(b"\xfd")
            except OSError:
                pass
    except (OSError,ValueError,KeyError) as error:
        controller.reason="controller_error"
        controller.event("error",message=str(error))
    finally:
        if conn:
            conn.close()  # live bridge detaches in place even on disconnect
        background.close(wait=False)
        writer.shutdown(wait=True)
        telemetry.close()
        if configuration:
            restore_configuration(configuration)
        # Leave the user's game running, with ordinary keyboard control restored.
        write_json(output,journal())
    print(json.dumps({"result":str(output),"reason":controller.reason,"confirmed_berries":sorted(controller.confirmed),
                      "rooms":sorted(controller.visited),"game_left_running":process is not None and process.poll() is None},ensure_ascii=False),flush=True)
    return 0 if controller.reason in {"time_budget","chapter_complete","user_stopped","user_detached","detached"} else 3
