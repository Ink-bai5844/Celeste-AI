"""On-demand target assistance. Passive input is the default in every state."""
from __future__ import annotations

from concurrent.futures import Future, ProcessPoolExecutor
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import multiprocessing
import os
from pathlib import Path
from queue import Empty, Full, Queue
import socket
import struct
from threading import Event, Lock, Thread
import time

from .anchor import can_plan_while_idle
from .assist_validation import (AssistValidationError,validate_tile_target,make_assist_anchor,
                                observe_assist_anchor,admit_assist_candidate,_same,
                                cassette_comparison_snapshot,cassette_pickup_required)
from .assist_planning import target_reached,route_strategy_specs
from .assist_routes import (normalize_route,start_route_progress,advance_route_progress,
                            route_complete)
from .maps import load_map
from .mac_activity import MacActivity
from .playback import read_message, recv_exact
from .runtime_state import (state_from_observation,cassette_snapshot_from_observation,
                            falling_snapshot_from_observation,RuntimeStateError)
from .physics import DASH,Simulator,SimState
from .dash_preference import dash_preference_key,dash_usage
from .search_progress import MAX_CORRIDOR_PREVIEWS

PASSIVE=128
_worker_epoch=None
_worker_warm_ready=None
_worker_warm_release=None
_worker_progress_queue=None


def _bounded_strategy_results(values):
    """Freeze small failure evidence once, never retain worker trajectories."""
    if not isinstance(values,(list,tuple)):
        return []
    omitted={"actions","expected","anchor_state","final_state","points","paths",
             "observation","source_observation","model_state"}
    def compact(value,remaining,depth=4):
        if remaining[0]<=0 or depth<0:
            return None
        remaining[0]-=1
        if value is None or type(value) is bool:
            return value
        if type(value) in (int,float):
            return value if abs(value)<1e100 and math.isfinite(value) else None
        if isinstance(value,str):
            return value[:160]
        if isinstance(value,(list,tuple)):
            return [compact(item,remaining,depth-1) for item in value[:12] if remaining[0]>0]
        if isinstance(value,dict):
            result={}
            for key,item in value.items():
                if len(result)>=16 or remaining[0]<=0:
                    break
                if isinstance(key,str) and key not in omitted:
                    result[key[:80]]=compact(item,remaining,depth-1)
            return result
        return None
    fields=("strategy","reason","search_status","error","wall_seconds","expansions","simulated_frames",
            "best_prediction","strategy_details")
    rows=[]
    size=2
    for value in values[:48]:
        if not isinstance(value,dict):
            continue
        row={}
        for key in fields:
            item=value.get(key)
            expected=(str if key in {"strategy","reason","search_status","error"} else
                      dict if key in {"best_prediction","strategy_details"} else (int,float))
            if key in value and isinstance(item,expected):
                row[key]=compact(item,[64])
        encoded=lambda:len(json.dumps(row,ensure_ascii=False,allow_nan=False).encode("utf-8"))+2
        for key in ("strategy_details","best_prediction"):
            if encoded()>4096 or size+encoded()>65536:
                row.pop(key,None)
        amount=encoded()
        if size+amount>65536:
            break
        rows.append(row)
        size+=amount
    return rows


def _topology_preview(value):
    """Keep coarse guidance bounded and separate from executable predictions."""
    if not isinstance(value,dict) or not isinstance(value.get("corridors"),(list,tuple)):
        return None
    if not value["corridors"]:
        return dict(active_id=None,corridors=[])
    corridors=[]
    identifiers=set()
    for row in value["corridors"][:MAX_CORRIDOR_PREVIEWS]:
        if not isinstance(row,dict):
            continue
        identity=row.get("id")
        status=row.get("status")
        if (not isinstance(identity,str) or not 1<=len(identity)<=80 or identity in identifiers
                or not isinstance(status,str) or status not in {"pending","trying","failed","verified"}
                or not isinstance(row.get("points"),(list,tuple))):
            continue
        points=[]
        invalid_point=False
        for point in row["points"][:60]:
            if (not isinstance(point,dict) or not isinstance(point.get("room"),str)
                    or any(type(point.get(axis)) not in (int,float)
                           or not math.isfinite(point[axis]) for axis in ("x","y"))):
                invalid_point=True
                break
            points.append({key:point[key] for key in ("room","x","y")})
        if invalid_point or len(points)<2:
            continue
        reason=row.get("reason","")
        corridors.append(dict(id=identity,status=status,
                              reason=reason[:160] if isinstance(reason,str) else "",points=points))
        for key in ("turns", "expansions"):
            count = row.get(key)
            if type(count) is int and 0 <= count <= 10**9:
                corridors[-1][key] = count
        identifiers.add(identity)
    if not corridors:
        return None
    active=value.get("active_id")
    return dict(active_id=active if isinstance(active,str) and active in identifiers else None,
                corridors=corridors)


def default_assist_workers():
    return max(1,min(12,(os.cpu_count() or 2)//2))


def _initialize_search_worker(epoch,warm_ready=None,warm_release=None,progress_queue=None):
    global _worker_epoch,_worker_warm_ready,_worker_warm_release,_worker_progress_queue
    _worker_epoch=epoch
    _worker_warm_ready=warm_ready
    _worker_warm_release=warm_release
    _worker_progress_queue=progress_queue
    if progress_queue is not None:
        # Preview frames are disposable. Worker shutdown must never wait for
        # the viewer to drain a pipe after the search was cancelled.
        progress_queue.cancel_join_thread()
    try:
        if os.name=="nt":
            import ctypes
            kernel=ctypes.WinDLL("kernel32",use_last_error=True)
            kernel.GetCurrentProcess.restype=ctypes.c_void_p
            kernel.SetPriorityClass.argtypes=[ctypes.c_void_p,ctypes.c_uint]
            kernel.SetPriorityClass(kernel.GetCurrentProcess(),0x4000)  # BELOW_NORMAL
        else:
            os.nice(5)
    except (OSError,AttributeError):
        pass


def _warm_search_worker(timeout):
    """Hold one task in every process until the startup caller releases all."""
    if _worker_warm_ready is None or _worker_warm_release is None:
        raise RuntimeError("search worker warmup was not initialized")
    with _worker_warm_ready.get_lock():
        _worker_warm_ready.value+=1
    if not _worker_warm_release.wait(timeout):
        raise TimeoutError("search worker warmup barrier timed out")
    return os.getpid()


class _SearchSuperseded(Exception):
    pass


def _anchor_job(config):
    """Forecast in a prewarmed process, never sharing the bridge thread's GIL."""
    from .room_planning import _context
    generation=config["generation"]
    if _worker_epoch is not None and _worker_epoch.value!=generation:
        raise _SearchSuperseded()
    map_data,_=_context(config["map_path"])
    simulator=Simulator(map_data)
    original=simulator.step
    calls=0
    def checked_step(*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls%64==1 and _worker_epoch is not None and _worker_epoch.value!=generation:
            raise _SearchSuperseded()
        return original(*args,**kwargs)
    simulator.step=checked_step
    return make_assist_anchor(simulator,config["observation"],
        request_id=config["request_id"],future_frames=config["frames"])


def _target_job(config):
    from .assist_planning import plan_to_target,plan_to_route
    from .room_planning import _context
    from .search_progress import progress_reporter
    config=dict(config)
    generation=config.pop("_generation",None)
    def cancelled():
        return generation is not None and _worker_epoch is not None and _worker_epoch.value!=generation
    if cancelled():
        return dict(can_execute=False,reached=False,reason="search_cancelled")
    _,simulator=_context(config["map_path"])
    original=simulator.step
    had_override="step" in simulator.__dict__
    calls=0
    def check_step_cancelled():
        nonlocal calls
        calls+=1
        if calls%64==1 and cancelled():
            raise _SearchSuperseded()
    def checked_step(*args,**kwargs):
        check_step_cancelled()
        return original(*args,**kwargs)
    checked_step.cache_hit_guard=check_step_cancelled
    simulator.step=checked_step
    started=time.monotonic()
    def publish_progress(packet):
        if _worker_progress_queue is None or cancelled():
            return
        value=dict(packet,generation=generation,worker_id=os.getpid(),
                   strategy=config.get("strategy",f"legacy_{config.get('variant',0)}"),
                   elapsed=round(time.monotonic()-started,3),updated_at=time.monotonic())
        try:
            _worker_progress_queue.put_nowait(value)
        except (Full,OSError,ValueError):
            pass
    preview_route=config.get("route") or {"waypoints":[config["target"]],
                           "segments":[{"no_dash":False,"no_ground":False}]}
    try:
        with progress_reporter(publish_progress if _worker_progress_queue is not None else None,
                               config["start_state"],preview_route):
            return plan_to_route(**config) if "route" in config else plan_to_target(**config)
    except _SearchSuperseded:
        return dict(can_execute=False,reached=False,reason="search_cancelled")
    finally:
        if had_override:
            simulator.step=original
        else:
            del simulator.step


class TargetPlanner:
    """Bounded portfolio, preferring fewer and later dashes on complete routes."""
    def __init__(self,workers=None,seconds=8,expansions=10000,*,pool=None,clock=time.monotonic):
        workers=default_assist_workers() if workers is None else workers
        if type(workers) is not int or not 1<=workers<=61:
            raise ValueError("workers must be an integer in 1..61")
        context=multiprocessing.get_context("spawn")
        self.cancel_epoch=context.Value("q",0)
        self.warm_ready=context.Value("i",0)
        self.warm_release=context.Event()
        self.warm_release.set()
        self.warmup_result=None
        self.progress_queue=context.Queue(maxsize=workers*4)
        self._progress_packets=Queue(maxsize=workers*4)
        self._progress_stop=Event()
        self.search_previews={}
        self.pool=pool if pool is not None else ProcessPoolExecutor(max_workers=workers,mp_context=context,
                    initializer=_initialize_search_worker,
                    initargs=(self.cancel_epoch,self.warm_ready,self.warm_release,self.progress_queue))
        self.workers,self.seconds,self.expansions=workers,seconds,expansions
        self.clock=clock
        self.jobs={}
        self.forecasts=set()
        self.epoch=0
        self.active=False
        self.results=[]
        self.pending=deque()
        self.specs=[]
        self.completed=0
        self.submitted=0
        self.winner=None
        self.map_data=None
        self.hard_deadline=None
        self.deadline=0
        self.preference_candidate=None
        self.preference_until=None
        self._progress_reader=Thread(target=self._read_progress,
            args=(self.progress_queue,self._progress_packets,self._progress_stop),
            daemon=True,name="assist-preview-reader")
        self._progress_reader.start()

    def warmup(self,timeout=20,stop=None):
        """Prestart every process before a game connection can wait for input.

        Constructors stay lazy for offline callers and injected test pools.
        This explicit startup operation must run before any search is active.
        The barrier guarantees distinct processes rather than twelve fast probes
        accidentally executing in only one already-started worker.
        """
        if self.warmup_result is not None:
            return dict(self.warmup_result)
        if self.active or any(not future.done() for future in self.jobs):
            raise RuntimeError("worker warmup must precede search")
        if not math.isfinite(timeout) or timeout<=0:
            raise ValueError("warmup timeout must be positive and finite")
        started=time.monotonic()
        deadline=started+timeout
        futures=[]
        self.warm_ready.value=0
        self.warm_release.clear()
        try:
            for _ in range(self.workers):
                if stop is not None and stop.is_set():
                    return dict(workers=self.workers,ready=self.warm_ready.value,cancelled=True,
                                seconds=round(time.monotonic()-started,4))
                futures.append(self.pool.submit(_warm_search_worker,timeout+5))
            while self.warm_ready.value<self.workers:
                if stop is not None and stop.is_set():
                    return dict(workers=self.workers,ready=self.warm_ready.value,cancelled=True,
                                seconds=round(time.monotonic()-started,4))
                for future in futures:
                    if future.done():
                        future.result()  # Report broken initialization immediately.
                if time.monotonic()>=deadline:
                    raise TimeoutError(f"search worker warmup timed out ({self.warm_ready.value}/{self.workers} ready)")
                time.sleep(.01)
            self.warm_release.set()
            pids={future.result(timeout=max(.01,deadline-time.monotonic())) for future in futures}
            if len(pids)!=self.workers:
                raise RuntimeError("worker warmup did not reach every process")
            self.warmup_result=dict(workers=self.workers,ready=len(pids),worker_pids=sorted(pids),
                                    seconds=round(time.monotonic()-started,4),cancelled=False)
            return dict(self.warmup_result)
        finally:
            self.warm_release.set()
            for future in futures:
                if not future.done():
                    future.cancel()

    def set_map(self,map_data):
        self.map_data=map_data

    def set_deadline(self,deadline):
        self.hard_deadline=deadline

    def _generation(self):
        self.epoch+=1
        self.cancel_epoch.value=self.epoch

    def _prune_stale(self):
        self.forecasts={future for future in self.forecasts if not future.done()}
        for future,metadata in list(self.jobs.items()):
            if metadata["epoch"]!=self.epoch and future.done():
                del self.jobs[future]

    def cancel(self):
        self._generation()
        self.search_previews.clear()
        self.active=False
        self.results=[]
        self.preference_candidate=None
        self.preference_until=None
        self.pending.clear()
        for future in self.forecasts:
            future.cancel()
        self.forecasts={future for future in self.forecasts if not future.done()}
        for future in list(self.jobs):
            future.cancel()
            if future.done():
                del self.jobs[future]

    def forecast_anchor(self,map_path,observation,request_id,frames):
        self._prune_stale()
        future=self.pool.submit(_anchor_job,dict(map_path=str(map_path),
            observation=deepcopy(observation),request_id=request_id,frames=frames,
            generation=self.epoch))
        self.forecasts.add(future)
        return future

    def request(self,map_path,state,target,attempt,forbidden):
        self._prune_stale()
        if self.active:
            return False
        self._generation()
        self.search_previews.clear()
        self.active=True
        self.results=[]
        self.preference_candidate=None
        self.preference_until=None
        self.pending.clear()
        self.completed=self.submitted=0
        self.winner=None
        now=self.clock()
        self.deadline=self.hard_deadline if self.hard_deadline is not None else now+90
        if "waypoints" in target:
            specs=route_strategy_specs(target,state,self.map_data)
            # Round-robin families instead of filling every first slot with
            # slight variations of one algorithm.
            families={}
            for spec in specs:
                families.setdefault(spec.get("portfolio_family",spec["family"]),deque()).append(spec)
            specs=[]
            while any(families.values()):
                for values in families.values():
                    if values:
                        specs.append(values.popleft())
            if attempt and specs:
                offset=(attempt*self.workers)%len(specs)
                specs=specs[offset:]+specs[:offset]
            # A single worker and every retry must still reach the corridor
            # portfolio before spending the request on another long search.
            # A proven refill-guidance proposal also retains an early slot on
            # retries. Every other recipe keeps its relative rotated order.
            specs.sort(key=lambda spec:0 if spec["name"]=="ordered_corridor" else
                       1 if spec.get("portfolio_family")=="refill_relay" else 2)
        else:
            specs=[dict(name=f"legacy_{attempt*self.workers+i}",family="legacy",
                        variant=attempt*self.workers+i) for i in range(self.workers)]
        self.specs=specs
        ordered_full_budget=(len(specs)<=self.workers and specs
                             and all(spec["family"]=="ordered" for spec in specs))
        for spec in specs:
            family=spec["family"]
            scale=(1.7 if family in {"support_graph","zip_graph"} else
                   1.5 if family in {"wall_relay","spring_relay","underhang_relay"} else
                   1.25 if family=="rested_support" else 1)
            scale=max(scale,spec.get("budget_scale",1))
            time_scale=max(scale,spec.get("time_budget_scale",scale))
            config=dict(map_path=str(map_path),start_state=state.to_dict(),
                        seconds=(max(.01,min(self.deadline-now-1,spec.get("max_seconds",self.deadline-now-1))) if ordered_full_budget else
                                 min(spec.get("max_seconds",20),self.seconds*time_scale*(1+attempt*.25))),
                        expansions=int(self.expansions*scale),variant=spec["variant"],
                        forbidden=list(forbidden),_generation=self.epoch)
            config["route" if "waypoints" in target else "target"]=deepcopy(target)
            if "waypoints" in target:
                config["max_attempts"]=1
                config["strategy"]=spec["name"]
            self.pending.append((spec["name"],config))
        self._fill_slots()
        return True

    def _fill_slots(self):
        self._prune_stale()
        while (self.active and self.pending and self.clock()<self.deadline
                and sum(not future.done() for future in self.jobs)<self.workers):
            name,config=self.pending.popleft()
            remaining=self.deadline-self.clock()
            if remaining<=.01:
                self.pending.appendleft((name,config))
                break
            config["seconds"]=min(config["seconds"],remaining)
            try:
                future=self.pool.submit(_target_job,config)
            except Exception as error:
                self.completed+=1
                self.results.append(dict(can_execute=False,reason="worker_error",strategy=name,error=str(error)))
                continue
            self.jobs[future]=dict(epoch=self.epoch,strategy=name)
            self.submitted+=1

    def poll(self):
        self._drain_progress()
        if not self.active:
            self._prune_stale()
            return None
        for future,metadata in list(self.jobs.items()):
            if metadata["epoch"]!=self.epoch or not future.done():
                continue
            del self.jobs[future]
            self.completed+=1
            try:
                result=future.result()
                if not isinstance(result,dict):
                    raise ValueError("invalid worker response")
                if result.get("can_execute"):
                    actions=result.get("actions")
                    expected=result.get("expected")
                    if (result.get("can_execute") is not True or result.get("reached") is not True
                            or not isinstance(actions,list) or not actions
                            or any(type(action) is not int or not 0<=action<=127 for action in actions)
                            or not isinstance(result.get("anchor_state"),dict)
                            or not isinstance(expected,list) or len(expected)!=len(actions)+1):
                        raise ValueError("malformed executable route")
                    result["dash_preference"]=dash_usage(result)
                    result["_dash_preference_key"]=dash_preference_key(result)
                result.setdefault("strategy",metadata["strategy"])
                self.results.append(result)
            except Exception as error:
                self.results.append(dict(can_execute=False,reason="worker_error",strategy=metadata["strategy"],error=str(error)))
        if (self.results and not self.pending
                and not any(metadata["epoch"]==self.epoch for metadata in self.jobs.values())
                and all(result.get("reason")=="unsupported_room_mechanics" for result in self.results)):
            diagnostics=self._failure_diagnostics()
            self.cancel()
            return dict(can_execute=False,reason="unsupported_room_mechanics",
                        terminal_failure=True,budget_exhausted=False,
                        strategy_results=diagnostics)
        if self.clock()>=self.deadline:
            if self.preference_candidate is not None:
                return self._choose_candidate(self.preference_candidate)
            diagnostics=self._failure_diagnostics()
            self.cancel()
            return dict(can_execute=False,reason="no_route_within_budget",budget_exhausted=True,
                        strategy_results=diagnostics)
        ready=[r for r in self.results if r.get("can_execute") is True and r.get("reached") is True
               and isinstance(r.get("actions"),list)]
        if ready:
            chosen=min(ready,key=lambda result:result["_dash_preference_key"])
            self.preference_candidate=chosen
            if self.preference_until is None:
                self.preference_until=min(self.clock()+1.,self.deadline-.1)
            # Zero is already the minimum resource use. Otherwise allow one
            # bounded comparison window for alternative arrivals/backtracking;
            # a complete fallback is retained if those searches fail or expire.
            outstanding=self.pending or any(metadata["epoch"]==self.epoch
                                             for metadata in self.jobs.values())
            if (chosen["dash_preference"]["dash_count"]==0 or not outstanding
                    or self.clock()>=self.preference_until):
                return self._choose_candidate(chosen)
        self._fill_slots()
        if self.pending or any(metadata["epoch"]==self.epoch for metadata in self.jobs.values()):
            return None
        if self.preference_candidate is not None:
            return self._choose_candidate(self.preference_candidate)
        reasons=[r.get("reason","no_route") for r in self.results]
        self.active=False
        return dict(can_execute=False,reason=reasons[0] if reasons else "no_route_within_budget",
                    details=reasons,strategy_results=self._failure_diagnostics(),
                    budget_exhausted=self.deadline-self.clock()<=1.05)

    def _choose_candidate(self,chosen):
        chosen={key:value for key,value in chosen.items() if key!="_dash_preference_key"}
        self.winner=chosen.get("strategy")
        self.cancel()
        return chosen

    def finish_preference_window(self):
        # Exact-frame handoffs have priority over optional quality comparison.
        # Also applies to a result first drained by poll on that same frame.
        self.preference_until=self.clock()

    def _failure_diagnostics(self):
        # Keep bounded search evidence in the live trace without copying routes
        # or complete worker state onto the input bridge's critical path.
        results=[dict(result,search_status="finished") for result in self.results]
        for future,metadata in self.jobs.items():
            if metadata["epoch"]!=self.epoch:
                continue
            if not future.done():
                results.append(dict(strategy=metadata["strategy"],
                    search_status="running" if future.running() else "submitted",
                    reason="search_in_progress_at_stop" if future.running() else "search_queued_in_worker"))
                continue
            try:
                result=future.result()
                if isinstance(result,dict):
                    results.append(dict(result,strategy=result.get("strategy",metadata["strategy"]),
                                        search_status="finished"))
            except Exception as error:
                results.append(dict(strategy=metadata["strategy"],search_status="finished",
                                    reason="worker_error",error=str(error)))
        results.extend(dict(strategy=name,search_status="not_started",
                            reason="strategy_not_started") for name,_ in self.pending)
        return _bounded_strategy_results(results)

    def failure_diagnostics(self):
        """Snapshot finished, running and unstarted work without waiting or submitting."""
        return self._failure_diagnostics()

    @staticmethod
    def _read_progress(source,packets,stop):
        # multiprocessing.Queue.get_nowait() can block after poll sees only a
        # frame header: recv_bytes still waits for the entire pipe payload.
        # Only this daemon may touch that IPC read. The socket thread consumes
        # completed objects from the local bounded Queue instead.
        while not stop.is_set():
            try:
                packet=source.get(timeout=.1)
            except Empty:
                continue
            except (EOFError,OSError,ValueError):
                return
            if stop.is_set():
                return
            try:
                packets.put_nowait(packet)
            except Full:
                # Previews are disposable: retain the newest completed packet
                # without waiting for a game frame to make queue space.
                try:
                    packets.get_nowait()
                except Empty:
                    pass
                try:
                    packets.put_nowait(packet)
                except Full:
                    pass

    def _drain_progress(self):
        # Bound work on the 50ms input bridge: heavy paths are sampled in the
        # workers, and at most eight small preview packets are admitted per tick.
        names={spec["name"] for spec in self.specs}
        for _ in range(8):
            try:
                packet=self._progress_packets.get_nowait()
            except (Empty,OSError,ValueError):
                break
            if (not isinstance(packet,dict) or packet.get("generation")!=self.epoch
                    or packet.get("strategy") not in names
                    or type(packet.get("worker_id")) is not int):
                continue
            raw_paths=packet.get("paths")
            if not isinstance(raw_paths,(list,tuple)):
                raw_paths=[]
            topology=_topology_preview(packet.get("topology"))
            paths=[]
            for path in raw_paths[:2]:
                if not isinstance(path,dict):
                    continue
                raw_points=path.get("points")
                if not isinstance(raw_points,(list,tuple)):
                    continue
                points=[]
                for point in raw_points[:120]:
                    if (not isinstance(point,dict) or not isinstance(point.get("room"),str)
                            or any(type(point.get(axis)) not in (int,float)
                                   or not math.isfinite(point[axis]) for axis in ("x","y"))):
                        continue
                    points.append({key:point[key] for key in ("room","x","y")})
                if points:
                    paths.append(dict(kind=path.get("kind","best"),points=points,
                                      waypoints_reached=path.get("waypoints_reached",0),
                                      local_leg=bool(path.get("local_leg",False))))
            if not paths and topology is None:
                continue
            keep=("generation","worker_id","strategy","elapsed","expansions",
                  "simulated_frames","waypoints_reached","updated_at")
            self.search_previews[packet["worker_id"]]={**{key:packet[key] for key in keep if key in packet},
                                                       "paths":paths}
            if topology is not None:
                self.search_previews[packet["worker_id"]]["topology"]=topology

    def previews(self):
        # Entries are replaced wholesale after admission and are read-only to
        # the GUI, avoiding a deep copy of every polyline on every game frame.
        return list(self.search_previews.values())

    def stats(self):
        running=[metadata["strategy"] for future,metadata in self.jobs.items()
                 if metadata["epoch"]==self.epoch and not future.done()]
        return dict(workers=self.workers,active=len(running),completed=self.completed,
                    total=len(self.specs),queued=len(self.pending),submitted=self.submitted,
                    strategies=[spec["name"] for spec in self.specs],
                    active_strategies=running,winner=self.winner)

    def close(self):
        self._progress_stop.set()
        self.cancel()
        self.pool.shutdown(wait=False,cancel_futures=True)
        # Never join the reader here: a worker terminated halfway through a
        # framed pipe write can leave recv_bytes waiting indefinitely. It is a
        # daemon and will discard an eventual completion after this stop flag.
        self.progress_queue.cancel_join_thread()
        self.progress_queue.close()


@dataclass
class AssistOptions:
    max_search_seconds:float=120
    max_attempts:int=8
    max_execution_retries:int=8
    tolerance:float=10
    max_execution_seconds:float=30
    future_anchor_frames:int=1800
    max_recovery_seconds:float=30


class AssistController:
    """One explicitly confirmed target; never chooses a destination by itself."""
    def __init__(self,map_data,planner,options=None,on_event=None):
        self.map=map_data
        self.simulator=Simulator(map_data)
        self.planner=planner
        if callable(getattr(planner,"set_map",None)):
            planner.set_map(map_data)
        self.options=options or AssistOptions()
        self.on_event=on_event or (lambda item:None)
        self.mode="manual"
        self.message="手动游玩：依次选择路线点，设置分段限制后确认"
        self.target=None
        self.request_id=0
        self.started=0
        self.attempts=0
        self.execution_retries=0
        self.plan=None
        self.index=0
        self.anchor=None
        self.forbidden=set()
        self.observation=None
        self.result=None
        self.strategy_results=[]
        self.events=[]
        self.stable_frames=0
        self.execution_human_epoch=None
        self.execution_spawn=None
        self.executing_started=0
        self.execution_limit=self.options.max_execution_seconds
        self.route_request=None
        self.progress=None
        self.execution_previous=None
        self.execution_deaths=None
        self.pending_candidate=None
        self.anchor_forecast=None
        self.request_human_epoch=None
        self.recovery=None
        self.execution_warnings=[]
        self.warning_kinds=set()

    def clear_forecast(self):
        # A running forecast is read-only and bounded. Discarding its Future
        # prevents a cancelled/replaced request from using the result.
        if self.anchor_forecast is not None:
            self.anchor_forecast.cancel()
        self.anchor_forecast=None
        self.pending_candidate=None

    def prepare_anchor(self,obs):
        """Forecast clocks/subpixel drift off the latency-sensitive socket."""
        room=self.map.rooms[obs["room"]]
        timed=any(entity.kind=="cassetteBlock" for entity in room.entities)
        if timed:
            try:
                current=state_from_observation(self.map,obs)
                timed=not getattr(current,"cassette_stopped",False)
            except RuntimeStateError:
                # Preserve the asynchronous strict error path for unsupported
                # or incomplete active-room observations.
                pass
        if not timed and self.anchor_forecast is None:
            try:
                return make_assist_anchor(self.simulator,obs,request_id=self.request_id)
            except AssistValidationError as error:
                if error.reason!="timed_projection_required":
                    raise
        if self.anchor_forecast is None:
            # A retry must still reach its forecast frame before the request's
            # wall-clock deadline; a fresh fixed 30s wait can exceed that budget.
            remaining=self.options.max_search_seconds-(time.monotonic()-self.started)
            frames=max(1,min(self.options.future_anchor_frames,int(max(0,remaining-.5)*60)))
            forecast_anchor=getattr(self.planner,"forecast_anchor",None)
            if callable(forecast_anchor):
                self.anchor_forecast=forecast_anchor(self.map.source,obs,
                    self.request_id,frames)
                self.event("anchor_forecast_started",source_frame=obs["frame"],
                           future_frames=frames,worker="process")
                return None
            # Small injected planners used by offline callers may have no pool.
            # The live TargetPlanner always uses its prewarmed worker processes.
            future=Future()
            future.set_running_or_notify_cancel()
            self.anchor_forecast=future
            source=deepcopy(obs)
            request_id=self.request_id
            def forecast():
                try:
                    # This simulator is private to this forecast thread.
                    anchor=make_assist_anchor(Simulator(self.map),source,
                        request_id=request_id,future_frames=frames)
                    future.set_result(anchor)
                except Exception as error:
                    future.set_exception(error)
            Thread(target=forecast,daemon=True,name="assist-clock-forecast").start()
            self.event("anchor_forecast_started",source_frame=obs["frame"],
                       future_frames=frames,worker="injected_planner_thread")
            return None
        if not self.anchor_forecast.done():
            return None
        future=self.anchor_forecast
        self.anchor_forecast=None
        anchor=future.result()
        validity=observe_assist_anchor(self.map,anchor,obs,request_id=self.request_id)
        if not validity["valid"]:
            raise AssistValidationError(validity["reason"])
        self.event("anchor_forecast_ready",source_frame=anchor["source_frame"],
                   target_frame=anchor["target_frame"])
        return anchor

    def event(self,name,**data):
        value=dict(event=name,request_id=self.request_id,mode=self.mode,**data)
        self.events.append(value)
        self.events=self.events[-100:]
        self.on_event(value)

    def confirm(self,target):
        self.strategy_results=[]
        self.clear_forecast()
        self.route_request=None
        self.progress=None
        self.execution_previous=None
        self.planner.cancel()
        self.request_id+=1
        try:
            self.target=validate_tile_target(self.map,target)
        except AssistValidationError as error:
            self.target=None
            self.finish(False,"所选位置无效或被永久实心地形占据")
            return
        self.plan=None
        self.anchor=None
        self.index=0
        self.attempts=0
        self.execution_retries=0
        self.forbidden=set()
        self.recovery=None
        self.execution_warnings=[]
        self.warning_kinds=set()
        self.request_human_epoch=(self.observation or {}).get("human_input_epoch")
        self.started=time.monotonic()
        if callable(getattr(self.planner,"set_deadline",None)):
            self.planner.set_deadline(self.started+self.options.max_search_seconds)
        self.mode="waiting"
        self.message="等待玩家落稳；按任意游戏操作键可取消"
        self.result=None
        self.stable_frames=0
        self.event("target_confirmed",target=self.target)

    def confirm_route(self,route):
        """Confirm an entire no-death itinerary; edits never mutate this snapshot."""
        self.strategy_results=[]
        try:
            selected=normalize_route(self.map,route)
        except (AssistValidationError,ValueError,TypeError) as error:
            self.request_id+=1
            self.route_request=None
            self.target=None
            self.progress=None
            self.finish(False,f"路线或分段限制无效：{error}")
            return
        self.confirm(selected["waypoints"][-1])
        if self.target is None:
            return
        self.route_request=deepcopy(selected)
        self.message=f"已确认 {len(selected['waypoints'])} 个路线点；等待稳定起点后搜索整条路线"
        self.event("route_confirmed",route=self.route_request)
        try:
            current=state_from_observation(self.map,self.observation) if self.observation else None
        except RuntimeStateError:
            current=None
        for index,target in enumerate(selected["waypoints"]):
            if cassette_pickup_required(self.map,target,current):
                self.finish(False,f"第 {index+1} 个目标必须触发磁带拾取剧情，当前尚不支持；磁带需手动拾取")
                return

    def cancel(self,reason="已取消辅助，手动控制"):
        self.clear_forecast()
        self.planner.cancel()
        self.plan=None
        self.anchor=None
        self.recovery=None
        self.mode="manual"
        self.message=reason
        self.result="cancelled"
        self.event("cancelled",message=reason)

    def finish(self,success,reason):
        if success:
            self.strategy_results=[]
        self.clear_forecast()
        self.planner.cancel()
        self.plan=None
        self.anchor=None
        self.recovery=None
        self.mode="success" if success else "failed"
        self.result="reached" if success else reason
        self.message=("已连续到达全部路线点，手动控制" if self.route_request else "已到达所选格子，手动控制") if success else f"辅助失败：{reason}。已交还控制"
        self.event("target_reached" if success else "target_failed",reason=reason)
        return PASSIVE

    def execution_warning(self,kind,message,obs,expected=None):
        """A prediction difference is diagnostic, not an input cancellation."""
        if kind not in self.warning_kinds:
            self.warning_kinds.add(kind)
            warning=dict(kind=kind,action_index=self.index,message=message,
                         observed={key:obs.get(key) for key in ("room","x","y")})
            if expected is not None:
                warning["predicted"]={key:expected.get(key) for key in ("room","x","y")}
            self.execution_warnings.append(warning)
            self.event("execution_prediction_warning",**warning)
        self.message=f"{message}；继续当前完整动作，玩家操作可立即取消"

    def blacklist_execution(self,reason):
        if self.plan and self.plan.get("actions"):
            from .room_planning import action_fingerprint
            fingerprint=action_fingerprint(self.plan["actions"])
            self.forbidden.add(fingerprint)
            self.event("execution_candidate_failed",reason=reason,fingerprint=fingerprint,
                       executed_frames=self.index,planned_frames=len(self.plan["actions"]))
            self.execution_retries+=1
        self.plan=None
        self.index=0
        self.progress=None
        self.execution_previous=None

    def restart_search_round(self,reason):
        """Each retry retains the complete route but gets a fresh bounded search."""
        self.clear_forecast()
        self.planner.cancel()
        self.anchor=None
        self.stable_frames=0
        self.progress=None
        self.execution_previous=None
        if (self.attempts>=self.options.max_attempts
                or self.execution_retries>=self.options.max_execution_retries):
            return self.finish(False,"已用完本次候选与重试预算；尚未找到成功路线，不代表目标不可达")
        self.started=time.monotonic()
        if callable(getattr(self.planner,"set_deadline",None)):
            self.planner.set_deadline(self.started+self.options.max_search_seconds)
        self.mode="waiting"
        self.message="保留原整条路线，等待稳定起点后尝试下一条候选"
        self.event("automatic_route_retry",reason=reason,next_attempt=self.attempts+1,
                   excluded_candidates=len(self.forbidden))
        return PASSIVE

    def recover_execution(self,reason,obs,*,allow_completion=False):
        self.clear_forecast()
        self.planner.cancel()
        self.anchor=None
        self.mode="waiting"
        self.stable_frames=0
        self.recovery=dict(reason=reason,started=time.monotonic(),allow_completion=allow_completion)
        if not allow_completion:
            self.blacklist_execution(reason)
        self.message="等待自然复活或落稳，再自动尝试下一条完整路线"
        self.event("execution_recovery_started",reason=reason,allow_completion=allow_completion)
        return PASSIVE

    def observe_recovery(self,obs):
        recovery=self.recovery
        if recovery["allow_completion"]:
            interrupted=(obs.get("dead") or (self.execution_spawn is not None and obs.get("spawn_id")!=self.execution_spawn)
                         or (self.execution_deaths is not None and obs.get("deaths")!=self.execution_deaths))
            if not interrupted and self.route_request:
                self.progress=advance_route_progress(self.route_request,self.progress,
                    self.execution_previous,obs,None)
                self.execution_previous=deepcopy(obs)
                interrupted=bool(self.progress.violation)
                if route_complete(self.route_request,self.progress):
                    return self.finish(True,"reached_after_action_tail")
            elif not interrupted and target_reached(obs,self.target):
                return self.finish(True,"reached_after_action_tail")
            if interrupted:
                self.blacklist_execution("death_or_constraint_during_settle")
                recovery["allow_completion"]=False
        stable=(not obs.get("dead") and can_plan_while_idle(obs))
        self.stable_frames=self.stable_frames+1 if stable else 0
        if self.stable_frames>=3:
            if self.plan:
                self.blacklist_execution(recovery["reason"])
            self.recovery=None
            result=self.restart_search_round(recovery["reason"])
            # The three actual stable observations remain valid; dispatch on
            # the next observation instead of waiting another three ticks.
            if self.mode=="waiting":
                self.stable_frames=3
            return result
        if time.monotonic()-recovery["started"]>self.options.max_recovery_seconds:
            if self.plan:
                self.blacklist_execution(recovery["reason"])
            return self.finish(False,"等待自然复活或稳定起点超时，尚无法开始下一条候选")
        return PASSIVE

    def next_action(self):
        action=self.plan["actions"][self.index]
        if (self.route_request and self.progress is not None
                and self.progress.index<len(self.route_request["segments"])
                and self.route_request["segments"][self.progress.index]["no_dash"] and action&DASH):
            return self.recover_execution("路线下一帧包含当前分段禁止的冲刺",self.observation)
        self.index+=1
        return action

    def observe(self,obs):
        self.observation=obs
        if self.mode in {"manual","success","failed"}:
            return PASSIVE
        if self.request_human_epoch is None:
            self.request_human_epoch=obs.get("human_input_epoch")
        if (obs.get("human_input_mask",0) or obs.get("human_menu_input")
                or (self.request_human_epoch is not None and obs.get("human_input_epoch")!=self.request_human_epoch)):
            if self.anchor is not None:
                self.event("anchor_invalidated",reason="runtime_epoch_changed")
            self.cancel("检测到玩家操作，已取消辅助")
            return PASSIVE
        if obs.get("in_cutscene"):
            self.cancel("剧情正在进行，已交还手动控制")
            return PASSIVE
        if self.recovery is not None:
            return self.observe_recovery(obs)
        if self.mode=="executing" and (obs.get("dead")
                or (self.execution_spawn is not None and obs.get("spawn_id")!=self.execution_spawn)
                or (self.execution_deaths is not None and obs.get("deaths")!=self.execution_deaths)):
            return self.recover_execution("本次路线发生死亡或自然复活",obs)
        if self.mode=="executing" and self.plan:
            expected=self.plan["expected"][min(self.index,len(self.plan["expected"])-1)]
            expected_room=self.map.rooms.get(expected["room"])
            falling_ids=({f"{expected_room.name}:{entity.id}" for entity in expected_room.entities
                          if entity.kind=="fallingBlock"} if expected_room else set())
            if falling_ids:
                try:
                    predicted=sorted([list(row) for row in expected.get("mechanisms",())
                                      if row[0] in falling_ids],key=lambda row:row[0])
                    if {row[0] for row in predicted}!=falling_ids or len(predicted)!=len(falling_ids):
                        raise ValueError("missing predicted falling block")
                    actual=falling_snapshot_from_observation(self.map,obs)
                except (RuntimeStateError,ValueError,TypeError,KeyError,IndexError):
                    self.execution_warning("falling_telemetry","坠落方块遥测或预测不完整",obs,expected)
                else:
                    if actual!=predicted:
                        self.execution_warning("falling_state","实际坠落方块状态偏离预测",obs,expected)
            if expected.get("cassette_clock") is not None or expected.get("cassette_stopped",False):
                try:
                    actual=cassette_snapshot_from_observation(self.map,obs)
                    keys=("cassette_clock","cassette_blocks","player_depth","player_actual_depth")
                    predicted={key:expected[key] for key in keys}
                    if expected.get("cassette_stopped",False):
                        predicted["cassette_stopped"]=True
                    matched=_same(cassette_comparison_snapshot(actual),
                                  cassette_comparison_snapshot(predicted))
                except (RuntimeStateError,ValueError,TypeError,KeyError):
                    self.execution_warning("cassette_telemetry","磁带节拍或方块遥测不完整",obs,expected)
                else:
                    if not matched:
                        self.execution_warning("cassette_state","实际磁带节拍或方块状态偏离预测",obs,expected)
        if self.route_request:
            if self.mode=="executing":
                previous=self.execution_previous
                mask=self.plan["actions"][self.index-1] if self.plan and self.index else None
                self.progress=advance_route_progress(self.route_request,self.progress,previous,obs,mask)
                self.execution_previous=deepcopy(obs)
                if self.progress.violation:
                    reason={"dash_forbidden":"发生了禁止的冲刺", "landing_forbidden":"发生了禁止的落地",
                            "player_died":"玩家死亡", "invalid_input_mask":"动作格式无效"}.get(self.progress.violation,self.progress.violation)
                    return self.recover_execution(f"第 {self.progress.index+1} 段未满足限制：{reason}",obs)
                if route_complete(self.route_request,self.progress):
                    return self.finish(True,"reached")
            else:
                current=start_route_progress(self.route_request,obs)
                if not current.violation and route_complete(self.route_request,current):
                    self.progress=current
                    return self.finish(True,"reached")
        elif target_reached(obs,self.target):
            return self.finish(True,"reached")
        if self.mode!="executing" and time.monotonic()-self.started>self.options.max_search_seconds:
            # This deadline precedes poll(); snapshot finished work before
            # finish() cancels the generation and discards the planner results.
            diagnostics=getattr(self.planner,"failure_diagnostics",None)
            if callable(diagnostics):
                self.strategy_results=_bounded_strategy_results(diagnostics())
            if not self.attempts or self.anchor is None:
                return self.finish(False,"本轮时间预算内未获得可规划的稳定起点")
            return self.restart_search_round("本轮时间预算内未找到可执行路线")
        if self.mode=="executing" and time.monotonic()-self.executing_started>self.execution_limit:
            return self.recover_execution("执行时间预算耗尽",obs)
        if self.plan:
            expected=self.plan["expected"][min(self.index,len(self.plan["expected"])-1)]
            mismatch=(expected["room"]!=obs["room"] or
                      math.hypot(expected["x"]-obs["x"],expected["y"]-obs["y"])>self.options.tolerance)
            if mismatch:
                self.execution_warning("player_position","实际轨迹偏离预测",obs,expected)
            if self.index<len(self.plan["actions"]):
                return self.next_action()
            return self.recover_execution("完整动作已执行，但游戏尚未确认全部目标",obs,allow_completion=True)
        if self.anchor is not None:
            validity=observe_assist_anchor(self.map,self.anchor,obs,request_id=self.request_id)
            if not validity["valid"]:
                self.planner.cancel()
                self.pending_candidate=None
                self.anchor=None
                self.mode="waiting"
                self.message="玩家或机关状态已变化，等待新的稳定起点"
                self.event("anchor_invalidated",reason=validity["reason"])
        if (self.anchor is not None and self.anchor["mode"]=="future"
                and obs["frame"]==self.anchor["target_frame"]):
            finish_window=getattr(self.planner,"finish_preference_window",None)
            if callable(finish_window):
                finish_window()
        result=self.pending_candidate
        if result is None:
            result=self.planner.poll()
        if result is not None:
            if result.get("can_execute") and (not isinstance(result.get("anchor_state"),dict)
                    or not isinstance(result.get("actions"),list) or not isinstance(result.get("expected"),list)
                    or len(result["expected"])!=len(result["actions"])+1
                    or any(type(action) is not int or not 0<=action<=127 for action in result["actions"])):
                return self.finish(False,"搜索结果未通过起点与动作校验")
            if self.route_request and result.get("can_execute") and (
                    not result.get("reached") or result.get("route")!=self.route_request
                    or not isinstance(result.get("waypoint_indices"),list)
                    or len(result["waypoint_indices"])!=len(self.route_request["waypoints"])
                    or any(type(i) is not int or not 0<=i<=len(result["actions"]) for i in result["waypoint_indices"])
                    or result["waypoint_indices"]!=sorted(result["waypoint_indices"])
                    or result["waypoint_indices"][-1]!=len(result["actions"])):
                return self.finish(False,"搜索结果未通过整条路线与顺序校验")
            admission=(admit_assist_candidate(self.map,self.anchor,obs,request_id=self.request_id,
                                              candidate_state=result.get("anchor_state"))
                       if self.anchor is not None else {"accepted":False})
            if admission.get("waiting") and result.get("can_execute"):
                self.pending_candidate=result
                remaining=max(0,math.ceil((self.anchor["target_frame"]-obs["frame"])/60))
                self.message=f"路线已验证，约 {remaining} 秒后按预测状态出发；玩家操作可立即取消"
                return PASSIVE
            if admission.get("waiting") and not result.get("can_execute"):
                # A failed search has no actions to admit and need not keep
                # waiting for the forecast frame to report its failure.
                admission={"accepted":True}
            self.pending_candidate=None
            if not admission["accepted"]:
                self.anchor=None
                self.mode="waiting"
                self.message="起点已变化，按当前位置重新计算"
                self.event("stale_candidate_discarded")
            elif not result.get("can_execute"):
                self.strategy_results=_bounded_strategy_results(result.get("strategy_results",[]))
                self.anchor=None
                self.mode="waiting"
                self.event("offline_attempt_failed",reason=result.get("reason"),
                           strategy_results=self.strategy_results)
                if (result.get("terminal_failure")
                        and result.get("reason")=="unsupported_room_mechanics"):
                    return self.finish(False,"目标区域包含尚未支持的机关，暂时无法验证完整路线")
                return self.restart_search_round("本轮未找到新的完整候选")
            else:
                from .room_planning import action_fingerprint
                if action_fingerprint(result["actions"]) in self.forbidden:
                    self.event("failed_candidate_excluded",fingerprint=action_fingerprint(result["actions"]))
                    return self.restart_search_round("搜索再次返回已实际失败的候选")
                self.strategy_results=[]
                self.plan=result
                self.index=0
                self.mode="executing"
                self.executing_started=time.monotonic()
                self.execution_warnings=[]
                self.warning_kinds=set()
                # Camera transitions/freeze are not player-update frames. Give
                # a fully planned route time for those native animations too.
                self.execution_limit=max(self.options.max_execution_seconds,
                    len(result["actions"])/60+5+2*len(result.get("boundary_indices",[]))) if self.route_request else self.options.max_execution_seconds
                self.execution_human_epoch=obs.get("human_input_epoch")
                self.execution_spawn=obs.get("spawn_id")
                self.execution_deaths=obs.get("deaths")
                self.execution_previous=deepcopy(obs)
                if self.route_request:
                    self.progress=start_route_progress(self.route_request,obs)
                    if self.progress.violation:
                        return self.finish(False,"起点不满足路线分段限制")
                self.message="辅助执行中；按任意游戏操作键或取消按钮即可收回控制"
                self.event("execute_target_route",frames=len(result["actions"]))
                if result["actions"]:
                    return self.next_action()
                return self.finish(target_reached(obs,self.target),"目标未被实际触达")
        stable=can_plan_while_idle(obs) and not obs.get("human_input_mask",0) and not obs.get("human_menu_input")
        self.stable_frames=self.stable_frames+1 if stable else 0
        if not stable and self.anchor_forecast is not None:
            self.clear_forecast()
        if self.anchor is not None and self.anchor["mode"]=="idle" and not stable:
            self.planner.cancel()
            self.anchor=None
            self.mode="waiting"
            self.message="玩家仍在移动，等待新的稳定起点"
        if self.anchor is None and self.stable_frames>=3:
            if self.attempts>=self.options.max_attempts:
                return self.finish(False,"本次未找到可执行路线")
            try:
                anchor=self.prepare_anchor(obs)
            except (AssistValidationError,ValueError,KeyError,TypeError):
                self.message="等待玩家和机关状态稳定，真人输入仍然有效"
                return PASSIVE
            except Exception as error:
                self.event("anchor_forecast_failed",error_type=type(error).__name__,error=str(error))
                return self.finish(False,"本次起点预测失败，请重新确认路线")
            if anchor is None:
                self.message="正在后台预测精确起点，玩家操作可立即取消"
                return PASSIVE
            state=SimState.from_dict(anchor["state"])
            if self.planner.request(self.map.source,state,self.route_request or self.target,self.attempts,self.forbidden):
                self.strategy_results=[]
                self.attempts+=1
                self.anchor=anchor
                self.mode="planning"
                self.message=f"后台搜索第 {self.attempts} 组路线；玩家操作可立即取消"
                self.event("background_target_search",attempt=self.attempts,room=obs["room"],x=obs["x"],y=obs["y"])
        return PASSIVE

    def status(self):
        return dict(mode=self.mode,message=self.message,target=self.target,result=self.result,
                    strategy_results=self.strategy_results,
                    execution_warnings=self.execution_warnings,excluded_candidates=len(self.forbidden),
                    recovering=self.recovery is not None,
                    attempts=self.attempts,request_id=self.request_id,events=self.events,
                    observation=self.observation,route=self.plan.get("expected",[]) if self.plan else [],
                    route_index=self.index,route_request=deepcopy(self.route_request),
                    waypoints=deepcopy(self.route_request["waypoints"]) if self.route_request else [],
                    segments=deepcopy(self.route_request["segments"]) if self.route_request else [],
                    waypoint_index=self.progress.index if self.progress is not None else 0,
                    route_active=bool(self.route_request and self.mode in {"waiting","planning","executing"}),
                    search_previews=self.planner.previews() if callable(getattr(self.planner,"previews",None)) else [],
                    search_stats=self.planner.stats() if callable(getattr(self.planner,"stats",None)) else {})


class SharedAssistState:
    def __init__(self):
        self.lock=Lock()
        self.value=dict(connected=False,mode="disconnected",message="等待辅助模式的游戏连接")
        self.commands=Queue()
        self.stop=Event()

    def update(self,**values):
        with self.lock:
            self.value.update(values)

    def snapshot(self):
        with self.lock:
            return dict(self.value)


def map_for_observation(game_dir,obs):
    maps=Path(game_dir)/"Content"/"Maps"
    name=obs.get("map_name")
    if name:
        candidate=(maps/(str(name).removesuffix(".bin")+".bin")).resolve()
        if candidate.parent==maps.resolve() and candidate.is_file():
            return candidate
    names={0:"Intro",1:"ForsakenCity",2:"OldSite",3:"CelestialResort",4:"GoldenRidge",
           5:"MirrorTemple",6:"Reflection",7:"Summit",8:"Epilogue",9:"Core",10:"LostLevels"}
    area,mode=int(obs["area_id"]),int(obs["area_mode"])
    for path in maps.glob("*.bin"):
        stem=path.stem
        suffix=stem.split("-",1)[-1]
        prefix=stem.split("-",1)[0]
        candidate_mode=1 if prefix.endswith("H") else 2 if prefix.endswith("X") else 0
        if suffix==names.get(area) and candidate_mode==mode:
            return path.resolve()
    raise ValueError("无法找到当前章节的本地地图文件")


def serve_assist(shared,game_dir,port,options=None,workers=None,plan_seconds=8,on_trace=None):
    planner=TargetPlanner(workers=workers,seconds=plan_seconds)
    controller=None
    connection=None
    def trace(kind,**values):
        if on_trace is not None:
            try:
                on_trace(dict(kind=kind,time=time.time(),**values))
            except Exception as error:
                # Developer diagnostics must never interrupt physical passthrough.
                shared.update(trace_error=str(error))
    try:
        shared.update(connected=False,mode="starting",
                      message=f"正在准备 {planner.workers} 个后台搜索进程；游戏仍可手动操作")
        trace("worker_warmup_started",workers=planner.workers)
        warmed=planner.warmup(stop=shared.stop)
        trace("worker_warmup_completed",**warmed)
        if shared.stop.is_set():
            return
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET,getattr(socket,"SO_EXCLUSIVEADDRUSE",socket.SO_REUSEADDR),1)
            listener.bind(("127.0.0.1",port))
            listener.listen(2)
            listener.settimeout(.5)
            shared.update(message=f"等待游戏连接（本机端口 {port}）；进入关卡后自动连接")
            while not shared.stop.is_set():
                try:
                    connection,_=listener.accept()
                except socket.timeout:
                    continue
                activity=MacActivity("Celeste real-time input bridge")
                try:
                    activity.start()
                    trace("realtime_activity",**activity.status())
                    connection.settimeout(2)
                    connection.setsockopt(socket.IPPROTO_TCP,socket.TCP_NODELAY,1)
                    if recv_exact(connection,8)!=b"LIVEINFO":
                        raise ValueError("游戏仍是旧自动闯关模式，请用辅助机启动器更新桥接")
                    size=struct.unpack("<I",recv_exact(connection,4))[0]
                    if not 0<size<=1024*1024:
                        raise ValueError("无效的游戏握手")
                    hello=json.loads(recv_exact(connection,size))
                    if hello.get("assist_protocol_version",0)<1:
                        raise ValueError("当前游戏桥版本较旧：辅助模式需重新启动更新后的游戏")
                    source=map_for_observation(game_dir,hello)
                    map_data=load_map(source)
                    controller=AssistController(map_data,planner,options,
                                                on_event=lambda event:trace("event",event=event))
                    while not shared.commands.empty():
                        try:
                            shared.commands.get_nowait()
                        except Empty:
                            break
                    connection.sendall(b"\x00")
                    trace("hello",observation=hello,map_path=str(source))
                    connection.settimeout(30)
                    shared.update(connected=True,map_path=str(source),map_data=map_data,**controller.status())
                    import select
                    while not shared.stop.is_set():
                        if not select.select([connection],[],[],.25)[0]:
                            continue
                        frame_started=time.perf_counter()
                        kind,obs=read_message(connection)
                        frame_read=time.perf_counter()
                        if kind=="done":
                            connection.sendall(b"\x00")
                            controller.cancel("游戏已交还手动控制；点击右上角「重新连接」恢复连接")
                            break
                        while True:
                            try:
                                command=shared.commands.get_nowait()
                            except Empty:
                                break
                            guard=command.get("guard")
                            if guard is not None and (not isinstance(guard,dict)
                                    or guard.get("request_id")!=controller.request_id
                                    or guard.get("room")!=obs.get("room")
                                    or guard.get("spawn_id")!=obs.get("spawn_id")):
                                controller.event("command_rejected",command_id=command.get("command_id",command.get("id")),
                                                 reason="current_state_changed")
                                continue
                            if command["kind"]=="target":
                                controller.confirm(command["target"])
                            elif command["kind"]=="route":
                                controller.confirm_route(command["route"])
                            elif command["kind"]=="cancel":
                                controller.cancel()
                        current_source=map_for_observation(game_dir,obs)
                        if current_source!=source:
                            controller.cancel("章节已变化，已取消原目标")
                            source=current_source
                            map_data=load_map(source)
                            controller=AssistController(map_data,planner,options,
                                                        on_event=lambda event:trace("event",event=event))
                            shared.update(map_path=str(source),map_data=map_data)
                        previous_mode=controller.mode
                        commands_finished=time.perf_counter()
                        action=controller.observe(obs)
                        observation_finished=time.perf_counter()
                        connection.sendall(bytes([action]))
                        response_sent=time.perf_counter()
                        if on_trace is not None:
                            phases=dict(read_ms=(frame_read-frame_started)*1000,
                                commands_ms=(commands_finished-frame_read)*1000,
                                controller_ms=(observation_finished-commands_finished)*1000,
                                send_ms=(response_sent-observation_finished)*1000)
                            if max(phases.values())>10:
                                trace("bridge_slow_frame",request_id=controller.request_id,
                                      frame=obs.get("frame"),mode=controller.mode,**phases)
                            if controller.mode=="executing" and previous_mode!="executing":
                                trace("plan",request_id=controller.request_id,candidate=controller.plan)
                            trace("state",observation=obs,action=action,mode=controller.mode,
                                  request_id=controller.request_id,route_index=controller.index,
                                  waypoint_index=controller.progress.index if controller.progress else 0)
                        shared.update(connected=True,**controller.status())
                        previous_frame_finished=time.perf_counter()
                        if on_trace is not None and previous_frame_finished-response_sent>.01:
                            trace("bridge_slow_publish",request_id=controller.request_id,
                                  frame=obs.get("frame"),mode=controller.mode,
                                  milliseconds=(previous_frame_finished-response_sent)*1000)
                except (OSError,ValueError,KeyError) as error:
                    trace("connection_error",error_type=type(error).__name__,error=str(error),
                          errno=getattr(error,"errno",None),winerror=getattr(error,"winerror",None),
                          request_id=controller.request_id if controller else None,
                          mode=controller.mode if controller else None)
                    if controller:
                        controller.cancel(f"连接断开：{error}")
                    shared.update(connected=False,mode="disconnected",message=str(error))
                finally:
                    activity.close()
                    trace("realtime_activity_ended",**activity.status())
                    trace("disconnected")
                    if connection:
                        connection.close()
                        connection=None
                    planner.cancel()
                    while not shared.commands.empty():
                        try:
                            shared.commands.get_nowait()
                        except Empty:
                            break
                    if controller:
                        shared.update(connected=False,**controller.status())
    except (OSError,RuntimeError) as error:
        trace("startup_error",error_type=type(error).__name__,error=str(error))
        shared.update(connected=False,mode="failed",message=f"服务无法启动：{error}")
    finally:
        if connection:
            connection.close()
        planner.close()


def start_assist_service(shared,game_dir,port,**kwargs):
    thread=Thread(target=serve_assist,args=(shared,game_dir,port),kwargs=kwargs,daemon=True,name="celeste-assist-bridge")
    thread.start()
    return thread
