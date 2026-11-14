"""Best-effort, bounded visualization of offline search and coarse corridors.

This module publishes positions only. A preview is never an executable plan.
Physical ``paths`` contain actual simulator states. Geometric ``topology`` is
kept separate and never contributes to public-waypoint completion counts.
"""
from collections import OrderedDict
from copy import deepcopy
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import replace
import math
import time

from .physics import SimState

_ACTIVE=ContextVar("celeste_search_progress",default=None)
MAX_CORRIDOR_PREVIEWS=24


def _point(state):
    defaults=dict(room=None,x=None,y=None,ducking=False,on_ground=False,vy=0.,state=0,
                  dash_frames=0,dash_buffer=0,dead=False,dashes=0)
    if isinstance(state,dict):
        return {key:state.get(key,value) for key,value in defaults.items()}
    return {key:getattr(state,key,value) for key,value in defaults.items()}


def _key(state):
    if isinstance(state,dict):
        state=SimState.from_dict(state)
    return replace(state,frame=0,reason="")


def _join(prefix,points):
    result=list(prefix)
    for state in points:
        point=_point(state)
        if not result or point!=result[-1]:
            result.append(point)
    return result


def _route_marks(points,route):
    from .assist_routes import start_route_progress,advance_route_progress
    if not points:
        return 0,set()
    progress=start_route_progress(route,points[0])
    marks={0} if progress.index else set()
    for position in range(1,len(points)):
        after=advance_route_progress(route,progress,points[position-1],points[position])
        if after.index>progress.index:
            marks.add(position)
        progress=after
    return progress.index,marks


def _decimate(points,route,limit=120):
    if not points:
        return [],0,False
    reached,marks=_route_marks(points,route)
    required={0,len(points)-1}|marks
    for index in range(1,len(points)):
        if points[index]["room"]!=points[index-1]["room"]:
            required.update((index-1,index))
    truncated=False
    if len(required)>limit:
        # Keep a genuine anchored prefix if an unusually cyclic candidate has
        # more transition endpoints than the payload limit can represent.
        cutoff=sorted(required)[limit-1]
        points=points[:cutoff+1];required={index for index in required if index<=cutoff}
        reached,_=_route_marks(points,route);truncated=True
    available=[index for index in range(len(points)) if index not in required]
    slots=limit-len(required)
    if slots>=len(available):
        required.update(available)
    elif slots>0:
        required.update(available[min(len(available)-1,int((i+.5)*len(available)/slots))]
                        for i in range(slots))
    return [points[index] for index in sorted(required)],reached,truncated


class _Reporter:
    def __init__(self,callback,start_state,route,interval):
        self.callback=callback
        self.start=SimState.from_dict(start_state) if isinstance(start_state,dict) else start_state
        self.start_key=_key(self.start)
        points=route["waypoints"] if "waypoints" in route else [route]
        limits=route.get("segments",[{} for _ in points])
        self.route=dict(waypoints=points,segments=[{flag:segment.get(flag,False)
                            for flag in ("no_dash","no_ground")} for segment in limits])
        self.interval=max(.1,float(interval))
        self.last_emit=float("-inf")
        self.last_topology_emit=float("-inf")
        self.expansions=self.simulated=0
        self.prefixes=OrderedDict()
        self.prefix_points=0
        self.topology=None
        self.last_paths=[]

    def compose(self,start,points):
        identity=_key(start)
        if identity==self.start_key:
            return _join([_point(self.start)],points),False
        remembered=self.prefixes.get(identity)
        if remembered is None:
            return _join([],[start]+list(points)),True
        self.prefixes.move_to_end(identity)
        prefix,local=remembered
        return _join(prefix,points),local

    def remember(self,start,end,points):
        combined,local=self.compose(start,points)
        # Short prefixes are retained verbatim. Route marks are only needed
        # when points must be removed, or when a preview is actually emitted.
        if len(combined)>240:
            combined,_,truncated=_decimate(combined,self.route,limit=240)
            if truncated:
                return
        identity=_key(end)
        old=self.prefixes.pop(identity,None)
        if old is not None:
            self.prefix_points-=len(old[0])
        self.prefixes[identity]=(combined,local);self.prefix_points+=len(combined)
        while len(self.prefixes)>128 or self.prefix_points>20000:
            _,(removed,_)=self.prefixes.popitem(last=False)
            self.prefix_points-=len(removed)


@contextmanager
def progress_reporter(callback,start_state,route,interval=.4):
    """Install one worker's optional callback; accepts SimState or its dict.

    The callback should publish with put_nowait. Preview construction and callback
    errors are isolated from planning; no drawing failure may reject a route.
    """
    context=None if callback is None else _Reporter(callback,start_state,route,interval)
    token=_ACTIVE.set(context)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


class _Tracker:
    def __init__(self,context,start):
        self.context,self.start=context,start
        self.expansions=self.simulated=0

    def report(self,builder,expansions,simulated_frames,*,remember=False):
        """builder yields (kind, final_state, actual_state_knots), at most two."""
        context=self.context
        context.expansions+=max(0,expansions-self.expansions)
        context.simulated+=max(0,simulated_frames-self.simulated)
        self.expansions,self.simulated=expansions,simulated_frames
        now=time.monotonic()
        due=now-context.last_emit>=context.interval
        if not due and not remember:
            return
        try:
            if not due:
                candidate=next(iter(builder()),None)
                if candidate is not None:
                    _,end,points=candidate
                    context.remember(self.start,end,points)
                return
            candidates=list(builder())[:2]
            context.last_emit=now
            paths=[]
            for kind,_,points in candidates:
                combined,local=context.compose(self.start,points)
                sampled,reached,truncated=_decimate(combined,context.route)
                paths.append(dict(kind=kind,points=[{key:point[key] for key in ("room","x","y")}
                                                   for point in sampled],
                                  waypoints_reached=0 if local else reached,local_leg=local,
                                  truncated=truncated))
            if remember and candidates:
                _,end,points=candidates[0]
                context.remember(self.start,end,points)
            context.last_paths=deepcopy(paths)
            packet=dict(expansions=context.expansions,simulated_frames=context.simulated,
                        waypoints_reached=max((path["waypoints_reached"] for path in paths),default=0),
                        paths=paths)
            if context.topology is not None:
                packet["topology"]=deepcopy(context.topology)
            context.callback(packet)
        except Exception:
            # Telemetry remains strictly optional and cannot alter solver results.
            return


def search_tracker(start_state):
    context=_ACTIVE.get()
    return None if context is None else _Tracker(context,start_state)


def remember_segment(start_state,end_state,states):
    """Compose a real ride/settling segment between two search legs."""
    context=_ACTIVE.get()
    if context is not None:
        try:
            context.remember(start_state,end_state,states)
        except Exception:
            pass


def dash_break_conditions(corridor):
    """Copy bounded geometric requirements without accepting actions or success."""
    raw=corridor.get("conditions",()) if isinstance(corridor,dict) else ()
    if not isinstance(raw,(list,tuple)):
        return []
    result=[]
    seen=set()
    def finite(value):
        try:
            return type(value) in (float,int) and math.isfinite(value)
        except (OverflowError,ValueError):
            return False
    for value in raw[:16]:
        if not isinstance(value,dict) or value.get("kind")!="dash_break" or value.get("can_dash") is not True:
            continue
        identity,room=value.get("identity"),value.get("room")
        rect,approach,direction=value.get("rect"),value.get("approach"),value.get("dash_direction")
        if (not isinstance(identity,str) or not 0<len(identity)<=80 or identity in seen
                or not isinstance(room,str) or not 0<len(room)<=80
                or not isinstance(rect,(list,tuple)) or len(rect)!=4 or not all(map(finite,rect))
                or rect[2]<=rect[0] or rect[3]<=rect[1]
                or not isinstance(approach,dict) or not all(finite(approach.get(axis)) for axis in ("x","y"))
                or not isinstance(direction,dict) or not all(finite(direction.get(axis)) and
                    direction[axis] in (-1,0,1) for axis in ("x","y"))
                or direction["x"]==direction["y"]==0):
            continue
        result.append(dict(kind="dash_break",identity=identity,room=room,rect=list(rect),can_dash=True,
            approach={axis:approach[axis] for axis in ("x","y")},
            dash_direction={axis:direction[axis] for axis in ("x","y")}))
        seen.add(identity)
    return result


def publish_corridor_progress(corridors,active_id=None,*,throttle_seconds=0):
    """Publish the bounded corridor inventory, independently of real paths.

    Each item has id, pending/trying/failed/verified status, reason, and at most
    60 world-feet points. Callers may mark verified only after the complete
    public route passes uninterrupted physical replay, never after one portal.
    Updates are best-effort and cannot change search or execution decisions.
    """
    context=_ACTIVE.get()
    if context is None:
        return
    try:
        if not isinstance(corridors,(list,tuple)):
            return
        selected=[]
        identities=set()
        for corridor in corridors[:MAX_CORRIDOR_PREVIEWS]:
            if not isinstance(corridor,dict):
                continue
            identity=corridor.get("id")
            status=corridor.get("status")
            points=corridor.get("points")
            if (not isinstance(identity,str) or not identity or len(identity)>80 or identity in identities
                    or status not in {"pending","trying","failed","verified"}
                    or not isinstance(points,(list,tuple)) or len(points)<2):
                continue
            if any(not isinstance(point,dict) or not isinstance(point.get("room"),str)
                   or not point["room"] or any(type(point.get(axis)) not in (int,float)
                       or not math.isfinite(point[axis]) for axis in ("x","y")) for point in points):
                continue
            # Preserve exact geometric endpoints; do not manufacture physical
            # states, actions or public-waypoint claims from these vertices.
            if len(points)>60:
                points=[points[round(index*(len(points)-1)/59)] for index in range(60)]
            reason=corridor.get("reason","")
            selected.append(dict(id=identity,status=status,
                reason=reason[:160] if isinstance(reason,str) else "",
                points=[{key:point[key] for key in ("room","x","y")} for point in points]))
            conditions=dash_break_conditions(corridor)
            if conditions:
                selected[-1]["conditions"]=conditions
            for key in ("turns", "expansions"):
                count = corridor.get(key)
                if type(count) is int and 0 <= count <= 10**9:
                    selected[-1][key] = count
            identities.add(identity)
        active_id=active_id if active_id in identities else None
        if context.topology is not None and context.topology["active_id"]!=active_id:
            context.last_paths=[]
        context.topology=dict(active_id=active_id,corridors=selected)
        # Rotation updates the retained inventory immediately. Only transport
        # is throttled; later physical packets still carry the latest counts.
        now=time.monotonic()
        if (type(throttle_seconds) in (int,float) and math.isfinite(throttle_seconds)
                and now-context.last_topology_emit<max(0,throttle_seconds)):
            return
        context.last_topology_emit=now
        paths=deepcopy(context.last_paths)
        context.callback(dict(expansions=context.expansions,simulated_frames=context.simulated,
            waypoints_reached=max((path["waypoints_reached"] for path in paths),default=0),
            paths=paths,topology=deepcopy(context.topology)))
    except Exception:
        return
