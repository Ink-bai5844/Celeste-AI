"""Bounded, target-only offline planning for the human-operated assistant.

The supplied state is the actual player state, never a spawn or a teleport.
The sole objective is the explicitly selected world-map tile. A partial search
is diagnostic only: executable output must replay alive into the requested tile.
"""
from __future__ import annotations

from collections import deque, OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
import heapq
import json
import math
import time

from .assist_validation import AssistValidationError, validate_tile_target
from .physics import SimState
from .planner import (Goal, Heuristic, SearchResult, macros, state_key, search_leg, coverage_goals,
                      resume_search_alternatives, room_is_supported,terminal_goal_crossing_allowed)
from .assist_routes import (normalize_route, start_route_progress, advance_route_progress,
                            route_complete)
from .room_planning import _context, action_fingerprint
from .routes import snapshot
from .strategies import search_goal
from .assist_strategies import (route_strategy_specs, strategy_spec, search_support_graph,
                                search_zip_chain,search_wall_relay,search_spring_relay,
                                search_underhang_relay)
from .search_progress import search_tracker,remember_segment
from .safe_handoff import neutral_handoff_safe
from .underhang_moves import (route_has_underhang, underhang_edges,
                              underhang_detour_edges, underhang_climb_macros)


def _step_started_dash(simulator, before, after):
    """Read the simulator's transient event before another rollout overwrites it."""
    return bool(getattr(simulator,"last_step_started_dash",
                        before.state!=2 and after.state==2))


def target_reached(state, target: dict) -> bool:
    """Check actual/simulated player collider centre against the selected tile.

    Accept either SimState, its serialized dictionary, or live telemetry. The
    controller can use exactly this predicate for actual success verification.
    """
    if target.get("transition_to"):
        from .assist_targets import transition_target_reached
        return transition_target_reached(state,target)
    if isinstance(state,dict):
        room,x,y=state.get("room"),state.get("x"),state.get("y")
        dead=state.get("dead",False)
        ducking=state.get("ducking",state.get("model_state",{}).get("ducking",False))
    else:
        room,x,y=state.room,state.x,state.y
        dead,ducking=state.dead,state.ducking
    if dead or room!=target.get("room") or x is None or y is None:
        return False
    centre_y=float(y)-(3 if ducking else 5.5)
    if target.get("internal_refill_ready"):
        dashes=(state.get("dashes",state.get("model_state",{}).get("dashes",0))
                if isinstance(state,dict) else state.dashes)
        return dashes>0 and math.hypot(float(x)-float(target["x"]),
                                      centre_y-float(target["y"]))<16
    size=float(target.get("tile_size",8))
    return (float(target["x"])<=float(x)<float(target["x"])+size
            and float(target["y"])<=centre_y<float(target["y"])+size)


@dataclass(frozen=True)
class TileTargetGoal(Goal):
    tile_x: float=0
    tile_y: float=0
    tile_size: int=8
    source_room: str=""
    transition_to: str=""
    edge: str=""
    arrival_bounds: tuple=()
    exit_region: tuple=()
    internal_refill_ready: bool=False
    crossing_only: bool=False
    terminal_entry: object=None
    handoff_frames: int=0
    handoff_simulator: object=field(default=None,repr=False,compare=False,hash=False)

    def region_members(self):
        """Return individual lanes, keeping gaps and terminal permissions exact."""
        if self.terminal_entry is not None:
            certificate=self.terminal_entry
            return tuple(dict(x=lane.tile_x,y=lane.tile_y,
                              arrival_bounds=lane.arrival_bounds)
                         for lane in (getattr(certificate,"lanes",()) or (certificate,)))
        return tuple(dict(x=x,y=y,arrival_bounds=bounds) for x,y,bounds in self.exit_region)

    def waypoint(self,state):
        members=self.region_members()
        if not members:
            return self.x,self.y
        points=[]
        for member in members:
            left,top,right,bottom=member["arrival_bounds"]
            if state.room==self.source_room:
                # The longitudinal coordinate is the actual clear lane, which
                # can be narrower than a tile near a permanent obstruction.
                x,y=member["x"]+4,member["y"]+9.5
                if self.edge in {"top","bottom"}:
                    x=(left+right)/2
                else:
                    y=(top+bottom)/2+5.5
            else:
                x,y=(left+right)/2,(top+bottom)/2+5.5
            points.append((x,y))
        return min(points,key=lambda point:(math.hypot(state.x-point[0],state.y-point[1]),point))

    def geometric_hit(self,state):
        if self.crossing_only and self.transition_to and state.reason!="transition":
            # A positional hit several updates after entering through another
            # lane is not the selected exit. Physics clears this event marker
            # at every update; the route replay also verifies the source room.
            return False
        return target_reached(state,{"room":self.source_room or self.room,"x":self.tile_x,
                                     "y":self.tile_y,"tile_size":self.tile_size,
                                     "transition_to":self.transition_to,"edge":self.edge,
                                     "arrival_bounds":self.arrival_bounds,
                                     **({"exit_region":self.region_members()} if self.exit_region else {}),
                                     "internal_refill_ready":self.internal_refill_ready})

    def achieved(self,state):
        return self.geometric_hit(state) and (not self.handoff_frames or
               neutral_handoff_safe(self.handoff_simulator,state,self.handoff_frames))

    def first_hit_rejected(self,state):
        return bool(self.handoff_frames and self.geometric_hit(state) and
                    not neutral_handoff_safe(self.handoff_simulator,state,self.handoff_frames))


def _selected_goal(target):
    if not isinstance(target,dict) or not isinstance(target.get("room"),str):
        raise ValueError("target must contain a room name and world tile origin x/y")
    size=target.get("tile_size",8)
    if isinstance(size,bool) or size!=8:
        raise ValueError("Celeste map tiles have tile_size=8")
    try:
        x,y=float(target["x"]),float(target["y"])
    except (KeyError,TypeError,ValueError) as error:
        raise ValueError("target x/y must be finite world tile origins") from error
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("target x/y must be finite world tile origins")
    normalized={"room":target["room"],"x":x,"y":y,"tile_size":8}
    normalized.update({key:deepcopy(target[key]) for key in ("transition_to","edge","arrival_bounds","exit_region","internal_refill_ready") if key in target})
    bounds=tuple(target.get("arrival_bounds",()))
    gx,gy=((bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2+5.5) if len(bounds)==4 else (x+4,y+9.5)
    goal=TileTargetGoal(kind="target",room=target.get("transition_to") or target["room"],x=gx,y=gy,
                        id=f"target:{target['room']}:{x:g}:{y:g}",
                        tile_x=x,tile_y=y,tile_size=8,source_room=target["room"],
                        transition_to=target.get("transition_to", ""),edge=target.get("edge", ""),
                        arrival_bounds=bounds,
                        exit_region=tuple((member["x"],member["y"],tuple(member["arrival_bounds"]))
                                          for member in target.get("exit_region",())),
                        internal_refill_ready=bool(target.get("internal_refill_ready")))
    return goal,normalized


def plan_to_target(map_path: str, start_state: dict, target: dict,
                   seconds: float=12, expansions: int=6000, variant: int=0,
                   forbidden: list[str] | None=None, max_attempts: int=4,
                   max_frames: int=900) -> dict:
    """Search only the selected tile, using bounded offline recipe retries.

    ``target`` uses world tile-origin coordinates: {room,x,y,tile_size:8}.
    Solid/out-of-bounds/disconnected selections fail explicitly. Success means
    the normal/ducked player collider centre enters that tile, not a substituted
    nearby point. Every executable frame is replayed from ``start_state``.

    No game is contacted, no respawn is requested, and no berry/exploration goal
    is chosen. Actual execution success must be checked with target_reached().
    """
    if not math.isfinite(seconds) or seconds<=0:
        raise ValueError("seconds must be positive and finite")
    for name,value in (("expansions",expansions),("max_attempts",max_attempts),("max_frames",max_frames)):
        if isinstance(value,bool) or not isinstance(value,int) or value<1:
            raise ValueError(f"{name} must be a positive integer")
    goal,selected=_selected_goal(target)
    data,simulator=_context(map_path)
    start=SimState.from_dict(start_state)
    anchor=json.loads(json.dumps(start.to_dict()))
    started=time.monotonic()
    deadline=started+seconds
    blacklist=set(forbidden or [])
    attempts=[]
    expanded=simulated=0
    best=None

    def response(status,reason,**extra):
        result=dict(status=status,reason=reason,target=selected,primary_goal=goal.key,
                    actions=[],expected=[snapshot(start)],final_state=anchor,
                    anchor_state=anchor,reached=False,can_execute=False,runtime_verified=False,
                    allow_room_continuation=False,boundary_indices=[],
                    fingerprint=action_fingerprint([]),attempts=attempts,
                    expansions=expanded,simulated_frames=simulated,
                    wall_seconds=round(time.monotonic()-started,3),
                    objective="user_selected_tile",search_scope="offline_only",
                    requires_room_respawn=False)
        result.update(extra)
        return result

    if start.room not in data.rooms:
        return response("invalid_start","unknown_start_room")
    if start.dead:
        return response("invalid_start","player_is_dead")
    try:
        selected=validate_tile_target(data,selected)
    except AssistValidationError as error:
        return response("invalid_target",error.reason)
    goal,selected=_selected_goal(selected)
    reachable={start.room}
    queue=deque([start.room])
    while queue:
        name=queue.popleft()
        for neighbor in data.neighbors(name):
            if neighbor.spawns and neighbor.name not in reachable:
                reachable.add(neighbor.name)
                queue.append(neighbor.name)
    if goal.room not in reachable:
        return response("invalid_target","target_room_unreachable")
    for name in {start.room,goal.room}:
        if not room_is_supported(simulator,name,start):
            return response("unsupported","unsupported_room_mechanics",
                            capabilities=simulator.capability_report(name))
    if start.cassette_clock is not None and not goal.transition_to:
        goal=replace(goal,handoff_frames=120,handoff_simulator=simulator)
    if goal.achieved(start):
        return response("already_reached","already_at_target",reached=True)
    if blacklist:
        resume_search_alternatives(simulator,start)

    for index in range(max_attempts):
        now=time.monotonic()
        if now>=deadline or expanded>=expansions:
            break
        remaining=max_attempts-index
        attempt_end=now+(deadline-now)/remaining
        attempt_expansions=max(1,(expansions-expanded)//remaining)
        result=search_goal(simulator,start,goal,deadline=attempt_end,
                           max_expansions=attempt_expansions,max_frames=max_frames,
                           variant=(variant+index)%12,allow_unsupported=False)
        expanded+=result.expansions
        simulated+=result.simulated_frames
        attempts.append(dict(variant=(variant+index)%12,reached=result.reached,
                             reason=result.reason,expansions=result.expansions,
                             frames=len(result.actions)))
        distance=math.hypot(result.state.x-goal.x,result.state.y-goal.y)
        if not result.state.dead and (best is None or distance<best[0]):
            best=(distance,result.state)
        if not result.reached:
            continue
        state=start
        actions=[]
        expected=[snapshot(start)]
        boundaries=[]
        dash_frames=[];dash_progress=[];travel=0.
        invalid_input=False
        for mask in result.actions:
            if type(mask) is not int or not 0<=mask<=127:
                invalid_input=True
                break
            previous_state=state
            previous=state.room
            state=simulator.step(state,mask)
            simulated+=1
            if state.room==previous_state.room:
                travel+=math.hypot(state.x-previous_state.x,state.y-previous_state.y)
            if _step_started_dash(simulator,previous_state,state):
                dash_frames.append(len(actions)+1);dash_progress.append(travel)
            if state.dead:
                break
            actions.append(mask)
            expected.append(snapshot(state))
            if state.room!=previous:
                boundaries.append(len(actions))
            if goal.achieved(state) or goal.first_hit_rejected(state):
                break
        fingerprint=action_fingerprint(actions)
        if invalid_input or state.dead or not goal.achieved(state):
            attempts[-1]["replay_rejected"]=True
            if invalid_input:
                attempts[-1]["reason"]="invalid_input_mask"
            continue
        if fingerprint in blacklist:
            attempts[-1]["reason"]="forbidden_route"
            resume_search_alternatives(simulator,start)
            continue
        return response("ready","offline_target_route_found",actions=actions,expected=expected,
                        final_state=json.loads(json.dumps(state.to_dict())),reached=True,
                        can_execute=bool(actions),fingerprint=fingerprint,
                        dash_start_frames=dash_frames,dash_start_progress=dash_progress,
                        boundary_indices=boundaries,allow_room_continuation=bool(boundaries),
                        frames=len(actions),safe_endpoint=bool(state.on_ground and abs(state.vx)<.01))
    preview=None if best is None else dict(room=best[1].room,x=best[1].x,y=best[1].y,
                                         distance=round(best[0],2))
    return response("no_route","no_route_within_budget",best_prediction=preview,
                    budget_exhausted=True)


class _OrderedFrontier:
    """Optional progress-phase quotas retain upstream alternative arrivals."""
    def __init__(self,diverse=False):
        self.diverse=diverse
        self.queues={}
        self.count=0
        self.pops=0

    def push(self,score,index,phase):
        heapq.heappush(self.queues.setdefault(phase if self.diverse else 0,[]),(score,index))
        self.count+=1

    def pop(self):
        phases=sorted(key for key,queue in self.queues.items() if queue)
        phase=(phases[(self.pops//3)%len(phases)] if self.diverse and self.pops%3==2 else phases[-1])
        self.pops+=1
        self.count-=1
        return heapq.heappop(self.queues[phase])

    def trim(self,limit):
        active=[key for key,queue in self.queues.items() if queue]
        quota=max(1,limit//max(1,len(active)))
        self.count=0
        for phase in active:
            self.queues[phase]=heapq.nsmallest(quota,self.queues[phase])
            heapq.heapify(self.queues[phase])
            self.count+=len(self.queues[phase])

    def __len__(self):
        return self.count


_ORDERED_SEARCH_SESSIONS=OrderedDict()


def _wall_setup_macros(simulator,state,heuristic):
    """Retain climb/wait/launch choices that initially move away from a goal.

    A depleted air refill can require catching the wall behind the player,
    climbing while it recharges and jumping outward. Every emitted frame still
    passes the ordinary physics and route-constraint monitor.
    """
    if state.dead or state.on_ground or state.dashes or not hasattr(simulator,"_wall"):
        return
    target_x,_=heuristic.waypoint(state)
    outward=1 if target_x>=state.x else 2
    wall_direction=-1 if outward==1 else 1
    if not any(simulator._wall(state.room,state.x,state.y,wall_direction,distance)
               for distance in (1,6,12,18,24)):
        return
    toward=2 if wall_direction<0 else 1
    # Normal outward wall-jump speed settles to 90px/s. Sample nearby coast
    # times, then release movement to descend into the selected tile. Keeping
    # this setup together prevents the frontier from starving its temporarily
    # higher/farther intermediate states behind already-too-low arrivals.
    coast=max(0,min(72,round(abs(target_x-state.x)/1.5)))
    for duration in (24,48,72,100):
        setup=(64|4|toward,)*duration+(16|outward,)*12
        for offset in (-12,-8,0):
            inputs=setup+(outward,)*max(0,coast+offset)+(0,)*72
            yield inputs,len(inputs)


def _refill_guidance_route(data,start,route):
    """Optional internal refill milestones; the public route never changes.

    This recipe is reserved for unrestricted segments. Constrained routes keep
    their original uninterrupted monitor, including no-ground takeoff history.
    Other portfolio workers search the original goals without these extra
    requirements, so an unnecessary refill cannot exclude their valid routes.
    """
    if any(any(segment.values()) for segment in route["segments"]):
        return route,list(range(1,len(route["waypoints"])+1))
    waypoints=[];segments=[];explicit_indices=[]
    room_name=start.room
    x,y=start.x,start.y-5.5
    for target,limits in zip(route["waypoints"],route["segments"]):
        tx,ty=target["x"]+4,target["y"]+4
        dx,dy=tx-x,ty-y
        length2=dx*dx+dy*dy
        candidates=[]
        if room_name==target["room"] and not target.get("transition_to") and length2>96**2:
            room=data.rooms[room_name]
            for entity in room.entities:
                if entity.kind!="refill":
                    continue
                along=((entity.x-x)*dx+(entity.y-y)*dy)/length2
                distance=abs((entity.x-x)*dy-(entity.y-y)*dx)/math.sqrt(length2)
                if not .08<along<.92 or distance>48:
                    continue
                if min(math.hypot(entity.x-x,entity.y-y),math.hypot(entity.x-tx,entity.y-ty))<20:
                    continue
                point={"room":room_name,"x":room.x+math.floor((entity.x-room.x)/8)*8,
                       "y":room.y+math.floor((entity.y-room.y)/8)*8,"tile_size":8}
                try:
                    point=validate_tile_target(data,point)
                except AssistValidationError:
                    continue
                point["internal_refill_ready"]=True
                candidates.append((along,str(entity.id),point))
        for _,_,point in sorted(candidates)[:3]:
            if not waypoints or point!=waypoints[-1]:
                waypoints.append(point);segments.append(dict(limits))
        waypoints.append(target);segments.append(dict(limits));explicit_indices.append(len(waypoints))
        room_name=target.get("transition_to",target["room"])
        x,y=tx,ty
    return {"waypoints":waypoints,"segments":segments},explicit_indices


def _search_ordered_route(simulator, start, route, goals, *, deadline,
                          max_expansions, max_frames, variant, forbidden,key_precision="coarse",
                          profile="default"):
    """Try a bounded zero-dash route before relaxing that search preference.

    This does not alter segment restrictions. Failure is only a budget result;
    the relaxed frontier can revisit every earlier waypoint with different
    momentum/resources, including spending a dash before a difficult suffix.
    """
    options=dict(max_frames=max_frames,variant=variant,forbidden=forbidden,
                 key_precision=key_precision,profile=profile)
    remaining=max(0.,deadline-time.monotonic())
    all_forbidden=all(segment["no_dash"] for segment in route["segments"])
    probe=None
    if not all_forbidden and remaining>.02 and max_expansions>1:
        probe=_search_ordered_route_pass(simulator,start,route,goals,
                deadline=min(deadline,time.monotonic()+min(.35,remaining*.12)),
                max_expansions=max(1,min(1200,max_expansions//5)),allow_dash=False,**options)
        if probe[2]:
            probe[-1].update(dash_preference="zero_dash_first",zero_dash_probe_reached=True)
            return probe
    result=_search_ordered_route_pass(simulator,start,route,goals,deadline=deadline,
            max_expansions=max(0,max_expansions-(probe[3] if probe else 0)),
            allow_dash=not all_forbidden,**options)
    actions,state,reached,expanded,simulated,phase,details=result
    details.update(dash_preference=("zero_dash_then_relaxed" if probe else
                                  "segment_no_dash" if all_forbidden else "relaxed_without_probe"),
                   zero_dash_probe_reached=False,
                   zero_dash_probe_expansions=probe[3] if probe else 0,
                   zero_dash_probe_simulated_frames=probe[4] if probe else 0)
    return (actions,state,reached,expanded+(probe[3] if probe else 0),
            simulated+(probe[4] if probe else 0),phase,details)


def _search_ordered_route_pass(simulator, start, route, goals, *, deadline,
                          max_expansions, max_frames, variant, forbidden,key_precision="coarse",
                          profile="default",allow_dash=True):
    """One frontier spans all waypoints, retaining alternate arrivals upstream.

    No leg is committed before its suffix is solved. In particular, two arrivals
    at one waypoint with different velocity, dash state or airborne phase remain
    distinct search nodes and can lead to different legal suffixes.
    """
    data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    preview=search_tracker(start)
    original_frame=start.frame
    start=replace(start,frame=0,reason="")
    if not hasattr(simulator,"_planning_identity"):
        simulator._planning_identity=object()
    archive_key=(simulator._planning_identity,start,json.dumps(route,sort_keys=True),
                 variant,max_frames,key_precision,profile,allow_dash)
    session=_ORDERED_SEARCH_SESSIONS.pop(archive_key,None)
    # A directed exit must first approach its source room, even when the player
    # starts in the destination. Merely approaching the destination's entry band
    # would otherwise attract the search to a point that cannot count as a hit.
    if session:
        heuristics,nodes,frontier,costs,best_index,best_rank,completed=session[:7]
        zip_trials,zip_stages=session[7:] if len(session)>7 else ({},[])
    else:
        heuristics=[Heuristic(data,replace(goal,room=goal.source_room,
                        x=goal.tile_x+4,y=goal.tile_y+9.5) if goal.transition_to else goal)
                    for goal in goals]
        progress=start_route_progress(route,start)
        # Extra node fields retain actual dash events and physical travel. The
        # latter breaks ties toward later movement without rewarding idle waits.
        nodes=[(start,progress,-1,(),(),0,0.,0.)]
        best_index=0
        completed=None
        zip_trials,zip_stages={},[]
    def key(state,phase):
        physical=(replace(state,frame=0,reason="") if key_precision=="exact" else state_key(state))
        return phase.index,phase.left_ground,physical
    def score(state,phase):
        if route_complete(route,phase):
            return 0.0
        distance=heuristics[phase.index](state)
        value=distance+64*(len(goals)-phase.index-1)
        if profile=="lookahead" and phase.index+1<len(goals) and distance<64:
            next_goal=goals[phase.index+1]
            value+=.2*heuristics[phase.index+1](state)
            if (next_goal.x-state.x)*state.vx<0:
                value+=abs(state.vx)*.12
        return value
    def actions_for(index):
        runs=[]
        while nodes[index][2]>=0:
            _,_,parent,inputs=nodes[index][:4]
            runs.append(inputs)
            index=parent
        return [mask for run in reversed(runs) for mask in run]
    def preview_paths(preferred=None):
        choices=[("best",best_index if preferred is None else preferred)]
        candidates=[queue[0] for queue in frontier.queues.values() if queue]
        if candidates:
            other=min(candidates)[1]
            if other!=choices[0][1]:
                choices.append(("alternative",other))
        for kind,index in choices:
            end=nodes[index][0];runs=[]
            while nodes[index][2]>=0:
                node=nodes[index]
                runs.append(list(node[4] if len(node)>4 else ())+[node[0]])
                index=node[2]
            yield kind,end,[nodes[0][0]]+[state for run in reversed(runs) for state in run]
    if not session:
        initial_score=score(start,progress)
        frontier=_OrderedFrontier(profile in {"phase_diverse","semantic"})
        frontier.push(initial_score,0,progress.index)
        costs={key(start,progress):(0,0)}
        best_rank=(progress.index,-initial_score)
    expanded=simulated=0
    nested_expanded=nested_simulated=0
    support={start.room:room_is_supported(simulator,start.room,start)}

    def finish(index,reached,actions=None):
        if len(nodes)<=80000:
            _ORDERED_SEARCH_SESSIONS[archive_key]=(heuristics,nodes,frontier,costs,best_index,best_rank,
                                                   index if reached else None,zip_trials,zip_stages)
            while len(_ORDERED_SEARCH_SESSIONS)>6 or (len(_ORDERED_SEARCH_SESSIONS)>1 and
                    sum(len(value[1]) for value in _ORDERED_SEARCH_SESSIONS.values())>60000):
                _ORDERED_SEARCH_SESSIONS.popitem(last=False)
        state,phase,_,_=nodes[index][:4]
        state=replace(state,frame=state.frame+original_frame)
        if preview is not None:
            preview.report(lambda:preview_paths(index),expanded-nested_expanded,
                           simulated-nested_simulated,remember=True)
        details=dict(resumed_frontier=bool(session),frontier_nodes=len(frontier),stored_nodes=len(nodes),
                     waypoints_reached=phase.index,profile=profile,key_precision=key_precision,
                     actual_dash_count=nodes[index][5],dash_movement_progress=nodes[index][6],
                     zip_stages=[stage for stage in zip_stages if stage.get("family")!="support_graph"])
        support_stages=[stage for stage in zip_stages if stage.get("family")=="support_graph"]
        if support_stages:
            details["support_stages"]=support_stages
        return actions or [],state,reached,expanded,simulated,phase,details

    if completed is not None:
        actions=actions_for(completed)
        if action_fingerprint(actions) not in forbidden:
            return finish(completed,True,actions)
    while frontier and expanded<max_expansions and time.monotonic()<deadline:
        if preview is not None:
            preview.report(preview_paths,expanded-nested_expanded,simulated-nested_simulated)
        priority,index=frontier.pop()
        state,phase,_,_=nodes[index][:4]
        elapsed=state.frame-start.frame
        if (nodes[index][5],elapsed)>costs.get(key(state,phase),(math.inf,math.inf)) or elapsed>=max_frames:
            continue
        if route_complete(route,phase):
            actions=actions_for(index)
            if action_fingerprint(actions) not in forbidden:
                return finish(index,True,actions)
            continue
        expanded+=1
        heuristic=heuristics[phase.index]
        limits=route["segments"][phase.index]
        proposals=list(macros(state,heuristic,variant,simulator))
        # Reserve a small number of complete mechanism-guided proposals per
        # phase. They remain ordinary edges in this same frontier: another
        # waypoint arrival can still backtrack after a failed suffix. The
        # shared per-frame monitor below remains the sole progress authority.
        zip_attempt=zip_trials.get(phase.index,0)
        unrestricted=not any(any(segment.values()) for segment in route["segments"])
        cassette_guidance=state.cassette_clock is not None
        underhang_guidance=unrestricted and route_has_underhang(route,state,data)
        if (allow_dash and profile=="semantic" and unrestricted
                and getattr(heuristic.goal,"internal_refill_ready",False)):
            # A refill relay needs coordinated jump/dash approaches, not just
            # short inputs ranked by distance to its tile. Keep several real
            # pickup states (including different momentum/cooldowns) as edges
            # in this same ordered frontier. Public hits and restrictions are
            # still checked on every frame below; no resource is assigned.
            trial_key=("refill",phase.index)
            if zip_trials.get(trial_key,0)<3 and max_expansions-expanded>0:
                from .refill_moves import refill_approach_edges
                zip_trials[trial_key]=zip_trials.get(trial_key,0)+1
                counters=dict(expansions=0,simulated_frames=0)
                contacts=refill_approach_edges(simulator,state,
                    replace(heuristic.goal,kind="refill"),
                    deadline=min(deadline,time.monotonic()+.5),
                    max_frames=min(180,max_frames-elapsed),stats=counters,
                    max_expansions=min(256,max_expansions-expanded),
                    max_edges=6,remember_results=False)
                expanded+=counters["expansions"];simulated+=counters["simulated_frames"]
                nested_expanded+=counters["expansions"];nested_simulated+=counters["simulated_frames"]
                proposals=[(tuple(actions),len(actions)) for actions,_,_ in contacts]+proposals
                zip_stages.append(dict(phase=phase.index,family="refill_approach",
                    reached=bool(contacts),arrivals=len(contacts)))
        if profile=="semantic" and underhang_guidance:
            proposals=list(underhang_climb_macros(simulator,state,heuristic.goal))+proposals
            trial_key=("underhang",phase.index)
            if zip_trials.get(trial_key,0)<3:
                zip_trials[trial_key]=zip_trials.get(trial_key,0)+1
                counters=dict(expansions=0,simulated_frames=0,
                              max_expansions=max(0,max_expansions-expanded))
                guided=[]
                edge_deadline=min(deadline,time.monotonic()+.25)
                for generator in (underhang_edges,underhang_detour_edges):
                    extra=dict(allow_dash=allow_dash) if generator is underhang_detour_edges else {}
                    for actions,end,label in generator(simulator,state,heuristic.goal,
                            deadline=edge_deadline,max_frames=min(220,max_frames-elapsed),stats=counters,**extra):
                        guided.append((tuple(actions),len(actions)))
                        zip_stages.append(dict(phase=phase.index,family="underhang",reached=True,
                                               frames=len(actions),edge=label))
                expanded+=counters["expansions"];simulated+=counters["simulated_frames"]
                nested_expanded+=counters["expansions"];nested_simulated+=counters["simulated_frames"]
                proposals=guided+proposals
        descending=heuristic.goal.y>state.y+32
        support_attempt_limit=1 if descending else 3
        if (profile=="semantic" and unrestricted and cassette_guidance and not state.on_ground
                and nodes[index][2]>=0 and nodes[nodes[index][2]][1].index<phase.index):
            # A safely reached aerial waypoint may coast onto a real support.
            # Offer that complete physical edge before spending a support-graph
            # search on a state with no grounded takeoff. The ordered monitor
            # below still checks every frame and any intervening waypoint hit.
            settled=state
            coast=[]
            for _ in range(min(120,max_frames-elapsed)):
                if time.monotonic()>=deadline:
                    break
                settled=simulator.step(settled,0)
                simulated+=1
                coast.append(0)
                if settled.dead or settled.room!=state.room:
                    break
                if settled.on_ground and settled.vy>=0:
                    proposals=[(tuple(coast),len(coast))]+proposals
                    break
        if (allow_dash and profile=="semantic" and zip_attempt<support_attempt_limit and unrestricted and cassette_guidance
                and state.on_ground and state.vy>=0 and max_expansions-expanded>0):
            zip_trials[phase.index]=zip_attempt+1
            remaining=deadline-time.monotonic()
            # These are proposals in the same ordered frontier, not separately
            # executable legs. An intermediate safe arrival is useful near a
            # pickup; the exact waypoint state still starts the next segment.
            guided=replace(heuristic.goal,handoff_frames=120,handoff_simulator=simulator)
            leg=search_support_graph(simulator,state,guided,
                    # A descent already has the ordered input frontier. A short
                    # relay proposal is useful, but repeated long ascent-style
                    # staging can consume its entire continuation budget.
                    deadline=min(deadline,time.monotonic()+min(1. if descending else 8.,max(0,remaining)*.6)),
                    max_expansions=min(4000,max_expansions-expanded),
                    max_frames=max_frames-elapsed,variant=4,
                    mode="underhang" if guided.y>state.y+32 else "climb")
            expanded+=leg.expansions;simulated+=leg.simulated_frames
            nested_expanded+=leg.expansions;nested_simulated+=leg.simulated_frames
            zip_stages.append(dict(phase=phase.index,family="support_graph",reached=leg.reached,
                    frames=len(leg.actions),stages=getattr(leg,"strategy_details",{}).get("internal_stages",[])))
            if leg.reached and leg.actions:
                proposals=[(tuple(leg.actions),len(leg.actions))]+proposals
        elif (allow_dash and profile=="semantic" and zip_attempt<3 and not cassette_guidance and not underhang_guidance
                and not any(any(segment.values()) for segment in route["segments"])
                and any(entity.kind=="zipMover" and entity.nodes
                        for entity in data.rooms[state.room].entities)
                and max_expansions-expanded>0):
            zip_trials[phase.index]=zip_attempt+1
            remaining=deadline-time.monotonic()
            guided=heuristic.goal
            leg=search_zip_chain(simulator,state,guided,
                    deadline=min(deadline,time.monotonic()+min(8.,max(0,remaining)*.3)),
                    max_expansions=min(4000,max_expansions-expanded),
                    max_frames=max_frames-elapsed,variant=0,
                    ride_frames=(24,16,8)[zip_attempt])
            expanded+=leg.expansions;simulated+=leg.simulated_frames
            nested_expanded+=leg.expansions;nested_simulated+=leg.simulated_frames
            zip_stages.append(dict(phase=phase.index,ride_frames=(24,16,8)[zip_attempt],
                          reached=leg.reached,frames=len(leg.actions),
                          stages=getattr(leg,"strategy_details",{}).get("internal_stages",[])))
            if leg.actions:
                candidates=[leg.actions]+list(getattr(leg,"alternative_actions",[]))[:2]
                proposals=[(tuple(actions),len(actions)) for actions in candidates]+proposals
        if profile=="micro" and (heuristic(state)<40 or abs(state.vy)<30):
            target_x,_=heuristic.waypoint(state)
            direction=1 if target_x>=state.x else 2
            proposals=[(mask,1) for mask in (0,direction,3-direction,16|direction,64|direction,32|4|direction)]+proposals
        if profile in {"refill_wait","semantic"}:
            waits=sorted({min(120,max(1,timer)) for _,timer in state.refill_timers if 0<timer<1000})
            previous_goal=goals[phase.index-1] if phase.index else None
            setups=(list(_wall_setup_macros(simulator,state,heuristic))
                    if previous_goal and abs(state.x-previous_goal.x)<32
                    and abs(state.y-previous_goal.y)<24 else [])
            proposals=setups+[(mask,duration) for duration in waits for mask in (0,64)]+proposals
        # All profiles try ordinary motion first; semantic proposals remain
        # full physical edges and are checked against every route restriction.
        proposals.sort(key=lambda item:any(action&32 for action in item[0])
                       if isinstance(item[0],tuple) else bool(item[0]&32))
        for mask,duration in proposals:
            masks=mask if isinstance(mask,tuple) else [mask]*duration
            if not allow_dash and any(action&32 for action in masks):
                continue
            if limits["no_dash"] and not isinstance(mask,tuple) and mask&32:
                continue
            successor,current=state,phase
            inputs=[]
            crossings=[]
            dash_count,dash_progress,travel=nodes[index][5:8]
            for action in masks[:max_frames-elapsed]:
                if time.monotonic()>=deadline:
                    break
                previous=successor
                successor=simulator.step(previous,action)
                if successor.room==previous.room:
                    travel+=math.hypot(successor.x-previous.x,successor.y-previous.y)
                if _step_started_dash(simulator,previous,successor):
                    dash_count+=1
                    dash_progress+=travel
                    if not allow_dash:
                        simulated+=1
                        break
                if successor.room!=previous.room:
                    crossings.extend((previous,successor))
                simulated+=1
                inputs.append(action)
                current=advance_route_progress(route,current,previous,successor,action)
                if successor.room not in support:
                    support[successor.room]=room_is_supported(simulator,successor.room,successor)
                terminal_entry=(route_complete(route,current) and
                     terminal_goal_crossing_allowed(simulator,goals[-1],previous,successor))
                if current.violation or (not support[successor.room] and not terminal_entry):
                    break
                if current.index!=phase.index:
                    # Re-evaluate both goals and constraints immediately at the
                    # exact hit frame, retaining this state's full momentum.
                    break
            if (not inputs or current.violation or not allow_dash and dash_count
                    or (not support.get(successor.room,False) and not terminal_entry)):
                continue
            if route_complete(route,current) and goals[-1].first_hit_rejected(successor):
                continue
            cost=successor.frame-start.frame
            history_cost=(dash_count,cost)
            successor_key=key(successor,current)
            if history_cost>=costs.get(successor_key,(math.inf,math.inf)):
                continue
            costs[successor_key]=history_cost
            child=len(nodes)
            nodes.append((successor,current,index,tuple(inputs),tuple(crossings),dash_count,dash_progress,travel))
            h=score(successor,current)
            if route_complete(route,current):
                actions=actions_for(child)
                if action_fingerprint(actions) not in forbidden:
                    return finish(child,True,actions)
                continue
            rank=(current.index,-h)
            if rank>best_rank:
                best_index,best_rank=child,rank
            weight=.03 if variant%2==0 else .18
            frontier.push(cost*weight+h*(1.4+(variant%6)*.15)+24*dash_count
                          -.04*min(64.,dash_progress),child,current.index)
        # A deadline can interrupt one macro before later proposals from this
        # node have been expanded. Keep that node for the next slice; already
        # explored successor keys prune duplicate edges on resumption.
        if time.monotonic()>=deadline:
            frontier.push(priority,index,phase.index)
        if len(frontier)>(80000 if profile=="wide" else 40000):
            frontier.trim(60000 if profile=="wide" else 20000)
    return finish(best_index,False)


@dataclass(frozen=True)
class RestingSupportGoal(Goal):
    required_dashes: int=1

    def achieved(self,state):
        return (not state.dead and state.room==self.room and state.on_ground
                and abs(state.x-self.x)<.01 and abs(state.y-self.y)<1
                and abs(state.vx)<.01 and abs(state.vy)<.01
                and state.dashes>=self.required_dashes and state.stamina>=110
                and state.prev_input==0)


def _search_single_assist(simulator,start,goal,*,deadline,max_expansions,max_frames,variant,
                          support_index=None,suffix_kind=None):
    """Use optional internal ledges while keeping one user-selected destination.

    Distance-field progress can penalize the higher ledge needed to climb out of
    a pocket. A dedicated recipe retains these temporarily backward/upward
    choices instead of restricting supports to immediately improving distance.
    The proposal is accepted only after the whole route passes ordered replay.
    """
    data=simulator.map_data if hasattr(simulator,"map_data") else simulator.map
    if goal.transition_to:
        # The ordinary multi-room heuristic aims at the boundary at the current
        # height. For a selected exit *lane* that can steer beneath a closed wall
        # indefinitely. Guide toward the explicit source-side opening instead;
        # TileTargetGoal.achieved still requires the actual destination entry.
        goal=replace(goal,room=goal.source_room,x=goal.tile_x+4,y=goal.tile_y+9.5)
    candidates=[]
    distant=(goal.room!=start.room or math.hypot(goal.x-start.x,goal.y-start.y)>96)
    if variant%3==1 and distant:
        for support in coverage_goals(data.rooms[start.room]):
            rise=start.y-support.y
            horizontal=abs(support.x-start.x)
            if 24<=rise<=128 and horizontal<=160:
                candidates.append((math.hypot(horizontal,rise)-rise*2,support.key,support))
    if candidates:
        candidates.sort(key=lambda row:row[:2])
        choice=(variant//6) if support_index is None else support_index
        support=candidates[choice%min(3,len(candidates))][-1]
        if variant%6==4:
            # Merely brushing a ledge within the ordinary 12px surface tolerance
            # can leave a different velocity/resource phase for a 2px change of
            # initial position. This recipe explicitly searches a rested landing
            # near the support centre; it never assigns coordinates or timers.
            support=RestingSupportGoal("surface",support.room,support.x,support.y,
                                       id=support.key+":rested",
                                       required_dashes=getattr(simulator,"max_dashes",1))
        now=time.monotonic()
        support_fraction=.6 if isinstance(support,RestingSupportGoal) else .3
        first=search_leg(simulator,start,support,deadline=now+(deadline-now)*support_fraction,
                         max_expansions=max(1,max_expansions//3),max_frames=max_frames,
                         variant=4 if isinstance(support,RestingSupportGoal) else 0,
                         allow_unsupported=False)
        if first.reached and len(first.actions)<max_frames:
            # First contact sets on_ground at the end of the update; native
            # dash refill happens on the following grounded NormalUpdate. A
            # manually selected intermediate point used to get that free tick
            # while the human confirmed the next command. Keep it inside this
            # single continuous route using real neutral simulation instead.
            settled=first.state
            settling=[]
            settling_states=[settled]
            maximum=getattr(simulator,"max_dashes",1)
            def ready_on_support(state):
                return (state.dashes>=maximum and state.stamina>=110
                        and abs(state.vx)<.01 and abs(state.vy)<.01 and state.prev_input==0)
            if settled.on_ground and settled.vy>=0 and not ready_on_support(settled):
                for _ in range(min(40,max_frames-len(first.actions))):
                    successor=simulator.step(settled,0)
                    if (successor.dead or successor.room!=settled.room
                            or not successor.on_ground):
                        break
                    settled=successor
                    settling.append(0)
                    settling_states.append(settled)
                    if ready_on_support(settled):
                        remember_segment(first.state,settled,settling_states)
                        first=replace(first,actions=first.actions+settling,state=settled,
                                      simulated_frames=first.simulated_frames+len(settling))
                        break
            # Alternate a direct suffix with refill/landing strategies. A high
            # ledge can make an air refill accessible even when it was not a
            # useful first subgoal from the original low starting position.
            semantic_suffix=variant%6==4 and suffix_kind!="direct"
            suffix_search=search_goal if semantic_suffix else search_leg
            second=suffix_search(simulator,first.state,goal,deadline=deadline,
                                max_expansions=max(1,max_expansions-first.expansions),
                                max_frames=max_frames-len(first.actions),
                                variant=5 if semantic_suffix else 0,allow_unsupported=False)
            return SearchResult(first.actions+second.actions,second.state,second.reached,
                                first.expansions+second.expansions,
                                first.simulated_frames+second.simulated_frames,second.reason)
        fallback=search_leg(simulator,start,goal,deadline=deadline,
                            max_expansions=max(1,max_expansions-first.expansions),
                            max_frames=max_frames,variant=0,allow_unsupported=False)
        return replace(fallback,expansions=first.expansions+fallback.expansions,
                       simulated_frames=first.simulated_frames+fallback.simulated_frames)
    if (variant%3==2 or variant%6==4) and suffix_kind!="direct":
        return search_goal(simulator,start,goal,deadline=deadline,max_expansions=max_expansions,
                           max_frames=max_frames,variant=5 if variant%6==4 else variant,
                           allow_unsupported=False)
    return search_leg(simulator,start,goal,deadline=deadline,max_expansions=max_expansions,
                      max_frames=max_frames,variant=variant-(variant%3),allow_unsupported=False)


def _search_named_strategy(simulator,start,goal,spec,*,deadline,max_expansions,max_frames):
    options=dict(deadline=deadline,max_expansions=max_expansions,max_frames=max_frames)
    name,family=spec["name"],spec["family"]
    if family=="rested_support":
        return _search_single_assist(simulator,start,goal,variant=4,
                    support_index=spec["support_index"],
                    suffix_kind="direct" if name.endswith("direct") else "semantic",**options)
    guided=(replace(goal,room=goal.source_room,x=goal.tile_x+4,y=goal.tile_y+9.5)
            if goal.transition_to else goal)
    if family=="direct":
        return search_leg(simulator,start,guided,variant=spec["variant"],allow_unsupported=False,**options)
    if family=="semantic":
        return search_goal(simulator,start,guided,variant=spec["variant"],allow_unsupported=False,**options)
    if family=="wall_relay":
        return search_wall_relay(simulator,start,guided,variant=spec["variant"],**options)
    if family=="spring_relay":
        return search_spring_relay(simulator,start,guided,variant=spec["variant"],**options)
    if family=="underhang_relay":
        return search_underhang_relay(simulator,start,guided,variant=spec["variant"],**options)
    if family in {"support_graph","zip_graph"}:
        if name in {"zip_board_ride","zip_chain_fine","zip_chain_semantic"}:
            return search_zip_chain(simulator,start,guided,variant=spec["variant"],**options)
        mode={"zip_board_ride":"zip","zip_chain_graph":"zip_chain"}.get(name,name.removeprefix("support_beam_"))
        return search_support_graph(simulator,start,guided,mode=mode,variant=spec["variant"],**options)
    raise ValueError(f"Strategy {name} needs the ordered route frontier")


def plan_to_route(map_path: str, start_state: dict, route: dict,
                  seconds: float=12, expansions: int=6000, variant: int=0,
                  forbidden: list[str] | None=None, max_attempts: int=4,
                  max_frames: int=1800, strategy: str | None=None) -> dict:
    """Plan an entire ordered, constrained route without any intermediate reset.

    Every segment starts at the previous waypoint's actual simulated hit state.
    A successful return includes a second, uninterrupted frame-by-frame replay
    proving order, survival, and all segment constraints from the supplied anchor.
    No prefix is executable when the complete route cannot be found in budget.
    """
    if not math.isfinite(seconds) or seconds<=0:
        raise ValueError("seconds must be positive and finite")
    for name,value in (("expansions",expansions),("max_attempts",max_attempts),("max_frames",max_frames)):
        if type(value) is not int or value<1:
            raise ValueError(f"{name} must be a positive integer")
    spec=strategy_spec(strategy) if strategy is not None else None
    data,simulator=_context(map_path)
    start=SimState.from_dict(start_state)
    anchor=json.loads(json.dumps(start.to_dict()))
    started=time.monotonic()
    deadline=started+seconds
    attempts=[]
    expanded=simulated=0
    selected=route
    details={}

    def response(status,reason,**extra):
        result=dict(status=status,reason=reason,route=selected,actions=[],
                    expected=[snapshot(start)],final_state=anchor,anchor_state=anchor,
                    reached=False,can_execute=False,runtime_verified=False,
                    boundary_indices=[],waypoint_indices=[],allow_room_continuation=False,
                    fingerprint=action_fingerprint([]),attempts=attempts,expansions=expanded,
                    simulated_frames=simulated,wall_seconds=round(time.monotonic()-started,3),
                    objective="user_selected_route",search_scope="offline_only",
                    requires_room_respawn=False,strategy=strategy or "legacy_variants",
                    strategy_family=spec["family"] if spec else "legacy",
                    strategy_details=details)
        result.update(extra)
        return result

    if start.room not in data.rooms or start.dead:
        return response("invalid_start","player_is_dead" if start.dead else "unknown_start_room")
    try:
        selected=normalize_route(data,route)
    except AssistValidationError as error:
        return response("invalid_target",error.reason)
    unrestricted_single=(len(selected["waypoints"])==1 and not any(selected["segments"][0].values()))
    if spec and spec["family"]!="ordered" and not unrestricted_single:
        return response("invalid_strategy","strategy_requires_unrestricted_single_target")
    progress=start_route_progress(selected,start)
    if progress.violation:
        return response("invalid_start",progress.violation)
    if route_complete(selected,progress):
        return response("already_reached","already_at_route",reached=True,
                        waypoint_indices=[0]*len(selected["waypoints"]))
    goals=[replace(_selected_goal(point)[0],crossing_only=bool(point.get("transition_to")))
           for point in selected["waypoints"]]
    if start.cassette_clock is not None and not goals[-1].transition_to:
        goals[-1]=replace(goals[-1],handoff_frames=120,handoff_simulator=simulator)
    reachable={start.room}
    queue=deque([start.room])
    while queue:
        name=queue.popleft()
        for neighbor in data.neighbors(name):
            if neighbor.spawns and neighbor.name not in reachable:
                reachable.add(neighbor.name)
                queue.append(neighbor.name)
    if any(goal.room not in reachable for goal in goals):
        return response("invalid_target","target_room_unreachable")
    terminal_certificate=None
    last=selected["waypoints"][-1]
    if last.get("transition_to") and not room_is_supported(simulator,last["transition_to"]):
        from .terminal_entry import terminal_entry_certificate
        terminal_certificate=terminal_entry_certificate(simulator,last,is_terminal=True)
        if terminal_certificate is not None:
            goals[-1]=replace(goals[-1],terminal_entry=terminal_certificate)
    required_rooms={start.room,*(goal.room for goal in goals[:-1])}
    if terminal_certificate is None:
        required_rooms.add(goals[-1].room)
    for name in required_rooms:
        if not room_is_supported(simulator,name,start):
            return response("unsupported","unsupported_room_mechanics",
                            capabilities=simulator.capability_report(name))
    blacklist=set(forbidden or [])
    best=None
    def corridor_verdict(success,reason):
        if not (spec and spec["name"]=="ordered_corridor" and details.get("topology")):
            return
        from .search_progress import publish_corridor_progress
        topology=deepcopy(details["topology"])
        selected_corridor=details.get("selected_corridor_id")
        for corridor in topology.get("corridors",[]):
            if corridor.get("id")==selected_corridor:
                corridor["status"]="verified" if success else "failed"
                corridor["reason"]=reason
        details["topology"]=topology
        publish_corridor_progress(topology.get("corridors",[]),active_id=selected_corridor)

    for attempt in range(max_attempts):
        now=time.monotonic()
        if now>=deadline or expanded>=expansions:
            break
        remaining=max_attempts-attempt
        attempt_variant=spec["variant"] if spec else (variant+attempt)%12
        attempt_deadline=now+(deadline-now)/remaining
        attempt_expansions=max(1,(expansions-expanded)//remaining)
        if spec and spec["name"] in {"ordered_mechanism","ordered_mechanism_capability"}:
            from .mechanism_planning import search_mechanism_routes
            actions,final,reached,used,frames,phase,details=search_mechanism_routes(
                simulator,start,selected,goals,deadline=attempt_deadline,
                max_expansions=attempt_expansions,max_frames=max_frames,
                variant=attempt_variant,use_detours=spec["name"]=="ordered_mechanism",forbidden=blacklist)
        elif spec and spec["name"]=="ordered_regional":
            from .regional_planning import search_regional_routes
            actions,final,reached,used,frames,phase,details=search_regional_routes(
                simulator,start,selected,goals,deadline=attempt_deadline,
                max_expansions=attempt_expansions,max_frames=max_frames,
                variant=attempt_variant,forbidden=blacklist)
        elif spec and spec["name"]=="ordered_corridor":
            from .corridor_planning import search_corridor_routes
            actions,final,reached,used,frames,phase,details=search_corridor_routes(
                simulator,start,selected,goals,deadline=attempt_deadline,
                max_expansions=attempt_expansions,max_frames=max_frames,
                variant=attempt_variant,forbidden=blacklist)
        elif (unrestricted_single and (not spec or spec["family"]!="ordered")
                and (not goals[0].transition_to or start.room==goals[0].source_room)):
            # A lone target benefits from the established geometry strategies:
            # a ledge, refill or spring can be a useful *internal* search stage.
            # The user still selects just one point and the resulting complete
            # route must pass the same uninterrupted ordered replay below.
            if spec:
                candidate=_search_named_strategy(simulator,start,goals[0],spec,deadline=attempt_deadline,
                                                max_expansions=attempt_expansions,max_frames=max_frames)
            else:
                candidate=_search_single_assist(simulator,start,goals[0],deadline=attempt_deadline,
                                               max_expansions=attempt_expansions,max_frames=max_frames,
                                               variant=attempt_variant)
            details=getattr(candidate,"strategy_details",{})
            actions,final,reached=candidate.actions,candidate.state,candidate.reached
            used,frames=candidate.expansions,candidate.simulated_frames
            phase=start_route_progress(selected,start)
            if candidate.reached:
                # Diagnostic only; actual directed-exit/order admission remains
                # the second replay below, including the source crossing frame.
                phase=type(phase)(len(goals),not final.on_ground)
        else:
            guidance=selected
            guidance_goals=goals
            explicit_indices=None
            profile=spec.get("ordered_profile","default") if spec else "default"
            if profile=="semantic":
                guidance,explicit_indices=_refill_guidance_route(data,start,selected)
                guidance_goals=[replace(goal,crossing_only=bool(goal.transition_to))
                                for goal in (_selected_goal(point)[0] for point in guidance["waypoints"])]
                if terminal_certificate is not None:
                    guidance_goals[-1]=replace(guidance_goals[-1],terminal_entry=terminal_certificate)
            actions,final,reached,used,frames,phase,details=_search_ordered_route(
                simulator,start,guidance,guidance_goals,deadline=attempt_deadline,
                max_expansions=attempt_expansions,max_frames=max_frames,
                variant=attempt_variant,forbidden=blacklist,
                key_precision=spec.get("key_precision","coarse") if spec else "coarse",
                profile=profile)
            if explicit_indices is not None:
                details.update(internal_waypoints=len(guidance["waypoints"])-len(selected["waypoints"]),
                               guidance_waypoints=guidance["waypoints"],
                               guidance_waypoints_reached=phase.index)
                phase=type(phase)(sum(index<=phase.index for index in explicit_indices),
                                  phase.left_ground,phase.violation)
                details["waypoints_reached"]=phase.index
        expanded+=used
        simulated+=frames
        attempts.append(dict(variant=attempt_variant,strategy=strategy or "legacy_variants",reached=reached,
                             expansions=used,frames=len(actions),waypoints_reached=phase.index))
        if best is None or phase.index>best[0]:
            best=(phase.index,final)
        if not reached:
            continue
        state=start
        phase=start_route_progress(selected,state)
        expected=[snapshot(state)]
        waypoints=[0]*phase.index
        boundaries=[]
        dash_frames=[];dash_progress=[];travel=0.
        replay=[]
        replay_support={start.room:True}
        for action in actions:
            previous=state
            state=simulator.step(previous,action)
            simulated+=1
            if state.room==previous.room:
                travel+=math.hypot(state.x-previous.x,state.y-previous.y)
            if _step_started_dash(simulator,previous,state):
                dash_frames.append(len(replay)+1);dash_progress.append(travel)
            before_index=phase.index
            phase=advance_route_progress(selected,phase,previous,state,action)
            if phase.violation:
                break
            if state.room not in replay_support:
                replay_support[state.room]=room_is_supported(simulator,state.room,state)
            if (not replay_support[state.room] and not (route_complete(selected,phase)
                    and terminal_goal_crossing_allowed(simulator,goals[-1],previous,state))):
                break
            replay.append(action)
            expected.append(snapshot(state))
            waypoints.extend([len(replay)]*(phase.index-before_index))
            if state.room!=previous.room:
                boundaries.append(len(replay))
            if route_complete(selected,phase):
                break
        fingerprint=action_fingerprint(replay)
        if (not route_complete(selected,phase) or len(replay)!=len(actions)
                or fingerprint in blacklist):
            attempts[-1]["replay_rejected"]=True
            corridor_verdict(False,"整条路线复验未通过")
            continue
        if goals[-1].first_hit_rejected(state):
            attempts[-1]["replay_rejected"]="unsafe_cassette_handoff"
            corridor_verdict(False,"终点松手后的安全检查未通过")
            continue
        corridor_verdict(True,"整条路线已通过逐帧重放")
        return response("ready","offline_route_found",actions=replay,expected=expected,
                        final_state=json.loads(json.dumps(state.to_dict())),reached=True,
                        can_execute=bool(replay),waypoint_indices=waypoints,
                        boundary_indices=boundaries,allow_room_continuation=bool(boundaries),
                        fingerprint=fingerprint,frames=len(replay),
                        dash_start_frames=dash_frames,dash_start_progress=dash_progress,
                        neutral_handoff_frames=goals[-1].handoff_frames,
                        terminal_entry_only=terminal_certificate is not None)
    return response("no_route","no_route_within_budget",budget_exhausted=True,
                    best_prediction=None if best is None else dict(waypoints_reached=best[0],
                                  room=best[1].room,x=best[1].x,y=best[1].y))
